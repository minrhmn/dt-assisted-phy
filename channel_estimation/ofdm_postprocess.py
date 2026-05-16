#!/usr/bin/env python3
"""
ofdm_postprocess_v3.py — Process one RX capture -> SNR + BER.
Supports QPSK and 16-QAM (reads modulation from TX npz).

Usage:
  python3 ofdm_postprocess_v3.py --rx rx_capture.npz --tx tx_waveform_qpsk_50m.npz
  python3 ofdm_postprocess_v3.py --rx rx_capture.npz --tx tx_waveform_16qam_25m.npz --noise rx_noise.npz --plot
"""

import argparse, os, sys
import numpy as np
from scipy.signal import fftconvolve, find_peaks
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demod16 import get_demod as _get_16qam_demod
from load_capture import load_samples

parser = argparse.ArgumentParser(description="OFDM post-processing v3 (multi-mod)")
parser.add_argument("--rx", type=str, required=True)
parser.add_argument("--tx", type=str, required=True)
parser.add_argument("--noise", type=str, default=None, help="Noise-only capture for power-based SNR")
parser.add_argument("--skip-frames", type=int, default=5)
parser.add_argument("--plot", action="store_true")
args = parser.parse_args()

# ── Load ─────────────────────────────────────────────────────────────

samples, rx_data = load_samples(args.rx)
rate = float(rx_data["rate"])

tx_data = np.load(args.tx)
sc2_freq = tx_data["sc2_freq"]
p0_freq = tx_data["p0_freq"]
p1_freq = tx_data["p1_freq"]
data_freq = tx_data["data_freq"]
data_bits = tx_data["data_bits"]
occupied_pos = tx_data["occupied_pos"]
const_map = tx_data["const_map"]
FFT = int(tx_data["fft"])
CP = int(tx_data["cp"])
FRAME_LEN = int(tx_data["frame_len"])
N_DATA_SYM = int(tx_data["n_data_sym"])
MOD = str(tx_data["mod"])
BPS = int(tx_data["bits_per_symbol"])
CONST_CONV = str(tx_data["const_convention"]) if "const_convention" in tx_data.files else "legacy"
_demod16 = _get_16qam_demod(CONST_CONV)

SYM = FFT + CP
occ_bins = occupied_pos % FFT
N_OCC = len(occ_bins)

sc2_td = np.fft.ifft(sc2_freq, n=FFT)
sc2_template = np.concatenate([sc2_td[-CP:], sc2_td]).astype(np.complex64)

print(f"RX: {len(samples)} samples ({len(samples)/rate:.3f} s)")
print(f"TX: {MOD.upper()}, {BPS} bps, FFT={FFT}, rate={rate/1e6:.0f} MHz, frame={FRAME_LEN}")

# ── Noise PSD ────────────────────────────────────────────────────────

noise_psd = None
if args.noise:
    ns, nd = load_samples(args.noise)
    n_blocks = len(ns) // FFT
    noise_fft = np.zeros(N_OCC, dtype=np.float64)
    for b in range(n_blocks):
        X = np.fft.fft(ns[b*FFT:(b+1)*FFT])
        noise_fft += np.abs(X[occ_bins])**2
    noise_psd = noise_fft / n_blocks
    print(f"Noise floor: {10*np.log10(np.mean(noise_psd)+1e-20):.1f} dB")

# ── Demapper ─────────────────────────────────────────────────────────

def qam_demod(syms):
    """Hard-decision demapper for QPSK or 16-QAM."""
    if BPS == 2:  # QPSK
        bits = np.zeros(len(syms) * 2, dtype=np.int8)
        bits[0::2] = (syms.imag < 0).astype(np.int8)
        bits[1::2] = (syms.real < 0).astype(np.int8)
        return bits
    else:  # 16-QAM — dispatched to legacy or gray demapper by TX convention
        return _demod16(syms)

# ── Reconstruct TX symbols ──────────────────────────────────────────

X_true = np.zeros((N_DATA_SYM, N_OCC), dtype=np.complex64)
for d in range(N_DATA_SYM):
    bits = data_bits[d]
    if BPS == 2:
        idx = bits[0::2] * 2 + bits[1::2]
    else:
        idx = bits[0::4] * 8 + bits[1::4] * 4 + bits[2::4] * 2 + bits[3::4]
    X_true[d] = const_map[idx]

# ── SC2 xcorr frame sync ────────────────────────────────────────────

xcorr = np.abs(fftconvolve(samples, np.conj(sc2_template[::-1]), mode='valid'))
peak_thresh = 0.5 * np.max(xcorr)
peaks, _ = find_peaks(xcorr, height=peak_thresh, distance=int(FRAME_LEN * 0.8))
frame_starts = peaks
spacings = np.diff(peaks)
print(f"\nSync: {len(peaks)} frames")
if len(spacings) > 0:
    print(f"  Spacing: mean={np.mean(spacings):.1f}, std={np.std(spacings):.1f}, expected={FRAME_LEN}")

# ── CFO estimation ───────────────────────────────────────────────────

cfos = []
for fs in frame_starts:
    if fs + FRAME_LEN > len(samples):
        continue
    p0_seg = samples[fs + 1*SYM + CP : fs + 1*SYM + CP + FFT]
    p1_seg = samples[fs + 14*SYM + CP : fs + 14*SYM + CP + FFT]
    H0 = np.fft.fft(p0_seg)[occ_bins] / p0_freq[occ_bins]
    H1 = np.fft.fft(p1_seg)[occ_bins] / p1_freq[occ_bins]
    delta_phase = np.angle(np.sum(H1 * np.conj(H0)))
    delta_t = 13 * SYM / rate
    cfos.append(delta_phase / (2 * np.pi * delta_t))

cfo_global = np.median(cfos) if cfos else 0.0
print(f"CFO: {cfo_global:.1f} Hz")
n_vec = np.arange(len(samples))
samples_corr = samples * np.exp(-1j * 2 * np.pi * cfo_global / rate * n_vec)

# ── Frame extraction ─────────────────────────────────────────────────

valid = (frame_starts >= 0) & (frame_starts + FRAME_LEN <= len(samples_corr))
frame_starts = frame_starts[valid][args.skip_frames:]
n_frames = len(frame_starts)

Y_p0 = np.zeros((n_frames, FFT), dtype=np.complex64)
Y_p1 = np.zeros((n_frames, FFT), dtype=np.complex64)
Y_data = np.zeros((n_frames, N_DATA_SYM, FFT), dtype=np.complex64)

for i, fs in enumerate(frame_starts):
    Y_p0[i] = np.fft.fft(samples_corr[fs + 1*SYM + CP : fs + 1*SYM + CP + FFT])
    Y_p1[i] = np.fft.fft(samples_corr[fs + 14*SYM + CP : fs + 14*SYM + CP + FFT])
    for d in range(N_DATA_SYM):
        sym_idx = 2 + d
        Y_data[i, d] = np.fft.fft(samples_corr[fs + sym_idx*SYM + CP : fs + sym_idx*SYM + CP + FFT])

print(f"Extracted {n_frames} frames")

# ── LS channel estimation ────────────────────────────────────────────

H_ls = Y_p0[:, occ_bins] / p0_freq[occ_bins]

# ── H-variance SNR ───────────────────────────────────────────────────

H_ref = H_ls[0]
H_aligned = np.zeros_like(H_ls)
for i in range(n_frames):
    phi = np.angle(np.sum(H_ls[i] * np.conj(H_ref)))
    H_aligned[i] = H_ls[i] * np.exp(-1j * phi)
H_mean = np.mean(H_aligned, axis=0)
H_var = np.mean(np.abs(H_aligned - H_mean)**2, axis=0)
snr_hvar = np.mean(np.abs(H_mean)**2 / (H_var + 1e-20))
ebn0_hvar = snr_hvar / (BPS * (SYM / FFT))
ebn0_hvar_db = 10 * np.log10(ebn0_hvar + 1e-20)

# ── Power-based SNR ──────────────────────────────────────────────────

snr_pwr_db = np.nan
ebn0_pwr_db = np.nan
if noise_psd is not None:
    rx_pwr = np.mean(np.abs(Y_data[:, :, occ_bins])**2, axis=(0, 1))
    snr_pwr_sc = np.maximum((rx_pwr - noise_psd) / (noise_psd + 1e-20), 1e-10)
    snr_pwr = np.mean(snr_pwr_sc)
    snr_pwr_db = 10 * np.log10(snr_pwr + 1e-20)
    ebn0_pwr = snr_pwr / (BPS * (SYM / FFT))
    ebn0_pwr_db = 10 * np.log10(ebn0_pwr + 1e-20)

# ── CPE + ZF + BER + EVM ─────────────────────────────────────────────

total_err_cpe = 0
total_err_nocpe = 0
total_bits = 0
evm_sig = 0.0
evm_err = 0.0

for i in range(n_frames):
    H_p1_obs = Y_p1[i, occ_bins] / p1_freq[occ_bins]
    phi_p1 = np.angle(np.sum(H_p1_obs * np.conj(H_ls[i])))

    for d in range(N_DATA_SYM):
        sym_idx = 2 + d
        alpha = (sym_idx - 1) / 13
        phi_d = alpha * phi_p1
        Y_d = Y_data[i, d, occ_bins]

        Y_corrected = Y_d * np.exp(-1j * phi_d)
        X_hat = Y_corrected / (H_ls[i] + 1e-20)
        bits_hat = qam_demod(X_hat)
        total_err_cpe += np.sum(bits_hat != data_bits[d])

        evm_err += np.sum(np.abs(X_hat - X_true[d])**2)
        evm_sig += np.sum(np.abs(X_true[d])**2)

        X_hat_nocpe = Y_d / (H_ls[i] + 1e-20)
        bits_nocpe = qam_demod(X_hat_nocpe)
        total_err_nocpe += np.sum(bits_nocpe != data_bits[d])

        total_bits += len(data_bits[d])

ber_cpe = total_err_cpe / total_bits if total_bits > 0 else 1.0
ber_nocpe = total_err_nocpe / total_bits if total_bits > 0 else 1.0
snr_evm = evm_sig / (evm_err + 1e-20)
ebn0_evm = snr_evm / (BPS * (SYM / FFT))
ebn0_evm_db = 10 * np.log10(ebn0_evm + 1e-20)

# ── Results ──────────────────────────────────────────────────────────

pwr_str = f"{snr_pwr_db:.1f}" if not np.isnan(snr_pwr_db) else "N/A"
pwr_ebn0_str = f"{ebn0_pwr_db:.1f}" if not np.isnan(ebn0_pwr_db) else "N/A"

print(f"\n{'='*60}")
print(f"RESULTS: {MOD.upper()}, {rate/1e6:.0f} MHz ({n_frames} frames, {total_bits} bits)")
print(f"{'='*60}")
print(f"  H-var SNR:   {10*np.log10(snr_hvar+1e-20):.1f} dB   Eb/N0: {ebn0_hvar_db:.1f} dB")
print(f"  EVM SNR:     {10*np.log10(snr_evm+1e-20):.1f} dB   Eb/N0: {ebn0_evm_db:.1f} dB")
print(f"  Power SNR:   {pwr_str} dB   Eb/N0: {pwr_ebn0_str} dB")
print(f"  BER w/ CPE:  {ber_cpe:.6f}")
print(f"  BER w/o CPE: {ber_nocpe:.6f}")
print(f"{'='*60}")

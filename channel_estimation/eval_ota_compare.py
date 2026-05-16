#!/usr/bin/env python3
"""OTA comparison: 5 receivers — hybrid R_ee strategy.

Receivers:
  1. LS                  — P0-P1 interpolated baseline
  2. LMMSE               — Empirical R_hh from 20 sounding positions
  3. DT-Derived LMMSE    — Global R_hh from dense RT grid + Q-D model
  4. DT-Assisted LS      — Per-position H_dt + empirical R_ee
  5. DT-Assisted LMMSE   — Per-position H_dt + empirical R_ee (full matrix)

Usage:
    python eval_ota_compare.py                       # all positions, both mods
    python eval_ota_compare.py --pos p1 --mod qpsk   # single
    python eval_ota_compare.py --bw 25m              # different bandwidth
"""

import os, sys, json, argparse, glob, re, time
import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import fftconvolve, find_peaks

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)

from config.ofdm_params import (FFT, CP, SYM, N_OCC, N_DATA_SYM, N_GRID_SYM,
                    OCC_BINS, FRAME_SYMS, BW_OPTIONS, BW_NORM,
                    DATA_DIR, MODELS_DIR, RESULTS_DIR,
                    tx_waveform_path, ota_captures_dir,
                    error_stats_path)
CIR_MAX_TAP = 50
FRAME_LEN = FRAME_SYMS * SYM
RESULTS_V4_DIR = '/home/native/project/results_v4'

ALL_POS = ['p1', 'p2', 'p3', 'p4', 'p5']
ALL_MOD = ['qpsk', '16qam']

RECEIVER_NAMES = [
    'ls',
    'lmmse_empirical',
    'dt_derived_lmmse',
    'dt_assisted_ls',
    'dt_assisted_lmmse',
]

QAM16_THRESH = 2.0 / np.sqrt(10)

_P0_OFF = 1 * SYM + CP
_P1_OFF = 14 * SYM + CP
_DATA_OFFS = np.array([(2 + d) * SYM + CP for d in range(N_DATA_SYM)])
_SYM_OFFS = np.concatenate([[_P0_OFF], _DATA_OFFS, [_P1_OFF]])
_FFT_IDX = np.arange(FFT)


# ── Helpers ──────────────────────────────────────────────────────────────

def extract_txg(path):
    m = re.search(r'txg(\d+)', os.path.basename(path))
    return int(m.group(1)) if m else -1


def load_samples(rx_path):
    d = np.load(rx_path)
    if 'samples' in d:
        return d['samples'].astype(np.complex64).flatten(), float(d['rate'])
    if 'samples_i16' in d:
        raw = d['samples_i16'].astype(np.float32) * float(d['sample_scale'])
        return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64), float(d['rate'])
    raise ValueError(f'Unknown format: {list(d.keys())}')


def cir_to_cfr(a_re, a_im, tau, bw_hz=50e6):
    a = (a_re + 1j * a_im).astype(np.complex64)
    freqs = np.fft.fftfreq(FFT, d=1.0 / bw_hz)
    f_occ = freqs[OCC_BINS]
    H = np.sum(a[:, None] * np.exp(-1j * 2 * np.pi * f_occ[None, :] * tau[:, None]),
               axis=0)
    return H.astype(np.complex64)


def align_timing_phase(H_prior, H_target):
    """Align H_prior to H_target in CIR timing and first-path phase.

    Does NOT scale energy — returns H at its original power level.
    """
    cir_p = np.array(ifft(H_prior))
    cir_t = np.array(ifft(H_target))
    pk_p = int(np.argmax(np.abs(cir_p[:CIR_MAX_TAP])))
    pk_t = int(np.argmax(np.abs(cir_t[:CIR_MAX_TAP])))
    shift = pk_t - pk_p
    if shift != 0:
        cir_p = np.roll(cir_p, shift)
        if shift > 0:
            cir_p[:shift] = 0
        else:
            cir_p[shift:] = 0
    cir_p *= np.exp(-1j * (np.angle(cir_p[pk_t]) - np.angle(cir_t[pk_t])))
    cir_p[CIR_MAX_TAP:] = 0
    return fft(cir_p).astype(np.complex128)


def compute_noise_psd(noise_path):
    samples, _ = load_samples(noise_path)
    n_blocks = len(samples) // FFT
    psd = np.zeros(N_OCC, dtype=np.float64)
    for b in range(n_blocks):
        X = np.fft.fft(samples[b * FFT:(b + 1) * FFT])
        psd += np.abs(X[OCC_BINS])**2
    return psd / n_blocks


def compute_tau_rms(h_occ):
    h_full = np.zeros(FFT, dtype=np.complex64)
    h_full[OCC_BINS] = h_occ
    cir = np.fft.ifft(h_full)
    p = np.abs(cir[:CIR_MAX_TAP])**2
    tot = np.sum(p) + 1e-30
    taps = np.arange(CIR_MAX_TAP, dtype=np.float64)
    mu = np.sum(p * taps) / tot
    return float(np.sqrt(np.sum(p * (taps - mu)**2) / tot))


# ── LMMSE infrastructure ────────────────────────────────────────────────

def build_lmmse_filter(eigvecs, eigvals, N0):
    gamma = eigvals / (eigvals + N0 + 1e-30)
    W = (eigvecs * gamma[None, :]) @ eigvecs.conj().T
    post_eig = eigvals * N0 / (eigvals + N0 + 1e-30)
    err_var = np.real(np.diag((eigvecs * post_eig[None, :]) @ eigvecs.conj().T))
    return W.astype(np.complex128), err_var.astype(np.float64)


def compute_r_hh_empirical(bw_label):
    bw_mhz = int(float(bw_label.replace('m', '')))
    bw_tag = f'bw{bw_mhz}p0'
    H_all = []
    for i in range(1, 21):
        fpath = os.path.join(RESULTS_V4_DIR, f'posp{i}', f'{bw_tag}.npz')
        if not os.path.exists(fpath):
            continue
        d = np.load(fpath)
        cfr = d['cfr_avg'].astype(np.complex128)
        cir = np.fft.ifft(cfr)
        H = fft(np.pad(cir[:CIR_MAX_TAP], (0, FFT - CIR_MAX_TAP)))[OCC_BINS]
        H_all.append(H)
    H_all = np.array(H_all)
    R_sample = (H_all.conj().T @ H_all) / len(H_all)
    mu = np.real(np.trace(R_sample)) / N_OCC
    return (R_sample + mu * np.eye(N_OCC)).astype(np.complex128)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos', default=None)
    parser.add_argument('--mod', default=None)
    parser.add_argument('--bw', default='50m')
    parser.add_argument('--skip-frames', type=int, default=5)
    args = parser.parse_args()

    positions = [args.pos] if args.pos else ALL_POS
    mods = [args.mod] if args.mod else ALL_MOD
    bw_label = args.bw
    bw_hz = BW_OPTIONS[bw_label]

    print(f'OTA Comparison — {bw_label}, positions={positions}, mods={mods}')
    print(f'Receivers: {RECEIVER_NAMES}\n')

    # ── Receiver 2: Empirical R_hh from 20 sounding positions ──
    print('Building R_hh (Empirical from sounding)...')
    R_emp = compute_r_hh_empirical(bw_label)
    eigvals_emp, eigvecs_emp = np.linalg.eigh(R_emp)
    eigvals_emp = np.maximum(eigvals_emp[::-1].real, 0.0)
    eigvecs_emp = eigvecs_emp[:, ::-1]
    print(f'  trace={np.sum(eigvals_emp):.4f} '
          f'({10 * np.log10(np.sum(eigvals_emp) / N_OCC + 1e-30):.1f} dB/sc)')

    # ── Receiver 3: Global R_hh from dense grid + Q-D ──
    rhh_path = os.path.join(DATA_DIR, f'rhh_qd_global_{bw_label}.npz')
    if not os.path.exists(rhh_path):
        print(f'  ERROR: {rhh_path} not found. Run precompute_rhh_qd.py first.')
        return
    rhh_data = np.load(rhh_path)
    rhh_eigvals_raw = rhh_data['R_hh_eigvals'].astype(np.float64)
    rhh_eigvecs = rhh_data['R_hh_eigvecs'].astype(np.complex128)
    rhh_trace_per_sc = float(rhh_data['trace_per_sc'])
    # Normalize: eigenvalues such that trace/N = 1
    rhh_eigvals_norm = rhh_eigvals_raw / rhh_trace_per_sc
    print(f'  R_hh global: trace/sc={rhh_trace_per_sc:.2e} '
          f'({10 * np.log10(rhh_trace_per_sc + 1e-30):.1f} dB), '
          f'{int(rhh_data["n_positions"])} grid positions')

    # ── Receivers 4 & 5: Empirical R_ee for DT-Assisted ──
    err_path = error_stats_path(bw_label)
    if os.path.exists(err_path):
        err_data = np.load(err_path)
        U_ee_emp = err_data['R_EE_eigvecs'].astype(np.complex128)
        lam_ee_emp = err_data['R_EE_eigvals'].astype(np.float64)
        P_cal = float(err_data['P_cal'])
        ree_trace_emp = float(np.sum(lam_ee_emp)) / N_OCC
        print(f'  Empirical R_ee: trace/sc={ree_trace_emp:.4f}, P_cal={P_cal:.4f}, '
              f'eff_rank_90={int(err_data["eff_rank_90"])}')
    else:
        U_ee_emp = lam_ee_emp = None
        P_cal = 1.0
        print(f'  WARNING: {err_path} not found')

    # ── D7R CIR for 5 OTA positions ──
    cir_path = os.path.join(DATA_DIR, 'cir_ota_d7r_cal.npz')
    cir_data = np.load(cir_path) if os.path.exists(cir_path) else None
    print(f'  D7R CIR: {"loaded" if cir_data else "NOT FOUND"}')

    # ── Process all captures ──
    all_results = []
    sync_ok = 0
    sync_fail = 0
    t0 = time.time()

    for pos in positions:
        H_dt_occ_rt = None
        P_dt_rt = None

        if cir_data is not None and f'{pos}_a_re' in cir_data:
            H_dt_occ_rt = cir_to_cfr(cir_data[f'{pos}_a_re'],
                                      cir_data[f'{pos}_a_im'],
                                      cir_data[f'{pos}_tau'],
                                      bw_hz=bw_hz).astype(np.complex128)
            P_dt_rt = float(np.mean(np.abs(H_dt_occ_rt)**2))
            print(f'\n{pos}: P_dt_rt={10 * np.log10(P_dt_rt + 1e-30):.1f} dB')

        for mod in mods:
            bps = 4 if mod == '16qam' else 2
            tx_data = dict(np.load(tx_waveform_path(mod, bw_label)))
            data_dir = ota_captures_dir(pos, bw_label, mod)

            noise_path = os.path.join(data_dir, f'rx_noise_{bw_label}.npz')
            if not os.path.exists(noise_path):
                print(f'  SKIP {pos}/{mod}: no noise file')
                continue
            noise_psd = compute_noise_psd(noise_path)
            N0_noise = float(np.mean(noise_psd))

            rx_files = sorted(glob.glob(os.path.join(data_dir, 'rx_*_txg*.npz')))
            rx_files = [f for f in rx_files if os.path.getsize(f) > 5e6]
            rx_files.sort(key=extract_txg, reverse=True)

            for rx_path in rx_files:
                txg = extract_txg(rx_path)
                result = process_capture_v2(
                    rx_path, tx_data,
                    H_dt_occ_rt, P_dt_rt,
                    N0_noise, noise_psd,
                    eigvecs_emp, eigvals_emp,
                    rhh_eigvecs, rhh_eigvals_norm,
                    U_ee_emp, lam_ee_emp, P_cal,
                    mod, bw_label,
                    skip_frames=args.skip_frames,
                )

                if result is None:
                    sync_fail += 1
                    print(f'  {pos}/{mod}/txg{txg}: SYNC FAIL')
                    continue
                sync_ok += 1

                result['position'] = pos
                result['modulation'] = mod
                result['file'] = os.path.basename(rx_path)
                all_results.append(result)

                ber_strs = []
                mse_strs = []
                for name in RECEIVER_NAMES:
                    b = result['ber'].get(name, float('nan'))
                    ber_strs.append(f'{name}={b:.5f}')
                    m = result['mse'].get(name, float('nan'))
                    nmse_db = 10 * np.log10(m / (result['h_power'] + 1e-30) + 1e-30)
                    mse_strs.append(f'{name}={nmse_db:.1f}')
                print(f'  {pos}/{mod}/txg{txg}: Eb/N0={result["ebn0_db"]:.1f}dB, '
                      f'corr={result["dt_corr"]:.2f}, '
                      f'alpha_cap={result.get("alpha_cap_db", 0):.1f}dB, '
                      f'{result["n_frames"]}fr')
                print(f'    BER: {", ".join(ber_strs)}')
                print(f'    NMSE(dB): {", ".join(mse_strs)}')

    elapsed = time.time() - t0
    print(f'\n{"="*140}')
    print(f'SYNC: {sync_ok}/{sync_ok + sync_fail} passed '
          f'({100 * sync_ok / max(sync_ok + sync_fail, 1):.0f}%)')
    print(f'Total time: {elapsed:.1f}s')

    # ── Summary tables ──
    if not all_results:
        print('No results.')
        return

    for mod in mods:
        mod_results = [r for r in all_results if r['modulation'] == mod]
        if not mod_results:
            continue

        mod_results.sort(key=lambda r: (r['position'], r['ebn0_db']))

        print(f'\n{"="*140}')
        print(f'  {mod.upper()} — {len(mod_results)} captures')
        print(f'{"="*140}')
        hdr = f'{"Pos":>4s} {"TxG":>4s} {"Eb/N0":>7s} {"Corr":>5s}'
        for name in RECEIVER_NAMES:
            hdr += f' {name:>18s}'
        print(hdr)
        print('-' * 140)

        for r in mod_results:
            line = (f'{r["position"]:>4s} {r["tx_gain"]:>4d} '
                    f'{r["ebn0_db"]:>7.1f} {r["dt_corr"]:>5.2f}')
            for name in RECEIVER_NAMES:
                b = r['ber'].get(name, float('nan'))
                line += f' {b:>18.5f}'
            print(line)

        print(f'\n  Per-position mean BER ({mod.upper()}):')
        hdr2 = f'  {"Pos":>4s} {"N":>3s} {"Eb/N0":>8s}'
        for name in RECEIVER_NAMES:
            hdr2 += f' {name:>18s}'
        print(hdr2)
        for pos in positions:
            pos_r = [r for r in mod_results if r['position'] == pos]
            if not pos_r:
                continue
            snrs = [r['ebn0_db'] for r in pos_r]
            line = f'  {pos:>4s} {len(pos_r):>3d} {np.mean(snrs):>6.1f}dB'
            for name in RECEIVER_NAMES:
                bers = [r['ber'][name] for r in pos_r
                        if not np.isnan(r['ber'].get(name, float('nan')))]
                if bers:
                    line += f' {np.mean(bers):>18.5f}'
                else:
                    line += f' {"N/A":>18s}'
            print(line)

        print(f'\n  Per-position mean NMSE dB ({mod.upper()}):')
        hdr3 = f'  {"Pos":>4s} {"N":>3s} {"Eb/N0":>8s}'
        for name in RECEIVER_NAMES:
            hdr3 += f' {name:>12s}'
        print(hdr3)
        for pos in positions:
            pos_r = [r for r in mod_results if r['position'] == pos]
            if not pos_r:
                continue
            snrs = [r['ebn0_db'] for r in pos_r]
            line = f'  {pos:>4s} {len(pos_r):>3d} {np.mean(snrs):>6.1f}dB'
            for name in RECEIVER_NAMES:
                nmses = [10 * np.log10(r['mse'][name] / (r['h_power'] + 1e-30) + 1e-30)
                         for r in pos_r if name in r['mse']]
                if nmses:
                    line += f' {np.mean(nmses):>12.1f}'
                else:
                    line += f' {"N/A":>12s}'
            print(line)

    # ── Save ──
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f'ota_comparison_{bw_label}.json')
    with open(out_path, 'w') as f:
        json.dump({
            'bw': bw_label,
            'receivers': RECEIVER_NAMES,
            'sync_ok': sync_ok,
            'sync_fail': sync_fail,
            'captures': all_results,
        }, f, indent=2)
    print(f'\nSaved -> {out_path}')


def process_capture_v2(rx_path, tx_data,
                       H_dt_occ_rt, P_dt_rt,
                       N0_noise, noise_psd,
                       eigvecs_emp, eigvals_emp,
                       rhh_eigvecs, rhh_eigvals_norm,
                       U_ee_emp, lam_ee_emp, P_cal,
                       modulation, bw_label,
                       skip_frames=5):
    """Process one OTA capture with 5 receivers.

    DT-Derived LMMSE: global R_hh from Q-D, scaled by P_signal (noise-corrected).
    DT-Assisted LS/LMMSE: empirical R_ee, H_dt energy-matched to H_meas.
    """
    bps = 4 if modulation == '16qam' else 2
    p0_freq = tx_data['p0_freq']
    p1_freq = tx_data['p1_freq']
    p0_occ = p0_freq[OCC_BINS].astype(np.complex128)
    p1_occ = p1_freq[OCC_BINS].astype(np.complex128)
    data_bits_all = np.array(tx_data['data_bits'], dtype=np.int8)

    # ── Robust sync ──
    preamble_td = np.fft.ifft(tx_data['sc2_freq']).astype(np.complex64)
    template = np.concatenate([preamble_td[-CP:], preamble_td])
    samples, rate = load_samples(rx_path)

    xcorr = np.abs(fftconvolve(samples, np.conj(template[::-1]), mode='valid'))
    peak_thresh = 0.5 * np.max(xcorr)
    peaks, _ = find_peaks(xcorr, height=peak_thresh,
                          distance=int(FRAME_LEN * 0.8))

    valid_mask = (peaks >= 0) & (peaks + FRAME_LEN <= len(samples))
    peaks = peaks[valid_mask]
    if len(peaks) < skip_frames + 5:
        return None

    # ── CFO estimation ──
    cfos = []
    for fs in peaks:
        p0_seg = samples[fs + _P0_OFF: fs + _P0_OFF + FFT]
        p1_seg = samples[fs + _P1_OFF: fs + _P1_OFF + FFT]
        H0 = np.fft.fft(p0_seg)[OCC_BINS] / p0_freq[OCC_BINS]
        H1 = np.fft.fft(p1_seg)[OCC_BINS] / p1_freq[OCC_BINS]
        cfos.append(np.angle(np.sum(H1 * np.conj(H0))) / (2 * np.pi * 13 * SYM / rate))
    cfo_global = np.median(cfos) if cfos else 0.0
    n_vec = np.arange(len(samples))
    samples_corr = samples * np.exp(-1j * 2 * np.pi * cfo_global / rate * n_vec).astype(np.complex64)

    frame_starts = peaks[skip_frames:]
    if len(frame_starts) > 500:
        frame_starts = frame_starts[:500]
    nf = len(frame_starts)
    if nf < 5:
        return None

    # ── Frame extraction ──
    abs_idx = (frame_starts[:, None, None]
               + _SYM_OFFS[None, :, None]
               + _FFT_IDX[None, None, :])
    td = samples_corr[abs_idx]
    cfo_corr = np.exp((-1j * 2 * np.pi * cfo_global / rate)
                      * abs_idx.astype(np.float64)).astype(np.complex64)
    fd = np.fft.fft(td * cfo_corr, axis=-1)

    H_ls_p0 = (fd[:, 0, OCC_BINS] / p0_occ[None, :]).astype(np.complex128)
    H_ls_p1 = (fd[:, 13, OCC_BINS] / p1_occ[None, :]).astype(np.complex128)
    Y_data_occ = fd[:, 1:13, OCC_BINS].astype(np.complex128)

    # ── Phase alignment across frames ──
    H_ref = H_ls_p0[0]
    phase_offsets = np.angle(np.sum(H_ls_p0 * np.conj(H_ref[None, :]), axis=1))
    H_aligned = H_ls_p0 * np.exp(-1j * phase_offsets[:, None])
    H_mean = np.mean(H_aligned, axis=0).astype(np.complex64)
    h_power = float(np.mean(np.abs(H_mean)**2))

    P_signal = max(h_power - N0_noise, 1e-30)

    # ── SNR ──
    rx_psd = np.mean(np.abs(Y_data_occ)**2, axis=(0, 1))
    snr_per_sc = np.maximum((rx_psd - noise_psd) / (noise_psd + 1e-20), 1e-10)
    snr_mean = float(np.mean(snr_per_sc))
    ebn0_db = float(10 * np.log10(snr_mean / (bps * SYM / FFT) + 1e-20))

    # ── Time interpolation weights ──
    alphas_time = np.arange(1, N_DATA_SYM + 1).astype(np.float64) / (N_GRID_SYM - 1)
    t = alphas_time[None, :, None]

    # ── Receiver 1: LS (interpolated P0-P1) ──
    H_ls_data = H_ls_p0[:, None, :] * (1 - t) + H_ls_p1[:, None, :] * t

    # ── Receiver 2: LMMSE (empirical R_hh) ──
    W_emp, ev_emp = build_lmmse_filter(eigvecs_emp, eigvals_emp, N0_noise)
    H_lmmse_p0 = H_ls_p0 @ W_emp.T
    H_lmmse_p1 = H_ls_p1 @ W_emp.T
    H_emp_data = H_lmmse_p0[:, None, :] * (1 - t) + H_lmmse_p1[:, None, :] * t

    # ── Receiver 3: DT-Derived LMMSE (global R_hh, scaled by P_signal) ──
    lam_rhh_scaled = rhh_eigvals_norm * P_signal
    W_rhh, ev_rhh = build_lmmse_filter(rhh_eigvecs, lam_rhh_scaled, N0_noise)
    H_rhh_p0 = H_ls_p0 @ W_rhh.T
    H_rhh_p1 = H_ls_p1 @ W_rhh.T
    H_rhh_data = H_rhh_p0[:, None, :] * (1 - t) + H_rhh_p1[:, None, :] * t

    # ── DT prior alignment ──
    has_dt = H_dt_occ_rt is not None and P_dt_rt is not None

    H_dtls_data = H_ls_data
    H_dtlm_data = H_ls_data
    dt_corr = 0.0
    ev_dtls = 0.0
    ev_dtlm = 0.0

    if has_dt:
        # DT-Assisted: energy-match H_dt to H_meas, empirical R_ee scaled by h_power/P_cal
        H_mean_full = np.zeros(FFT, dtype=np.complex64)
        H_mean_full[OCC_BINS] = H_mean
        H_dt_full_rt = np.zeros(FFT, dtype=np.complex128)
        H_dt_full_rt[OCC_BINS] = H_dt_occ_rt.astype(np.complex128)

        H_dt_full_aligned = align_timing_phase(H_dt_full_rt, H_mean_full)
        H_dt_aligned_occ = H_dt_full_aligned[OCC_BINS]
        P_dt_aligned = float(np.mean(np.abs(H_dt_aligned_occ)**2))
        scale_energy = np.sqrt(h_power / (P_dt_aligned + 1e-30))
        H_dt_ota = (H_dt_aligned_occ * scale_energy).astype(np.complex128)

        dt_corr = float(np.abs(np.vdot(H_dt_ota, H_mean.astype(np.complex128)))
                        / (np.linalg.norm(H_dt_ota) * np.linalg.norm(H_mean) + 1e-30))

        scale_ratio = h_power / P_cal

        if U_ee_emp is not None and lam_ee_emp is not None:
            lam_ee_scaled = lam_ee_emp * scale_ratio

            # Receiver 4: DT-Assisted LS (per-SC Wiener with empirical R_ee)
            R_ee_diag = np.sum(np.abs(U_ee_emp)**2 * lam_ee_scaled[None, :], axis=1)
            w_ls = R_ee_diag / (R_ee_diag + N0_noise + 1e-20)
            ev_dtls_arr = R_ee_diag * N0_noise / (R_ee_diag + N0_noise + 1e-20)
            ev_dtls = float(np.mean(ev_dtls_arr))

            # Receiver 5: DT-Assisted LMMSE (full matrix with empirical R_ee)
            gamma_ee = lam_ee_scaled / (lam_ee_scaled + N0_noise + 1e-20)
            W_ee = (U_ee_emp * gamma_ee[None, :]) @ U_ee_emp.conj().T
            ev_dtlm = float(np.sum(lam_ee_scaled * N0_noise /
                                   (lam_ee_scaled + N0_noise + 1e-20))) / N_OCC

            H_dt_frames = H_dt_ota[None, :] * np.exp(1j * phase_offsets[:, None])

            H_dtls_data = ((1 - w_ls)[None, None, :] * H_dt_frames[:, None, :]
                           + w_ls[None, None, :] * H_ls_data)

            res_flat = (H_ls_data - H_dt_frames[:, None, :]).reshape(-1, N_OCC)
            e_flat = res_flat @ W_ee.T
            H_dtlm_data = H_dt_frames[:, None, :] + e_flat.reshape(nf, N_DATA_SYM, N_OCC)

    ev_emp_mean = float(np.mean(ev_emp))
    ev_rhh_mean = float(np.mean(ev_rhh))

    receivers = {
        'ls':                (H_ls_data,     N0_noise),
        'lmmse_empirical':   (H_emp_data,    N0_noise + ev_emp_mean),
        'dt_derived_lmmse':  (H_rhh_data,    N0_noise + ev_rhh_mean),
        'dt_assisted_ls':    (H_dtls_data,   N0_noise + ev_dtls if has_dt else N0_noise),
        'dt_assisted_lmmse': (H_dtlm_data,   N0_noise + ev_dtlm if has_dt else N0_noise),
    }

    # ── MSE (per-frame phase-aligned reference) ──
    H_ref_frames = (H_mean.astype(np.complex128)[None, :]
                    * np.exp(1j * phase_offsets[:, None]))
    mse_results = {}
    for name, (H_est, _) in receivers.items():
        mse_results[name] = float(np.mean(np.abs(H_est - H_ref_frames[:, None, :])**2))

    # ── BER ──
    ber_results = {}
    for name, (H_est, noise_reg) in receivers.items():
        X_hat = (np.conj(H_est) * Y_data_occ / (np.abs(H_est)**2 + noise_reg)).astype(np.complex64)

        if modulation == '16qam':
            b0 = (X_hat.real > 0).astype(np.int8)
            b1 = (np.abs(X_hat.real) < QAM16_THRESH).astype(np.int8)
            b2 = (X_hat.imag > 0).astype(np.int8)
            b3 = (np.abs(X_hat.imag) < QAM16_THRESH).astype(np.int8)
            bits_hat = np.stack([b0, b1, b2, b3], axis=-1).reshape(nf, N_DATA_SYM, -1)
        else:
            b0 = (X_hat.imag < 0).astype(np.int8)
            b1 = (X_hat.real < 0).astype(np.int8)
            bits_hat = np.stack([b0, b1], axis=-1).reshape(nf, N_DATA_SYM, -1)

        n_errs = int(np.sum(bits_hat != data_bits_all[None, :, :]))
        ber_results[name] = n_errs

    total_bits = nf * N_DATA_SYM * data_bits_all.shape[1]

    ber_final = {name: float(errs) / max(total_bits, 1)
                 for name, errs in ber_results.items()}

    return {
        'ber': ber_final,
        'mse': mse_results,
        'h_power': h_power,
        'n_frames': nf,
        'n_bits': total_bits,
        'ebn0_db': ebn0_db,
        'h_power_db': float(10 * np.log10(h_power + 1e-30)),
        'P_signal_db': float(10 * np.log10(P_signal + 1e-30)),
        'alpha_cap_db': float(10 * np.log10(h_power / P_dt_rt + 1e-30)) if has_dt else float('nan'),
        'dt_corr': dt_corr,
        'cfo_hz': float(cfo_global),
        'tx_gain': extract_txg(rx_path),
    }


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Cramér-Rao Bounds for OTA channel estimation.

Computes three theoretical CRB curves (vs SNR) and overlays empirical MSE
from OTA captures for all estimators.

Bounds:
  1. CRB (unstructured) — no structure exploited, = LS variance
  2. CRB (L-tap)        — finite CIR length L, deterministic parameter
  3. BCRB (DT prior)    — Bayesian CRB with R_EE from calibrated DT

Reference: Kay (1993) ch. 3, 15; Biguesh & Gershman (2006); van Trees (2002).
"""

import os, sys, json, glob, re, time, argparse
import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import fftconvolve, find_peaks

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from config.ofdm_params import (FFT, CP, SYM, N_OCC, N_DATA_SYM, N_GRID_SYM, OCC_BINS,
                    FRAME_SYMS, BW_OPTIONS, DATA_DIR, RESULTS_DIR,
                    tx_waveform_path, ota_captures_dir, error_stats_path)

CIR_MAX_TAP = 50
FRAME_LEN = FRAME_SYMS * SYM
RESULTS_V4_DIR = '/home/native/project/results_v4'

ALL_POS = ['p1', 'p2', 'p3', 'p4', 'p5']
ALL_BW  = ['20m', '25m', '50m']


# ────────────── Theoretical CRB curves ──────────────

def crb_unstructured(snr_lin):
    """CRB for unstructured frequency-domain estimation (= LS variance).

    NMSE = 1/SNR  (unit-power pilots |X_p|^2 = 1)
    """
    return 1.0 / snr_lin


def crb_structured(snr_lin, L, N_sc=N_OCC):
    """CRB for L-tap CIR model (deterministic, no statistical prior).

    NMSE = (L/N_sc) / SNR
    Exploits that H lives in L-dimensional subspace of N_sc.
    """
    return (L / N_sc) / snr_lin


def bcrb_dt(snr_lin, eigvals, N_sc=N_OCC):
    """Bayesian CRB with DT prior R_EE = U diag(eigvals) U^H.

    BCRB_avg = (1/N_sc) * sum_i lambda_i * sigma_w^2 / (lambda_i + sigma_w^2)
             = (1/N_sc) * sum_i lambda_i / (1 + lambda_i * SNR / P_h)

    This equals the DT-Assisted LMMSE error variance — the LMMSE estimator
    is efficient and achieves this bound.
    """
    # sigma_w^2 = P_h / SNR  where P_h is channel power
    # But eigvals are for R_EE (error covariance), not R_HH
    # BCRB = (1/N_sc) sum_i  lam_i * sigma_w^2 / (lam_i + sigma_w^2)
    # We need sigma_w^2 as a function of SNR.
    # SNR = P_h / sigma_w^2 → sigma_w^2 = P_h / SNR
    # But CRB should be normalized: NMSE = MSE / P_h
    #
    # For the BCRB with DT prior, the error covariance depends on
    # the actual noise power, not SNR. We parameterize:
    #   sigma_w^2 sweeps, and we compute NMSE = MSE / P_h
    #
    # Since we compute CRB vs SNR (dB), and SNR = P_h / sigma_w^2:
    #   sigma_w^2 = P_h / SNR_lin
    #   MSE = (1/N_sc) sum_i lam_i * sigma_w^2 / (lam_i + sigma_w^2)
    #   NMSE = MSE / P_h = (1/N_sc) sum_i lam_i / (lam_i * SNR_lin / P_h + 1) / SNR_lin
    #
    # But we need to scale eigvals. In our system, eigvals of R_EE are
    # computed from the DT error (H_meas - H_rt). The P_h varies per capture.
    # For a fair bound, we use the average P_cal as normalization.
    #
    # Simplest: compute MSE(sigma_w^2) and then NMSE = MSE / P_h
    # The scale_ratio P_h/P_cal rescales eigvals per capture.
    # For the theoretical curve, use eigvals as-is (average R_EE).

    nmse = np.zeros_like(snr_lin)
    for i, snr in enumerate(snr_lin):
        sigma_w2 = 1.0 / snr  # assuming P_h = 1 (normalized)
        # Scale eigvals to unit channel power
        lam_norm = eigvals / np.sum(eigvals) * N_sc  # rough normalization
        # Actually: keep eigvals as-is and use P_h from P_cal
        mse = np.mean(eigvals * sigma_w2 / (eigvals + sigma_w2 + 1e-30))
        nmse[i] = mse  # already per-SC average, divide by P_h later
    return nmse


def compute_bcrb_curve(snr_db, eigvals, P_cal):
    """Compute BCRB NMSE curve properly accounting for channel power scaling."""
    snr_lin = 10**(snr_db / 10)
    sigma_w2 = P_cal / snr_lin  # sigma_w^2 = P_h / SNR

    nmse = np.zeros(len(snr_db))
    for i in range(len(snr_db)):
        # Per-eigenvalue Bayesian MSE
        mse_per_eig = eigvals * sigma_w2[i] / (eigvals + sigma_w2[i] + 1e-30)
        mse_avg = np.mean(mse_per_eig)  # average over N_sc eigenvalues
        nmse[i] = mse_avg / P_cal  # normalize by channel power
    return nmse


# ────────────── Empirical MSE from OTA ──────────────

def load_samples(rx_path):
    d = np.load(rx_path)
    if 'samples' in d:
        return d['samples'].astype(np.complex64).flatten(), float(d['rate'])
    raw = d['samples_i16'].astype(np.float32) * float(d['sample_scale'])
    return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64), float(d['rate'])


def extract_txg(path):
    m = re.search(r'txg(\d+)', os.path.basename(path))
    return int(m.group(1)) if m else -1


def compute_noise_psd(noise_path, chunk=8192):
    samples, _ = load_samples(noise_path)
    n = len(samples) // FFT
    psd = np.zeros(N_OCC, dtype=np.float64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        blk = samples[s * FFT:e * FFT].reshape(e - s, FFT)
        psd += np.sum(np.abs(np.fft.fft(blk, axis=-1)[:, OCC_BINS])**2, axis=0)
    return psd / n


def align_prior_to_target(H_prior, H_target):
    cir_p = np.array(ifft(H_prior))
    cir_t = np.array(ifft(H_target))
    pk_p = int(np.argmax(np.abs(cir_p[:CIR_MAX_TAP])))
    pk_t = int(np.argmax(np.abs(cir_t[:CIR_MAX_TAP])))
    shift = pk_t - pk_p
    if shift != 0:
        cir_p = np.roll(cir_p, shift)
        if shift > 0: cir_p[:shift] = 0
        else: cir_p[shift:] = 0
    cir_p *= np.exp(-1j * (np.angle(cir_p[pk_t]) - np.angle(cir_t[pk_t])))
    cir_p[CIR_MAX_TAP:] = 0
    H_out = fft(cir_p).astype(np.complex128)
    scale = np.sqrt(np.sum(np.abs(H_target)**2) / (np.sum(np.abs(H_out)**2) + 1e-30))
    return (H_out * scale).astype(np.complex64)


def compute_p_cal(bw_label):
    bw_mhz = int(float(bw_label.replace('m', '')))
    bw_tag = f'bw{bw_mhz}p0'
    powers = []
    for i in range(1, 21):
        fp = os.path.join(RESULTS_V4_DIR, f'posp{i}', f'{bw_tag}.npz')
        if not os.path.exists(fp): continue
        d = np.load(fp)
        cfr = d['cfr_avg'].astype(np.complex128)
        cir = np.fft.ifft(cfr)
        H = fft(np.pad(cir[:CIR_MAX_TAP], (0, FFT - CIR_MAX_TAP))).astype(np.complex64)
        powers.append(float(np.mean(np.abs(H)**2)))
    return float(np.mean(powers)) if powers else 1.0


def compute_r_hh_empirical(bw_label):
    """R_hh from 20 measured calibration channels with Ledoit-Wolf shrinkage."""
    bw_mhz = int(float(bw_label.replace('m', '')))
    bw_tag = f'bw{bw_mhz}p0'
    H_all = []
    for i in range(1, 21):
        fp = os.path.join(RESULTS_V4_DIR, f'posp{i}', f'{bw_tag}.npz')
        if not os.path.exists(fp): continue
        d = np.load(fp)
        cfr = d['cfr_avg'].astype(np.complex128)
        cir = np.fft.ifft(cfr)
        H_all.append(fft(np.pad(cir[:CIR_MAX_TAP], (0, FFT - CIR_MAX_TAP)))[OCC_BINS])
    H_all = np.array(H_all)
    R = (H_all.conj().T @ H_all) / len(H_all)
    mu = np.real(np.trace(R)) / N_OCC
    return (R + 0.1 * mu * np.eye(N_OCC)).astype(np.complex128)


def compute_empirical_mse(pos, bw='50m', mod='qpsk', max_sync_frames=400, R_hh=None):
    """Extract per-capture empirical MSE for all estimators."""
    data_dir = ota_captures_dir(pos, bw, mod)
    tx_data = dict(np.load(tx_waveform_path(mod, bw)))
    bps = 4 if mod == '16qam' else 2

    noise_path = os.path.join(data_dir, f'rx_noise_{bw}.npz')
    if not os.path.exists(noise_path):
        return []
    noise_psd = compute_noise_psd(noise_path)
    N0_noise = float(np.mean(noise_psd))

    P_cal = compute_p_cal(bw)

    H_dt_full = None
    hdt_path = os.path.join(DATA_DIR, f'rt_hdt_{pos}_{bw}.npy')
    if os.path.exists(hdt_path):
        H_dt_full = np.load(hdt_path)

    lam_eig, U_eig = None, None
    err_p = error_stats_path(bw)
    if os.path.exists(err_p):
        d = np.load(err_p)
        if 'R_EE_eigvecs' in d and 'R_EE_eigvals' in d:
            U_eig = d['R_EE_eigvecs'].astype(np.complex128)
            lam_eig = d['R_EE_eigvals'].astype(np.float64)

    p0_occ = tx_data['p0_freq'][OCC_BINS].astype(np.complex128)
    p1_occ = tx_data['p1_freq'][OCC_BINS].astype(np.complex128)
    preamble_td = np.fft.ifft(tx_data['sc2_freq']).astype(np.complex64)
    template = np.concatenate([preamble_td[-CP:], preamble_td])

    _P0_OFF = 1 * SYM + CP
    _P1_OFF = 14 * SYM + CP
    _FFT_IDX = np.arange(FFT)
    _ALPHAS = np.arange(1, N_DATA_SYM + 1, dtype=np.float64) / (N_GRID_SYM - 1)

    rx_files = sorted(glob.glob(os.path.join(data_dir, 'rx_*_txg*.npz')))
    rx_files = [f for f in rx_files if os.path.getsize(f) > 5e6]
    rx_files.sort(key=extract_txg, reverse=True)

    results = []
    for rx_path in rx_files:
        samples, rate = load_samples(rx_path)

        sync_len = min(len(samples), (max_sync_frames + 50) * FRAME_LEN)
        xcorr = np.abs(fftconvolve(samples[:sync_len],
                                   np.conj(template[::-1]), mode='valid'))
        peaks, _ = find_peaks(xcorr, height=0.5 * np.max(xcorr),
                              distance=int(FRAME_LEN * 0.8))
        if len(peaks) < 15:
            continue
        # Reject captures with irregular frame spacing (bad sync)
        diffs = np.diff(peaks)
        spacing_ok = np.abs(diffs - FRAME_LEN) < 0.05 * FRAME_LEN
        if np.mean(spacing_ok) < 0.8:
            continue

        valid = peaks[(peaks >= 0) & (peaks + FRAME_LEN <= len(samples))][5:]
        if len(valid) < 200:
            continue
        nf = len(valid)

        # Batch LS estimation
        p0_idx = valid[:, None] + _P0_OFF + _FFT_IDX[None, :]
        p1_idx = valid[:, None] + _P1_OFF + _FFT_IDX[None, :]
        H_ls_p0 = np.fft.fft(samples[p0_idx], axis=-1)[:, OCC_BINS] / p0_occ
        H_ls_p1 = np.fft.fft(samples[p1_idx], axis=-1)[:, OCC_BINS] / p1_occ

        # Phase alignment → H_mean as ground truth proxy
        H_ref = H_ls_p0[0]
        phase_off = np.angle(np.sum(H_ls_p0 * np.conj(H_ref[None, :]), axis=1))
        H_aligned = H_ls_p0 * np.exp(-1j * phase_off[:, None])
        H_mean = np.mean(H_aligned, axis=0)

        h_power = float(np.mean(np.abs(H_mean)**2))
        if h_power < 1e-6:
            continue

        # SNR
        snr_lin = h_power / (N0_noise + 1e-30)
        snr_db = float(10 * np.log10(snr_lin + 1e-30))

        # MSE computation: use H_mean (frame-averaged) as ground truth
        # This is a biased estimate of true MSE but standard in OTA literature
        mse = {}

        # 1. LS (P0 only)
        mse['ls'] = float(np.mean(np.abs(H_aligned - H_mean[None, :])**2))

        # 2. DT-assisted estimators
        if H_dt_full is not None and lam_eig is not None and U_eig is not None:
            H_mean_full = np.zeros(FFT, dtype=np.complex64)
            H_mean_full[OCC_BINS] = H_mean.astype(np.complex64)
            H_dt_aligned = align_prior_to_target(H_dt_full, H_mean_full)
            H_dt_occ = H_dt_aligned[OCC_BINS].astype(np.complex128)

            scale_ratio = h_power / P_cal
            R_sc = np.sum(np.abs(U_eig)**2 * lam_eig[None, :], axis=1) * scale_ratio
            w_sc = R_sc / (R_sc + N0_noise + 1e-20)

            # DT-Assisted LS: per-frame
            H_dt_fr = H_dt_occ[None, :] * np.exp(1j * phase_off[:, None])
            H_dtls = w_sc * H_aligned + (1 - w_sc) * H_dt_occ[None, :]
            mse['dt_ls'] = float(np.mean(np.abs(H_dtls - H_mean[None, :])**2))

            # DT-Assisted LMMSE: Wiener filter
            lam_s = lam_eig * scale_ratio
            gamma = lam_s / (lam_s + N0_noise + 1e-20)
            W_lmmse = (U_eig * gamma[None, :]) @ U_eig.conj().T
            e_hat = (H_aligned - H_dt_occ[None, :]) @ W_lmmse.T
            H_lmmse = H_dt_occ[None, :] + e_hat
            mse['dt_lmmse'] = float(np.mean(np.abs(H_lmmse - H_mean[None, :])**2))

        # LMMSE with R_hh from calibration data (scale to match OTA power)
        if R_hh is not None:
            R_hh_s = R_hh * (h_power / P_cal)
            W_rhh = R_hh_s @ np.linalg.inv(R_hh_s + N0_noise * np.eye(N_OCC))
            H_lmmse_rhh = H_aligned @ W_rhh.T
            mse['lmmse_rhh'] = float(np.mean(np.abs(H_lmmse_rhh - H_mean[None, :])**2))

        # NMSE
        nmse = {k: v / h_power for k, v in mse.items()}

        results.append({
            'file': os.path.basename(rx_path),
            'tx_gain': extract_txg(rx_path),
            'snr_db': snr_db,
            'h_power': h_power,
            'n_frames': nf,
            'mse': mse,
            'nmse': nmse,
        })

    return results


def compute_L_eff(bw='50m'):
    """Compute effective CIR length from measured calibration channels."""
    bw_mhz = int(float(bw.replace('m', '')))
    bw_tag = f'bw{bw_mhz}p0'
    L_list = []
    for i in range(1, 21):
        fp = os.path.join(RESULTS_V4_DIR, f'posp{i}', f'{bw_tag}.npz')
        if not os.path.exists(fp):
            continue
        d = np.load(fp)
        cfr = d['cfr_avg'].astype(np.complex128)
        cir = np.fft.ifft(cfr)
        p = np.abs(cir[:CIR_MAX_TAP])**2
        cumsum = np.cumsum(p) / (np.sum(p) + 1e-30)
        L95 = int(np.searchsorted(cumsum, 0.95)) + 1
        L_list.append(L95)
    return int(np.median(L_list)) if L_list else 10


def main():
    parser = argparse.ArgumentParser(description='CRB computation and MSE comparison')
    parser.add_argument('--bw', default='50m', choices=ALL_BW)
    parser.add_argument('--pos', nargs='+', default=ALL_POS)
    args = parser.parse_args()
    bw = args.bw

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. System parameters ──
    L_eff = compute_L_eff(bw)
    P_cal = compute_p_cal(bw)

    lam_eig = None
    err_p = error_stats_path(bw)
    if os.path.exists(err_p):
        d = np.load(err_p)
        if 'R_EE_eigvals' in d:
            lam_eig = np.sort(d['R_EE_eigvals'].astype(np.float64))[::-1]

    R_hh = compute_r_hh_empirical(bw)
    lam_rhh = np.sort(np.real(np.linalg.eigvalsh(R_hh)))[::-1]

    print(f'System parameters:')
    print(f'  N_SC={N_OCC}, L_eff={L_eff} (95% CIR energy), P_cal={P_cal:.4f}')
    if lam_eig is not None:
        n_nonzero = int(np.sum(lam_eig > 1e-6))
        cumsum = np.cumsum(lam_eig) / np.sum(lam_eig)
        r95 = int(np.searchsorted(cumsum, 0.95)) + 1
        print(f'  R_EE: {n_nonzero} nonzero eigenvalues, eff_rank_95%={r95}')
    n_nonzero_rhh = int(np.sum(lam_rhh > 1e-6))
    cumsum_rhh = np.cumsum(lam_rhh) / np.sum(lam_rhh)
    r95_rhh = int(np.searchsorted(cumsum_rhh, 0.95)) + 1
    print(f'  R_hh: {n_nonzero_rhh} nonzero eigenvalues, eff_rank_95%={r95_rhh}')

    # ── 2. Theoretical CRB curves ──
    snr_db = np.linspace(-5, 35, 200)
    snr_lin = 10**(snr_db / 10)

    crb = {}
    crb['unstructured'] = 1.0 / snr_lin
    crb['structured'] = (L_eff / N_OCC) / snr_lin

    if lam_eig is not None:
        # BCRB: sweep sigma_w^2, compute per-eigenvalue MSE
        bcrb = np.zeros(len(snr_db))
        for i in range(len(snr_db)):
            sigma_w2 = P_cal / snr_lin[i]
            mse_eig = lam_eig * sigma_w2 / (lam_eig + sigma_w2 + 1e-30)
            bcrb[i] = np.mean(mse_eig) / P_cal
        crb['bcrb_dt'] = bcrb

    # BCRB with R_hh from calibration data
    bcrb_rhh = np.zeros(len(snr_db))
    for i in range(len(snr_db)):
        sigma_w2 = P_cal / snr_lin[i]
        mse_eig = lam_rhh * sigma_w2 / (lam_rhh + sigma_w2 + 1e-30)
        bcrb_rhh[i] = np.mean(mse_eig) / P_cal
    crb['bcrb_rhh'] = bcrb_rhh

    print(f'\nTheoretical bounds at SNR=15 dB:')
    snr15 = 10**(15/10)
    print(f'  CRB (unstructured): NMSE = {10*np.log10(1/snr15):.1f} dB')
    print(f'  CRB ({L_eff}-tap):       NMSE = {10*np.log10(L_eff/N_OCC/snr15):.1f} dB')
    if lam_eig is not None:
        sw2 = P_cal / snr15
        bcrb15 = np.mean(lam_eig * sw2 / (lam_eig + sw2)) / P_cal
        print(f'  BCRB (DT prior):    NMSE = {10*np.log10(bcrb15):.1f} dB')
    sw2 = P_cal / snr15
    bcrb_rhh15 = np.mean(lam_rhh * sw2 / (lam_rhh + sw2)) / P_cal
    print(f'  BCRB (R_hh):        NMSE = {10*np.log10(bcrb_rhh15):.1f} dB')

    # ── 3. Empirical MSE from OTA captures ──
    print(f'\nExtracting empirical MSE from OTA captures...')
    emp_results = {}
    for pos in args.pos:
        print(f'\n  {pos}:')
        res = compute_empirical_mse(pos, bw=bw, R_hh=R_hh)
        if res:
            emp_results[pos] = res
            for r in res:
                nmse_s = ' '.join(f'{k}={10*np.log10(v+1e-30):.1f}' for k, v in r['nmse'].items())
                print(f'    TXg={r["tx_gain"]:>3d}  SNR={r["snr_db"]:>5.1f} dB  '
                      f'nf={r["n_frames"]}  NMSE: {nmse_s}')

    # ── 4. Save results ──
    out = {
        'system': {
            'N_SC': N_OCC, 'L_eff': L_eff, 'P_cal': P_cal, 'bw': bw,
            'R_EE_rank_95': int(np.searchsorted(np.cumsum(lam_eig) / np.sum(lam_eig), 0.95) + 1) if lam_eig is not None else None,
        },
        'crb_curves': {
            'snr_db': snr_db.tolist(),
            'unstructured_nmse_db': (10 * np.log10(crb['unstructured'])).tolist(),
            'structured_nmse_db': (10 * np.log10(crb['structured'])).tolist(),
        },
        'empirical': {pos: res for pos, res in emp_results.items()},
    }
    if 'bcrb_dt' in crb:
        out['crb_curves']['bcrb_dt_nmse_db'] = (10 * np.log10(crb['bcrb_dt'] + 1e-30)).tolist()
    if 'bcrb_rhh' in crb:
        out['crb_curves']['bcrb_rhh_nmse_db'] = (10 * np.log10(crb['bcrb_rhh'] + 1e-30)).tolist()

    out_path = os.path.join(RESULTS_DIR, f'crb_mse_{bw}.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\nSaved → {out_path}')

    # ── 5. Generate figure ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))

        # Theoretical curves
        ax.plot(snr_db, 10 * np.log10(crb['unstructured']),
                color='#555555', ls='--', lw=2, label='CRB')
        if 'bcrb_dt' in crb:
            ax.plot(snr_db, 10 * np.log10(crb['bcrb_dt'] + 1e-30),
                    color='#555555', ls='-', lw=2.5, label=r'BCRB ($R_{EE}$)')
        if 'bcrb_rhh' in crb:
            ax.plot(snr_db, 10 * np.log10(crb['bcrb_rhh'] + 1e-30),
                    color='#888888', ls=':', lw=2, label=r'BCRB ($R_{HH}$)')

        # Empirical MSE — per-position lines for each estimator
        est_styles = {
            'ls':        {'color': 'C0', 'marker': 'o', 'label': 'LS'},
            'lmmse_rhh': {'color': 'C2', 'marker': 's', 'label': 'LMMSE'},
            'dt_ls':     {'color': 'C3', 'marker': '^', 'label': 'DT-Assisted LS'},
            'dt_lmmse':  {'color': 'C1', 'marker': 'D', 'label': 'DT-Assisted LMMSE'},
        }

        pos_markers = {'p1': 'o', 'p2': 's', 'p3': '^', 'p4': 'D', 'p5': 'v'}

        for est_key, style in est_styles.items():
            snrs_all, nmses_all = [], []
            for pos, res_list in emp_results.items():
                for r in res_list:
                    if est_key in r['nmse']:
                        snrs_all.append(r['snr_db'])
                        nmses_all.append(10 * np.log10(r['nmse'][est_key] + 1e-30))

            if snrs_all:
                ax.scatter(snrs_all, nmses_all, marker=style['marker'],
                          s=35, alpha=0.55, color=style['color'],
                          label=style['label'], edgecolors='white',
                          linewidths=0.3, zorder=2)

        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel('NMSE (dB)', fontsize=12)
        ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(-10, 25)
        ax.set_ylim(-20, 10)
        ax.tick_params(labelsize=10)
        fig.tight_layout()

        fig_path = os.path.join(os.path.dirname(__file__), '..', 'figures', 'mse',
                                f'crb_mse_ota_{bw}.png')
        fig.savefig(fig_path, dpi=200, bbox_inches='tight')
        print(f'Figure → {fig_path}')
        plt.close(fig)

    except ImportError:
        print('matplotlib not available, skipping figure')


if __name__ == '__main__':
    main()

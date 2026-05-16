#!/usr/bin/env python3
"""Compute empirical DT error statistics from 20 measured positions.

For each BW and each position: load raw CIR → CFR, align H_dt to H_meas,
compute error covariance in CIR domain, transform to 192×192 freq-domain R_EE.

All 20 positions are used uniformly (no train/test split — R_ee is an
environment-level statistic, not a per-position learned parameter).

Uses pre-computed D7+refr CIR data from cir_measured_d7r_cal.npz.
"""

import os, sys
import numpy as np
from scipy.fft import fft, ifft

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from config.ofdm_params import (FFT, OCC_BINS, DATA_DIR,
                    SOUNDING_DIR, RX_POSITIONS,
                    BW_OPTIONS, measured_path, error_stats_path,
                    load_cir_as_cfr)

ALL_POSITIONS = list(RX_POSITIONS.keys())
CIR_MAX_TAP = 50
N_SC = FFT


def load_measured_h(pos_key, bw='bw50p0'):
    """Load measured H from sounding dataset, truncate CIR to 50 taps."""
    fpath = os.path.join(SOUNDING_DIR, pos_key, f'{bw}.npz')
    d = np.load(fpath)
    cfr = d['cfr_avg'].astype(np.complex128)
    cir = ifft(cfr)
    cir_trunc = np.zeros(N_SC, dtype=np.complex128)
    cir_trunc[:CIR_MAX_TAP] = cir[:CIR_MAX_TAP]
    return fft(cir_trunc).astype(np.complex64), cir[:CIR_MAX_TAP]


def align_and_scale(H_sim, H_true, cir_ref=None):
    """Align simulated CFR to measured: peak shift, phase, CIR truncation, energy scale."""
    cir_sim = np.array(ifft(H_sim))
    if cir_ref is None:
        cir_ref = np.array(ifft(H_true))

    pk_s = int(np.argmax(np.abs(cir_sim[:CIR_MAX_TAP])))
    pk_m = int(np.argmax(np.abs(cir_ref[:CIR_MAX_TAP])))
    shift = pk_m - pk_s
    if shift != 0:
        cir_sim = np.roll(cir_sim, shift)
        if shift > 0:
            cir_sim[:shift] = 0
        else:
            cir_sim[shift:] = 0

    cir_sim *= np.exp(-1j * (np.angle(cir_sim[pk_m]) - np.angle(cir_ref[pk_m])))
    cir_sim[CIR_MAX_TAP:] = 0

    H_out = fft(cir_sim).astype(np.complex64)
    scale = np.sqrt(np.sum(np.abs(H_true)**2) / (np.sum(np.abs(H_out)**2) + 1e-30))
    return H_out * scale


def compute_stats_for_bw(bw_label):
    """Compute DT error stats for a single BW."""
    bw_key = f'bw{bw_label.replace("m", "")}p0'

    cir_file = measured_path(bw_label)
    if not os.path.exists(cir_file):
        print(f'  ERROR: {cir_file} not found. Run compute_hdt_ota_positions.py first.')
        return None

    errors_cir = []
    nmse_list = []
    corr_list = []
    pos_labels = []

    print(f'\n--- BW: {bw_label} ({bw_key}) ---')

    for pos_key in ALL_POSITIONS:
        try:
            H_dt_raw = load_cir_as_cfr(cir_file, pos_key, bw_label).astype(np.complex128)
        except KeyError:
            print(f'  {pos_key}: not in CIR data, skipping')
            continue

        H_meas, cir_meas = load_measured_h(pos_key, bw=bw_key)
        H_dt = align_and_scale(H_dt_raw, H_meas, cir_meas)

        error = H_meas.astype(np.complex128) - H_dt.astype(np.complex128)
        cir_error = ifft(error)

        nmse = float(10 * np.log10(np.sum(np.abs(error)**2) /
                                    (np.sum(np.abs(H_meas)**2) + 1e-30)))
        corr = float(np.abs(np.sum(H_dt * np.conj(H_meas))) /
                      (np.linalg.norm(H_dt) * np.linalg.norm(H_meas) + 1e-30))

        errors_cir.append(cir_error)
        nmse_list.append(nmse)
        corr_list.append(corr)
        pos_labels.append(pos_key)

        print(f'  {pos_key:8s}  corr={corr:.3f}  nmse={nmse:+.2f} dB')

    errors_cir = np.array(errors_cir)
    R_ee_delay = np.mean(np.abs(errors_cir)**2, axis=0)

    # CIR covariance (50×50)
    cov = np.zeros((CIR_MAX_TAP, CIR_MAX_TAP), dtype=np.complex128)
    for e in errors_cir:
        e_trunc = e[:CIR_MAX_TAP]
        cov += np.outer(e_trunc, np.conj(e_trunc))
    cov /= len(errors_cir)
    eigenvalues = np.sort(np.real(np.linalg.eigvalsh(cov)))[::-1]
    cumulative = np.cumsum(eigenvalues) / (np.sum(eigenvalues) + 1e-30)
    eff_rank_90 = int(np.searchsorted(cumulative, 0.9)) + 1
    eff_rank_95 = int(np.searchsorted(cumulative, 0.95)) + 1
    eff_rank_99 = int(np.searchsorted(cumulative, 0.99)) + 1

    nmse_arr = np.array(nmse_list)
    corr_arr = np.array(corr_list)

    # Full 192×192 frequency-domain R_EE: F_occ @ cov @ F_occ^H
    n_taps = cov.shape[0]
    tap_indices = np.arange(n_taps)
    F_occ = np.exp(-1j * 2 * np.pi * OCC_BINS[:, None] * tap_indices[None, :] / N_SC)
    R_EE_freq = F_occ @ cov @ F_occ.conj().T

    eig_vals_freq, eig_vecs_freq = np.linalg.eigh(R_EE_freq)
    eig_vals_freq = eig_vals_freq[::-1].real
    eig_vecs_freq = eig_vecs_freq[:, ::-1]
    eig_vals_freq = np.maximum(eig_vals_freq, 0.0)

    cumulative_freq = np.cumsum(eig_vals_freq) / (np.sum(eig_vals_freq) + 1e-30)
    eff_rank_freq_90 = int(np.searchsorted(cumulative_freq, 0.9)) + 1
    eff_rank_freq_95 = int(np.searchsorted(cumulative_freq, 0.95)) + 1

    # Average power across positions for scale reference (P_cal)
    p_cal = float(np.mean([np.mean(np.abs(
        load_measured_h(k, bw=bw_key)[0])**2) for k in pos_labels]))

    print(f'  Positions: {len(pos_labels)}')
    print(f'  NMSE range: [{nmse_arr.min():.2f}, {nmse_arr.max():.2f}] dB, '
          f'mean: {nmse_arr.mean():.2f} dB')
    print(f'  Corr range: [{corr_arr.min():.3f}, {corr_arr.max():.3f}]')
    print(f'  Effective rank (90/95/99%): {eff_rank_90}/{eff_rank_95}/{eff_rank_99} taps')
    print(f'  Freq-domain R_EE eff rank (90/95%): {eff_rank_freq_90}/{eff_rank_freq_95}')
    print(f'  P_cal: {10*np.log10(p_cal+1e-30):.1f} dB')

    out_path = error_stats_path(bw_label)
    np.savez_compressed(
        out_path,
        R_ee_delay=R_ee_delay.astype(np.float64),
        eigenvalues=eigenvalues.astype(np.float64),
        cov_matrix=cov.astype(np.complex128),
        R_EE_freq=R_EE_freq.astype(np.complex128),
        R_EE_eigvals=eig_vals_freq.astype(np.float64),
        R_EE_eigvecs=eig_vecs_freq.astype(np.complex128),
        nmse_per_pos=nmse_arr,
        corr_per_pos=corr_arr,
        pos_labels=np.array(pos_labels),
        P_cal=np.float64(p_cal),
        eff_rank_90=eff_rank_90,
        eff_rank_95=eff_rank_95,
        eff_rank_99=eff_rank_99,
    )
    print(f'  Saved to {out_path}')
    return out_path


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f'Computing DT error stats for {len(ALL_POSITIONS)} positions '
          f'across {len(BW_OPTIONS)} BWs...')
    print(f'CIR source: {measured_path("50m")}')

    for bw_label in sorted(BW_OPTIONS.keys()):
        compute_stats_for_bw(bw_label)

    print('\nDone.')


if __name__ == '__main__':
    main()

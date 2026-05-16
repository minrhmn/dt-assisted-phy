#!/usr/bin/env python3
"""Precompute global R_hh from dense RT grid + Q-D model.

R_hh = (1/N) * sum_p [ H_dt(p) H_dt(p)^H + R_ee_analytic(p) ]

Averaged over all 3004 grid positions. One matrix per BW.
Output: data/rhh_qd_global_{20,25,50}m.npz
"""

import os, sys, time
import numpy as np

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)


from config.ofdm_params import FFT, OCC_BINS, BW_OPTIONS, DATA_DIR
from channel_model.general_qd_channel import GeneralQDChannel

N_OCC = len(OCC_BINS)
QD_MODEL_PATH = os.path.join(DATA_DIR, 'general_qd_env_model_d7r_cal.npz')
CIR_GRID_PATH = os.path.join(DATA_DIR, 'cir_grid_d7r_cal.npz')


def cir_to_cfr_occ(a_re, a_im, tau, bw_hz):
    a = (a_re + 1j * a_im).astype(np.complex128)
    freqs = np.fft.fftfreq(FFT, d=1.0 / bw_hz)
    f_occ = freqs[OCC_BINS]
    return np.sum(a[:, None] * np.exp(-1j * 2 * np.pi * f_occ[None, :] * tau[:, None]),
                  axis=0)


def precompute_rhh(bw_label, bw_hz, qd, cir_data, pos_keys):
    R_hh_sum = np.zeros((N_OCC, N_OCC), dtype=np.complex128)
    p_dt_list = []
    n_valid = 0

    for pos in pos_keys:
        a_re = cir_data[f'{pos}_a_re']
        a_im = cir_data[f'{pos}_a_im']
        tau = cir_data[f'{pos}_tau']

        H_dt = cir_to_cfr_occ(a_re, a_im, tau, bw_hz)
        p_dt = float(np.mean(np.abs(H_dt) ** 2))
        p_dt_list.append(p_dt)

        R_ee, _ = qd.compute_ree_analytic(H_dt, bw_hz=bw_hz)
        R_hh_sum += np.outer(H_dt, np.conj(H_dt)) + R_ee
        n_valid += 1

    R_hh = R_hh_sum / n_valid
    trace_per_sc = np.real(np.trace(R_hh)) / N_OCC

    eigvals, eigvecs = np.linalg.eigh(R_hh)
    eigvals = eigvals[::-1].real
    eigvecs = eigvecs[:, ::-1]
    eigvals = np.maximum(eigvals, 0.0)

    out_path = os.path.join(DATA_DIR, f'rhh_qd_global_{bw_label}.npz')
    np.savez_compressed(
        out_path,
        R_hh_eigvals=eigvals.astype(np.float64),
        R_hh_eigvecs=eigvecs.astype(np.complex128),
        trace_per_sc=np.float64(trace_per_sc),
        n_positions=n_valid,
        bw_hz=np.float64(bw_hz),
        p_dt_mean=np.float64(np.mean(p_dt_list)),
        p_dt_std=np.float64(np.std(p_dt_list)),
    )

    cum = np.cumsum(eigvals) / (np.sum(eigvals) + 1e-30)
    eff90 = int(np.searchsorted(cum, 0.9)) + 1
    eff95 = int(np.searchsorted(cum, 0.95)) + 1

    print(f'  {bw_label}: {n_valid} positions, '
          f'trace/sc={trace_per_sc:.2e} ({10 * np.log10(trace_per_sc + 1e-30):.1f} dB), '
          f'eff_rank 90/95%={eff90}/{eff95}')
    print(f'    P_dt mean={10 * np.log10(np.mean(p_dt_list) + 1e-30):.1f} dB, '
          f'top 5 eigvals: {eigvals[:5]}')
    print(f'    Saved: {out_path}')
    return out_path


def main():
    print('Loading Q-D model...')
    qd = GeneralQDChannel.from_env_model(QD_MODEL_PATH)
    print(qd.summary())
    print()

    print('Loading dense grid CIR...')
    cir_data = np.load(CIR_GRID_PATH)
    pos_keys = sorted(set(
        k.replace('_a_re', '').replace('_a_im', '').replace('_tau', '')
        for k in cir_data.keys()
    ))
    print(f'  {len(pos_keys)} positions\n')

    for bw_label, bw_hz in sorted(BW_OPTIONS.items()):
        t0 = time.time()
        precompute_rhh(bw_label, bw_hz, qd, cir_data, pos_keys)
        print(f'    Time: {time.time() - t0:.1f}s\n')

    print('Done.')


if __name__ == '__main__':
    main()

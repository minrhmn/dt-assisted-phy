#!/usr/bin/env python3
"""Fig 4 — Calibration: (a) UPES loss convergence, (b) alpha convergence,
plus CFR overlay figures for positions 2, 10, 19.

    python figures/gen_fig_calibration.py
"""
import os, json
import numpy as np
from numpy.fft import fft, ifft
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'font.size': 6,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'text.usetex': False,
    'axes.labelsize': 5.5,
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'legend.fontsize': 5,
    'figure.dpi': 300,
})

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
OUT_DIR = os.path.join(_ROOT, 'output')
CIR_MAX_TAP = 50


def align_and_scale(H_sim, H_true):
    cir_sim = np.array(ifft(H_sim))
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


occ = np.concatenate([np.arange(32, 128), np.arange(129, 225)])
freq_axis = (occ - 128) * (50e6 / 256) / 1e6


def cir_to_cfr(a_re, a_im, tau, bw, n_fft):
    from scipy.fft import fftfreq
    freqs = fftfreq(n_fft, d=1.0 / bw)
    a = (a_re + 1j * a_im).astype(np.complex64)
    phase = np.exp(-1j * 2 * np.pi * np.outer(freqs, tau)).astype(np.complex64)
    return (phase @ a).astype(np.complex64)


def calc_nmse_corr(h_est, h_ref):
    nmse = 10 * np.log10(np.sum(np.abs(h_ref - h_est)**2) / np.sum(np.abs(h_ref)**2))
    corr = np.abs(np.sum(h_ref * np.conj(h_est))) / (np.linalg.norm(h_ref) * np.linalg.norm(h_est))
    return nmse, corr


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Part 1: UPES loss + alpha convergence ──
    cal_path = os.path.join(_ROOT, 'data', 'calibration', 'material_calibration_upes_perpos_d7refr.json')
    with open(cal_path) as f:
        cal = json.load(f)

    loss_history = cal['loss_history']
    alpha_db_history = cal['alpha_db_history']
    best_iter = cal['best_iteration']
    best_loss = cal['best_loss']

    fig1, (ax_loss, ax_alpha) = plt.subplots(1, 2, figsize=(4.8, 1.6))

    iters = np.arange(len(loss_history))

    ax_loss.plot(iters, loss_history, 'k-', lw=0.7)
    ax_loss.axvline(best_iter, color='#d62728', lw=0.5, ls='--', alpha=0.7)
    ax_loss.plot(best_iter, best_loss, 'o', color='#d62728', ms=2.5, zorder=5)
    ax_loss.annotate(f'Best: {best_loss:.0f}\n(iter {best_iter})',
                     xy=(best_iter, best_loss),
                     xytext=(best_iter - 80, best_loss + 8000),
                     fontsize=4.5, color='#d62728',
                     arrowprops=dict(arrowstyle='->', color='#d62728', lw=0.4))
    ax_loss.set_xlabel('Iteration')
    ax_loss.set_ylabel('UPES loss')
    ax_loss.set_title('(a) Loss convergence', fontsize=6, pad=4)
    ax_loss.grid(True, alpha=0.3, linewidth=0.3)
    ax_loss.tick_params(axis='both', length=1.5, width=0.3, pad=1)
    for spine in ax_loss.spines.values():
        spine.set_linewidth(0.3)

    ax_alpha.plot(iters, alpha_db_history, color='#1f77b4', lw=0.7)
    ax_alpha.axhline(alpha_db_history[-1], color='gray', lw=0.4, ls=':', alpha=0.6)
    ax_alpha.text(len(iters) * 0.55, alpha_db_history[-1] + 0.003,
                  f'$\\alpha$ = {alpha_db_history[-1]:.1f} dB',
                  fontsize=4.5, color='gray', va='bottom')
    ax_alpha.set_xlabel('Iteration')
    ax_alpha.set_ylabel('$\\alpha$ (dB)')
    ax_alpha.set_title('(b) Global power scale', fontsize=6, pad=4)
    ax_alpha.grid(True, alpha=0.3, linewidth=0.3)
    ax_alpha.tick_params(axis='both', length=1.5, width=0.3, pad=1)
    for spine in ax_alpha.spines.values():
        spine.set_linewidth(0.3)

    plt.tight_layout(pad=0.4)
    fig1.savefig(os.path.join(OUT_DIR, 'fig_cal_loss.pdf'), bbox_inches='tight', pad_inches=0.02)
    fig1.savefig(os.path.join(OUT_DIR, 'fig_cal_loss.png'), bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.close(fig1)
    print("Saved fig_cal_loss.{pdf,png}")

    # ── Part 2: CFR overlays ──
    CIR_PRECAL  = os.path.join(_ROOT, 'data', 'channel', 'cir_measured_d7r.npz')
    CIR_POSTCAL = os.path.join(_ROOT, 'data', 'channel', 'cir_measured_d7r_cal.npz')
    MEAS_DIR    = os.path.join(_ROOT, 'data', 'sounding')

    if not os.path.exists(CIR_PRECAL):
        print(f"Skipping CFR overlays: {CIR_PRECAL} not found")
        return
    if not os.path.exists(MEAS_DIR):
        print(f"Skipping CFR overlays: {MEAS_DIR} not found")
        print("  (Requires measured sounding data in data/sounding/)")
        return

    cir_pre = np.load(CIR_PRECAL)
    cir_post = np.load(CIR_POSTCAL)
    BW, N_FFT = 50e6, 256

    for pos in ['posp2', 'posp10', 'posp19']:
        meas_path = os.path.join(MEAS_DIR, f'{pos}_bw50.npz')
        if not os.path.exists(meas_path):
            print(f"  Skipping {pos}: measured data not found")
            continue

        meas_data = np.load(meas_path)
        h_meas_full = meas_data['cfr_per_frame'].mean(axis=0)

        h_pre_full = cir_to_cfr(cir_pre[f'{pos}_a_re'], cir_pre[f'{pos}_a_im'],
                                 cir_pre[f'{pos}_tau'], BW, N_FFT)
        h_post_full = cir_to_cfr(cir_post[f'{pos}_a_re'], cir_post[f'{pos}_a_im'],
                                  cir_post[f'{pos}_tau'], BW, N_FFT)

        h_pre_aligned = align_and_scale(h_pre_full, h_meas_full)
        h_post_aligned = align_and_scale(h_post_full, h_meas_full)

        h_meas = h_meas_full[occ]
        h_pre = h_pre_aligned[occ]
        h_post = h_post_aligned[occ]

        nmse_pre, corr_pre = calc_nmse_corr(h_pre, h_meas)
        nmse_post, corr_post = calc_nmse_corr(h_post, h_meas)

        fig, ax = plt.subplots(figsize=(2.4, 1.6))

        mag_meas = 20 * np.log10(np.abs(h_meas) + 1e-10)
        mag_pre = 20 * np.log10(np.abs(h_pre) + 1e-10)
        mag_post = 20 * np.log10(np.abs(h_post) + 1e-10)

        ax.plot(freq_axis, mag_meas, color='green', lw=0.6, alpha=0.9, label='Measured')
        ax.plot(freq_axis, mag_pre, color='#d62728', lw=0.5, alpha=0.6, linestyle='--',
                label=f'Pre-cal (NMSE={nmse_pre:.1f} dB, $\\rho$={corr_pre:.2f})')
        ax.plot(freq_axis, mag_post, color='#1f77b4', lw=0.5, alpha=0.8,
                label=f'Post-cal (NMSE={nmse_post:.1f} dB, $\\rho$={corr_post:.2f})')

        pos_num = pos.replace('posp', '')
        ax.set_title(f'Position {pos_num}', fontsize=7, pad=4)
        ax.set_xlabel('Frequency offset (MHz)')
        ax.set_ylabel('$|H[k]|$ (dB)')
        ax.set_xlim([freq_axis[0], freq_axis[-1]])
        ax.grid(True, alpha=0.3, linewidth=0.3)
        ax.legend(loc='lower left', fontsize=4.5)
        ax.tick_params(axis='both', length=1.5, width=0.3, pad=1)
        for spine in ax.spines.values():
            spine.set_linewidth(0.3)

        plt.tight_layout(pad=0.3)
        fig.savefig(os.path.join(OUT_DIR, f'fig_cal_cfr_{pos}.pdf'), bbox_inches='tight', pad_inches=0.02)
        fig.savefig(os.path.join(OUT_DIR, f'fig_cal_cfr_{pos}.png'), bbox_inches='tight', pad_inches=0.02, dpi=300)
        plt.close(fig)
        print(f"Saved fig_cal_cfr_{pos}.{{pdf,png}}  NMSE: {nmse_pre:.1f} -> {nmse_post:.1f} dB")


if __name__ == '__main__':
    main()

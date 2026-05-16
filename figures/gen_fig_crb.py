#!/usr/bin/env python3
"""Fig 9 — BCRB + empirical NMSE: 3 panels (20/25/50 MHz).

Two Bayesian CRB curves: BCRB(R_EE) with DT-assist, BCRB(R_HH) without.
Empirical scatter: LS, LMMSE, DT-Assisted LS, DT-Assisted LMMSE.

    python figures/gen_fig_crb.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
RESULTS_DIR = os.path.join(_ROOT, 'data', 'ota_results')
DATA_DIR = os.path.join(_ROOT, 'data', 'channel')
OUT_DIR = os.path.join(_ROOT, 'output')

N_OCC = 192
BWS = ['20m', '25m', '50m']
BW_LABELS = {'20m': '20 MHz', '25m': '25 MHz', '50m': '50 MHz'}

EST_STYLES = {
    'ls':        {'color': '#1f77b4', 'marker': 'o', 'label': 'LS'},
    'lmmse_rhh': {'color': '#2ca02c', 'marker': 'd', 'label': 'LMMSE'},
    'dt_ls':     {'color': '#d62728', 'marker': 's', 'label': 'DT-Assisted LS'},
    'dt_lmmse':  {'color': '#bcbd22', 'marker': 'D', 'label': 'DT-Assisted LMMSE'},
}


def main():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'STIX', 'STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
        'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    })

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    snr_db_th = np.linspace(-10, 30, 300)
    snr_lin_th = 10**(snr_db_th / 10)

    for col, bw in enumerate(BWS):
        ax = axes[col]
        path = os.path.join(RESULTS_DIR, f'crb_mse_{bw}.json')
        data = json.load(open(path))
        emp = data['empirical']

        err_npz = np.load(os.path.join(DATA_DIR, f'dt_error_stats_{bw}.npz'))
        lam_ee = np.sort(err_npz['R_EE_eigvals'].astype(np.float64))[::-1]
        P_cal = float(err_npz['P_cal'])

        rhh_path = os.path.join(DATA_DIR, f'rhh_measured_{bw}.npz')
        if os.path.exists(rhh_path):
            lam_hh = np.load(rhh_path)['eigvals']
        else:
            rhh_qd = np.load(os.path.join(DATA_DIR, f'rhh_qd_global_{bw}.npz'))
            lam_hh = np.sort(np.real(rhh_qd['eigvals']))[::-1]

        # BCRB with DT-assist (R_EE)
        bcrb_ee = np.array([np.mean(lam_ee * (P_cal / s) / (lam_ee + P_cal / s + 1e-30)) / P_cal
                            for s in snr_lin_th])
        ax.plot(snr_db_th, 10 * np.log10(bcrb_ee + 1e-30),
                color='#555555', ls='-', lw=2.5,
                label='BCRB (with DT-assist)' if col == 0 else None)

        # BCRB without DT-assist (R_HH)
        bcrb_hh = np.array([np.mean(lam_hh * (P_cal / s) / (lam_hh + P_cal / s + 1e-30)) / P_cal
                            for s in snr_lin_th])
        ax.plot(snr_db_th, 10 * np.log10(bcrb_hh + 1e-30),
                color='#888888', ls=':', lw=2,
                label='BCRB (without DT-assist)' if col == 0 else None)

        for est_key, sty in EST_STYLES.items():
            snrs, nmses = [], []
            for pos, res_list in emp.items():
                for r in res_list:
                    if est_key in r['nmse']:
                        snrs.append(r['snr_db'])
                        nmses.append(10 * np.log10(r['nmse'][est_key] + 1e-30))
            if snrs:
                ax.scatter(snrs, nmses, marker=sty['marker'],
                           s=40, alpha=0.6, color=sty['color'],
                           label=sty['label'] if col == 0 else None,
                           edgecolors='white', linewidths=0.3, zorder=2)

        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Normalized MSE (dB)')
        ax.set_title(BW_LABELS[bw], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.25)
        ax.set_xlim(-10, 25)
        ax.set_ylim(-20, 10)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=10,
               bbox_to_anchor=(0.5, -0.02), frameon=False,
               columnspacing=1.2, handletextpad=0.4)
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.18, top=0.92, wspace=0.28)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, 'fig_crb_mse_ota.png')
    fig.savefig(out, dpi=200, bbox_inches='tight', pad_inches=0.08)
    print(f'Saved -> {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()

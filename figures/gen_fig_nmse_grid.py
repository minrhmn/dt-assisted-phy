#!/usr/bin/env python3
"""Fig 7 — OTA NMSE vs Eb/N0 grid (3 BWs x 5 positions) + gain over LS.

All 5 receivers. QPSK only.

    python figures/gen_fig_nmse_grid.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
RESULTS_DIR = os.path.join(_ROOT, 'data', 'ota_results')
OUT_DIR = os.path.join(_ROOT, 'output')

POSITIONS = ['p1', 'p2', 'p3', 'p4', 'p5']
POS_LABELS = {
    'p1': 'P1 (LOS)', 'p2': 'P2 (LOS)', 'p3': 'P3 (LOS)',
    'p4': 'P4 (LOS)', 'p5': 'P5 (NLOS)',
}
BWS = ['50m', '25m', '20m']
BW_LABELS = {'50m': '50 MHz', '25m': '25 MHz', '20m': '20 MHz'}

SYNC_FAILURES_50M = {
    ('p2', 'qpsk'):  {13, 17, 19},
    ('p2', '16qam'): {13, 15, 17, 19, 21, 23},
    ('p3', 'qpsk'):  {13, 15, 17, 19, 21},
    ('p3', '16qam'): {13, 15, 17, 19, 21, 23, 25},
    ('p4', 'qpsk'):  {13, 15, 17, 19},
    ('p4', '16qam'): {13, 15, 17},
    ('p5', 'qpsk'):  {13, 15},
    ('p5', '16qam'): {13, 15, 17, 19, 21},
}

RECEIVERS = {
    'ls':                 {'color': '#1f77b4', 'ls': '-',  'mk': '^',  'lw': 1.5, 'ms': 5, 'label': 'LS'},
    'lmmse_empirical':    {'color': '#2ca02c', 'ls': '-',  'mk': 'd',  'lw': 1.5, 'ms': 5, 'label': 'LMMSE'},
    'dt_derived_lmmse':   {'color': '#9467bd', 'ls': '--', 'mk': 'p',  'lw': 1.8, 'ms': 6, 'label': 'DT-Derived LMMSE'},
    'dt_assisted_ls':     {'color': '#d62728', 'ls': '-',  'mk': 's',  'lw': 1.8, 'ms': 6, 'label': 'DT-Assisted LS'},
    'dt_assisted_lmmse':  {'color': '#bcbd22', 'ls': '-',  'mk': 'D',  'lw': 2.0, 'ms': 6, 'label': 'DT-Assisted LMMSE'},
}


def load_captures(bw, pos, mod='qpsk'):
    path = os.path.join(RESULTS_DIR, f'ota_comparison_{bw}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    bad_gains = SYNC_FAILURES_50M.get((pos, mod), set()) if bw == '50m' else set()
    caps = [c for c in data['captures']
            if c['position'] == pos and c['modulation'] == mod
            and c.get('tx_gain', -1) not in bad_gains]
    return caps if caps else None


def extract_nmse(captures):
    caps = sorted(captures, key=lambda c: c['ebn0_db'])
    ebn0 = np.array([c['ebn0_db'] for c in caps])
    h_power = np.array([c['h_power'] for c in caps])
    dt_corr = np.mean([c['dt_corr'] for c in caps])
    nmse = {}
    for rx in RECEIVERS:
        vals = np.array([c['mse'].get(rx, np.nan) for c in caps])
        nmse[rx] = 10 * np.log10(vals / (h_power + 1e-30) + 1e-30)
    return ebn0, nmse, dt_corr


def nmse_grid():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'STIX', 'STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
        'legend.fontsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    })

    n_rows, n_cols = len(BWS), len(POSITIONS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 9.5))

    for row, bw in enumerate(BWS):
        for col, pos in enumerate(POSITIONS):
            ax = axes[row, col]
            caps = load_captures(bw, pos)
            if caps is None:
                ax.text(0.5, 0.5, 'no data', transform=ax.transAxes, ha='center', va='center')
                continue

            ebn0, nmse, dt_corr = extract_nmse(caps)
            for rx, sty in RECEIVERS.items():
                show_label = (row == 0 and col == 0)
                ax.plot(ebn0, nmse[rx], color=sty['color'], ls=sty['ls'],
                        marker=sty['mk'], ms=sty['ms'], lw=sty['lw'],
                        label=sty['label'] if show_label else None, markevery=1)

            ax.set_xlim(ebn0[0] - 0.5, ebn0[-1] + 0.5)
            ax.grid(True, alpha=0.25, which='both')
            ax.set_xlabel('$E_b/N_0$ (dB)')
            ax.set_ylabel('NMSE (dB)')
            if row == 0:
                ax.set_title(f'{POS_LABELS[pos]} ($\\rho$={dt_corr:.2f})',
                             fontsize=10, fontweight='bold')
            if col == 0:
                ax.annotate(BW_LABELS[bw], xy=(0, 0.5), xycoords='axes fraction',
                            xytext=(-55, 0), textcoords='offset points',
                            fontsize=12, fontweight='bold', rotation=90, ha='center', va='center')
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(length=2)

    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.10, top=0.93, wspace=0.32, hspace=0.38)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, 0.0), frameon=False, columnspacing=1.0, handletextpad=0.4)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, 'fig_ota_nmse_grid.png')
    fig.savefig(out, dpi=200, bbox_inches='tight', pad_inches=0.08)
    print(f'Saved -> {out}')
    plt.close(fig)


def gain_grid():
    """NMSE gain (dB) over LS vs Eb/N0 — 3x5 grid, 4 receivers."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'STIX', 'STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
        'legend.fontsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    })

    gain_receivers = {k: v for k, v in RECEIVERS.items() if k != 'ls'}
    n_rows, n_cols = len(BWS), len(POSITIONS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 9.5))

    for row, bw in enumerate(BWS):
        for col, pos in enumerate(POSITIONS):
            ax = axes[row, col]
            caps = load_captures(bw, pos)
            if caps is None:
                ax.text(0.5, 0.5, 'no data', transform=ax.transAxes, ha='center', va='center')
                continue

            ebn0, nmse, dt_corr = extract_nmse(caps)
            nmse_ls = nmse['ls']
            for rx, sty in gain_receivers.items():
                gain_db = nmse_ls - nmse[rx]
                show_label = (row == 0 and col == 0)
                ax.plot(ebn0, gain_db, color=sty['color'], ls=sty['ls'],
                        marker=sty['mk'], ms=sty['ms'], lw=sty['lw'],
                        label=sty['label'] if show_label else None, markevery=1)

            ax.axhline(0, color='black', lw=0.8, ls='--')
            ax.set_xlim(ebn0[0] - 0.5, ebn0[-1] + 0.5)
            ax.grid(True, alpha=0.25, which='both')
            ax.set_xlabel('$E_b/N_0$ (dB)')
            ax.set_ylabel('Gain over LS (dB)')
            if row == 0:
                ax.set_title(f'{POS_LABELS[pos]} ($\\rho$={dt_corr:.2f})',
                             fontsize=10, fontweight='bold')
            if col == 0:
                ax.annotate(BW_LABELS[bw], xy=(0, 0.5), xycoords='axes fraction',
                            xytext=(-55, 0), textcoords='offset points',
                            fontsize=12, fontweight='bold', rotation=90, ha='center', va='center')
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(length=2)

    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.10, top=0.93, wspace=0.32, hspace=0.38)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 0.0), frameon=False, columnspacing=1.0, handletextpad=0.4)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, 'fig_ota_nmse_gain_grid.png')
    fig.savefig(out, dpi=200, bbox_inches='tight', pad_inches=0.08)
    print(f'Saved -> {out}')
    plt.close(fig)


if __name__ == '__main__':
    nmse_grid()
    gain_grid()

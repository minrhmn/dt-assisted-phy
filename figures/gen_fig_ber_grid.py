#!/usr/bin/env python3
"""Fig 10 — BER vs Eb/N0: 4x5 grid (50/25 MHz x QPSK/16-QAM x 5 positions).

Receivers: LS, LMMSE, DT-Assisted LS, DT-Assisted LMMSE, Neural RX.

    python figures/gen_fig_ber_grid.py
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

RECEIVERS = {
    'ls':                {'color': 'C0', 'ls': '-',  'mk': '^',  'lw': 1.5, 'ms': 5, 'label': 'LS'},
    'lmmse_empirical':   {'color': 'C2', 'ls': '-',  'mk': 'd',  'lw': 1.5, 'ms': 5, 'label': 'LMMSE'},
    'dt_assisted_ls':    {'color': 'C3', 'ls': '-',  'mk': 's',  'lw': 1.8, 'ms': 6, 'label': 'DT-Assisted LS'},
    'dt_assisted_lmmse': {'color': 'C1', 'ls': '-',  'mk': 'D',  'lw': 2.0, 'ms': 6, 'label': 'DT-Assisted LMMSE'},
    'neural_rx':         {'color': 'C4', 'ls': '--', 'mk': 'o',  'lw': 2.0, 'ms': 6, 'label': 'Neural RX'},
}

CONFIGS = [('50m', 'qpsk'), ('50m', '16qam'), ('25m', 'qpsk'), ('25m', '16qam')]


def load_data(pos, bw, mod):
    lm_path = os.path.join(RESULTS_DIR, f'eval_ota_lmmse_{pos}_{bw}_{mod}.json')
    if not os.path.exists(lm_path):
        return None
    with open(lm_path) as f:
        lm = json.load(f)

    nr_path = os.path.join(RESULTS_DIR, 'ota_neural_rx_summary.json')
    nr_caps = None
    if os.path.exists(nr_path):
        with open(nr_path) as f:
            nr = json.load(f)
        try:
            nr_caps = nr['bandwidths'][bw]['positions'][pos][mod]['captures']
        except KeyError:
            pass

    captures = sorted(lm['captures'], key=lambda c: c['tx_gain'])
    ebn0 = np.array([c['ebn0_db'] for c in captures])
    dt_corr = np.mean([c['dt_corr'] for c in captures])

    ber = {}
    for rx in ['ls', 'lmmse_empirical', 'dt_assisted_ls', 'dt_assisted_lmmse']:
        ber[rx] = np.array([c['ber'].get(rx, np.nan) for c in captures])

    ber['neural_rx'] = np.full(len(captures), np.nan)
    if nr_caps is not None and len(nr_caps) == len(captures):
        for i in range(len(captures)):
            ber['neural_rx'][i] = nr_caps[i]['ber_nr']

    return ebn0, ber, dt_corr


def main():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'STIX', 'STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 11,
        'legend.fontsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    })

    n_rows, n_cols = len(CONFIGS), len(POSITIONS)
    pw, ph = 3.6, 3.6 / 1.5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(pw * n_cols + 1.2, ph * n_rows))

    for row, (bw, mod) in enumerate(CONFIGS):
        for col, pos in enumerate(POSITIONS):
            ax = axes[row, col]
            data = load_data(pos, bw, mod)
            if data is None:
                ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
                continue

            ebn0, ber, dt_corr = data
            for rx, sty in RECEIVERS.items():
                vals = ber[rx]
                valid = ~np.isnan(vals) & (vals > 0)
                if not np.any(valid):
                    continue
                show_label = (row == 0 and col == 0)
                ax.semilogy(ebn0[valid], vals[valid],
                            color=sty['color'], ls=sty['ls'], marker=sty['mk'],
                            ms=sty['ms'], lw=sty['lw'],
                            label=sty['label'] if show_label else None, markevery=2)

            all_vals = np.concatenate([v[~np.isnan(v) & (v > 0)] for v in ber.values()])
            y_bot = 1e-3 if np.any(all_vals < 0.01) else 1e-2
            ax.set_ylim(y_bot, 0.5)
            ax.set_xlim(ebn0[0] - 0.5, ebn0[-1] + 0.5)
            ax.grid(True, alpha=0.25, which='both')

            if row == n_rows - 1:
                ax.set_xlabel('$E_b/N_0$ (dB)')
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel('BER')
            else:
                ax.set_yticklabels([])
            if row == 0:
                ax.set_title(f'{pos} ($\\rho$={dt_corr:.2f})', fontsize=11)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(length=2)

    fig.tight_layout(rect=[0.05, 0.06, 1, 1], h_pad=0.4, w_pad=0.3)

    for bw_idx, bw_text in enumerate(['50 MHz', '25 MHz']):
        top_row = bw_idx * 2
        bot_row = bw_idx * 2 + 1
        y_top = axes[top_row, 0].get_position().y1
        y_bot = axes[bot_row, 0].get_position().y0
        fig.text(0.005, (y_top + y_bot) / 2, bw_text, va='center', ha='center',
                 fontsize=15, fontweight='bold', rotation=90)

    mod_labels = ['QPSK', '16-QAM', 'QPSK', '16-QAM']
    for row in range(n_rows):
        y_top = axes[row, 0].get_position().y1
        y_bot = axes[row, 0].get_position().y0
        fig.text(0.03, (y_top + y_bot) / 2, mod_labels[row], va='center', ha='center',
                 fontsize=13, rotation=90)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=13,
               bbox_to_anchor=(0.53, -0.01), frameon=False,
               columnspacing=1.2, handletextpad=0.5)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, 'fig_ber_ota_grid.png')
    fig.savefig(out, dpi=200, bbox_inches='tight', pad_inches=0.15)
    print(f'Saved -> {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()

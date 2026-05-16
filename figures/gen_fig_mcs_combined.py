#!/usr/bin/env python3
"""Fig 11 — MCS adaptation: throughput bar chart + 11-entry MCS table.

    python figures/gen_fig_mcs_combined.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DATA_DIR = os.path.join(_ROOT, 'data', 'mcs')
OUT_DIR = os.path.join(_ROOT, 'output')

EXP_ORDER = ['posp1', 'posp3', 'posp10', 'posp11', 'posp19']
POS_LABELS = {
    'posp1': 'Pos 1', 'posp3': 'Pos 3', 'posp10': 'Pos 10',
    'posp11': 'Pos 11', 'posp19': 'Pos 19',
}
LOS_STATUS = {
    'posp1': 'LOS', 'posp3': 'LOS', 'posp10': 'NLOS',
    'posp11': 'LOS', 'posp19': 'NLOS',
}

N_OCC, FFT, CP, BW = 192, 256, 64, 50e6
N_DATA_SYM, N_TOTAL_SYM = 12, 14
BW_EFF_MHZ = N_OCC * (BW / FFT) * (N_DATA_SYM / N_TOTAL_SYM) * (FFT / (FFT + CP)) / 1e6

MCS_TABLE_11 = {
    1:  dict(mod_order=4,   code_rate=240, mod_name='QPSK'),
    3:  dict(mod_order=4,   code_rate=256, mod_name='QPSK'),
    4:  dict(mod_order=4,   code_rate=308, mod_name='QPSK'),
    6:  dict(mod_order=4,   code_rate=449, mod_name='QPSK'),
    9:  dict(mod_order=4,   code_rate=679, mod_name='QPSK'),
    10: dict(mod_order=16,  code_rate=340, mod_name='16QAM'),
    13: dict(mod_order=16,  code_rate=490, mod_name='16QAM'),
    16: dict(mod_order=16,  code_rate=658, mod_name='16QAM'),
    19: dict(mod_order=64,  code_rate=517, mod_name='64QAM'),
    22: dict(mod_order=64,  code_rate=666, mod_name='64QAM'),
    25: dict(mod_order=64,  code_rate=822, mod_name='64QAM'),
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'STIX', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'legend.fontsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
})

C_3GPP, C_3GPP_DARK = '#4472C4', '#2F5496'
C_DT, C_DT_DARK = '#ED7D31', '#C55A11'


def draw_throughput(ax, data):
    results = data['results']
    n = len(EXP_ORDER)
    x = np.arange(n)
    w = 0.32

    tput_se_lm = [results[k]['3GPP_InF_LMMSE']['throughput'] for k in EXP_ORDER]
    tput_se_dt = [results[k]['DT_Assisted_QD']['throughput'] for k in EXP_ORDER]
    tput_lm = [t * BW_EFF_MHZ for t in tput_se_lm]
    tput_dt = [t * BW_EFF_MHZ for t in tput_se_dt]
    mcs_lm = [results[k]['3GPP_InF_LMMSE']['mcs'] for k in EXP_ORDER]
    mcs_dt = [results[k]['DT_Assisted_QD']['mcs'] for k in EXP_ORDER]

    ax.bar(x - w/2, tput_lm, w, label='3GPP InF-SL',
           color=C_3GPP, edgecolor=C_3GPP_DARK, linewidth=0.6, zorder=3)
    ax.bar(x + w/2, tput_dt, w, label='DT + Q-D',
           color=C_DT, edgecolor=C_DT_DARK, linewidth=0.6, zorder=3)

    ymax = max(max(tput_lm), max(tput_dt))
    for i in range(n):
        ax.text(x[i] - w/2, tput_lm[i] + 0.01 * ymax, f'MCS {mcs_lm[i]}',
                ha='center', va='bottom', fontsize=7.5, color=C_3GPP_DARK, fontweight='semibold')
        ax.text(x[i] + w/2, tput_dt[i] + 0.01 * ymax, f'MCS {mcs_dt[i]}',
                ha='center', va='bottom', fontsize=7.5, color=C_DT_DARK, fontweight='semibold')

        top = max(tput_lm[i], tput_dt[i])
        bracket_y = top + 0.16 * ymax
        if mcs_dt[i] != mcs_lm[i]:
            gain = (tput_dt[i] / tput_lm[i] - 1) * 100
            ax.plot([x[i] - w/2, x[i] - w/2, x[i] + w/2, x[i] + w/2],
                    [bracket_y - 0.015*ymax, bracket_y, bracket_y, bracket_y - 0.015*ymax],
                    color='#444444', lw=0.8, zorder=4)
            ax.text(x[i], bracket_y + 0.01 * ymax, f'+{gain:.0f}%',
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#222222')
        else:
            ax.text(x[i], bracket_y - 0.02 * ymax, 'TIE',
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color='#666666')

    snrs = [results[k]['snr_db'] for k in EXP_ORDER]
    xlabels = [f'{POS_LABELS[k]} ({LOS_STATUS[k]})\nSNR = {snrs[i]:+.1f} dB'
               for i, k in enumerate(EXP_ORDER)]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8.5)
    ax.set_ylabel('Throughput (Mbps)', fontsize=11)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.95, edgecolor='#CCCCCC')
    ax.set_ylim(0, ymax * 1.50)
    ax.grid(True, axis='y', alpha=0.25, zorder=0)
    ax.set_axisbelow(True)


def draw_mcs_table(ax):
    ax.axis('off')
    col_labels = ['MCS', 'Mod.', 'Rate', 'Mbps']
    rows = []
    for idx in sorted(MCS_TABLE_11.keys()):
        e = MCS_TABLE_11[idx]
        se = int(np.log2(e['mod_order'])) * e['code_rate'] / 1024
        mbps = se * BW_EFF_MHZ
        rows.append([str(idx), e['mod_name'], f'{e["code_rate"]}/1024', f'{mbps:.1f}'])

    table = ax.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.50)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#BBBBBB')
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor('#E8E8E8')
            cell.set_text_props(fontweight='bold', fontsize=9.5)
        else:
            cell.set_facecolor('white')
    ax.set_title('MCS Table  —  3GPP TS 38.214', fontsize=10, style='italic', pad=10)


def main():
    with open(os.path.join(DATA_DIR, 'mcs_integrated_results.json')) as f:
        data = json.load(f)

    fig = plt.figure(figsize=(11, 5.0))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[2.2, 1],
                  left=0.07, right=0.97, top=0.93, bottom=0.13, wspace=0.08)

    draw_throughput(fig.add_subplot(gs[0, 0]), data)
    draw_mcs_table(fig.add_subplot(gs[0, 1]))

    os.makedirs(OUT_DIR, exist_ok=True)
    out_png = os.path.join(OUT_DIR, 'fig_mcs_combined.png')
    out_pdf = os.path.join(OUT_DIR, 'fig_mcs_combined.pdf')
    fig.savefig(out_png, dpi=250, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(out_pdf, bbox_inches='tight', pad_inches=0.05)
    print(f'Saved {out_png}')
    print(f'Saved {out_pdf}')
    plt.close(fig)


if __name__ == '__main__':
    main()

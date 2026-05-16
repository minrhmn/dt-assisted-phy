#!/usr/bin/env python3
"""Fig 3 — R_ee eigenvalue spectrum: cumulative energy at 20/25/50 MHz.

    python figures/gen_fig_ree_eigenspectrum.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'font.size': 3.5,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'text.usetex': False,
    'axes.labelsize': 3.5,
    'xtick.labelsize': 3,
    'ytick.labelsize': 3,
    'legend.fontsize': 3,
    'figure.dpi': 600,
})

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DATA_DIR = os.path.join(_ROOT, 'data', 'channel')
OUT_DIR = os.path.join(_ROOT, 'output')

fig, ax = plt.subplots(figsize=(2.4, 1.2))

colors = {'50': '#1f77b4', '25': '#ff7f0e', '20': '#2ca02c'}
markers = {'50': 'o', '25': 's', '20': '^'}

n_show = 20

for bw in ['50', '25', '20']:
    d = np.load(os.path.join(DATA_DIR, f'dt_error_stats_{bw}m.npz'))
    eigvals = np.sort(d['R_EE_eigvals'])[::-1]
    eigvals_pos = eigvals[:n_show]
    total = eigvals_pos.sum()
    cumulative = np.cumsum(eigvals_pos) / total * 100

    idx = np.arange(1, n_show + 1)
    ax.plot(idx, cumulative, color=colors[bw], marker=markers[bw],
            markersize=1.8, linewidth=0.7, label=f'{bw} MHz')

ax.axhline(y=95, color='gray', linestyle='--', linewidth=0.4, alpha=0.7)
ax.text(16, 92, '$r_{\\rm eff}$ (95%)', fontsize=3, color='gray', ha='center')

for bw, reff in [('50', 8), ('25', 5), ('20', 4)]:
    ax.plot(reff, 95, marker='x', color=colors[bw], markersize=3.5, markeredgewidth=1.0, zorder=5)

ax.set_xlabel('Eigenvalue index $i$')
ax.set_ylabel('Cumulative energy (%)')
ax.set_xlim(0.5, n_show + 0.5)
ax.set_ylim(50, 101)
ax.set_xticks([1, 5, 10, 15, 20])
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3, linewidth=0.3)
ax.tick_params(axis='both', length=1.5, width=0.3, pad=1)
for spine in ax.spines.values():
    spine.set_linewidth(0.3)

plt.tight_layout(pad=0.2)

os.makedirs(OUT_DIR, exist_ok=True)
plt.savefig(os.path.join(OUT_DIR, 'fig_ree_eigenspectrum.pdf'), bbox_inches='tight', pad_inches=0.02)
plt.savefig(os.path.join(OUT_DIR, 'fig_ree_eigenspectrum.png'), bbox_inches='tight', pad_inches=0.02, dpi=600)
print(f'Saved output/fig_ree_eigenspectrum.{{pdf,png}}')

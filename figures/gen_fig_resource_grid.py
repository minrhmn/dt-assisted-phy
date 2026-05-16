#!/usr/bin/env python3
"""Fig 8 — OFDM resource grid: 16 symbols x 256 subcarriers.

Dual-mode: uses Sionna ResourceGrid if available, otherwise builds grid directly.

    python figures/gen_fig_resource_grid.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
OUT_DIR = os.path.join(_ROOT, 'output')

FFT_SIZE = 256
CP_LEN = 64
NUM_OFDM_SYMBOLS = 16
NUM_GUARD_LEFT = 32
NUM_GUARD_RIGHT = 31
PILOT_OFDM_INDICES = [1, 14]
N_OCCUPIED = 192

DATA_TYPE = 0
PILOT_TYPE = 1
NULL_SC_TYPE = 2
DC_NULL_TYPE = 3
PREAMBLE_TYPE = 4
GUARD_SYM_TYPE = 5


def build_grid_sionna():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import warnings
    warnings.filterwarnings('ignore')
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from sionna.phy.ofdm import ResourceGrid

    rg = ResourceGrid(
        num_ofdm_symbols=NUM_OFDM_SYMBOLS,
        fft_size=FFT_SIZE,
        subcarrier_spacing=195312.5,
        num_tx=1, num_streams_per_tx=1,
        cyclic_prefix_length=CP_LEN,
        num_guard_carriers=(NUM_GUARD_LEFT, NUM_GUARD_RIGHT),
        dc_null=True,
        pilot_pattern="kronecker",
        pilot_ofdm_symbol_indices=PILOT_OFDM_INDICES,
    )
    return rg.build_type_grid()[0, 0].numpy()


def build_grid_manual():
    grid = np.zeros((NUM_OFDM_SYMBOLS, FFT_SIZE), dtype=np.float32)
    for sc in range(NUM_GUARD_LEFT):
        grid[:, sc] = NULL_SC_TYPE
    for sc in range(FFT_SIZE - NUM_GUARD_RIGHT, FFT_SIZE):
        grid[:, sc] = NULL_SC_TYPE
    grid[:, 128] = DC_NULL_TYPE
    for sym in PILOT_OFDM_INDICES:
        for sc in range(NUM_GUARD_LEFT, FFT_SIZE - NUM_GUARD_RIGHT):
            if sc != 128:
                grid[sym, sc] = PILOT_TYPE
    return grid


def main():
    try:
        base_grid = build_grid_sionna()
    except (ImportError, ModuleNotFoundError):
        print("Sionna not available, using manual grid construction")
        base_grid = build_grid_manual()

    grid = base_grid.copy().astype(np.float32)
    for sc in range(FFT_SIZE):
        if grid[0, sc] == DATA_TYPE:
            grid[0, sc] = PREAMBLE_TYPE
    for sc in range(FFT_SIZE):
        if grid[15, sc] in (DATA_TYPE, PILOT_TYPE):
            grid[15, sc] = GUARD_SYM_TYPE

    cmap = mcolors.ListedColormap([
        '#88c999', '#ff7f0e', '#9467bd', '#e377c2', '#1f77b4', '#7f7f7f',
    ])
    norm = mcolors.BoundaryNorm([0, 1, 2, 3, 4, 5, 6], cmap.N)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(grid.T, interpolation='nearest', origin='lower',
              cmap=cmap, norm=norm, aspect='auto')

    sym_labels = ['PRE', 'P0'] + [f'D{i}' for i in range(12)] + ['P1', 'GRD']
    ax.set_xticks(range(NUM_OFDM_SYMBOLS))
    ax.set_xticklabels(sym_labels, fontsize=8, rotation=45, ha='right')
    ax.set_xlabel('OFDM Symbol', fontsize=11)

    sc_ticks = [0, NUM_GUARD_LEFT, 128, FFT_SIZE - NUM_GUARD_RIGHT - 1, FFT_SIZE - 1]
    sc_labels = ['0', f'{NUM_GUARD_LEFT}', '128 (DC)', f'{FFT_SIZE-NUM_GUARD_RIGHT-1}', f'{FFT_SIZE-1}']
    ax.set_yticks(sc_ticks)
    ax.set_yticklabels(sc_labels, fontsize=8)
    ax.set_ylabel('Subcarrier Index', fontsize=11)

    for y in [NUM_GUARD_LEFT - 0.5, FFT_SIZE - NUM_GUARD_RIGHT - 0.5, 128]:
        ax.axhline(y=y, color='white', linewidth=0.5, linestyle='--', alpha=0.6)

    ax.set_title('OFDM Resource Grid', fontsize=12, fontweight='bold')

    patches = [
        mpatches.Patch(color='#88c999', label='Data'),
        mpatches.Patch(color='#ff7f0e', label='Pilot'),
        mpatches.Patch(color='#1f77b4', label='Preamble'),
        mpatches.Patch(color='#7f7f7f', label='Guard'),
        mpatches.Patch(color='#9467bd', label='Null Subcarrier'),
        mpatches.Patch(color='#e377c2', label='DC Null'),
    ]
    ax.legend(handles=patches, loc='center left', bbox_to_anchor=(1.02, 0.5),
              fontsize=11, frameon=False)

    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, 'ofdm_resource_grid.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()

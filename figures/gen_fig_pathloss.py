#!/usr/bin/env python3
"""Fig 2 — Path loss: Digital Twin (D7+refr, calibrated) vs measured vs analytic.

SA measurements anchored to Sionna RT at d=1 m.

Dual-mode: runs Sionna RT if available, otherwise loads precomputed data
from data/calibration/pathloss_d7r.npz.

    python figures/gen_fig_pathloss.py
"""
import os, sys, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'font.size': 8,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'text.usetex': False,
    'axes.labelsize': 7,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'legend.fontsize': 6,
    'figure.dpi': 300,
})

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
OUT_DIR = os.path.join(_ROOT, 'output')
PRECOMPUTED = os.path.join(_ROOT, 'data', 'calibration', 'pathloss_d7r.npz')

FC = 3.5e9
FFT_SIZE = 256
BW_HZ = 50e6

LOS_TX = [7.9, -0.1, 1.0]
LOS_RX_X = np.arange(6.9, -10.4, -0.5)
LOS_RX_Y = -0.1
LOS_RX_Z = 1.2

PWR_MEAS_SA = np.array([
    -32.7, -42.7, -42.0, -46.0, -41.5,
    -49.6, -50.0, -47.1, -45.5, -49.5,
    -58.8, -60.0, -52.5, -45.7, -45.0,
    -45.5, -47.0, -48.1, -64.0, -62.8,
    -54.9, -50.8, -53.6, -50.8, -53.0,
    -55.1, -62.1, -52.7, -62.9, -53.6,
    -53.9, -60.0, -56.8, -60.0, -64.0,
])
D_MEAS_SA = np.linspace(1.0, 18.0, 35)


def fit_pl(d, pl):
    a, b = np.polyfit(np.log10(d), pl, 1)
    n = a / 10.0
    resid = pl - (a * np.log10(d) + b)
    return n, b, np.std(resid, ddof=1)


def run_rt_sweep():
    """Run D7R path loss sweep using Sionna RT."""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import sionna
    from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver
    import drjit as dr
    sys.path.insert(0, _ROOT)
    from config.scene_config import apply_calibration, RT_CONFIG

    SCENE_PATH = os.path.join(_ROOT, '..', 'weeks_hall_refined', 'weeks_hall_refined.xml')
    SAMPLES = 1_000_000
    RT_KW = {**RT_CONFIG, 'samples_per_src': SAMPLES, 'max_num_paths_per_src': SAMPLES}

    print("Running D7R path loss sweep...")
    scene = load_scene(SCENE_PATH, merge_shapes=False)
    scene.frequency = FC
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0,
                                 horizontal_spacing=0, pattern='iso', polarization='H')
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0,
                                 horizontal_spacing=0, pattern='iso', polarization='H')
    apply_calibration(scene)
    scene.add(Transmitter('tx', position=LOS_TX))
    scene.add(Receiver('rx', position=[0, LOS_RX_Y, LOS_RX_Z]))
    solver = PathSolver()
    freqs = np.fft.fftfreq(FFT_SIZE, d=1.0 / BW_HZ)

    dists_rt, pls_rt = [], []
    for x in LOS_RX_X:
        scene.receivers['rx'].position = [float(x), LOS_RX_Y, LOS_RX_Z]
        paths = solver(scene, **RT_KW)
        H = np.squeeze(paths.cfr(frequencies=freqs, normalize_delays=True,
                                  normalize=False, out_type='numpy'))
        pg = np.mean(np.abs(H)**2)
        pls_rt.append(-10 * np.log10(pg + 1e-30))
        dists_rt.append(np.sqrt((LOS_TX[0] - float(x))**2 +
                                (LOS_TX[1] - LOS_RX_Y)**2 +
                                (LOS_TX[2] - LOS_RX_Z)**2))

    del scene, solver
    dr.flush_malloc_cache()

    dists_rt = np.array(dists_rt)
    pls_rt = np.array(pls_rt)

    np.savez(PRECOMPUTED, dists_rt=dists_rt, pls_rt=pls_rt)
    print(f"Saved precomputed data -> {PRECOMPUTED}")
    return dists_rt, pls_rt


def load_precomputed():
    d = np.load(PRECOMPUTED)
    return d['dists_rt'], d['pls_rt']


def main():
    if os.path.exists(PRECOMPUTED):
        print(f"Loading precomputed data from {PRECOMPUTED}")
        dists_rt, pls_rt = load_precomputed()
    else:
        try:
            dists_rt, pls_rt = run_rt_sweep()
        except ImportError:
            print("ERROR: Sionna not available and no precomputed data found.")
            print(f"  Expected: {PRECOMPUTED}")
            print("  Run this script with Sionna installed to generate precomputed data.")
            return

    N_DT, PL0_DT, SIG_DT = fit_pl(dists_rt, pls_rt)
    print(f"Digital Twin (D7R): n={N_DT:.3f}  sigma={SIG_DT:.2f} dB")

    PL_RT_1M = pls_rt[0]
    SA_OFFSET = PL_RT_1M - (PWR_MEAS_SA[0] - PWR_MEAS_SA[0])
    PL_SA = SA_OFFSET + (PWR_MEAS_SA[0] - PWR_MEAS_SA)
    N_SA, PL0_SA, SIG_SA = fit_pl(D_MEAS_SA, PL_SA)
    print(f"SA measured (anchored to RT@1m): n={N_SA:.3f}  sigma={SIG_SA:.2f} dB")

    fig, ax = plt.subplots(figsize=(3.5, 2.4))

    d_model = np.linspace(1, 18, 200)
    lam = 3e8 / FC
    fspl = 20 * np.log10(4 * np.pi * d_model / lam)
    inf_sl = 31.84 + 21.50 * np.log10(d_model) + 19 * np.log10(FC / 1e9)

    ax.plot(d_model, fspl, 'k--', lw=0.8, label='FSPL ($n$=2.0)')
    ax.plot(d_model, inf_sl, 'k:', lw=0.8, label='3GPP InF ($n$=2.15)')

    a_sa = 10 * N_SA
    ax.scatter(D_MEAS_SA, PL_SA, c='green', marker='o', s=12, zorder=5, alpha=0.6,
               label=f'Measured ($n$={N_SA:.2f})')
    ax.plot(d_model, a_sa * np.log10(d_model) + PL0_SA, 'g-', lw=1.0)

    a_dt = 10 * N_DT
    ax.scatter(dists_rt, pls_rt, c='#1f77b4', marker='s', s=9, zorder=4, alpha=0.5,
               label=f'Digital Twin ($n$={N_DT:.2f})')
    ax.plot(d_model, a_dt * np.log10(d_model) + PL0_DT, color='#1f77b4', lw=1.0)

    ax.set_xlabel('3D distance (m)')
    ax.set_ylabel('Path loss (dB)')
    ax.set_xlim([0.8, 18.5])
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(loc='lower right', fontsize=5.5)

    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUT_DIR, 'fig_pathloss.png'),
                bbox_inches='tight', pad_inches=0.03, dpi=300)
    print(f"Saved {OUT_DIR}/fig_pathloss.png")
    plt.close()


if __name__ == '__main__':
    main()

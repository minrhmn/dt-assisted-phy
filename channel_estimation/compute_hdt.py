#!/usr/bin/env python3
"""Compute raw RT CIR for 5 OTA + 20 measured positions (25 total).

Uses calibrated_scene_config.py (D7+refr) + nominal TX position.
Per-position solving (one RX at a time) to avoid path starvation.
Stores raw path amplitudes (a_re, a_im) and delays (tau) per position.

Output:
  data/cir_ota_d7r_cal.npz       — 5 OTA positions: {pos}_a_re, {pos}_a_im, {pos}_tau
  data/cir_measured_d7r_cal.npz  — 20 measured positions (regenerated)
"""

import os, sys
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings; warnings.filterwarnings('ignore')

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from config.ofdm_params import FC, DATA_DIR, SCENE_PATH, NOMINAL_TX_POS, RX_POSITIONS

from config.scene_config import apply_calibration, RT_CONFIG

from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver

OTA_POSITIONS = {
    'p1': [-2.18, 2.63, 1.10],
    'p2': [-5.15, -1.13, 1.10],
    'p3': [1.43, -0.83, 1.10],
    'p4': [-7.78, 1.18, 1.37],
    'p5': [-10.32, -1.15, 1.21],
}


def compute_per_position(scene, solver, positions):
    """Run RT per-position (one RX at a time) to avoid path starvation.

    Returns {key: (a_re, a_im, tau)} with non-zero paths only.
    """
    result = {}
    rx_name = 'rx_single'

    for key, pos in positions.items():
        if rx_name in scene.receivers:
            scene.receivers[rx_name].position = pos
        else:
            scene.add(Receiver(rx_name, position=pos))

        paths = solver(scene, **RT_CONFIG)

        a_re = np.array(paths.a[0].numpy())[0, 0, 0, 0, :]
        a_im = np.array(paths.a[1].numpy())[0, 0, 0, 0, :]
        tau = np.array(paths.tau.numpy())[0, 0, :]

        mask = (np.abs(a_re) + np.abs(a_im)) > 0
        result[key] = (a_re[mask].astype(np.float32),
                       a_im[mask].astype(np.float32),
                       tau[mask].astype(np.float32))

        pg = float(10 * np.log10(np.sum(a_re[mask]**2 + a_im[mask]**2) + 1e-30))
        n = int(mask.sum())
        tau_ns = tau[mask] * 1e9
        print(f'  {key}: {n} paths, PG={pg:.1f} dB, '
              f'tau=[{tau_ns.min():.1f}, {tau_ns.max():.1f}] ns' if n > 0 else
              f'  {key}: 0 paths')

    if rx_name in scene.receivers:
        scene.remove(rx_name)

    return result


def save_cir_npz(filepath, cir_dict):
    """Save {key: (a_re, a_im, tau)} to .npz."""
    save_dict = {}
    for key, (a_re, a_im, tau) in cir_dict.items():
        save_dict[f'{key}_a_re'] = a_re
        save_dict[f'{key}_a_im'] = a_im
        save_dict[f'{key}_tau'] = tau
    np.savez_compressed(filepath, **save_dict)


def main():
    print('Loading scene...')
    scene = load_scene(SCENE_PATH, merge_shapes=False)
    scene.frequency = FC
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0,
                                 horizontal_spacing=0, pattern='iso', polarization='H')
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0,
                                 horizontal_spacing=0, pattern='iso', polarization='H')

    applied = apply_calibration(scene)
    print(f'Applied calibration to {len(applied)} materials')

    scene.add(Transmitter('tx', position=NOMINAL_TX_POS))

    solver = PathSolver()
    solver.loop_mode = 'evaluated'

    os.makedirs(DATA_DIR, exist_ok=True)

    # --- 5 OTA positions (per-position solving) ---
    print(f'\nComputing RT for {len(OTA_POSITIONS)} OTA positions (per-position)...')
    ota_cir = compute_per_position(scene, solver, OTA_POSITIONS)
    ota_path = os.path.join(DATA_DIR, 'cir_ota_d7r_cal.npz')
    save_cir_npz(ota_path, ota_cir)
    print(f'Saved {ota_path}')

    # --- 20 measured positions (per-position solving) ---
    print(f'\nComputing RT for {len(RX_POSITIONS)} measured positions (per-position)...')
    meas_cir = compute_per_position(scene, solver, RX_POSITIONS)
    meas_path = os.path.join(DATA_DIR, 'cir_measured_d7r_cal.npz')
    save_cir_npz(meas_path, meas_cir)
    print(f'Saved {meas_path}')

    print(f'\nDone. {len(ota_cir) + len(meas_cir)} positions saved.')


if __name__ == '__main__':
    main()

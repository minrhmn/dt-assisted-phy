#!/usr/bin/env python3
"""UPES material calibration with per-position RT solver — no-lag α, cosine lr decay.

L_UPES = Σ_k Σ_f ( P_meas,k(f) - α · P_rt,k(f;θ) )²

Three-phase per iteration (20 solver calls, no α lag):
  Phase 1 (AD forward): solve all positions → store sim_pwr tensors in AD graph
  Phase 2 (detach): compute global α from current iteration's power
  Phase 3 (AD backward): compute loss with correct α → backward over stored tensors
"""

import os, sys

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import json, argparse, time
import numpy as np
from scipy.fft import fftfreq

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings; warnings.filterwarnings('ignore')

import drjit as dr
import mitsuba as mi
from sionna.rt import (load_scene, PlanarArray, Transmitter, Receiver,
                        PathSolver, RadioMaterial)
from sionna.rt.utils import sigmoid

FC          = 3.5e9
N_SC        = 256
MAX_EPS_R   = 15.0
MAX_SIGMA   = 2.0

DEFAULT_SCENE    = '/home/native/project/weeks_hall_refined/weeks_hall_refined.xml'
DEFAULT_MEAS_DIR = '/home/native/project/results_v4'
DEFAULT_POS_FILE = '/home/native/project/sionna/data_results/rx_positions_refined.json'
DEFAULT_OUTPUT   = '/home/native/project/sionna/data_results/material_calibration_upes_perpos_d7refr.json'

BW_CONFIGS = {
    'bw50p0': {'hz': 50e6, 'label': '50 MHz', 'weight': 1.0},
    'bw20p0': {'hz': 20e6, 'label': '20 MHz', 'weight': 1.0},
    'bw25p0': {'hz': 25e6, 'label': '25 MHz', 'weight': 1.0},
}

def logit_to_eps(logit):
    return 1.0 + sigmoid(logit) * (MAX_EPS_R - 1.0)

def logit_to_sigma(logit):
    return sigmoid(logit) * MAX_SIGMA

def val_to_logit(value, lo, hi):
    frac = np.clip((value - lo) / (hi - lo), 1e-6, 1 - 1e-6)
    return float(np.log(frac / (1 - frac)))

def load_positions(path):
    with open(path) as f:
        data = json.load(f)
    return data['tx'], data['rx']

def load_measurements(meas_dir, pos_keys, bw_label):
    meas = {}
    for key in pos_keys:
        fpath = os.path.join(meas_dir, key, f'{bw_label}.npz')
        if not os.path.exists(fpath):
            continue
        d = np.load(fpath)
        meas[key] = d['cfr_mag_sq_avg'].astype(np.float64)
    return meas


def main():
    parser = argparse.ArgumentParser(description='UPES material calibration (per-position RT + AD)')
    parser.add_argument('--scene',         default=DEFAULT_SCENE)
    parser.add_argument('--meas-dir',      default=DEFAULT_MEAS_DIR)
    parser.add_argument('--pos-file',      default=DEFAULT_POS_FILE)
    parser.add_argument('--output',        default=DEFAULT_OUTPUT)
    parser.add_argument('--lr',            type=float, default=0.01)
    parser.add_argument('--iterations',    type=int,   default=300)
    parser.add_argument('--positions',     default='1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20')
    parser.add_argument('--exclude-materials', default='ceiling-board')
    parser.add_argument('--bandwidths',    default='bw50p0,bw20p0,bw25p0')
    parser.add_argument('--seed',          type=int,   default=42)
    parser.add_argument('--no-scattering', action='store_true')
    parser.add_argument('--alpha-mode',    default='robust', choices=['ls', 'robust'])
    parser.add_argument('--outlier-db',    type=float, default=10.0,
                        help='Exclude observations with residual > this from gradient')
    parser.add_argument('--max-depth',     type=int, default=7)
    parser.add_argument('--refraction',    action='store_true', default=True)
    parser.add_argument('--no-refraction', action='store_true')
    args = parser.parse_args()

    use_refraction = args.refraction and not args.no_refraction
    RT_KW = dict(max_depth=args.max_depth, los=True, specular_reflection=True,
                 diffuse_reflection=False, refraction=use_refraction,
                 synthetic_array=True, seed=args.seed)
    print(f'RT config: max_depth={args.max_depth}, refraction={use_refraction}')

    active_keys = [f'posp{i}' for i in map(int, args.positions.split(','))]
    bw_labels = [b.strip() for b in args.bandwidths.split(',') if b.strip()]

    print('Loading positions...')
    tx_pos, rx_pos = load_positions(args.pos_file)

    print('Loading measurements...')
    bw_meas = {}
    for bw in bw_labels:
        bw_meas[bw] = load_measurements(args.meas_dir, active_keys, bw)
        n = sum(1 for k in active_keys if k in bw_meas[bw])
        print(f'  {bw}: {n} positions')
    active_keys = [k for k in active_keys if all(k in bw_meas[bw] for bw in bw_labels)]
    print(f'  Active positions: {len(active_keys)}')

    print('\nLoading scene...')
    scene = load_scene(args.scene, merge_shapes=False)
    scene.frequency = FC
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0,
                                 horizontal_spacing=0, pattern='iso', polarization='H')
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0,
                                 horizontal_spacing=0, pattern='iso', polarization='H')

    if not args.no_scattering:
        sys.path.insert(0, '/home/native/project/3d_model_processing')
        from material_config import apply_scattering
        applied = apply_scattering(scene)
        print(f'Applied baseline scattering to {len(applied)} materials')
    else:
        print('Scattering disabled')

    scene.add(Transmitter('tx', position=tx_pos))
    scene.add(Receiver('rx', position=[0, 0, 1.0]))
    print('Added single receiver (repositioned per position)')

    solver = PathSolver()
    solver.loop_mode = 'evaluated'

    print('\nScene materials:')
    scene_mats = {}
    mat_to_objs = {}
    for name, mat in scene.radio_materials.items():
        eps  = float(mat.relative_permittivity.numpy().item())
        sig  = float(mat.conductivity.numpy().item())
        is_metal = sig > 1e4
        scene_mats[name] = dict(eps_r=eps, sigma=sig, is_metal=is_metal)
        mat_to_objs[name] = [
            on for on, o in scene.objects.items()
            if o.radio_material is not None and o.radio_material.name == name
        ]
        tag = ' [metal]' if is_metal else ''
        print(f'  {name:25s}  ε_r={eps:6.2f}  σ={sig:12.6f}  ({len(mat_to_objs[name]):3d} objs){tag}')

    exclude = set(args.exclude_materials.split(',')) if args.exclude_materials else set()
    trainable_names = [n for n, m in scene_mats.items()
                       if not m['is_metal'] and n not in exclude]
    print(f'\nTrainable: {trainable_names}  ({2*len(trainable_names)} params)')

    opt = mi.ad.Adam(lr=args.lr)
    trainable_mats = {}
    for name in trainable_names:
        init = scene_mats[name]
        opt[f'le_{name}'] = mi.Float(val_to_logit(init['eps_r'], 1.0, MAX_EPS_R))
        opt[f'ls_{name}'] = mi.Float(val_to_logit(init['sigma'], 0.0, MAX_SIGMA))
        tmat = RadioMaterial(
            f'{name}_t',
            relative_permittivity=logit_to_eps(opt[f'le_{name}']),
            conductivity=logit_to_sigma(opt[f'ls_{name}']),
        )
        trainable_mats[name] = tmat
        for obj_name in mat_to_objs[name]:
            scene.objects[obj_name].radio_material = tmat

    for name in trainable_names:
        try:
            scene.remove(name)
        except Exception:
            pass

    bw_freqs = {}
    for bw in bw_labels:
        bw_freqs[bw] = fftfreq(N_SC, d=1.0 / BW_CONFIGS[bw]['hz'])

    n_obs = len(active_keys) * len(bw_labels)
    print(f'\nObservations: {n_obs * N_SC} ({len(active_keys)} pos × {len(bw_labels)} BWs × {N_SC} SCs)')

    hdr_names = trainable_names[:3]
    print(f'\n{"It":>4}  {"Loss":>12}  {"α":>12}  {"α(dB)":>8}  {"Incl":>4}', end='')
    for n in hdr_names:
        print(f'  {"ε_"+n[:8]:>10}  {"σ_"+n[:8]:>10}', end='')
    print()
    print('─' * (50 + 22 * len(hdr_names)))

    loss_history = []
    alpha_history = []
    best_loss, best_it = float('inf'), 0
    best_params = {}
    lr_init = args.lr
    lr_min  = args.lr * 0.01
    t0 = time.time()

    for it in range(args.iterations):
        # Cosine lr decay
        lr = lr_min + 0.5 * (lr_init - lr_min) * (1 + np.cos(np.pi * it / args.iterations))
        opt.set_learning_rate(lr)

        for name in trainable_names:
            trainable_mats[name].relative_permittivity = logit_to_eps(opt[f'le_{name}'])
            trainable_mats[name].conductivity          = logit_to_sigma(opt[f'ls_{name}'])

        # Phase 1: forward pass with AD — store sim_pwr tensors + collect detached for α
        stored = []  # list of (sim_pwr_ad, meas_pwr_np, weight)
        rt_pg_np = []
        meas_pg_np = []

        for key in active_keys:
            scene.receivers['rx'].position = rx_pos[key]
            paths = solver(scene, **RT_KW)
            for bw in bw_labels:
                w = BW_CONFIGS[bw]['weight']
                H_re, H_im = paths.cfr(frequencies=bw_freqs[bw], normalize_delays=True,
                                        normalize=False, out_type='drjit')
                sim_pwr = dr.square(H_re.array) + dr.square(H_im.array)
                dr.eval(sim_pwr)

                sim_pwr_np = dr.detach(sim_pwr).numpy()
                meas_pwr = bw_meas[bw][key]
                rt_pg_np.append(float(np.mean(sim_pwr_np)))
                meas_pg_np.append(float(np.mean(meas_pwr)))

                stored.append((sim_pwr, meas_pwr, w))
            dr.flush_malloc_cache()

        # Phase 2: compute α from current iteration's power (no lag)
        rt_pg = np.array(rt_pg_np)
        meas_pg = np.array(meas_pg_np)
        if args.alpha_mode == 'ls':
            alpha = float(np.sum(meas_pg * rt_pg) / (np.sum(rt_pg**2) + 1e-30))
        else:
            valid = rt_pg > 1e-30
            if np.any(valid):
                offsets = 10.0 * np.log10(meas_pg[valid] + 1e-30) - 10.0 * np.log10(rt_pg[valid] + 1e-30)
                offset_db = float(np.median(offsets))
                alpha = 10.0 ** (offset_db / 10.0)
        alpha_db = 10.0 * np.log10(alpha + 1e-30)
        n_included = len(stored)

        # Phase 3: compute loss and backward with correct α over stored tensors
        total_loss_val = 0.0
        alpha_f = mi.Float(alpha)
        for sim_pwr, meas_pwr, w in stored:
            meas_f = mi.Float(meas_pwr)
            residual = meas_f - alpha_f * sim_pwr
            partial_loss = w * dr.sum(dr.square(residual))
            dr.backward(partial_loss)
            total_loss_val += float(dr.detach(partial_loss).numpy().item())

        del stored
        dr.flush_malloc_cache()
        dr.flush_kernel_cache()

        opt.step()

        alpha_history.append({'alpha': alpha, 'alpha_db': alpha_db, 'lr': lr})
        loss_history.append(total_loss_val)
        if total_loss_val < best_loss:
            best_loss = total_loss_val
            best_it = it

        cur_params = {}
        for name in trainable_names:
            cur_params[name] = {
                'eps_r': float(dr.detach(logit_to_eps(opt[f'le_{name}'])).numpy().item()),
                'sigma': float(dr.detach(logit_to_sigma(opt[f'ls_{name}'])).numpy().item()),
            }
        if total_loss_val <= best_loss:
            best_params = cur_params

        if it < 5 or it % 10 == 0 or it == args.iterations - 1:
            print(f'{it:>4}  {total_loss_val:>12.2f}  {alpha:>12.2f}  {alpha_db:>8.2f}  {n_included:>4}', end='')
            for n in hdr_names:
                e = cur_params[n]['eps_r']
                s = cur_params[n]['sigma']
                print(f'  {e:>10.3f}  {s:>10.4f}', end='')
            print(f'  lr={lr:.5f}', flush=True)

    elapsed = time.time() - t0
    print(f'\nDone in {elapsed:.1f}s  ({elapsed/args.iterations:.1f}s/iter)')
    print(f'Best iteration: {best_it}, loss: {best_loss:.2f}')

    for name in trainable_names:
        trainable_mats[name].relative_permittivity = mi.Float(best_params[name]['eps_r'])
        trainable_mats[name].conductivity          = mi.Float(best_params[name]['sigma'])

    print('\n=== Calibrated Materials ===')
    for name in trainable_names:
        init  = scene_mats[name]
        final = best_params[name]
        print(f'  {name:25s}  ε_r: {init["eps_r"]:.3f} → {final["eps_r"]:.3f}  '
              f'σ: {init["sigma"]:.6f} → {final["sigma"]:.6f}')

    # ── Final evaluation (per-position) ───────────────────
    print('\n=== Final Per-Position Error ===')
    final_alpha = alpha_history[-1]['alpha']
    final_alpha_db = alpha_history[-1]['alpha_db']
    print(f'  Final α = {final_alpha:.4f} ({final_alpha_db:.2f} dB)')

    eval_results = {}
    all_errs = []
    print(f'\n  {"Pos":>8}  {"BW":>8}  {"PG_rt(dB)":>10}  {"α·PG_rt":>10}  '
          f'{"PG_meas":>10}  {"Err(dB)":>8}  {"UPES":>10}')
    print(f'  {"─"*8}  {"─"*8}  {"─"*10}  {"─"*10}  {"─"*10}  {"─"*8}  {"─"*10}')

    with dr.suspend_grad():
        for key in active_keys:
            eval_results[key] = {}
            scene.receivers['rx'].position = rx_pos[key]
            paths_eval = solver(scene, **RT_KW)
            for bw in bw_labels:
                H_re, H_im = paths_eval.cfr(frequencies=bw_freqs[bw], normalize_delays=True,
                                             normalize=False, out_type='drjit')
                h_re_np = H_re.array.numpy()
                h_im_np = H_im.array.numpy()
                sim_pwr = h_re_np**2 + h_im_np**2
                meas_pwr = bw_meas[bw][key]

                pg_rt_db = 10.0 * np.log10(np.mean(sim_pwr) + 1e-30)
                scaled_pg_db = 10.0 * np.log10(final_alpha * np.mean(sim_pwr) + 1e-30)
                pg_meas_db = 10.0 * np.log10(np.mean(meas_pwr) + 1e-30)
                err_db = scaled_pg_db - pg_meas_db
                upes = float(np.sum((meas_pwr - final_alpha * sim_pwr)**2))

                eval_results[key][bw] = {
                    'pg_rt_db': float(pg_rt_db),
                    'scaled_pg_db': float(scaled_pg_db),
                    'pg_meas_db': float(pg_meas_db),
                    'err_db': float(err_db),
                    'upes': float(upes),
                }
                all_errs.append(err_db)
                label = BW_CONFIGS[bw]['label']
                print(f'  {key:>8}  {label:>8}  {pg_rt_db:>10.2f}  {scaled_pg_db:>10.2f}  '
                      f'{pg_meas_db:>10.2f}  {err_db:>+8.2f}  {upes:>10.2f}')
            dr.flush_malloc_cache()

    errs = np.array(all_errs)
    print(f'\n  RMS PG error: {np.sqrt(np.mean(errs**2)):.3f} dB')
    print(f'  Mean PG error: {np.mean(errs):.3f} dB')
    print(f'  Max |PG error|: {np.max(np.abs(errs)):.3f} dB')

    # ── Figures ───────────────────────────────────────────
    import matplotlib.pyplot as plt
    fig_dir = os.path.join(os.path.dirname(os.path.abspath(args.output)),
                           '..', 'figures', '02_calibration', 'upes_perpos')
    os.makedirs(fig_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(loss_history, linewidth=1.0, color='black')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('UPES Loss')
    ax.set_title('Loss Convergence (per-position RT)')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    alphas_db = [a['alpha_db'] for a in alpha_history]
    ax.plot(alphas_db, linewidth=1.0, color='#2196F3')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('α (dB)')
    ax.set_title('Global Scale α')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    for bw in bw_labels:
        label = BW_CONFIGS[bw]['label']
        pg_meas = [eval_results[k][bw]['pg_meas_db'] for k in active_keys]
        pg_scaled = [eval_results[k][bw]['scaled_pg_db'] for k in active_keys]
        ax.scatter(pg_meas, pg_scaled, s=20, label=label, alpha=0.7)
    mn = min(ax.get_xlim()[0], ax.get_ylim()[0])
    mx = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([mn, mx], [mn, mx], 'k--', alpha=0.3, linewidth=0.8)
    ax.set_xlabel('Measured PG (dB)')
    ax.set_ylabel('α · RT PG (dB)')
    ax.set_title('Path Gain: Measured vs Scaled RT')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = os.path.join(fig_dir, 'upes_perpos_summary.png')
    fig.savefig(fig_path, dpi=150)
    print(f'\nFigures → {fig_path}')
    plt.close(fig)

    # ── Save ──────────────────────────────────────────────
    output = {
        'method': 'material_calibration_upes_perpos',
        'config': dict(lr=args.lr, iterations=args.iterations, seed=args.seed,
                       bandwidths=bw_labels, alpha_mode=args.alpha_mode,
                       outlier_db=args.outlier_db, solver='per_position',
                       positions=active_keys, scene=args.scene),
        'initial_params': {n: {k: v for k, v in scene_mats[n].items() if k != 'is_metal'}
                           for n in trainable_names},
        'final_params': best_params,
        'loss_history': loss_history,
        'alpha_history': [a['alpha'] for a in alpha_history],
        'alpha_db_history': [a['alpha_db'] for a in alpha_history],
        'best_iteration': best_it,
        'best_loss': best_loss,
        'elapsed_seconds': elapsed,
        'eval_results': eval_results,
        'final_stats': {
            'rms_pg_err_db': float(np.sqrt(np.mean(errs**2))),
            'mean_pg_err_db': float(np.mean(errs)),
            'max_abs_pg_err_db': float(np.max(np.abs(errs))),
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    print(f'Saved → {args.output}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Train the DT-augmented neural OFDM receiver (multi-BW, specular-only).

Usage:
    python train.py --modulation qpsk
    python train.py --modulation 16qam
    python train.py --no-dt-prior         # ablation
    python train.py --epochs 50 --batch 128
"""

import os, sys, json, time, argparse
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f'GPU(s) available: {[g.name for g in gpus]}')
else:
    print('WARNING: No GPU detected — training will be slow')

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from config.ofdm_params import (N_OCC, N_GRID_SYM, BITS_PER_SYM, DATA_DIR, MODELS_DIR,
                    RESULTS_DIR, ebnodb_to_noise_var, load_const_map,
                    dense_grid_path, error_stats_path)
from neural_receiver import NeuralOFDMReceiver, build_data_mask, masked_bce_loss
from neural_receiver.data_gen import OFDMDataGenerator


def cosine_lr(step, total_steps, lr_max=1e-3, lr_min=1e-5):
    progress = tf.cast(step, tf.float32) / tf.cast(total_steps, tf.float32)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + tf.math.cos(np.pi * progress))


def compute_ber(model, generator, snr_db, n_frames=200, data_mask_np=None,
                bits_per_sym=2):
    """Compute BER at a fixed SNR using DT channels."""
    total_bits = 0
    total_errors = 0

    if data_mask_np is None:
        data_mask_np = np.zeros((1, 1, N_GRID_SYM, 1), dtype=np.float32)
        data_mask_np[0, 0, 1:13, 0] = 1.0

    batch = min(n_frames, 64)
    n_batches = (n_frames + batch - 1) // batch

    # Temporarily fix SNR range so generate_batch produces the target SNR
    orig_range = generator.snr_range
    generator.snr_range = (snr_db, snr_db)

    for _ in range(n_batches):
        inputs, labels = generator.generate_batch(batch)

        logits = model(inputs, training=False).numpy()
        pred_bits = (logits > 0).astype(np.float32)

        mask = data_mask_np[0, 0, :, 0]
        for sym in range(N_GRID_SYM):
            if mask[sym] < 0.5:
                continue
            true_b = labels[:, :, sym, :]
            pred_b = pred_bits[:, :, sym, :]
            total_errors += int(np.sum(true_b != pred_b))
            total_bits += true_b.size

    generator.snr_range = orig_range
    return total_errors / max(total_bits, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--steps-per-epoch', type=int, default=500)
    parser.add_argument('--lr-max', type=float, default=1e-3)
    parser.add_argument('--lr-min', type=float, default=1e-5)
    parser.add_argument('--val-every', type=int, default=5)
    parser.add_argument('--res-blocks', type=int, default=3)
    parser.add_argument('--filters', type=int, default=64)
    parser.add_argument('--no-dt-prior', action='store_true')
    parser.add_argument('--modulation', type=str, default='qpsk',
                        choices=['qpsk', '16qam'])
    parser.add_argument('--channel-model', type=str, default='perturbation',
                        choices=['perturbation', 'qd'])
    parser.add_argument('--tag', type=str, default='')
    args = parser.parse_args()

    use_dt = not args.no_dt_prior
    mod_suffix = '_16qam' if args.modulation == '16qam' else ''
    tag_suffix = f'_{args.tag}' if args.tag else ''
    qd_suffix = '_qd' if args.channel_model == 'qd' else ''
    tag = f'neural_rx{qd_suffix}{tag_suffix}{mod_suffix}' + ('' if use_dt else '_nodt')

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Load multi-BW grids ---
    bw_grids = {}
    for bw_label in ['20m', '25m', '50m']:
        gp = dense_grid_path(bw_label)
        if os.path.exists(gp):
            bw_grids[bw_label] = dict(np.load(gp))
            print(f'Loaded {bw_label} grid: {len(bw_grids[bw_label])} positions')
        else:
            print(f'WARNING: {gp} not found, skipping {bw_label}')

    if not bw_grids:
        print('ERROR: No channel data. Run generate_dense_grid.py first.')
        sys.exit(1)

    # Per-BW error stats
    err_paths = {}
    for bw_label in bw_grids:
        ep = error_stats_path(bw_label)
        if os.path.exists(ep):
            err_paths[bw_label] = ep

    # tau_rms — prefer D7R data when using Q-D model
    tau_rms_data = None
    tau_rms_max = 1.0
    if args.channel_model == 'qd':
        tau_path = os.path.join(DATA_DIR, 'tau_rms_dense_grid_d7r.npz')
        meta_path = os.path.join(DATA_DIR, 'rt_hdt_dense_d7r_meta.json')
    else:
        tau_path = os.path.join(DATA_DIR, 'tau_rms_dense_grid.npz')
        meta_path = os.path.join(DATA_DIR, 'rt_hdt_dense_meta.json')
    if os.path.exists(tau_path):
        tau_rms_data = {k: float(v) for k, v in np.load(tau_path).items()}
        print(f'Loaded tau_rms: {len(tau_rms_data)} positions')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        tau_rms_max = meta.get('tau_rms_max', 1.0)
        print(f'tau_rms_max = {tau_rms_max:.2f}')

    _, n_bits = load_const_map(args.modulation)

    qd_env = None
    cir_data = None
    if args.channel_model == 'qd':
        
        qd_model_path = os.path.join(DATA_DIR, 'general_qd_env_model_d7r_cal.npz')
        from channel_model.general_qd_channel import GeneralQDChannel
        qd_env = GeneralQDChannel.from_env_model(qd_model_path)
        print(qd_env.summary())

        cir_grid_path = os.path.join(DATA_DIR, 'cir_grid_d7r_cal.npz')
        if os.path.exists(cir_grid_path):
            cir_data = dict(np.load(cir_grid_path))
            n_cir_pos = len([k for k in cir_data if k.endswith('_tau')])
            print(f'Using CIR grid: {n_cir_pos} positions (multi-BW on-the-fly)')
        else:
            print('ERROR: CIR grid not found at', cir_grid_path)
            sys.exit(1)
        err_paths = {}

    generator = OFDMDataGenerator(
        bw_grids=bw_grids if cir_data is None else None,
        error_stats_paths=err_paths if qd_env is None else None,
        snr_range=(-5, 25),
        nmse_range=(-10, 3),
        use_dt_prior=use_dt,
        modulation=args.modulation,
        tau_rms_data=tau_rms_data,
        tau_rms_max=tau_rms_max,
        qd_env=qd_env,
        cir_grid=cir_data,
    )

    # Build model
    model = NeuralOFDMReceiver(
        num_res_blocks=args.res_blocks,
        filters=args.filters,
        use_dt_prior=use_dt,
        n_bits=n_bits,
    )
    c_in = 10 if (use_dt and args.channel_model == 'qd') else (8 if use_dt else 4)
    dummy = tf.zeros((1, N_OCC, N_GRID_SYM, c_in))
    model(dummy, training=False)
    n_params = sum(p.numpy().size for p in model.trainable_variables)
    print(f'Model: {n_params:,} trainable params, DT prior={use_dt}, '
          f'mod={args.modulation}, n_bits={n_bits}')

    data_mask = build_data_mask()
    total_steps = args.epochs * args.steps_per_epoch
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr_max)
    global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

    @tf.function
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logits = model(inputs, training=True)
            loss = masked_bce_loss(labels, logits, data_mask)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    log = {
        'config': vars(args),
        'n_positions': generator.n_positions,
        'n_params': n_params,
        'bw_labels': generator.bw_labels,
        'epochs': [],
    }

    best_val_ber = 1.0
    val_snrs = [0, 10, 20]
    data_mask_np = data_mask.numpy()

    print(f'\nTraining {tag}: {args.epochs} epochs x {args.steps_per_epoch} steps, '
          f'batch={args.batch}\n')
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        t_epoch = time.time()

        for step in range(args.steps_per_epoch):
            inputs_np, labels_np = generator.generate_batch(args.batch)
            inputs_tf = tf.constant(inputs_np)
            labels_tf = tf.constant(labels_np)

            lr = cosine_lr(global_step, total_steps, args.lr_max, args.lr_min)
            optimizer.learning_rate.assign(lr)
            global_step.assign_add(1)

            loss = train_step(inputs_tf, labels_tf)
            epoch_losses.append(float(loss))

        avg_loss = np.mean(epoch_losses)
        elapsed = time.time() - t_epoch
        lr_now = float(optimizer.learning_rate)

        epoch_data = {
            'epoch': epoch,
            'loss': avg_loss,
            'lr': lr_now,
            'time_s': elapsed,
        }

        if epoch % args.val_every == 0 or epoch == 1:
            val_bers = {}
            for snr in val_snrs:
                ber = compute_ber(model, generator, snr, n_frames=200,
                                  data_mask_np=data_mask_np,
                                  bits_per_sym=n_bits)
                val_bers[f'ber_snr{snr}'] = ber
            epoch_data['val'] = val_bers

            avg_ber = np.mean(list(val_bers.values()))
            if avg_ber < best_val_ber:
                best_val_ber = avg_ber
                model.save(os.path.join(MODELS_DIR, f'{tag}_best.keras'))

            ber_str = ', '.join(f'{snr}dB:{val_bers[f"ber_snr{snr}"]:.4f}'
                                for snr in val_snrs)
            print(f'Epoch {epoch:3d}/{args.epochs} | loss={avg_loss:.5f} | '
                  f'lr={lr_now:.2e} | {elapsed:.1f}s | BER: {ber_str}')
        else:
            print(f'Epoch {epoch:3d}/{args.epochs} | loss={avg_loss:.5f} | '
                  f'lr={lr_now:.2e} | {elapsed:.1f}s')

        log['epochs'].append(epoch_data)

        if epoch % 20 == 0:
            model.save(os.path.join(MODELS_DIR, f'{tag}_epoch{epoch}.keras'))

    total_time = time.time() - t_start
    print(f'\nTraining complete in {total_time/3600:.1f}h')

    model.save(os.path.join(MODELS_DIR, f'{tag}_final.keras'))
    log['total_time_s'] = total_time
    log['best_val_ber'] = best_val_ber

    log_path = os.path.join(RESULTS_DIR, f'{tag}_training_log.json')
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f'Saved model to {MODELS_DIR}/{tag}_final.keras')
    print(f'Saved log to {log_path}')


if __name__ == '__main__':
    main()

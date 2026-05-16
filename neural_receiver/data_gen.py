"""Training data generator for the DT-augmented neural OFDM receiver.

Generates OFDM frames with:
  - Channels from the multi-BW dense RT grid (+ empirical DT perturbation)
  - Random QPSK/16QAM data symbols
  - Known P0/P1 pilots (exact values from TX waveform)
  - AWGN at random Eb/N0

Each sample produces:
  inputs:    (192, 14, 8)  float32
  labels:    (192, 14, bps) float32  (bits, only meaningful at data positions)
"""

import numpy as np
from scipy.fft import fft

import os, sys
_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from config.ofdm_params import (FFT, N_OCC, OCC_BINS, N_DATA_SYM, N_GRID_SYM,
                    QPSK_MAP, BITS_PER_SYM, DATA_DIR, BW_NORM,
                    load_pilots, load_const_map, ebnodb_to_noise_var)


class OFDMDataGenerator:
    """Generates batches of neural receiver training data (multi-BW).

    Supports two data modes:
      1. Pre-computed CFR grids (bw_grids): one H_occ per position per BW.
      2. Raw CIR grid (cir_grid): CIR stored once, CFR computed on-the-fly
         at any BW. This is the preferred mode for multi-BW training.
    """

    def __init__(self, bw_grids=None, error_stats_paths=None,
                 snr_range=(-5, 25), nmse_range=(-10, 3),
                 use_dt_prior=True, modulation='qpsk',
                 tau_rms_data=None, tau_rms_max=1.0,
                 qd_env=None, cir_grid=None):
        self.snr_range = snr_range
        self.nmse_range = nmse_range
        self.use_dt_prior = use_dt_prior
        self.modulation = modulation
        self.tau_rms_max = tau_rms_max

        self.const_map, self.bits_per_sym = load_const_map(modulation)

        self.cir_mode = cir_grid is not None
        if self.cir_mode:
            self._init_cir(cir_grid)
        elif bw_grids is not None:
            self._init_cfr(bw_grids)
        else:
            raise ValueError('Either bw_grids or cir_grid must be provided')

        self.p0, self.p1 = load_pilots(modulation)

        self.pilot_mask = np.zeros((N_OCC, N_GRID_SYM, 1), dtype=np.float32)
        self.pilot_mask[:, 0, 0] = 1.0
        self.pilot_mask[:, N_GRID_SYM - 1, 0] = 1.0

        self.tau_rms = {}
        if tau_rms_data is not None:
            for key, val in tau_rms_data.items():
                self.tau_rms[key] = float(val) / (tau_rms_max + 1e-30)

        self.error_stats = {}
        self.n_pert_taps = 10
        if error_stats_paths is not None:
            for bw_label, path in error_stats_paths.items():
                if path and os.path.exists(path):
                    d = np.load(path)
                    cov = d['cov_matrix']
                    try:
                        L = np.linalg.cholesky(
                            cov + 1e-12 * np.eye(cov.shape[0]))
                    except np.linalg.LinAlgError:
                        evals, evecs = np.linalg.eigh(cov)
                        evals = np.maximum(evals, 0)
                        L = evecs @ np.diag(np.sqrt(evals))
                    self.error_stats[bw_label] = {
                        'cov_L': L, 'n_taps': cov.shape[0],
                    }
                    self.n_pert_taps = cov.shape[0]
                    print(f'[DataGen] Loaded error stats for {bw_label}: '
                          f'{cov.shape[0]} taps')

        if not self.error_stats and qd_env is None:
            print('[DataGen] No error stats — using uncorrelated perturbation')

        self.qd_env = qd_env
        if qd_env is not None:
            if hasattr(qd_env, 'params'):
                p = qd_env.params
                print(f'[DataGen] General Q-D model: '
                      f'P_scat/P_dt=N({p.p_scat_dt_mean_db:.1f},'
                      f'{p.p_scat_dt_std_db:.1f})dB, '
                      f'ρ=N({p.rho_mean:.3f},{p.rho_std:.3f})')
            else:
                ps = qd_env['P_scatter_dist']
                rho = qd_env['rho_dist']
                print(f'[DataGen] Q-D channel model: '
                      f'P_scatter={ps[0]:.3f}±{ps[1]:.3f}, '
                      f'rho={rho[0]:.3f}±{rho[1]:.3f}')

        print(f'[DataGen] {len(self.bw_labels)} BWs, '
              f'{self.n_positions} positions, '
              f'{len(self.tau_rms)} tau_rms entries, '
              f'CIR mode={self.cir_mode}')

    def _init_cir(self, cir_grid):
        """Initialize from raw CIR data — CFR computed on-the-fly per BW."""
        all_keys = sorted(cir_grid.keys())
        pos_keys = sorted(set(k.rsplit('_', 1)[0]
                              for k in all_keys if k.endswith('_tau')))
        self.cir_pos_keys = pos_keys
        self.n_positions = len(pos_keys)

        max_paths = max(cir_grid[f'{pk}_tau'].shape[0] for pk in pos_keys)
        self.cir_a = np.zeros((self.n_positions, max_paths), dtype=np.complex64)
        self.cir_tau = np.zeros((self.n_positions, max_paths), dtype=np.float32)
        self.cir_n_paths = np.zeros(self.n_positions, dtype=np.int32)

        for i, pk in enumerate(pos_keys):
            a_re = cir_grid[f'{pk}_a_re']
            a_im = cir_grid[f'{pk}_a_im']
            tau = cir_grid[f'{pk}_tau']
            n = len(tau)
            self.cir_a[i, :n] = (a_re + 1j * a_im).astype(np.complex64)
            self.cir_tau[i, :n] = tau.astype(np.float32)
            self.cir_n_paths[i] = n

        from config.ofdm_params import BW_OPTIONS
        self.bw_labels = sorted(BW_OPTIONS.keys())
        self.bw_hz_map = {k: v for k, v in BW_OPTIONS.items()}

        self._precompute_phase_matrices()

        print(f'[DataGen] CIR mode: {self.n_positions} positions, '
              f'max {max_paths} paths, BWs={self.bw_labels}')

    def _precompute_phase_matrices(self):
        """Precompute exp(-j2pi*f*tau) phase matrices per BW for fast CFR."""
        self.phase_matrices = {}
        for bw_label, bw_hz in self.bw_hz_map.items():
            freqs = np.fft.fftfreq(FFT, d=1.0 / bw_hz)
            f_occ = freqs[OCC_BINS].astype(np.float32)
            self.phase_matrices[bw_label] = f_occ

    def _cir_to_cfr_batch(self, pos_idx, bw_label):
        """Convert CIR to CFR for a batch of position indices at given BW."""
        f_occ = self.phase_matrices[bw_label]
        a_batch = self.cir_a[pos_idx]          # (batch, max_paths)
        tau_batch = self.cir_tau[pos_idx]       # (batch, max_paths)
        phase = np.exp(-1j * 2 * np.pi
                       * f_occ[None, :, None]
                       * tau_batch[:, None, :]).astype(np.complex64)
        H = np.sum(a_batch[:, None, :] * phase, axis=2)  # (batch, N_OCC)
        return H.astype(np.complex64)

    def _init_cfr(self, bw_grids):
        """Initialize from pre-computed CFR grids (legacy path)."""
        self.bw_labels = sorted(bw_grids.keys())
        self.bw_data = {}
        for bw_label, cache in bw_grids.items():
            keys = list(cache.keys())
            sample = cache[keys[0]]
            if sample.shape[0] == N_OCC:
                h_occ = np.stack([cache[k].astype(np.complex64) for k in keys])
            else:
                h_occ = np.stack([
                    cache[k][OCC_BINS].astype(np.complex64) for k in keys
                ])
            self.bw_data[bw_label] = {'keys': keys, 'h_occ': h_occ}
        self.n_positions = len(self.bw_data[self.bw_labels[0]]['keys'])

    def _generate_perturbation(self, batch_size, h_dt_power, bw_label='50m'):
        """Generate structured CIR-domain perturbation.

        Returns: (batch_size, 192) complex64 perturbation in freq domain.
        """
        nmse_db = np.random.uniform(
            self.nmse_range[0], self.nmse_range[1], size=batch_size)
        target_power = h_dt_power * (10 ** (nmse_db / 10))

        stats = self.error_stats.get(bw_label)
        if stats is not None:
            n_taps = stats['n_taps']
            cov_L = stats['cov_L']
            z = (np.random.randn(batch_size, n_taps) +
                 1j * np.random.randn(batch_size, n_taps)) / np.sqrt(2)
            cir_pert = z @ cov_L.T
            pert_power = np.mean(np.abs(cir_pert)**2, axis=1) * FFT
            scale = np.sqrt(target_power / (pert_power + 1e-30))
            cir_pert *= scale[:, None]
        else:
            n_taps = self.n_pert_taps
            z = (np.random.randn(batch_size, n_taps) +
                 1j * np.random.randn(batch_size, n_taps)) / np.sqrt(2)
            pert_power = np.mean(np.abs(z)**2, axis=1) * FFT
            scale = np.sqrt(target_power / (pert_power + 1e-30))
            cir_pert = z * scale[:, None]

        cir_padded = np.zeros((batch_size, FFT), dtype=np.complex64)
        cir_padded[:, :n_taps] = cir_pert
        H_pert = np.fft.fft(cir_padded, axis=1)[:, OCC_BINS]
        return H_pert.astype(np.complex64)

    def _generate_qd_channel(self, batch_size, H_dt, bw_hz=None):
        """Generate channel via Q-D model: H_true = H_dt + H_dmc + fading.

        Supports both legacy dict format and GeneralQDChannel object.
        Returns: (batch_size, 192) complex64
        """
        if hasattr(self.qd_env, 'generate_batch'):
            return self.qd_env.generate_batch(H_dt, bw_hz=bw_hz)

        pdp_shape = self.qd_env['env_pdp_shape']
        P_scat_mean, P_scat_std = self.qd_env['P_scatter_dist']
        rho_mean, rho_std = self.qd_env['rho_dist']
        n_taps = len(pdp_shape)

        P_scatter = np.maximum(0.01,
            np.random.normal(P_scat_mean, P_scat_std, batch_size))

        scaled_pdp = pdp_shape[None, :] * P_scatter[:, None]
        z = (np.random.randn(batch_size, n_taps) +
             1j * np.random.randn(batch_size, n_taps)) / np.sqrt(2)
        cir_dmc = z * np.sqrt(scaled_pdp)

        cir_padded = np.zeros((batch_size, FFT), dtype=np.complex128)
        cir_padded[:, :n_taps] = cir_dmc
        H_dmc = np.fft.fft(cir_padded, axis=1)[:, OCC_BINS]

        h_static = H_dt.astype(np.complex128) + H_dmc
        P_static = np.mean(np.abs(h_static)**2, axis=1)

        rho = np.clip(np.random.normal(rho_mean, rho_std, batch_size),
                       0.05, 0.995)
        sigma_fading_sq = P_static * (1.0 / rho - 1.0)

        fading = np.sqrt(sigma_fading_sq[:, None] / 2) * (
            np.random.randn(batch_size, N_OCC) +
            1j * np.random.randn(batch_size, N_OCC))

        return (h_static + fading).astype(np.complex64)

    def generate_batch(self, batch_size):
        """Generate one training batch.

        Returns:
            inputs: (batch, 192, 14, 8) float32  [or (batch, 192, 14, 4) if no DT]
            labels: (batch, 192, 14, bps) float32
        """
        bw_label = np.random.choice(self.bw_labels)
        bw_val = BW_NORM[bw_label]

        if self.cir_mode:
            pos_idx = np.random.randint(0, self.n_positions, size=batch_size)
            H_dt = self._cir_to_cfr_batch(pos_idx, bw_label)
            pos_keys = [self.cir_pos_keys[i] for i in pos_idx]
        else:
            grid = self.bw_data[bw_label]
            keys = grid['keys']
            pos_idx = np.random.randint(0, len(keys), size=batch_size)
            H_dt = grid['h_occ'][pos_idx]
            pos_keys = [keys[i] for i in pos_idx]

        h_dt_power = np.mean(np.abs(H_dt)**2, axis=1)

        tau_rms_batch = np.array([
            self.tau_rms.get(k, 0.5) for k in pos_keys
        ], dtype=np.float32)

        if self.qd_env is not None:
            bw_hz = self.bw_hz_map.get(bw_label) if self.cir_mode else None
            H_applied = self._generate_qd_channel(batch_size, H_dt, bw_hz=bw_hz)
        else:
            H_pert = self._generate_perturbation(batch_size, h_dt_power, bw_label)
            H_applied = H_dt + H_pert

        bps = self.bits_per_sym
        bits = np.random.randint(0, 2,
            size=(batch_size, N_OCC, N_DATA_SYM, bps)).astype(np.float32)
        if bps == 2:
            sym_idx = (bits[..., 0] * 2 + bits[..., 1]).astype(int)
        else:
            sym_idx = (bits[..., 0] * 8 + bits[..., 1] * 4 +
                       bits[..., 2] * 2 + bits[..., 3]).astype(int)
        data_syms = self.const_map[sym_idx]

        X = np.zeros((batch_size, N_OCC, N_GRID_SYM), dtype=np.complex64)
        X[:, :, 0] = self.p0[None, :]
        X[:, :, 1:13] = data_syms
        X[:, :, 13] = self.p1[None, :]

        ebn0_db = np.random.uniform(
            self.snr_range[0], self.snr_range[1], size=batch_size)
        noise_var_unit = ebnodb_to_noise_var(ebn0_db, bits_per_sym=bps)

        H_broad = H_applied[:, :, None]
        Y_clean = H_broad * X

        h_power = np.mean(np.abs(H_applied)**2, axis=1)  # (batch,)
        noise_var = noise_var_unit * h_power

        noise = np.sqrt(noise_var[:, None, None] / 2) * (
            np.random.randn(batch_size, N_OCC, N_GRID_SYM) +
            1j * np.random.randn(batch_size, N_OCC, N_GRID_SYM)
        ).astype(np.complex64)
        Y = Y_clean + noise

        # Per-sample normalization: divide complex channels by sqrt(h_power)
        # to bring all inputs to O(1) scale while preserving frequency structure
        inv_scale = 1.0 / np.sqrt(np.maximum(h_power, 1e-30))  # (batch,)
        Y = Y * inv_scale[:, None, None]
        H_dt_norm = H_dt * inv_scale[:, None]

        if self.use_dt_prior:
            c_in = 10 if self.qd_env is not None else 8
            inputs = np.zeros((batch_size, N_OCC, N_GRID_SYM, c_in), dtype=np.float32)
            inputs[:, :, :, 0] = np.real(Y)
            inputs[:, :, :, 1] = np.imag(Y)
            inputs[:, :, :, 2] = np.real(H_dt_norm[:, :, None])
            inputs[:, :, :, 3] = np.imag(H_dt_norm[:, :, None])
            inputs[:, :, :, 4] = self.pilot_mask[None, :, :, 0]
            inputs[:, :, :, 5] = noise_var_unit[:, None, None]
            inputs[:, :, :, 6] = tau_rms_batch[:, None, None]
            inputs[:, :, :, 7] = bw_val
            if self.qd_env is not None:
                H_ls_p0 = Y[:, :, 0] / self.p0[None, :]
                H_ls_p1 = Y[:, :, 13] / self.p1[None, :]
                H_ls = (H_ls_p0 + H_ls_p1) / 2
                inputs[:, :, :, 8] = np.real(H_ls[:, :, None])
                inputs[:, :, :, 9] = np.imag(H_ls[:, :, None])
        else:
            c_in = 4
            inputs = np.zeros((batch_size, N_OCC, N_GRID_SYM, c_in), dtype=np.float32)
            inputs[:, :, :, 0] = np.real(Y)
            inputs[:, :, :, 1] = np.imag(Y)
            inputs[:, :, :, 2] = self.pilot_mask[None, :, :, 0]
            inputs[:, :, :, 3] = noise_var_unit[:, None, None]

        labels = np.zeros((batch_size, N_OCC, N_GRID_SYM, bps), dtype=np.float32)
        labels[:, :, 1:13, :] = bits

        return inputs, labels


def make_tf_dataset(generator, batch_size, prefetch=4):
    """Wrap generator as a tf.data.Dataset."""
    import tensorflow as tf

    def gen():
        while True:
            inputs, labels = generator.generate_batch(batch_size)
            yield inputs, labels

    bps = generator.bits_per_sym
    c_in = 10 if (generator.use_dt_prior and generator.qd_env is not None) else (8 if generator.use_dt_prior else 4)
    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, N_OCC, N_GRID_SYM, c_in), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size, N_OCC, N_GRID_SYM, bps), dtype=tf.float32),
        )
    )
    return ds.prefetch(prefetch)


if __name__ == '__main__':
    print('Loading dense grids...')
    from config.ofdm_params import dense_grid_path, error_stats_path

    bw_grids = {}
    for bw_label in ['20m', '25m', '50m']:
        gp = dense_grid_path(bw_label)
        if os.path.exists(gp):
            bw_grids[bw_label] = dict(np.load(gp))
            print(f'  {bw_label}: {len(bw_grids[bw_label])} positions')

    if not bw_grids:
        print('No channel data found. Run generate_dense_grid.py first.')
        sys.exit(1)

    err_paths = {}
    for bw_label in bw_grids:
        ep = error_stats_path(bw_label)
        if os.path.exists(ep):
            err_paths[bw_label] = ep

    tau_rms_data = None
    tau_rms_max = 1.0
    tau_path = os.path.join(DATA_DIR, 'tau_rms_dense_grid.npz')
    meta_path = os.path.join(DATA_DIR, 'rt_hdt_dense_meta.json')
    if os.path.exists(tau_path):
        tau_rms_data = {k: float(v) for k, v in np.load(tau_path).items()}
    if os.path.exists(meta_path):
        import json
        with open(meta_path) as f:
            meta = json.load(f)
        tau_rms_max = meta.get('tau_rms_max', 1.0)

    gen = OFDMDataGenerator(
        bw_grids, err_paths,
        tau_rms_data=tau_rms_data, tau_rms_max=tau_rms_max,
    )

    inputs, labels = gen.generate_batch(8)
    print(f'Inputs: {inputs.shape}, dtype={inputs.dtype}')
    print(f'Labels: {labels.shape}, dtype={labels.dtype}')
    print(f'Input channel ranges:')
    for c in range(inputs.shape[-1]):
        v = inputs[:, :, :, c]
        print(f'  ch{c}: [{v.min():.4f}, {v.max():.4f}]')
    print(f'Label sum at data pos: {labels[:, :, 1:13, :].sum():.0f} '
          f'(expected ~{8 * 192 * 12 * 2 * 0.5:.0f})')

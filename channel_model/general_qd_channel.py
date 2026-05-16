#!/usr/bin/env python3
"""General Q-D Channel Model for Indoor Factory Environments.

Model:
  H^(n)[k] = H_dt[k] + H_dmc[k] + ΔH^(n)[k]

  H_dt:     Deterministic from ray tracing (D-rays), raw CIR scale
  H_dmc:    Dense Multipath Component — exponential PDP, log-normal τ_rms
  ΔH^(n):   Frame-to-frame Rayleigh fading

Power convention:
  RT outputs raw CIR at ~-60 to -70 dB.  Global α (dB) maps RT→OTA scale:
    H_ota ≈ √α · H_rt
  The model works in **raw RT scale**.  At OTA inference time, scale
  input by 1/√α to enter RT space, or scale output by √α to get OTA space.

Environment parameters:
  α_db:           Global power scale RT→OTA [dB]
  μ_lgDS, σ_lgDS: Log-normal τ_rms distribution [log10(ns)]
  τ_scatter_ns:   Median scatter delay spread [ns]
  P_scat_dt_db:   P_scatter/P_dt ratio ~ N(μ, σ) [dB]
  ρ:              Static-to-total power ratio ~ N(μ, σ)

Parameter sources:
  1. from_sounding() — fit from OTA + RT at sounding positions (best)
  2. from_env_model() — load previously saved parameters
  3. from_hall_geometry() — 3GPP InF defaults from hall dims (no sounding)

Reference:
  3GPP TR 38.901 v17.0.0, Table 7.5-6 Part-3 (Indoor Factory)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import json


@dataclass
class EnvParams:
    """Environment-level channel model parameters."""
    alpha_db: float
    mu_lgDS: float
    sigma_lgDS: float
    tau_scatter_ns: float
    p_scat_dt_mean_db: float
    p_scat_dt_std_db: float
    rho_mean: float
    rho_std: float

    fc_hz: float = 3.5e9
    bw_hz: float = 50e6
    n_fft: int = 256
    n_taps: int = 50

    hall_dims: Optional[Tuple[float, float, float]] = None
    source: str = ''
    n_sounding_positions: int = 0


class GeneralQDChannel:
    """General Q-D channel model for indoor factory environments.

    All generation happens in raw RT power scale.
    Use alpha_scale() / inv_alpha_scale() to convert to/from OTA scale.
    """

    def __init__(self, params: EnvParams):
        self.params = params
        self._alpha_lin = 10 ** (params.alpha_db / 10)
        self._sqrt_alpha = np.sqrt(self._alpha_lin)
        self._ts_ns = 1e9 / params.bw_hz

    @property
    def sqrt_alpha(self):
        return self._sqrt_alpha

    @property
    def inv_sqrt_alpha(self):
        return 1.0 / self._sqrt_alpha

    # ── Constructors ─────────────────────────────────────────────────────

    @classmethod
    def from_hall_geometry(cls, length: float, width: float, height: float,
                           fc_hz: float = 3.5e9, bw_hz: float = 50e6,
                           n_fft: int = 256, condition: str = 'los',
                           alpha_db: float = 59.0):
        """Create model from hall dimensions using 3GPP InF defaults."""
        v = length * width * height
        s = 2 * (length * width + length * height + width * height)
        vs = v / s

        if condition == 'los':
            mu_lgds = np.log10(26 * vs + 14) - 9.35
            sigma_lgds = 0.15
            rho_mean, rho_std = 0.80, 0.10
            p_scat_mean, p_scat_std = 6.0, 3.0
        else:
            mu_lgds = np.log10(30 * vs + 32) - 9.44
            sigma_lgds = 0.19
            rho_mean, rho_std = 0.60, 0.15
            p_scat_mean, p_scat_std = 10.0, 4.0

        tau_rms_ns = 10 ** (mu_lgds + 9)
        tau_scatter_ns = tau_rms_ns * 3.0
        mu_lgDS_ns = mu_lgds + 9

        params = EnvParams(
            alpha_db=alpha_db,
            mu_lgDS=mu_lgDS_ns, sigma_lgDS=sigma_lgds,
            tau_scatter_ns=tau_scatter_ns,
            p_scat_dt_mean_db=p_scat_mean, p_scat_dt_std_db=p_scat_std,
            rho_mean=rho_mean, rho_std=rho_std,
            fc_hz=fc_hz, bw_hz=bw_hz, n_fft=n_fft,
            hall_dims=(length, width, height),
            source=f'3gpp_inf_{condition} (V/S={vs:.2f})',
        )
        return cls(params)

    @classmethod
    def from_sounding(cls, h_meas_list, h_dt_list,
                      alpha_db: float = 59.0,
                      rho_vals=None,
                      tau_rms_grid_ns=None,
                      fc_hz: float = 3.5e9, bw_hz: float = 50e6,
                      n_fft: int = 256, hall_dims=None):
        """Fit model from OTA sounding + RT at measured positions.

        Args:
            h_meas_list: List of (N_SC,) complex — time-averaged OTA CFR (OTA scale).
            h_dt_list: List of (N_SC,) complex — RT H_dt (raw RT scale).
            alpha_db: Global power scale from UPES calibration.
            rho_vals: List of measured ρ values (from doppler analysis).
            tau_rms_grid_ns: Array of τ_rms values from RT grid (for log-normal fit).
            hall_dims: Optional (L, W, H) for metadata.
        """
        inv_sqrt_alpha = 1.0 / np.sqrt(10 ** (alpha_db / 10))
        ts_ns = 1e9 / bw_hz
        n_pos = len(h_meas_list)

        p_scat_dt_db_list = []
        tau_scatter_list = []

        for i in range(n_pos):
            h_meas = np.asarray(h_meas_list[i], dtype=np.complex128)
            h_dt = np.asarray(h_dt_list[i], dtype=np.complex128)

            h_meas_rt = h_meas * inv_sqrt_alpha

            corr = np.sum(h_dt * np.conj(h_meas_rt))
            h_dt_aligned = h_dt * np.exp(-1j * np.angle(corr))

            h_error = h_meas_rt - h_dt_aligned
            p_scat = float(np.mean(np.abs(h_error)**2))
            p_dt = float(np.mean(np.abs(h_dt)**2))

            if p_dt > 1e-30:
                p_scat_dt_db_list.append(10 * np.log10(p_scat / p_dt))

            cir_err = np.fft.ifft(h_error)
            pdp = np.abs(cir_err[:50])**2
            pdp_total = np.sum(pdp)
            if pdp_total > 1e-30:
                pdp_norm = pdp / pdp_total
                taps = np.arange(50)
                t_mean = np.sum(pdp_norm * taps)
                t_rms = np.sqrt(np.sum(pdp_norm * (taps - t_mean)**2))
                tau_scatter_list.append(t_rms * ts_ns)

        # τ_rms log-normal from grid (if provided) or from scatter delays
        if tau_rms_grid_ns is not None:
            valid = tau_rms_grid_ns[tau_rms_grid_ns > 0.01]
            log_tau = np.log10(valid)
            mu_lgDS = float(np.mean(log_tau))
            sigma_lgDS = float(np.std(log_tau))
        else:
            mu_lgDS = 1.315
            sigma_lgDS = 0.234

        tau_scatter_ns = float(np.median(tau_scatter_list)) \
            if tau_scatter_list else 60.0

        p_scat_mean = float(np.mean(p_scat_dt_db_list)) \
            if p_scat_dt_db_list else 6.0
        p_scat_std = float(np.std(p_scat_dt_db_list)) \
            if p_scat_dt_db_list else 3.0

        if rho_vals is not None and len(rho_vals) > 0:
            rho_mean = float(np.mean(rho_vals))
            rho_std = float(np.std(rho_vals))
        else:
            rho_mean, rho_std = 0.842, 0.058

        params = EnvParams(
            alpha_db=alpha_db,
            mu_lgDS=mu_lgDS, sigma_lgDS=sigma_lgDS,
            tau_scatter_ns=tau_scatter_ns,
            p_scat_dt_mean_db=p_scat_mean, p_scat_dt_std_db=p_scat_std,
            rho_mean=rho_mean, rho_std=rho_std,
            fc_hz=fc_hz, bw_hz=bw_hz, n_fft=n_fft,
            hall_dims=hall_dims,
            source=f'sounding ({n_pos} pos)',
            n_sounding_positions=n_pos,
        )
        return cls(params)

    @classmethod
    def from_env_model(cls, path: str):
        """Load from saved model file."""
        d = np.load(path, allow_pickle=True)
        params = EnvParams(
            alpha_db=float(d['alpha_db']),
            mu_lgDS=float(d['mu_lgDS']),
            sigma_lgDS=float(d['sigma_lgDS']),
            tau_scatter_ns=float(d['tau_scatter_ns']),
            p_scat_dt_mean_db=float(d['p_scat_dt_mean_db']),
            p_scat_dt_std_db=float(d['p_scat_dt_std_db']),
            rho_mean=float(d['rho_mean']),
            rho_std=float(d['rho_std']),
            fc_hz=float(d['fc_hz']),
            bw_hz=float(d['bw_hz']),
            n_fft=int(d['n_fft']),
            n_taps=int(d['n_taps']),
            source=str(d['source']),
            n_sounding_positions=int(d.get('n_sounding_positions', 0)),
        )
        if 'hall_dims' in d and d['hall_dims'].size > 0:
            hd = d['hall_dims']
            params.hall_dims = (float(hd[0]), float(hd[1]), float(hd[2]))
        return cls(params)

    # ── Channel Generation ───────────────────────────────────────────────

    def _build_exponential_pdp(self, tau_rms_ns):
        """Build normalized exponential PDP for given τ_rms."""
        gamma = tau_rms_ns / self._ts_ns
        n = self.params.n_taps
        if gamma < 0.01:
            pdp = np.zeros(n)
            pdp[0] = 1.0
            return pdp
        pdp = np.exp(-np.arange(n) / gamma)
        pdp /= np.sum(pdp)
        return pdp

    def _draw_tau_rms(self, rng, size=None):
        """Draw τ_rms from log-normal distribution."""
        p = self.params
        log_tau = rng.normal(p.mu_lgDS, p.sigma_lgDS, size=size)
        return 10 ** log_tau  # ns

    def generate(self, h_dt, n_frames: int = 500, n_drops: int = 20,
                 rng=None):
        """Generate channel realizations at a single position (RT scale).

        Args:
            h_dt: (N_SC,) complex — raw RT channel prediction.
            n_frames: Temporal snapshots per drop.
            n_drops: Independent DMC realizations.
        Returns:
            List of n_drops arrays, each (n_frames, N_SC) complex128.
        """
        if rng is None:
            rng = np.random.default_rng()

        p = self.params
        h_dt = np.asarray(h_dt, dtype=np.complex128)
        n_sc = len(h_dt)
        p_dt = float(np.mean(np.abs(h_dt)**2))

        all_H = []
        for _ in range(n_drops):
            # Draw P_scatter relative to P_dt
            p_scat_db = rng.normal(p.p_scat_dt_mean_db, p.p_scat_dt_std_db)
            p_scat = p_dt * 10 ** (p_scat_db / 10)

            # Draw τ_rms for this drop's DMC
            tau_rms = self._draw_tau_rms(rng)
            pdp = self._build_exponential_pdp(tau_rms)

            # Generate DMC CIR
            scaled_pdp = pdp * p_scat
            cir = np.zeros(n_sc, dtype=np.complex128)
            for l in range(min(p.n_taps, n_sc)):
                if scaled_pdp[l] > 1e-30:
                    cir[l] = np.sqrt(scaled_pdp[l] / 2) * (
                        rng.standard_normal() + 1j * rng.standard_normal())
            h_dmc = np.fft.fft(cir)

            h_static = h_dt + h_dmc
            p_static = float(np.mean(np.abs(h_static)**2))

            # Frame-to-frame fading
            rho = np.clip(rng.normal(p.rho_mean, p.rho_std), 0.05, 0.995)
            sigma_sq = p_static * (1.0 / rho - 1.0)
            fading = np.sqrt(sigma_sq / 2) * (
                rng.standard_normal((n_frames, n_sc))
                + 1j * rng.standard_normal((n_frames, n_sc)))

            all_H.append(h_static[None, :] + fading)

        return all_H

    def generate_batch(self, h_dt_batch, rng=None, bw_hz=None):
        """Vectorized single-frame generation for training (RT scale).

        Args:
            h_dt_batch: (batch, N_OCC) complex — raw RT channels.
            bw_hz: Override bandwidth for DMC generation (default: model's BW).
        Returns:
            H_true: (batch, N_OCC) complex64
        """
        if rng is None:
            rng = np.random.default_rng()

        p = self.params
        batch_size, n_occ = h_dt_batch.shape
        h_dt = h_dt_batch.astype(np.complex128)
        p_dt = np.mean(np.abs(h_dt)**2, axis=1)  # (batch,)

        # P_scatter relative to P_dt (per sample)
        p_scat_db = rng.normal(p.p_scat_dt_mean_db, p.p_scat_dt_std_db,
                               batch_size)
        p_scat = p_dt * 10 ** (p_scat_db / 10)

        # Per-sample τ_rms → exponential PDP
        ts_ns = 1e9 / (bw_hz or p.bw_hz)
        tau_rms_arr = self._draw_tau_rms(rng, size=batch_size)
        gamma_arr = tau_rms_arr / ts_ns  # (batch,)

        tap_idx = np.arange(p.n_taps)[None, :]  # (1, n_taps)
        pdp_batch = np.exp(-tap_idx / np.maximum(gamma_arr[:, None], 0.01))
        pdp_batch /= np.sum(pdp_batch, axis=1, keepdims=True)

        # Scale by P_scatter
        scaled_pdp = pdp_batch * p_scat[:, None]  # (batch, n_taps)

        # DMC CIR
        z = (rng.standard_normal((batch_size, p.n_taps))
             + 1j * rng.standard_normal((batch_size, p.n_taps))) / np.sqrt(2)
        cir_dmc = z * np.sqrt(scaled_pdp)

        # CIR → CFR on occupied bins
        cir_padded = np.zeros((batch_size, p.n_fft), dtype=np.complex128)
        cir_padded[:, :p.n_taps] = cir_dmc
        occ_bins = np.array(list(range(p.n_fft - n_occ // 2, p.n_fft))
                            + list(range(1, n_occ // 2 + 1)))
        H_dmc = np.fft.fft(cir_padded, axis=1)[:, occ_bins]

        h_static = h_dt + H_dmc
        p_static = np.mean(np.abs(h_static)**2, axis=1)

        # Fading
        rho = np.clip(
            rng.normal(p.rho_mean, p.rho_std, batch_size), 0.05, 0.995)
        sigma_sq = p_static * (1.0 / rho - 1.0)
        fading = np.sqrt(sigma_sq[:, None] / 2) * (
            rng.standard_normal((batch_size, n_occ))
            + 1j * rng.standard_normal((batch_size, n_occ)))

        return (h_static + fading).astype(np.complex64)

    # ── Analytic R_ee ────────────────────────────────────────────────────

    def compute_ree_analytic(self, h_dt, bw_hz=None, n_fft=None,
                             occ_bins=None, rho_override=None):
        """Compute analytic R_ee from Q-D model parameters (blind).

        R_ee = F · diag(PDP · P_scatter) · F^H  +  P_total·(1/ρ - 1) · I
               └── structured DMC component ──┘    └── flat fading ────────┘

        Works at any BW — the PDP shape adapts to the delay resolution.

        Args:
            h_dt: (N_SC,) or (N_OCC,) complex — raw RT channel (unnormalized).
            bw_hz: Override bandwidth (default: model's fitted BW).
            n_fft: Override FFT size (default: model's n_fft).
            occ_bins: Occupied subcarrier indices (default: 192 OFDM bins).
            rho_override: Use measured ρ instead of environment mean.
        Returns:
            R_EE: (N_OCC, N_OCC) complex128 — error covariance.
            info: dict with P_scatter, rho, PDP, eigenvalues.
        """
        p = self.params
        bw = bw_hz or p.bw_hz
        nfft = n_fft or p.n_fft
        ts_ns = 1e9 / bw

        if occ_bins is None:
            n_occ = 192
            occ_bins = np.array(
                list(range(nfft - n_occ // 2, nfft)) +
                list(range(1, n_occ // 2 + 1)))
        n_occ = len(occ_bins)

        h_dt = np.asarray(h_dt, dtype=np.complex128)
        p_dt = float(np.mean(np.abs(h_dt)**2))

        p_scat = p_dt * 10 ** (p.p_scat_dt_mean_db / 10)

        tau_rms_ns = 10 ** p.mu_lgDS
        gamma = tau_rms_ns / ts_ns
        pdp = np.exp(-np.arange(p.n_taps) / max(gamma, 0.01))
        pdp /= np.sum(pdp)
        scaled_pdp = pdp * p_scat

        tap_idx = np.arange(p.n_taps)
        F_occ = np.exp(-1j * 2 * np.pi
                        * occ_bins[:, None] * tap_idx[None, :] / nfft)
        R_dmc = F_occ @ np.diag(scaled_pdp) @ F_occ.conj().T

        rho = rho_override if rho_override is not None else p.rho_mean
        p_total = p_dt + p_scat
        sigma_fading_sq = p_total * (1.0 / rho - 1.0)

        R_ee = R_dmc + sigma_fading_sq * np.eye(n_occ)

        eig_vals = np.sort(np.real(np.linalg.eigvalsh(R_ee)))[::-1]
        eig_vals = np.maximum(eig_vals, 0.0)

        info = {
            'P_dt': p_dt, 'P_scatter': p_scat, 'rho': rho,
            'sigma_fading_sq': sigma_fading_sq,
            'tau_rms_ns': tau_rms_ns, 'bw_hz': bw,
            'trace_R_ee': float(np.sum(eig_vals)),
            'eff_rank_90': int(np.searchsorted(
                np.cumsum(eig_vals) / (np.sum(eig_vals) + 1e-30), 0.9)) + 1,
        }
        return R_ee, info

    def compute_ree_eigen(self, h_dt, bw_hz=None, n_fft=None,
                          occ_bins=None, rho_override=None):
        """Analytic R_ee with eigendecomposition for efficient LMMSE.

        Returns (U, lambda, info) where R_ee = U @ diag(lambda) @ U^H.
        """
        R_ee, info = self.compute_ree_analytic(
            h_dt, bw_hz, n_fft, occ_bins, rho_override)
        eig_vals, eig_vecs = np.linalg.eigh(R_ee)
        eig_vals = eig_vals[::-1].real
        eig_vecs = eig_vecs[:, ::-1]
        eig_vals = np.maximum(eig_vals, 0.0)
        return eig_vecs, eig_vals, info

    # ── Scale helpers ────────────────────────────────────────────────────

    def to_ota_scale(self, H):
        """Scale RT-space channel to OTA power level."""
        return H * self._sqrt_alpha

    def to_rt_scale(self, H):
        """Scale OTA-space channel to raw RT power level."""
        return H / self._sqrt_alpha

    # ── I/O ──────────────────────────────────────────────────────────────

    def save(self, path: str):
        p = self.params
        np.savez_compressed(path,
            alpha_db=p.alpha_db,
            mu_lgDS=p.mu_lgDS, sigma_lgDS=p.sigma_lgDS,
            tau_scatter_ns=p.tau_scatter_ns,
            p_scat_dt_mean_db=p.p_scat_dt_mean_db,
            p_scat_dt_std_db=p.p_scat_dt_std_db,
            rho_mean=p.rho_mean, rho_std=p.rho_std,
            fc_hz=p.fc_hz, bw_hz=p.bw_hz, n_fft=p.n_fft, n_taps=p.n_taps,
            hall_dims=np.array(p.hall_dims) if p.hall_dims else np.array([]),
            source=str(p.source),
            n_sounding_positions=p.n_sounding_positions,
        )
        print(f'Saved: {path}')

    def summary(self) -> str:
        p = self.params
        lines = [
            f'General Q-D Channel Model',
            f'  Source: {p.source}',
            f'  fc = {p.fc_hz/1e9:.1f} GHz, BW = {p.bw_hz/1e6:.0f} MHz, '
            f'FFT = {p.n_fft}',
            f'  α = {p.alpha_db:.1f} dB (√α = {self._sqrt_alpha:.1f})',
        ]
        if p.hall_dims:
            L, W, H = p.hall_dims
            lines.append(f'  Hall: {L:.0f}x{W:.0f}x{H:.0f} m')
        lines += [
            f'  τ_rms ~ LogNormal: μ={p.mu_lgDS:.3f}, σ={p.sigma_lgDS:.3f} '
            f'(median {10**p.mu_lgDS:.1f} ns)',
            f'  τ_scatter = {p.tau_scatter_ns:.1f} ns (DMC decay)',
            f'  P_scatter/P_dt ~ N({p.p_scat_dt_mean_db:.1f}, '
            f'{p.p_scat_dt_std_db:.1f}) dB',
            f'  ρ ~ N({p.rho_mean:.3f}, {p.rho_std:.3f})',
            f'  PDP: exponential, Fading: Rayleigh',
        ]
        if p.n_sounding_positions:
            lines.append(
                f'  Fitted from {p.n_sounding_positions} sounding positions')
        return '\n'.join(lines)

    def __repr__(self):
        return self.summary()


def tau_rms_3gpp_inf(length, width, height, condition='los'):
    """3GPP TR 38.901 InF median RMS delay spread in nanoseconds."""
    v = length * width * height
    s = 2 * (length * width + length * height + width * height)
    vs = v / s
    if condition == 'los':
        mu = np.log10(26 * vs + 14) - 9.35
    else:
        mu = np.log10(30 * vs + 32) - 9.44
    return 10 ** (mu + 9)


if __name__ == '__main__':
    print('=== Test: from_hall_geometry (Weeks Hall) ===')
    m = GeneralQDChannel.from_hall_geometry(19, 13.5, 5)
    print(m)
    print()

    print('=== Test: generate ===')
    h_dt = np.random.randn(256) + 1j * np.random.randn(256)
    h_dt *= 1e-3
    H = m.generate(h_dt, n_frames=100, n_drops=3)
    for i, h in enumerate(H):
        p = 10*np.log10(np.mean(np.abs(h)**2)+1e-30)
        print(f'  Drop {i}: {h.shape}, P={p:.1f} dB')

    print('\n=== Test: generate_batch ===')
    h_batch = (np.random.randn(16, 192) + 1j*np.random.randn(16, 192)) * 1e-3
    Hb = m.generate_batch(h_batch.astype(np.complex64))
    print(f'  In: {h_batch.shape}, Out: {Hb.shape}')

    print('\n=== Test: save/load ===')
    m.save('/tmp/test_general_qd.npz')
    m2 = GeneralQDChannel.from_env_model('/tmp/test_general_qd.npz')
    print(m2)

    print('\n=== Test: scale helpers ===')
    print(f'  √α = {m.sqrt_alpha:.1f}, 1/√α = {m.inv_sqrt_alpha:.6f}')

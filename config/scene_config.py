"""
Calibrated scene configuration — UPES per-position, D7 + refraction.

4 materials calibrated via gradient-based UPES optimization across
20 RX positions at 3 bandwidths (20/25/50 MHz).
Source: data/calibration/material_calibration_upes_perpos_d7refr.json

RT config: max_depth=7, specular reflection + refraction,
diffuse_reflection=False. See RT_CONFIG dict.
"""

FC = 3.5e9

CALIBRATED_MATERIALS = {
    "concrete-wall":  dict(eps_r=2.706935, sigma=0.291850),
    "concrete-floor": dict(eps_r=5.183538, sigma=0.035833),
    "glass":          dict(eps_r=6.698977, sigma=0.008569),
    "wood":           dict(eps_r=2.095868, sigma=0.004722),
}

ITU_MATERIALS = {
    "ceiling-board":  dict(eps_r=1.4800, sigma=0.004229),
    "metal-trim":     dict(eps_r=1.0, sigma=1e7),
    "metal-deck":     dict(eps_r=1.0, sigma=1e7),
    "metal-machine":  dict(eps_r=1.0, sigma=1e7),
    "metal-door":     dict(eps_r=1.0, sigma=1e7),
    "metal-struct":   dict(eps_r=1.0, sigma=1e7),
    "metal-railing":  dict(eps_r=1.0, sigma=1e7),
    "metal-duct":     dict(eps_r=1.0, sigma=1e7),
}

MATERIALS = {**CALIBRATED_MATERIALS, **ITU_MATERIALS}

RT_CONFIG = dict(
    max_depth=7,
    los=True,
    specular_reflection=True,
    diffuse_reflection=False,
    refraction=True,
    synthetic_array=True,
    seed=42,
)


def apply_calibration(scene):
    """Apply calibrated + ITU material parameters (no scattering)."""
    applied = []
    for name, params in MATERIALS.items():
        mat = scene.radio_materials.get(name)
        if mat is None:
            continue
        mat.relative_permittivity = params["eps_r"]
        mat.conductivity = params["sigma"]
        applied.append(name)
    return applied

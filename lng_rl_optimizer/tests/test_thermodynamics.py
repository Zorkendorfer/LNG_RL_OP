import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.physics.thermodynamics import LNGComposition, compute_bog_rate_physics


def test_composition_sums_to_one():
    comp = LNGComposition()
    total = comp.methane + comp.ethane + comp.propane + comp.nitrogen
    assert abs(total - 1.0) < 1e-6


def test_bog_rate_physical_range():
    comp = LNGComposition()
    bog  = compute_bog_rate_physics(180.0, comp)
    assert 100 < bog < 400, f"BOG rate {bog:.1f} kg/h outside physical range"


def test_bog_rate_increases_with_heat():
    comp = LNGComposition()
    bog_low  = compute_bog_rate_physics(100.0, comp)
    bog_high = compute_bog_rate_physics(300.0, comp)
    assert bog_high > bog_low


def test_latent_heat_methane_rich():
    comp = LNGComposition(methane=0.95, ethane=0.03, propane=0.01, nitrogen=0.01)
    assert comp.latent_heat_kJ_kg > 490  # close to pure methane (509 kJ/kg)


def test_wobbe_index():
    comp = LNGComposition()
    wi = comp.wobbe_index
    assert 40 < wi < 60, f"Wobbe index {wi:.2f} outside typical LNG range"

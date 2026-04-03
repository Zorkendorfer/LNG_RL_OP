import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.physics.bog_model import BOGModel
from src.physics.thermodynamics import LNGComposition


def get_config():
    return yaml.safe_load(open("config/terminal.yaml"))


def test_sea_state_increases_bog():
    config = get_config()
    model  = BOGModel(config)
    comp   = LNGComposition()
    calm   = model.steady_state_bog(comp, 0.7, 10.0, sea_state=0.0)
    rough  = model.steady_state_bog(comp, 0.7, 10.0, sea_state=1.0)
    assert rough > calm, "Rough sea should produce more BOG"


def test_ambient_temperature_effect():
    config = get_config()
    model  = BOGModel(config)
    comp   = LNGComposition()
    cold   = model.steady_state_bog(comp, 0.7, -5.0, sea_state=0.0)
    warm   = model.steady_state_bog(comp, 0.7, 30.0, sea_state=0.0)
    assert warm > cold, "Higher ambient temperature should increase BOG"


def test_flash_bog_only_on_pressure_drop():
    config = get_config()
    model  = BOGModel(config)
    comp   = LNGComposition()
    # Pressure rise → no flash BOG
    assert model.flash_bog(100000, 5.0, comp) == 0.0
    # Pressure drop → flash BOG
    assert model.flash_bog(100000, -5.0, comp) > 0.0

# =========================================================
# Configuration loader
#
# Reads config/simulation_config.yaml and returns typed
# dataclass objects. Falls back to sensible defaults if
# the YAML file is not found.
# =========================================================

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.models.tyre import TyreCompound, TYRE_COMPOUNDS


_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "simulation_config.yaml"


@dataclass(frozen=True)
class SimulationConfig:
    step_size:      float = 5.0
    race_laps:      int   = 8
    multi_lap_laps: int   = 5


@dataclass(frozen=True)
class StrategyConfig:
    compounds:             tuple[TyreCompound, ...]
    pit_loss:              float
    min_stint_laps:        int
    max_stops:             int
    require_two_compounds: bool
    ranking_top_n:         int
    gap_top_n:             int
    pit_loss_sensitivity:  tuple[float, ...]


def load_config(
    path: str | Path | None = None,
) -> tuple[SimulationConfig, StrategyConfig]:
    """
    Load simulation and strategy configuration from a YAML file.

    Parameters
    ----------
    path : str | Path | None
        Path to the YAML config file.
        Defaults to config/simulation_config.yaml.

    Returns
    -------
    tuple[SimulationConfig, StrategyConfig]
    """
    config_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH

    if not config_path.exists():
        return _defaults()

    try:
        import yaml
    except ImportError:
        print(
            "Warning: PyYAML not installed. "
            "Using default configuration. "
            "Install with: pip install pyyaml"
        )
        return _defaults()

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    sim_data   = data.get("simulation", {})
    strat_data = data.get("strategy", {})

    simulation_config = SimulationConfig(
        step_size      = float(sim_data.get("step_size",      5.0)),
        race_laps      = int(sim_data.get("race_laps",        8)),
        multi_lap_laps = int(sim_data.get("multi_lap_laps",   5)),
    )

    raw_compounds = strat_data.get("compounds", ["soft", "medium", "hard"])
    compounds = tuple(
        TYRE_COMPOUNDS[name.lower()]
        for name in raw_compounds
        if name.lower() in TYRE_COMPOUNDS
    )

    strategy_config = StrategyConfig(
        compounds             = compounds,
        pit_loss              = float(strat_data.get("pit_loss",              22.0)),
        min_stint_laps        = int(strat_data.get("min_stint_laps",          2)),
        max_stops             = int(strat_data.get("max_stops",               2)),
        require_two_compounds = bool(strat_data.get("require_two_compounds",  True)),
        ranking_top_n         = int(strat_data.get("ranking_top_n",           10)),
        gap_top_n             = int(strat_data.get("gap_top_n",               5)),
        pit_loss_sensitivity  = tuple(
            float(v) for v in strat_data.get(
                "pit_loss_sensitivity", [18.0, 20.0, 22.0, 24.0, 26.0]
            )
        ),
    )

    return simulation_config, strategy_config


def _defaults() -> tuple[SimulationConfig, StrategyConfig]:
    from src.models.tyre import SOFT, MEDIUM, HARD
    return (
        SimulationConfig(),
        StrategyConfig(
            compounds             = (SOFT, MEDIUM, HARD),
            pit_loss              = 22.0,
            min_stint_laps        = 2,
            max_stops             = 2,
            require_two_compounds = True,
            ranking_top_n         = 10,
            gap_top_n             = 5,
            pit_loss_sensitivity  = (18.0, 20.0, 22.0, 24.0, 26.0),
        ),
    )

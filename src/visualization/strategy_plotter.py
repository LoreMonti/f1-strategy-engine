# =========================================================
# Strategy Plotter
#
# Comparison plots across a pool of RaceResult objects.
# =========================================================

from __future__ import annotations

import os
import matplotlib.pyplot as plt

from src.models.strategy import RaceResult


_OUTPUT_DIR = os.path.join("results", "plots")


class StrategyPlotter:
    """Static methods for visualising and comparing race strategies."""

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def plot_strategy_gaps(
        race_results: list[RaceResult],
        top_n: int = 5,
        virtual: bool = True,
        output_dir: str = _OUTPUT_DIR,
    ) -> None:
        """
        Gap to the fastest strategy, lap by lap.

        If virtual=True, pit losses are stripped so pure pace is visible.
        """
        selected  = race_results[:top_n]
        reference = selected[0]
        num_laps  = reference.num_laps

        def _gap(result: RaceResult, idx: int) -> float:
            if virtual:
                ref_t = _virtual_cum(reference, idx)
                t     = _virtual_cum(result,    idx)
            else:
                ref_t = reference.laps[idx].cumulative_time
                t     = result.laps[idx].cumulative_time
            return t - ref_t

        laps   = list(range(1, num_laps + 1))
        series = {r.strategy: [_gap(r, i) for i in range(num_laps)] for r in selected}

        title    = "Virtual Strategy Gap Evolution" if virtual else "Strategy Gap Evolution"
        filename = "virtual_strategy_gap_evolution.png" if virtual else "strategy_gap_evolution.png"

        _line_plot(
            laps=laps,
            series=series,
            xlabel="Lap",
            ylabel="Gap to Leader [s]",
            title=title,
            output_dir=output_dir,
            filename=filename,
        )
        print(f"Strategy gap plot saved → {filename}")

    @staticmethod
    def plot_strategy_lap_times(
        race_results: list[RaceResult],
        top_n: int = 5,
        output_dir: str = _OUTPUT_DIR,
    ) -> None:
        """Raw lap time evolution for the top strategies."""
        StrategyPlotter._plot_lap_metric(
            race_results=race_results,
            top_n=top_n,
            extractor=lambda lr: lr.raw_lap_time,
            ylabel="Raw Lap Time [s]",
            title="Strategy Raw Lap Time Evolution",
            filename="strategy_lap_times.png",
            output_dir=output_dir,
        )
        print("Strategy lap time plot saved.")

    @staticmethod
    def plot_strategy_tyre_wear(
        race_results: list[RaceResult],
        top_n: int = 5,
        output_dir: str = _OUTPUT_DIR,
    ) -> None:
        """End-of-lap tyre wear evolution."""
        StrategyPlotter._plot_lap_metric(
            race_results=race_results,
            top_n=top_n,
            extractor=lambda lr: lr.final_tyre_wear * 100.0,
            ylabel="Tyre Wear [%]",
            title="Strategy Tyre Wear Evolution",
            filename="strategy_tyre_wear.png",
            output_dir=output_dir,
        )
        print("Strategy tyre wear plot saved.")

    @staticmethod
    def plot_strategy_temperature(
        race_results: list[RaceResult],
        top_n: int = 5,
        output_dir: str = _OUTPUT_DIR,
    ) -> None:
        """End-of-lap tyre temperature evolution."""
        StrategyPlotter._plot_lap_metric(
            race_results=race_results,
            top_n=top_n,
            extractor=lambda lr: lr.final_tyre_temperature,
            ylabel="Tyre Temperature [°C]",
            title="Strategy Tyre Temperature Evolution",
            filename="strategy_tyre_temperature.png",
            output_dir=output_dir,
        )
        print("Strategy tyre temperature plot saved.")

    @staticmethod
    def plot_strategy_grip(
        race_results: list[RaceResult],
        top_n: int = 5,
        output_dir: str = _OUTPUT_DIR,
    ) -> None:
        """End-of-lap grip multiplier evolution."""
        StrategyPlotter._plot_lap_metric(
            race_results=race_results,
            top_n=top_n,
            extractor=lambda lr: lr.final_grip_multiplier,
            ylabel="Grip Multiplier",
            title="Strategy Grip Evolution",
            filename="strategy_grip.png",
            output_dir=output_dir,
        )
        print("Strategy grip plot saved.")

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _plot_lap_metric(
        race_results: list[RaceResult],
        top_n: int,
        extractor,
        ylabel: str,
        title: str,
        filename: str,
        output_dir: str,
    ) -> None:
        """
        Generic per-lap metric plot.

        extractor : callable(LapResult) -> float
            Function that pulls the desired value from each LapResult.
        """
        selected = race_results[:top_n]
        num_laps = selected[0].num_laps
        laps     = list(range(1, num_laps + 1))

        series = {
            r.strategy: [extractor(lr) for lr in r.laps]
            for r in selected
        }

        _line_plot(
            laps=laps,
            series=series,
            xlabel="Lap",
            ylabel=ylabel,
            title=title,
            output_dir=output_dir,
            filename=filename,
        )


# ------------------------------------------------------------------ #
# Module-level helpers                                                #
# ------------------------------------------------------------------ #

def _virtual_cum(result: RaceResult, lap_index: int) -> float:
    """Cumulative race time minus pit losses up to lap_index."""
    return result.laps[lap_index].cumulative_time - sum(
        lr.pit_time_loss for lr in result.laps[: lap_index + 1]
    )


def _line_plot(
    laps: list[int],
    series: dict[str, list[float]],
    xlabel: str,
    ylabel: str,
    title: str,
    output_dir: str,
    filename: str,
) -> None:
    """Shared line-plot renderer used by all strategy plot methods."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    for label, values in series.items():
        ax.plot(laps, values, marker="o", linewidth=2, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)

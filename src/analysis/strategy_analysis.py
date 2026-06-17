# =========================================================
# Strategy Analysis
#
# Post-processing tools for race strategy results:
# - pit window analysis
# - gap evolution (real and virtual)
# - undercut / overcut gain
# - pit-loss sensitivity
# =========================================================

from __future__ import annotations

from src.models.strategy import RaceResult, LapResult
from src.models.tyre import TyreCompound
from src.optimization.strategy_search import generate_and_simulate
from src.simulation.race_simulator import RaceSimulator


class StrategyAnalyzer:
    """
    Analyzes a pool of RaceResult objects.

    Parameters
    ----------
    race_results : list[RaceResult]
        Results to analyze, in any order. Internally sorted fastest-first.
    """

    def __init__(self, race_results: list[RaceResult]) -> None:
        self.race_results: list[RaceResult] = sorted(
            race_results, key=lambda r: r.total_time
        )

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def best_result(self) -> RaceResult:
        return self.race_results[0]

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pit_laps(result: RaceResult) -> tuple[int, ...]:
        return tuple(lr.lap for lr in result.laps if lr.pit_stop)

    @staticmethod
    def _compound_sequence(result: RaceResult) -> tuple[str, ...]:
        return tuple(stint.compound.name for stint in result.stints)

    @staticmethod
    def _pit_loss_so_far(result: RaceResult, lap_index: int) -> float:
        return sum(lr.pit_time_loss for lr in result.laps[: lap_index + 1])

    @classmethod
    def _virtual_cumulative_time(cls, result: RaceResult, lap_index: int) -> float:
        """Cumulative time minus all pit losses up to this lap."""
        return (
            result.laps[lap_index].cumulative_time
            - cls._pit_loss_so_far(result, lap_index)
        )

    def _grouped_by_compound_sequence(
        self,
    ) -> dict[tuple[str, ...], list[RaceResult]]:
        grouped: dict[tuple[str, ...], list[RaceResult]] = {}
        for result in self.race_results:
            key = self._compound_sequence(result)
            grouped.setdefault(key, []).append(result)
        return grouped

    # ------------------------------------------------------------------ #
    # Pit window analysis                                                  #
    # ------------------------------------------------------------------ #

    def print_pit_windows(self) -> None:
        """
        Group strategies by compound sequence and compare pit windows.
        """
        print("\n" + "=" * 50)
        print("PIT WINDOW ANALYSIS")
        print("=" * 50)

        for sequence, results in sorted(self._grouped_by_compound_sequence().items()):
            results_sorted = sorted(results, key=lambda r: r.total_time)
            best_time      = results_sorted[0].total_time
            sequence_name  = "-".join(sequence)

            print(f"\nCompound sequence: {sequence_name}")
            print("-" * 60)
            print(
                f"{'Pit Lap(s)':>12} | "
                f"{'Total [s]':>10} | "
                f"{'Delta [s]':>9} | "
                f"{'Strategy':<20}"
            )
            print("-" * 60)

            for result in results_sorted:
                pit_text = "/".join(f"L{p}" for p in self._pit_laps(result))
                delta    = result.total_time - best_time
                print(
                    f"{pit_text:>12} | "
                    f"{result.total_time:10.3f} | "
                    f"{delta:9.3f} | "
                    f"{result.strategy:<20}"
                )

    # ------------------------------------------------------------------ #
    # Gap evolution                                                        #
    # ------------------------------------------------------------------ #

    def print_gap_evolution(self, top_n: int = 5) -> None:
        """Lap-by-lap cumulative gap to the leader (includes pit losses)."""
        self._print_gap_table(
            top_n=top_n,
            title="STRATEGY GAP EVOLUTION",
            virtual=False,
        )

    def print_virtual_gap_evolution(self, top_n: int = 5) -> None:
        """
        Lap-by-lap virtual gap to the leader (pit losses stripped).

        Virtual time highlights pure race pace without the spike caused
        by pit-stop time losses.
        """
        self._print_gap_table(
            top_n=top_n,
            title="VIRTUAL STRATEGY GAP EVOLUTION",
            virtual=True,
        )

    def _print_gap_table(
        self,
        top_n: int,
        title: str,
        virtual: bool,
    ) -> None:
        selected  = self.race_results[:top_n]
        reference = selected[0]
        num_laps  = reference.num_laps

        print("\n" + "=" * 50)
        print(title)
        print("=" * 50)

        header = f"{'Lap':>3}"
        for r in selected:
            header += f" | {r.strategy:<18}"
        print(header)
        print("-" * len(header))

        for idx in range(num_laps):
            ref_time = (
                self._virtual_cumulative_time(reference, idx)
                if virtual
                else reference.laps[idx].cumulative_time
            )

            row = f"{idx + 1:3d}"
            for r in selected:
                t = (
                    self._virtual_cumulative_time(r, idx)
                    if virtual
                    else r.laps[idx].cumulative_time
                )
                row += f" | {t - ref_time:18.3f}"
            print(row)

    # ------------------------------------------------------------------ #
    # Undercut / overcut gain                                              #
    # ------------------------------------------------------------------ #

    def print_undercut_gain(self, only_one_stop: bool = True) -> None:
        """
        Quantify undercut/overcut gain relative to the best pit window
        for each compound sequence.
        """
        grouped: dict[tuple[str, ...], list[RaceResult]] = {}

        for result in self.race_results:
            pit_laps = self._pit_laps(result)
            if only_one_stop and len(pit_laps) != 1:
                continue
            key = self._compound_sequence(result)
            grouped.setdefault(key, []).append(result)

        print("\n" + "=" * 50)
        print("UNDERCUT / OVERCUT GAIN ANALYSIS")
        print("=" * 50)

        for sequence, results in sorted(grouped.items()):
            if len(results) < 2:
                continue

            results_sorted = sorted(results, key=lambda r: r.total_time)
            reference      = results_sorted[0]
            ref_pit        = self._pit_laps(reference)
            sequence_name  = "-".join(sequence)

            print(f"\nCompound sequence: {sequence_name}")
            print(
                "Reference pit window: "
                + "/".join(f"L{p}" for p in ref_pit)
            )
            print("-" * 90)
            print(
                f"{'Pit Lap(s)':>12} | "
                f"{'Peak Gain [s]':>13} | "
                f"{'Peak Loss [s]':>13} | "
                f"{'Final Delta [s]':>15} | "
                f"{'Strategy':<20}"
            )
            print("-" * 90)

            for result in results_sorted[1:]:
                virtual_gaps = [
                    self._virtual_cumulative_time(result, idx)
                    - self._virtual_cumulative_time(reference, idx)
                    for idx in range(result.num_laps)
                ]

                pit_text    = "/".join(f"L{p}" for p in self._pit_laps(result))
                peak_gain   = min(virtual_gaps)
                peak_loss   = max(virtual_gaps)
                final_delta = result.total_time - reference.total_time

                print(
                    f"{pit_text:>12} | "
                    f"{peak_gain:13.3f} | "
                    f"{peak_loss:13.3f} | "
                    f"{final_delta:15.3f} | "
                    f"{result.strategy:<20}"
                )

    # ------------------------------------------------------------------ #
    # Pit-loss sensitivity                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def print_pit_loss_sensitivity(
        race_simulator: RaceSimulator,
        num_laps: int,
        compounds: list[TyreCompound],
        pit_losses: list[float],
        step_size: float = 5.0,
        min_stint_laps: int = 2,
        max_stops: int = 2,
        require_two_compounds: bool = True,
    ) -> None:
        """
        Show how the optimal strategy changes as pit-loss varies.
        """
        print("\n" + "=" * 50)
        print("PIT LOSS SENSITIVITY ANALYSIS")
        print("=" * 50)
        print(
            f"{'Pit Loss [s]':>12} | "
            f"{'Best Strategy':<22} | "
            f"{'Total [s]':>10} | "
            f"{'Stops':>5}"
        )
        print("-" * 60)

        for pit_loss in pit_losses:
            results = generate_and_simulate(
                race_simulator=race_simulator,
                num_laps=num_laps,
                compounds=compounds,
                pit_loss=pit_loss,
                min_stint_laps=min_stint_laps,
                max_stops=max_stops,
                require_two_compounds=require_two_compounds,
                step_size=step_size,
            )
            best = results[0]
            print(
                f"{pit_loss:12.1f} | "
                f"{best.strategy:<22} | "
                f"{best.total_time:10.3f} | "
                f"{best.num_stops:5d}"
            )

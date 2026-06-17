# =========================================================
# Telemetry Plotter
#
# Plots for single-lap and multi-lap simulation results.
# =========================================================

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


_OUTPUT_DIR = os.path.join("results", "plots")


class TelemetryPlotter:
    """Static methods for visualising lap telemetry."""

    @staticmethod
    def plot_telemetry(telemetry: dict, output_dir: str = _OUTPUT_DIR) -> None:
        """
        Full telemetry dashboard for a single lap.

        Panels: speed, throttle/brake, gear, RPM, tyre wear/temp,
        downforce/drag, grip multiplier.
        """
        os.makedirs(output_dir, exist_ok=True)

        s    = telemetry["s"]
        fig  = plt.figure(figsize=(14, 20))
        gs   = gridspec.GridSpec(7, 1, hspace=0.45)

        panels = [
            (0, "Speed [km/h]",         [("v_kmh",          "royalblue",  "Speed")]),
            (1, "Throttle / Brake [%]",  [("throttle",       "limegreen",  "Throttle"),
                                          ("brake",           "crimson",    "Brake")]),
            (2, "Gear",                  [("gear",            "darkorange", "Gear")]),
            (3, "RPM",                   [("rpm",             "purple",     "RPM")]),
            (4, "Tyre Wear [%]",         [("front_tyre_wear", "steelblue",  "Front"),
                                          ("rear_tyre_wear",  "tomato",     "Rear")]),
            (5, "Tyre Temperature [°C]", [("front_tyre_temperature", "steelblue", "Front"),
                                          ("rear_tyre_temperature",  "tomato",    "Rear")]),
            (6, "Grip Multiplier",       [("front_grip_multiplier",  "steelblue", "Front"),
                                          ("rear_grip_multiplier",   "tomato",    "Rear")]),
        ]

        for row, ylabel, channels in panels:
            ax = fig.add_subplot(gs[row])
            for key, color, label in channels:
                data = telemetry.get(key, [])
                if key in ("front_tyre_wear", "rear_tyre_wear"):
                    data = np.array(data) * 100.0
                ax.plot(s, data, color=color, linewidth=1.5, label=label)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(True, alpha=0.3)
            if len(channels) > 1:
                ax.legend(fontsize=8, loc="upper right")

        fig.axes[-1].set_xlabel("Distance [m]")
        fig.suptitle("Lap Telemetry", fontsize=14, fontweight="bold")

        _save_and_close(fig, output_dir, "lap_telemetry.png")
        print("Telemetry plot saved.")

    @staticmethod
    def plot_multi_lap_summary(multi_lap_result: dict, output_dir: str = _OUTPUT_DIR) -> None:
        """
        Summary plots across multiple consecutive laps:
        lap time, tyre wear, tyre temperature, fuel mass.
        """
        os.makedirs(output_dir, exist_ok=True)

        laps      = [lap["lap"]                       for lap in multi_lap_result["laps"]]
        lap_times = [lap["lap_time"]                  for lap in multi_lap_result["laps"]]
        wear      = [lap["final_tyre_wear"] * 100.0   for lap in multi_lap_result["laps"]]
        temp      = [lap["final_tyre_temperature"]    for lap in multi_lap_result["laps"]]
        fuel      = [lap["final_fuel_mass"]           for lap in multi_lap_result["laps"]]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(
            f"Multi-Lap Summary — {multi_lap_result['tyre_compound']}",
            fontsize=13,
            fontweight="bold",
        )

        _bar_panel(axes[0, 0], laps, lap_times, "Lap",    "Lap Time [s]",        "royalblue",  "Lap Times")
        _bar_panel(axes[0, 1], laps, wear,      "Lap",    "Tyre Wear [%]",       "tomato",     "Tyre Wear")
        _bar_panel(axes[1, 0], laps, temp,      "Lap",    "Tyre Temperature [°C]","darkorange", "Tyre Temperature")
        _bar_panel(axes[1, 1], laps, fuel,      "Lap",    "Fuel Mass [kg]",      "seagreen",   "Fuel Mass")

        plt.tight_layout()
        _save_and_close(fig, output_dir, "multi_lap_summary.png")
        print("Multi-lap summary plot saved.")


# ------------------------------------------------------------------ #
# Private helpers                                                     #
# ------------------------------------------------------------------ #

def _bar_panel(
    ax: plt.Axes,
    x: list,
    y: list,
    xlabel: str,
    ylabel: str,
    color: str,
    title: str,
) -> None:
    ax.bar(x, y, color=color, alpha=0.8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)


def _save_and_close(fig: plt.Figure, output_dir: str, filename: str) -> None:
    fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)

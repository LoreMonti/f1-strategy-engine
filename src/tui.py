# =========================================================
# Interactive TUI — questionary + rich
#
# Launched automatically when main.py is run with no arguments.
# Returns a config dict identical in shape to argparse.Namespace
# so the rest of main() doesn't need to change.
# =========================================================

from __future__ import annotations

import glob
import os
import sys
from types import SimpleNamespace

import questionary
from questionary import Style
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box

console = Console()

# ── Questionary style (F1 dark theme) ────────────────────────────────────────
_STYLE = Style([
    ("qmark",        "fg:#e8002d bold"),
    ("question",     "bold"),
    ("answer",       "fg:#ffffff bold"),
    ("pointer",      "fg:#e8002d bold"),
    ("highlighted",  "fg:#e8002d bold"),
    ("selected",     "fg:#aaaaaa"),
    ("separator",    "fg:#444444"),
    ("instruction",  "fg:#888888"),
    ("text",         ""),
])

_TRACKS_DIR = "data/tracks"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _available_circuits() -> list[tuple[str, str]]:
    """Return list of (display_label, yaml_path) for all circuits in tracks dir."""
    paths = sorted(glob.glob(os.path.join(_TRACKS_DIR, "*.yaml")))
    result = []
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        # Pretty label: "silverstone_2024" → "Silverstone 2024"
        label = name.replace("_", " ").title()
        result.append((label, p))
    return result


def _print_header() -> None:
    title = Text()
    title.append("F1", style="bold red")
    title.append(" Lap Time Simulator", style="bold white")
    title.append(" & ", style="dim")
    title.append("Race Strategy Optimizer", style="bold white")

    console.print()
    console.print(Panel(
        title,
        subtitle="[dim]python main.py --help  for non-interactive mode[/dim]",
        border_style="red",
        padding=(0, 4),
    ))
    console.print()


def _print_summary(cfg: SimpleNamespace) -> None:
    """Print a recap table of the chosen settings before running."""
    t = Table(box=box.ROUNDED, border_style="dim", show_header=False, padding=(0, 2))
    t.add_column("Setting", style="dim")
    t.add_column("Value",   style="bold white")

    circuit_name = os.path.splitext(os.path.basename(cfg.circuit))[0].replace("_", " ").title()
    t.add_row("Circuit",    circuit_name)
    t.add_row("Solver",     "DP Optimizer" if cfg.solver == "dp" else "Brute-force")
    t.add_row("Search step", f"{cfg.step:.0f} m")
    t.add_row("Top N",      str(cfg.top_n) if cfg.top_n is not None else "from config")
    t.add_row("Multi-car",  f"Yes — {cfg.num_cars} cars" if cfg.multi_car else "No")
    _w = getattr(cfg, "wetness", None)
    if _w is None:
        t.add_row("Track",  "Circuit default (YAML weather)")
    elif _w > 0.0:
        _cond = "damp" if _w < 0.55 else "wet" if _w < 0.85 else "soaked"
        t.add_row("Track",  f"WET (static) — wetness {_w:.2f} ({_cond})")
    else:
        t.add_row("Track",  "Dry (forced)")
    if getattr(cfg, "sc_lap", None) is not None:
        t.add_row("Live SC",  f"decision at lap {cfg.sc_lap}")
    t.add_row("Output",     _output_label(cfg))
    t.add_row("Verbose",    "Yes" if cfg.verbose else "No")

    console.print(Panel(t, title="[bold]Run configuration[/bold]", border_style="red"))
    console.print()


def _output_label(cfg: SimpleNamespace) -> str:
    if cfg.dashboard:
        return f"Dashboard  (port {cfg.port})"
    if cfg.no_plots:
        return "Text only"
    return "PNG plots"


# ── Main TUI entry point ──────────────────────────────────────────────────────

def run_interactive() -> SimpleNamespace:
    """
    Show the interactive menu and return a SimpleNamespace that mirrors
    the argparse.Namespace produced by _parse_args().
    """
    _print_header()

    circuits = _available_circuits()
    if not circuits:
        console.print("[red]No circuit YAML files found in data/tracks/[/red]")
        sys.exit(1)

    # ── 1. Circuit ────────────────────────────────────────────────────
    circuit_choice = questionary.select(
        "Select circuit:",
        choices=[questionary.Choice(label, value=path) for label, path in circuits],
        style=_STYLE,
    ).ask()
    if circuit_choice is None:
        sys.exit(0)

    # ── 2. Solver ─────────────────────────────────────────────────────
    solver = questionary.select(
        "Select strategy solver:",
        choices=[
            questionary.Choice("DP Optimizer  — exact optimum, ~2-3 min",  value="dp"),
            questionary.Choice("Brute-force   — 300 sampled,   ~2-3 min",  value="brute"),
        ],
        style=_STYLE,
    ).ask()
    if solver is None:
        sys.exit(0)

    # ── 3. Search step ────────────────────────────────────────────────
    step_choice = questionary.select(
        "Strategy search step size:",
        choices=[
            questionary.Choice("50 m  — fast  (~1 min)",    value=50.0),
            questionary.Choice("25 m  — medium (~3 min)",   value=25.0),
            questionary.Choice("5 m   — full fidelity",     value=5.0),
        ],
        style=_STYLE,
    ).ask()
    if step_choice is None:
        sys.exit(0)

    # ── 4. Multi-car ──────────────────────────────────────────────────
    multi_car = questionary.confirm(
        "Enable multi-car simulation?",
        default=False,
        style=_STYLE,
    ).ask()
    if multi_car is None:
        sys.exit(0)

    num_cars = 5
    if multi_car:
        num_cars_str = questionary.select(
            "Number of cars:",
            choices=[
                questionary.Choice("3",  value=3),
                questionary.Choice("5",  value=5),
                questionary.Choice("8",  value=8),
                questionary.Choice("10", value=10),
            ],
            style=_STYLE,
        ).ask()
        if num_cars_str is None:
            sys.exit(0)
        num_cars = num_cars_str

    # ── 4c. Live Safety-Car decision (optional, slow) ─────────────────
    sc_lap = None
    live = questionary.confirm(
        "Run a live Safety-Car decision demo? (re-optimises mid-race; adds ~2-3 min)",
        default=False,
        style=_STYLE,
    ).ask()
    if live is None:
        sys.exit(0)
    if live:
        lap_str = questionary.text(
            "Safety Car deployed at lap:",
            validate=lambda v: (v.strip().isdigit() and int(v) >= 1)
                               or "Enter a positive lap number",
            style=_STYLE,
        ).ask()
        if lap_str is None:
            sys.exit(0)
        sc_lap = int(lap_str)

    # ── Weather: always the circuit's real conditions ─────────────────
    # The TUI always uses the circuit YAML weather (a dynamic Level B timeline
    # if present, e.g. the real Silverstone 2024 mixed race; else dry), so the
    # interactive run is always coherent with the real Grand Prix. What-if
    # overrides (forced dry / static wet) stay available via the CLI flags
    # --wetness and --weather-timeline. None → main() uses loader.weather_model().
    wetness = None

    # ── 5. Output mode ────────────────────────────────────────────────
    output = questionary.select(
        "Output mode:",
        choices=[
            questionary.Choice("Dashboard  — interactive browser plots", value="dashboard"),
            questionary.Choice("PNG plots  — saved to results/plots/",   value="plots"),
            questionary.Choice("Text only  — no plots",                  value="text"),
        ],
        style=_STYLE,
    ).ask()
    if output is None:
        sys.exit(0)

    port = 8050
    if output == "dashboard":
        port_str = questionary.select(
            "Dashboard port:",
            choices=[
                questionary.Choice("8050  (default)", value=8050),
                questionary.Choice("8051",            value=8051),
                questionary.Choice("8080",            value=8080),
            ],
            style=_STYLE,
        ).ask()
        if port_str is None:
            sys.exit(0)
        port = port_str

    # ── 6. Advanced options ───────────────────────────────────────────
    advanced = questionary.confirm(
        "Configure advanced options?",
        default=False,
        style=_STYLE,
    ).ask()
    if advanced is None:
        sys.exit(0)

    top_n   = None
    verbose = False

    if advanced:
        top_n_raw = questionary.select(
            "Top N strategies to display:",
            choices=[
                questionary.Choice("5  (default)", value=5),
                questionary.Choice("10",           value=10),
                questionary.Choice("15",           value=15),
                questionary.Choice("18 — All",     value=18),
            ],
            style=_STYLE,
        ).ask()
        if top_n_raw is None:
            sys.exit(0)
        top_n = top_n_raw

        verbose = questionary.confirm(
            "Print baseline lap & tyre degradation table?",
            default=False,
            style=_STYLE,
        ).ask()
        if verbose is None:
            sys.exit(0)

    # ── Build namespace ───────────────────────────────────────────────
    cfg = SimpleNamespace(
        circuit   = circuit_choice,
        solver    = solver,
        step      = step_choice,
        top_n     = top_n,
        multi_car = multi_car,
        num_cars  = num_cars,
        dashboard = (output == "dashboard"),
        no_plots  = (output == "text"),
        port      = port,
        verbose   = verbose,
        wetness   = wetness,
        sc_lap    = sc_lap,
    )

    console.print()
    _print_summary(cfg)

    confirm = questionary.confirm(
        "Start simulation?",
        default=True,
        style=_STYLE,
    ).ask()
    if not confirm:
        console.print("[dim]Aborted.[/dim]")
        sys.exit(0)

    console.print()
    return cfg

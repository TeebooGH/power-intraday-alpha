"""Market impact calibration and execution on continuous intraday power (simulation study)."""


def main() -> None:
    import argparse
    from pathlib import Path

    from .experiments import DEFAULT_OUT
    from .experiments import main as run

    parser = argparse.ArgumentParser(prog="power-intraday-alpha", description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"output directory for metrics.json and summary.png (default: {DEFAULT_OUT})")
    run(out=parser.parse_args().out)

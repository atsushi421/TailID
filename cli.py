"""Command-line interface for the TailID algorithm.

This module provides a CLI for running the TailID algorithm on data from
a text file, accepting method parameters from the user.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.tailid import MOS_DEFAULT, tail_id
from src.threshold_selection import select_threshold


def load_data_from_file(file_path: str) -> np.ndarray:
    """Load numerical data from a text file.

    The file should contain one numerical value per line.

    Args:
        file_path: Path to the text file containing the data.

    Returns:
        NumPy array of floating-point values.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid data.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    try:
        data = np.loadtxt(file_path, dtype=np.float64)
        if data.ndim == 0:
            data = np.array([float(data)])
        return data
    except ValueError as e:
        raise ValueError(f"Invalid data in file {file_path}: {e}") from e


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="tailid",
        description=(
            "TailID: Detect low-density mixtures in high-quantile tails "
            "for pWCET estimation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py data.txt --p_c1 0.95 --n_candidates 31
  python cli.py data.txt --p_c1 0.95 --n_candidates 51 --gamma 0.999

Data file format:
  The input file should contain one numerical value per line.
  Example:
    1.23
    4.56
    7.89
    ...
""",
    )

    parser.add_argument(
        "data_file",
        type=str,
        help="Path to the text file containing sample data",
    )

    parser.add_argument(
        "--p_c1",
        type=float,
        required=True,
        help=(
            "Candidate percentile (0 < p_c1 < 1). "
            "Defines the starting point for the candidate set."
        ),
    )

    parser.add_argument(
        "--n_candidates",
        type=int,
        required=True,
        help="Number of candidate thresholds to evaluate for p_m selection",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.9999,
        help=(
            "Confidence level (0 < gamma < 1). "
            "Controls detection sensitivity (default: 0.9999)"
        ),
    )

    parser.add_argument(
        "--mos",
        type=int,
        default=MOS_DEFAULT,
        help=(
            f"Minimum of Samples threshold for scenario classification "
            f"(default: {MOS_DEFAULT})"
        ),
    )

    return parser


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI.

    Args:
        args: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    try:
        data = load_data_from_file(parsed_args.data_file)
        print(f"Loaded {len(data)} data points from {parsed_args.data_file}")
        print()

        print("Selecting optimal p_m by minimizing EQMAE...")
        p_m = select_threshold(data, n_candidates=parsed_args.n_candidates)
        print(f"Selected p_m = {p_m:.4f}")
        print()

        result = tail_id(
            x=data,
            p_m=p_m,
            p_c1=parsed_args.p_c1,
            gamma=parsed_args.gamma,
            mos=parsed_args.mos,
        )

        print("=" * 60)
        print("TailID Analysis Result")
        print("=" * 60)
        print()
        print("Parameters:")
        print(f"  p_m (extreme value percentile): {p_m:.4f} (auto-selected)")
        print(f"  p_c1 (candidate percentile): {parsed_args.p_c1}")
        print(f"  n_candidates: {parsed_args.n_candidates}")
        print(f"  gamma (confidence level): {parsed_args.gamma}")
        print(f"  MoS (minimum of samples): {parsed_args.mos}")
        print()
        print("Results:")
        print(f"  Number of sensitive points: {len(result.sensitive_points)}")
        print(f"  Scenario: {result.scenario.name}")
        if result.tail_threshold is not None:
            print(f"  Tail threshold: {result.tail_threshold}")
        print()
        print("Interpretation:")
        print(f"  {result.message}")
        print()

        if len(result.sensitive_points) > 0:
            print("Sensitive points:")
            for i, point in enumerate(result.sensitive_points[:10]):
                print(f"  {i + 1}. {point}")
            if len(result.sensitive_points) > 10:
                print(f"  ... and {len(result.sensitive_points) - 10} more")
        print("=" * 60)

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

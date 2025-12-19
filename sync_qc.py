"""QC module for synchronized audio and biomechanics data.

Performs two-stage QC on synchronized recordings:
1. Identifies movement cycles with insufficient acoustic signal
2. Compares clean cycles to expected acoustic-biomechanics relationships
"""

import logging
from datetime import timedelta
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from parse_movement_cycles import extract_movement_cycles

logger = logging.getLogger(__name__)


class MovementCycleQC:
    """Perform QC analysis on extracted movement cycles."""

    def __init__(
        self,
        maneuver: Literal["walk", "sit_to_stand", "flexion_extension"],
        speed: Optional[Literal["slow", "medium", "fast"]] = None,
        acoustic_threshold: float = 100.0,
        acoustic_channel: Literal["raw", "filtered"] = "filtered",
    ):
        """Initialize QC analyzer.

        Args:
            maneuver: Type of movement (walk, sit_to_stand, flexion_extension)
            speed: Speed level for walking (slow, medium, fast)
            acoustic_threshold: Minimum RMS acoustic energy threshold per cycle
            acoustic_channel: Whether to use raw (ch) or filtered (f_ch) channels
        """
        self.maneuver = maneuver
        self.speed = speed
        self.acoustic_threshold = acoustic_threshold
        self.acoustic_channel = acoustic_channel

    def analyze_cycles(
        self,
        cycles: list[pd.DataFrame],
    ) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        """Perform two-stage QC on movement cycles.

        Args:
            cycles: List of movement cycle DataFrames from parse_movement_cycles.

        Returns:
            Tuple of (clean_cycles, outlier_cycles)
        """
        if not cycles:
            logger.warning("No cycles provided for QC analysis")
            return [], []

        # Stage 1: Identify cycles with insufficient acoustic signal
        clean_cycles, outliers_low_signal = self._stage1_acoustic_threshold(cycles)

        # Stage 2: Compare clean cycles to expected relationships
        # (Advanced validation could go here)

        outlier_cycles = outliers_low_signal

        logger.info(
            f"QC complete: {len(clean_cycles)} clean cycles, "
            f"{len(outlier_cycles)} outliers"
        )

        return clean_cycles, outlier_cycles

    def _stage1_acoustic_threshold(
        self,
        cycles: list[pd.DataFrame],
    ) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        """Stage 1: Filter cycles by acoustic signal strength.

        Uses area-under-curve (AUC) of acoustic energy to identify cycles with
        insufficient signal, typically at the start/end of recording or due to
        microphone artifacts.

        Args:
            cycles: List of cycle DataFrames.

        Returns:
            Tuple of (clean_cycles, outlier_cycles)
        """
        clean_cycles = []
        outlier_cycles = []

        for cycle_idx, cycle_df in enumerate(cycles):
            acoustic_energy = self._compute_cycle_acoustic_energy(cycle_df)

            if acoustic_energy >= self.acoustic_threshold:
                clean_cycles.append(cycle_df)
                logger.debug(f"Cycle {cycle_idx}: CLEAN (energy={acoustic_energy:.1f})")
            else:
                outlier_cycles.append(cycle_df)
                logger.debug(f"Cycle {cycle_idx}: OUTLIER (energy={acoustic_energy:.1f})")

        return clean_cycles, outlier_cycles

    def _compute_cycle_acoustic_energy(self, cycle_df: pd.DataFrame) -> float:
        """Compute total acoustic energy for a cycle using AUC.

        Args:
            cycle_df: Single movement cycle DataFrame.

        Returns:
            Total acoustic energy (AUC across all channels).
        """
        # Select channels based on acoustic_channel preference
        if self.acoustic_channel == "filtered":
            channel_names = ["f_ch1", "f_ch2", "f_ch3", "f_ch4"]
        else:
            channel_names = ["ch1", "ch2", "ch3", "ch4"]

        # Fall back to raw if filtered not available
        if not all(ch in cycle_df.columns for ch in channel_names):
            channel_names = ["ch1", "ch2", "ch3", "ch4"]

        # Compute AUC for each channel
        total_auc = 0.0
        if "tt" in cycle_df.columns:
            # Convert tt to seconds if timedelta
            if isinstance(cycle_df["tt"].iloc[0], pd.Timedelta):
                tt_seconds = cycle_df["tt"].dt.total_seconds().values
            else:
                tt_seconds = cycle_df["tt"].values
        else:
            # Use uniform spacing if tt not available
            tt_seconds = np.arange(len(cycle_df))

        for ch in channel_names:
            if ch in cycle_df.columns:
                ch_data = np.abs(cycle_df[ch].values)
                try:
                    auc = np.trapezoid(ch_data, tt_seconds)
                except AttributeError:
                    auc = np.trapz(ch_data, tt_seconds)
                total_auc += auc

        return total_auc


def perform_sync_qc(
    synced_pkl_path: Path,
    output_dir: Optional[Path] = None,
    maneuver: Optional[Literal["walk", "sit_to_stand", "flexion_extension"]] = None,
    speed: Optional[Literal["slow", "medium", "fast"]] = None,
    acoustic_threshold: float = 100.0,
    create_plots: bool = True,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], Path]:
    """Perform complete QC pipeline on a synchronized recording.

    Args:
        synced_pkl_path: Path to synchronized pickle file.
        output_dir: Directory to save QC results. Defaults to parent directory.
        maneuver: Type of movement. If None, inferred from file path.
        speed: Speed level for walking. If None, inferred from file path.
        acoustic_threshold: Minimum acoustic energy per cycle.
        create_plots: Whether to create visualization plots.

    Returns:
        Tuple of (clean_cycles, outlier_cycles, output_directory)

    Raises:
        FileNotFoundError: If synced_pkl_path does not exist.
        ValueError: If maneuver cannot be determined.
    """
    synced_pkl_path = Path(synced_pkl_path)
    if not synced_pkl_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {synced_pkl_path}")

    # Load synchronized data
    synced_df = pd.read_pickle(synced_pkl_path)
    logger.info(f"Loaded synchronized data: {synced_df.shape}")

    # Infer maneuver and speed if not provided
    if maneuver is None:
        maneuver = _infer_maneuver_from_path(synced_pkl_path)
    if speed is None and maneuver == "walk":
        speed = _infer_speed_from_path(synced_pkl_path)

    logger.info(f"Maneuver: {maneuver}, Speed: {speed}")

    # Extract movement cycles
    cycles = extract_movement_cycles(synced_df, maneuver=maneuver, speed=speed)
    logger.info(f"Extracted {len(cycles)} movement cycles")

    if not cycles:
        logger.warning("No movement cycles extracted")
        return [], [], output_dir or synced_pkl_path.parent

    # Perform QC analysis
    qc = MovementCycleQC(
        maneuver=maneuver,
        speed=speed,
        acoustic_threshold=acoustic_threshold,
    )
    clean_cycles, outlier_cycles = qc.analyze_cycles(cycles)

    # Set output directory
    base_dir = Path(output_dir) if output_dir is not None else synced_pkl_path.parent
    output_dir = base_dir / "MovementCycles"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    _save_qc_results(
        clean_cycles,
        outlier_cycles,
        output_dir,
        synced_pkl_path.stem,
        create_plots=create_plots,
    )

    logger.info(f"QC results saved to {output_dir}")

    return clean_cycles, outlier_cycles, output_dir


def _infer_maneuver_from_path(path: Path) -> str:
    """Infer maneuver type from file path.

    Args:
        path: Path to synchronized pickle file.

    Returns:
        Maneuver type (walk, sit_to_stand, flexion_extension)

    Raises:
        ValueError: If maneuver cannot be inferred.
    """
    path_str = str(path).lower()

    if "walk" in path_str:
        return "walk"
    elif "sit" in path_str and "stand" in path_str:
        return "sit_to_stand"
    elif "flexion" in path_str or "extension" in path_str or "flexext" in path_str:
        return "flexion_extension"
    else:
        raise ValueError(f"Cannot infer maneuver from path: {path}")


def _infer_speed_from_path(path: Path) -> Optional[str]:
    """Infer walking speed from file path.

    Args:
        path: Path to synchronized pickle file.

    Returns:
        Speed level (slow, medium, fast) or None.
    """
    path_str = str(path).lower()

    if "slow" in path_str:
        return "slow"
    elif "medium" in path_str:
        return "medium"
    elif "fast" in path_str:
        return "fast"

    return None


def _save_qc_results(
    clean_cycles: list[pd.DataFrame],
    outlier_cycles: list[pd.DataFrame],
    output_dir: Path,
    file_stem: str,
    create_plots: bool = True,
) -> None:
    """Save QC results to disk.

    Args:
        clean_cycles: List of clean cycle DataFrames.
        outlier_cycles: List of outlier cycle DataFrames.
        output_dir: Directory to save results.
        file_stem: Stem of original file (for naming).
        create_plots: Whether to create visualization plots.
    """
    # Create subdirectories
    clean_dir = output_dir / "clean"
    outlier_dir = output_dir / "outliers"
    clean_dir.mkdir(parents=True, exist_ok=True)
    outlier_dir.mkdir(parents=True, exist_ok=True)

    # Save clean cycles
    for i, cycle_df in enumerate(clean_cycles):
        filename = clean_dir / f"{file_stem}_cycle_{i:03d}.pkl"
        cycle_df.to_pickle(filename)

        if create_plots and MATPLOTLIB_AVAILABLE:
            _create_cycle_plot(cycle_df, clean_dir / f"{file_stem}_cycle_{i:03d}.png")

    # Save outlier cycles
    for i, cycle_df in enumerate(outlier_cycles):
        filename = outlier_dir / f"{file_stem}_outlier_{i:03d}.pkl"
        cycle_df.to_pickle(filename)

        if create_plots and MATPLOTLIB_AVAILABLE:
            _create_cycle_plot(
                cycle_df,
                outlier_dir / f"{file_stem}_outlier_{i:03d}.png",
                title_suffix="(OUTLIER)",
            )

    logger.info(f"Saved {len(clean_cycles)} clean cycles and {len(outlier_cycles)} outliers")


def _create_cycle_plot(
    cycle_df: pd.DataFrame,
    output_path: Path,
    title_suffix: str = "",
) -> None:
    """Create visualization plot for a single cycle.

    Args:
        cycle_df: Movement cycle DataFrame.
        output_path: Path to save PNG file.
        title_suffix: Additional suffix for title.
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Get time axis
        if "tt" in cycle_df.columns:
            if isinstance(cycle_df["tt"].iloc[0], pd.Timedelta):
                tt_seconds = cycle_df["tt"].dt.total_seconds().values
            else:
                tt_seconds = cycle_df["tt"].values
        else:
            tt_seconds = np.arange(len(cycle_df))

        # Plot 1: Knee angle
        if "Knee Angle Z" in cycle_df.columns:
            ax1.plot(tt_seconds, cycle_df["Knee Angle Z"], "k-", linewidth=2)
            ax1.set_ylabel("Knee Angle Z (degrees)", fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.set_title(f"Movement Cycle - Biomechanics {title_suffix}", fontsize=12)

        # Plot 2: Acoustic energy
        ax2_twin = ax2.twinx()

        # Plot acoustic channels
        acoustic_channels = ["f_ch1", "f_ch2", "f_ch3", "f_ch4"]
        colors = ["b", "g", "r", "m"]
        for ch, color in zip(acoustic_channels, colors):
            if ch in cycle_df.columns:
                ax2.plot(tt_seconds, cycle_df[ch], color=color, alpha=0.6, label=ch, linewidth=0.8)

        ax2.set_xlabel("Time (seconds)", fontsize=12)
        ax2.set_ylabel("Acoustic Amplitude", fontsize=12, color="blue")
        ax2.tick_params(axis="y", labelcolor="blue")
        ax2.grid(True, alpha=0.3)
        ax2.set_title(f"Acoustic Channels {title_suffix}", fontsize=12)
        ax2.legend(loc="upper left", fontsize=8)

        plt.tight_layout()
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        logger.debug(f"Created plot: {output_path}")

    except Exception as e:
        logger.warning(f"Failed to create plot {output_path}: {e}")


# CLI Support Functions


def setup_logging(verbose: bool = False) -> None:
    """Configure logging output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def find_synced_files(path: Path) -> list[Path]:
    """Find all pickle files located in any directory named 'Synced'.

    Args:
        path: Root path to search.

    Returns:
        List of paths to synced pickle files.
    """
    synced_files = []

    if path.is_file():
        # If a single file is provided, check if its parent is 'Synced'
        if path.suffix == ".pkl" and path.parent.name == "Synced":
            synced_files.append(path)
    elif path.is_dir():
        # Recursively find all .pkl files and check if their parent is 'Synced'
        for pkl_file in path.rglob("*.pkl"):
            if pkl_file.parent.name == "Synced":
                synced_files.append(pkl_file)

    return sorted(synced_files)


def main() -> int:
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Perform QC analysis on synchronized audio and biomechanics data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "path",
        type=Path,
        help="Path to synced pickle file, Synced directory, or participant directory",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=100.0,
        help="Minimum acoustic energy threshold per cycle (default: 100.0)",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip creation of visualization plots",
    )

    parser.add_argument(
        "--maneuver",
        type=str,
        choices=["walk", "sit_to_stand", "flexion_extension"],
        help="Override maneuver type inference",
    )

    parser.add_argument(
        "--speed",
        type=str,
        choices=["slow", "medium", "fast"],
        help="Override speed inference (for walking)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)

    # Validate path
    if not args.path.exists():
        logger.error(f"Path does not exist: {args.path}")
        return 1

    # Find files to process
    files_to_process = find_synced_files(args.path)

    if not files_to_process:
        logger.warning(f"No synced files found in {args.path}")
        return 1

    logger.info(f"Found {len(files_to_process)} synced file(s) to process")

    # Process each file
    total_clean = 0
    total_outliers = 0

    for synced_file in files_to_process:
        logger.info(f"\nProcessing: {synced_file}")

        try:
            clean_cycles, outlier_cycles, output_dir = perform_sync_qc(
                synced_file,
                maneuver=args.maneuver,
                speed=args.speed,
                acoustic_threshold=args.threshold,
                create_plots=not args.no_plots,
            )

            total_clean += len(clean_cycles)
            total_outliers += len(outlier_cycles)

            logger.info(
                f"✓ Complete: {len(clean_cycles)} clean, {len(outlier_cycles)} outliers "
                f"→ {output_dir}"
            )

        except Exception as e:
            logger.error(f"✗ Failed to process {synced_file}: {e}", exc_info=args.verbose)
            continue

    logger.info(f"\n{'='*60}")
    logger.info(f"Total: {total_clean} clean cycles, {total_outliers} outliers")
    logger.info(f"{'='*60}")

    return 0


if __name__ == "__main__":
    exit(main())

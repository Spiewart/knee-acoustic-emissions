"""Filename generation for pipeline output files.

Produces short, human-readable filenames. Uniqueness is enforced at the
database level via multi-column constraints (participant_id, knee, maneuver,
pass_number, speed, cycle_index) — NOT by filename alone.

Naming conventions:
    Sync files:    left_walk_p1_slow.pkl, right_fe.pkl
    Cycle files:   left_walk_p1_slow_c003.pkl, right_sts_c000.pkl
"""


def generate_sync_filename(
    knee: str,
    maneuver: str,
    pass_number: int | None = None,
    speed: str | None = None,
) -> str:
    """Generate a short sync output filename.

    Args:
        knee: Lowercase knee side ("left" or "right")
        maneuver: DB maneuver code ("fe", "sts", "walk")
        pass_number: Walk pass number (None for non-walk)
        speed: Walk speed category (None for non-walk)

    Returns:
        Filename with .pkl extension, e.g. "left_walk_p1_slow.pkl"
    """
    parts = [knee.lower(), maneuver.lower()]
    if pass_number is not None:
        parts.append(f"p{pass_number}")
    if speed is not None:
        parts.append(speed.lower())
    return "_".join(parts) + ".pkl"


def generate_cycle_filename(
    knee: str,
    maneuver: str,
    cycle_index: int,
    pass_number: int | None = None,
    speed: str | None = None,
) -> str:
    """Generate a short cycle output filename.

    Args:
        knee: Lowercase knee side ("left" or "right")
        maneuver: DB maneuver code ("fe", "sts", "walk")
        cycle_index: Zero-based cycle index
        pass_number: Walk pass number (None for non-walk)
        speed: Walk speed category (None for non-walk)

    Returns:
        Filename with .pkl extension, e.g. "left_walk_p1_slow_c003.pkl"
    """
    parts = [knee.lower(), maneuver.lower()]
    if pass_number is not None:
        parts.append(f"p{pass_number}")
    if speed is not None:
        parts.append(speed.lower())
    parts.append(f"c{cycle_index:03d}")
    return "_".join(parts) + ".pkl"

"""Tests for file naming utilities.

Verifies that generate_sync_filename and generate_cycle_filename produce
correct, lowercase, human-readable filenames for all maneuver types.
"""

from src.studies.file_naming import generate_cycle_filename, generate_sync_filename


class TestGenerateSyncFilename:
    """Verify sync filename generation for all maneuver types."""

    def test_walk_with_pass_and_speed(self):
        result = generate_sync_filename("left", "walk", pass_number=1, speed="slow")
        assert result == "left_walk_p1_slow.pkl"

    def test_walk_medium_speed(self):
        result = generate_sync_filename("right", "walk", pass_number=3, speed="medium")
        assert result == "right_walk_p3_medium.pkl"

    def test_fe_no_pass_no_speed(self):
        result = generate_sync_filename("left", "fe")
        assert result == "left_fe.pkl"

    def test_sts_no_pass_no_speed(self):
        result = generate_sync_filename("right", "sts")
        assert result == "right_sts.pkl"

    def test_always_lowercase(self):
        result = generate_sync_filename("Left", "Walk", pass_number=1, speed="Fast")
        assert result == "left_walk_p1_fast.pkl"

    def test_pkl_extension(self):
        result = generate_sync_filename("left", "walk", pass_number=1, speed="slow")
        assert result.endswith(".pkl")

    def test_pass_only_no_speed(self):
        result = generate_sync_filename("left", "walk", pass_number=5)
        assert result == "left_walk_p5.pkl"


class TestGenerateCycleFilename:
    """Verify cycle filename generation for all maneuver types."""

    def test_walk_cycle(self):
        result = generate_cycle_filename("left", "walk", 3, pass_number=1, speed="slow")
        assert result == "left_walk_p1_slow_c003.pkl"

    def test_fe_cycle(self):
        result = generate_cycle_filename("right", "fe", 0)
        assert result == "right_fe_c000.pkl"

    def test_sts_cycle(self):
        result = generate_cycle_filename("left", "sts", 12)
        assert result == "left_sts_c012.pkl"

    def test_cycle_index_zero_padded(self):
        result = generate_cycle_filename("left", "walk", 5, pass_number=1, speed="medium")
        assert result == "left_walk_p1_medium_c005.pkl"

    def test_large_cycle_index(self):
        result = generate_cycle_filename("left", "walk", 100, pass_number=1, speed="fast")
        assert result == "left_walk_p1_fast_c100.pkl"

    def test_always_lowercase(self):
        result = generate_cycle_filename("Right", "FE", 0)
        assert result == "right_fe_c000.pkl"

    def test_pkl_extension(self):
        result = generate_cycle_filename("left", "walk", 0, pass_number=1, speed="slow")
        assert result.endswith(".pkl")

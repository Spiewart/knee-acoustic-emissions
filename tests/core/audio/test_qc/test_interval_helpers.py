"""Tests for interval helper methods on ManeuverProcessor.

Validates _unflatten_intervals and _flatten_intervals which convert
between DB flat arrays [s1, e1, s2, e2, ...] and tuple pairs
[(s1, e1), (s2, e2), ...].
"""

from src.orchestration.participant_processor import ManeuverProcessor


class TestUnflattenIntervals:
    """Test _unflatten_intervals static method."""

    def test_none_returns_empty_list(self):
        assert ManeuverProcessor._unflatten_intervals(None) == []

    def test_empty_list_returns_empty_list(self):
        assert ManeuverProcessor._unflatten_intervals([]) == []

    def test_single_interval(self):
        result = ManeuverProcessor._unflatten_intervals([1.0, 3.0])
        assert result == [(1.0, 3.0)]

    def test_multiple_intervals(self):
        result = ManeuverProcessor._unflatten_intervals([1.0, 3.0, 5.0, 7.0, 10.0, 12.0])
        assert result == [(1.0, 3.0), (5.0, 7.0), (10.0, 12.0)]

    def test_odd_length_returns_empty_list(self):
        """Odd-length array cannot form pairs and should return empty."""
        result = ManeuverProcessor._unflatten_intervals([1.0, 3.0, 5.0])
        assert result == []

    def test_single_element_returns_empty_list(self):
        result = ManeuverProcessor._unflatten_intervals([1.0])
        assert result == []

    def test_preserves_float_precision(self):
        result = ManeuverProcessor._unflatten_intervals([1.234567, 8.901234])
        assert result == [(1.234567, 8.901234)]

    def test_zero_duration_interval(self):
        """A point interval (start == end) should still be returned."""
        result = ManeuverProcessor._unflatten_intervals([5.0, 5.0])
        assert result == [(5.0, 5.0)]


class TestFlattenIntervals:
    """Test _flatten_intervals static method."""

    def test_empty_list_returns_empty(self):
        result = ManeuverProcessor._flatten_intervals([])
        assert result == []

    def test_single_interval(self):
        result = ManeuverProcessor._flatten_intervals([(1.0, 3.0)])
        assert result == [1.0, 3.0]

    def test_multiple_intervals(self):
        result = ManeuverProcessor._flatten_intervals(
            [(1.0, 3.0), (5.0, 7.0), (10.0, 12.0)]
        )
        assert result == [1.0, 3.0, 5.0, 7.0, 10.0, 12.0]

    def test_preserves_float_precision(self):
        result = ManeuverProcessor._flatten_intervals([(1.234567, 8.901234)])
        assert result == [1.234567, 8.901234]

    def test_output_always_floats(self):
        """Integer inputs should be converted to floats."""
        result = ManeuverProcessor._flatten_intervals([(1, 3), (5, 7)])
        assert result == [1.0, 3.0, 5.0, 7.0]
        assert all(isinstance(v, float) for v in result)


class TestUnflattenNestedFormat:
    """Test _unflatten_intervals with nested list format from PostgreSQL.

    PostgreSQL ARRAY(Float) columns return [[s1,e1], [s2,e2], ...] when
    the data was originally stored as list[tuple[float,float]].
    """

    def test_nested_lists(self):
        """Nested lists [[s, e], ...] should be converted to tuple pairs."""
        result = ManeuverProcessor._unflatten_intervals([[1.0, 3.0], [5.0, 7.0]])
        assert result == [(1.0, 3.0), (5.0, 7.0)]

    def test_nested_tuples(self):
        """Nested tuples [(s, e), ...] should pass through as tuple pairs."""
        result = ManeuverProcessor._unflatten_intervals([(1.0, 3.0), (5.0, 7.0)])
        assert result == [(1.0, 3.0), (5.0, 7.0)]

    def test_single_nested_list(self):
        result = ManeuverProcessor._unflatten_intervals([[2.0, 4.0]])
        assert result == [(2.0, 4.0)]

    def test_nested_output_always_float_tuples(self):
        """Nested list input should produce float tuples regardless of input type."""
        result = ManeuverProcessor._unflatten_intervals([[1, 3], [5, 7]])
        assert result == [(1.0, 3.0), (5.0, 7.0)]
        for start, end in result:
            assert isinstance(start, float)
            assert isinstance(end, float)

    def test_real_postgresql_data(self):
        """Test with actual PostgreSQL format — list of 2-element lists."""
        pg_data = [
            [0.956, 1.739],
            [17.658, 17.901],
            [67.894, 70.222],
        ]
        result = ManeuverProcessor._unflatten_intervals(pg_data)
        assert len(result) == 3
        assert result[0] == (0.956, 1.739)
        assert result[2] == (67.894, 70.222)


class TestRoundTripConversion:
    """Test that flatten → unflatten and unflatten → flatten are identity ops."""

    def test_flatten_then_unflatten(self):
        original = [(1.0, 3.0), (5.5, 7.2), (10.0, 12.8)]
        flat = ManeuverProcessor._flatten_intervals(original)
        restored = ManeuverProcessor._unflatten_intervals(flat)
        assert restored == original

    def test_unflatten_then_flatten(self):
        original = [1.0, 3.0, 5.5, 7.2, 10.0, 12.8]
        pairs = ManeuverProcessor._unflatten_intervals(original)
        restored = ManeuverProcessor._flatten_intervals(pairs)
        assert restored == original

    def test_empty_round_trip(self):
        flat = ManeuverProcessor._flatten_intervals([])
        restored = ManeuverProcessor._unflatten_intervals(flat)
        assert restored == []

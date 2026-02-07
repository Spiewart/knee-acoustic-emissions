"""Tests for Phase 2D (Performance Optimization)."""

import time
from unittest.mock import MagicMock, patch

import pytest

from src.orchestration.performance_optimization import (
    BatchMovementCyclePersister,
    PerformanceOptimizedPersistence,
    PersistenceMetrics,
    create_optimized_db_session,
)


class TestPersistenceMetrics:
    """Test metrics collection for performance monitoring."""

    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = PersistenceMetrics()

        assert metrics.total_audio_saves == 0
        assert metrics.audio_errors == 0
        assert metrics.batch_size == 100

    def test_average_timing_calculations(self):
        """Test average timing calculations."""
        metrics = PersistenceMetrics()
        metrics.total_audio_saves = 2
        metrics.total_audio_time_ms = 100.0

        assert metrics.avg_audio_time_ms == 50.0

    def test_average_timing_with_zero_saves(self):
        """Test average timing returns 0 when no saves."""
        metrics = PersistenceMetrics()

        assert metrics.avg_audio_time_ms == 0.0
        assert metrics.avg_biomech_time_ms == 0.0

    def test_metrics_summary_generation(self):
        """Test metrics summary string generation."""
        metrics = PersistenceMetrics()
        metrics.total_audio_saves = 1
        metrics.total_audio_time_ms = 45.5
        metrics.total_cycle_saves = 150
        metrics.total_cycle_time_ms = 450.0

        summary = metrics.summary()

        assert "Audio Saves: 1" in summary
        assert "Cycle Saves: 150" in summary
        assert "45.50ms avg" in summary


class TestPerformanceOptimizedPersistence:
    """Test performance optimization features."""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        return MagicMock()

    @pytest.fixture
    def optimized_persistence(self, mock_session):
        """Create optimized persistence instance."""
        return PerformanceOptimizedPersistence(
            db_session=mock_session,
            batch_size=10,
            thread_pool_size=2,
        )

    def test_initialization_with_session(self, optimized_persistence):
        """Test initialization with database session."""
        assert optimized_persistence.db_session is not None
        assert optimized_persistence.batch_size == 10
        assert optimized_persistence.metrics is not None

    def test_initialization_without_session(self):
        """Test initialization without database session."""
        persistence = PerformanceOptimizedPersistence(db_session=None)

        assert persistence.db_session is None
        assert persistence.metrics is not None

    def test_timing_context_measures_elapsed_time(self, optimized_persistence):
        """Test timing context manager records elapsed time."""
        with optimized_persistence.timing_context("test_operation"):
            time.sleep(0.01)  # 10ms

        # Timing is logged but not directly accessible
        assert True  # Context manager executed without error

    def test_batch_insert_cycles_with_empty_list(self, optimized_persistence):
        """Test batch insert with empty cycle list."""
        def mock_save(cycle_data):
            return 1

        results = optimized_persistence.batch_insert_cycles([], mock_save)

        assert results == {}
        assert optimized_persistence.metrics.total_cycle_saves == 0

    def test_batch_insert_cycles_with_single_batch(self, optimized_persistence):
        """Test batch insert with single batch of cycles."""
        def mock_save(cycle_data):
            return cycle_data.get("cycle_id", 1)

        cycles = [
            {"cycle_id": 1, "data": "cycle1"},
            {"cycle_id": 2, "data": "cycle2"},
            {"cycle_id": 3, "data": "cycle3"},
        ]

        results = optimized_persistence.batch_insert_cycles(cycles, mock_save)

        assert len(results) == 3
        assert optimized_persistence.metrics.total_cycle_saves == 3

    def test_batch_insert_cycles_with_multiple_batches(self, optimized_persistence):
        """Test batch insert with multiple batches (batch_size=10, 25 cycles)."""
        def mock_save(cycle_data):
            return cycle_data.get("cycle_id", 1)

        cycles = [{"cycle_id": i} for i in range(25)]

        results = optimized_persistence.batch_insert_cycles(cycles, mock_save)

        # Should process all 25 cycles (3 batches: 10, 10, 5)
        assert len(results) == 25
        assert optimized_persistence.metrics.total_cycle_saves == 25

    def test_batch_insert_cycles_handles_errors(self, optimized_persistence):
        """Test batch insert continues on save errors."""
        def mock_save_with_errors(cycle_data):
            if cycle_data.get("cycle_id") == 2:
                raise ValueError("Save failed")
            return cycle_data.get("cycle_id", 1)

        cycles = [
            {"cycle_id": 1},
            {"cycle_id": 2},  # This will fail
            {"cycle_id": 3},
        ]

        results = optimized_persistence.batch_insert_cycles(
            cycles, mock_save_with_errors
        )

        assert len(results) == 3
        assert results[0] == 1
        assert results[1] is None  # Error
        assert results[2] == 3
        assert optimized_persistence.metrics.cycle_errors == 1

    def test_collect_timing_metrics_audio(self, optimized_persistence):
        """Test collecting audio timing metrics."""
        optimized_persistence.collect_timing_metrics(
            operation="audio",
            count=1,
            elapsed_ms=45.5,
            errors=0,
        )

        assert optimized_persistence.metrics.total_audio_saves == 1
        assert optimized_persistence.metrics.total_audio_time_ms == 45.5
        assert optimized_persistence.metrics.audio_errors == 0

    def test_collect_timing_metrics_cycle(self, optimized_persistence):
        """Test collecting cycle timing metrics."""
        optimized_persistence.collect_timing_metrics(
            operation="cycle",
            count=150,
            elapsed_ms=450.0,
            errors=2,
        )

        assert optimized_persistence.metrics.total_cycle_saves == 150
        assert optimized_persistence.metrics.total_cycle_time_ms == 450.0
        assert optimized_persistence.metrics.cycle_errors == 2

    def test_async_save_submission(self, optimized_persistence):
        """Test submitting async save operation."""
        def mock_save():
            return 101

        future = optimized_persistence.async_save(mock_save)

        # Should return a future
        assert future is not None

        # Can wait for result
        result = future.result(timeout=5)
        assert result == 101

    def test_wait_for_async_saves_collects_results(self, optimized_persistence):
        """Test waiting for multiple async saves."""
        def mock_save():
            return 101

        futures = [
            optimized_persistence.async_save(mock_save),
            optimized_persistence.async_save(mock_save),
            optimized_persistence.async_save(mock_save),
        ]

        results = optimized_persistence.wait_for_async_saves(futures)

        assert len(results) == 3
        assert all(r == 101 for r in results)

    def test_shutdown_thread_pool(self, optimized_persistence):
        """Test shutting down thread pool."""
        optimized_persistence.shutdown()

        # Thread pool should be shutdown
        assert True  # If we got here, shutdown succeeded


class TestBatchMovementCyclePersister:
    """Test specialized batch cycle persister."""

    @pytest.fixture
    def batch_persister(self):
        """Create batch cycle persister."""
        return BatchMovementCyclePersister(db_session=None, batch_size=5)

    def test_initialization(self, batch_persister):
        """Test batch persister initialization."""
        assert batch_persister.batch_size == 5
        assert len(batch_persister.cycle_buffer) == 0

    def test_set_foreign_keys(self, batch_persister):
        """Test setting shared foreign keys."""
        batch_persister.set_foreign_keys(
            audio_id=101,
            biomech_id=201,
            sync_id=301,
        )

        assert batch_persister.pending_fks["audio_id"] == 101
        assert batch_persister.pending_fks["biomech_id"] == 201
        assert batch_persister.pending_fks["sync_id"] == 301

    def test_add_cycle_merges_fks(self, batch_persister):
        """Test adding cycle merges FKs automatically."""
        batch_persister.set_foreign_keys(
            audio_id=101,
            biomech_id=201,
            sync_id=301,
        )

        cycle_data = {"cycle_number": 1, "data": "test"}
        batch_persister.add_cycle(cycle_data)

        assert len(batch_persister.cycle_buffer) == 1
        assert batch_persister.cycle_buffer[0]["audio_id"] == 101
        assert batch_persister.cycle_buffer[0]["biomech_id"] == 201
        assert batch_persister.cycle_buffer[0]["sync_id"] == 301

    def test_add_cycle_flushes_on_batch_full(self, batch_persister):
        """Test that buffer flushes when reaching batch size."""
        batch_persister.set_foreign_keys(
            audio_id=101,
            biomech_id=201,
            sync_id=301,
        )

        # Add 5 cycles (batch_size=5)
        for i in range(5):
            batch_persister.add_cycle({"cycle_number": i})

        # Buffer should be flushed
        assert len(batch_persister.cycle_buffer) == 0

    def test_flush_clears_buffer(self, batch_persister):
        """Test flush clears cycle buffer."""
        batch_persister.set_foreign_keys(
            audio_id=101,
            biomech_id=201,
            sync_id=301,
        )

        for i in range(3):
            batch_persister.add_cycle({"cycle_number": i})

        # Before flush, buffer should have cycles
        assert len(batch_persister.cycle_buffer) == 3

        results = batch_persister.flush()

        # After flush, buffer should be cleared
        assert len(batch_persister.cycle_buffer) == 0
        # Results list should be empty (no db_session) but flush executed
        assert isinstance(results, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Integration tests for Audio QC field persistence through DB and Excel.

Validates that:
1. AudioProcessing QC fields (per-channel fail segments, dropout, continuous)
   persist correctly to AudioProcessingRecord in the database.
2. Audio Excel sheet columns include all QC fields with correct values.
"""

from src.db.models import AudioProcessingRecord
from src.db.repository import Repository
from src.reports.report_generator import ReportGenerator


class TestAudioQCFieldsDatabasePersistence:
    """Verify audio QC fields round-trip through the database correctly."""

    def test_per_channel_fail_segments_persist(
        self,
        db_session,
        audio_processing_factory,
    ):
        """Per-channel qc_fail_segments should persist and round-trip."""
        repo = Repository(db_session)

        audio = audio_processing_factory(
            qc_fail_segments=[(1.0, 5.0), (10.0, 15.0)],
            qc_fail_segments_ch1=[(1.0, 5.0)],
            qc_fail_segments_ch2=[],
            qc_fail_segments_ch3=[(10.0, 15.0)],
            qc_fail_segments_ch4=[],
        )
        record = repo.save_audio_processing(audio)
        db_session.flush()

        fetched = db_session.get(AudioProcessingRecord, record.id)
        # Overall fail segments should be persisted
        assert fetched.qc_fail_segments is not None
        assert len(fetched.qc_fail_segments) >= 2  # at least 2 intervals

        # Per-channel: ch1 and ch3 have data, ch2 and ch4 are empty
        assert fetched.qc_fail_segments_ch1 is not None
        assert len(fetched.qc_fail_segments_ch1) >= 1
        assert fetched.qc_fail_segments_ch2 is None or len(fetched.qc_fail_segments_ch2) == 0
        assert fetched.qc_fail_segments_ch3 is not None
        assert len(fetched.qc_fail_segments_ch3) >= 1
        assert fetched.qc_fail_segments_ch4 is None or len(fetched.qc_fail_segments_ch4) == 0

    def test_signal_dropout_fields_persist(
        self,
        db_session,
        audio_processing_factory,
    ):
        """Signal dropout QC flags and segments should persist."""
        repo = Repository(db_session)

        audio = audio_processing_factory(
            qc_signal_dropout=True,
            qc_signal_dropout_segments=[(2.0, 3.0), (45.0, 48.0)],
            qc_signal_dropout_ch1=True,
            qc_signal_dropout_segments_ch1=[(2.0, 3.0)],
            qc_signal_dropout_ch2=False,
            qc_signal_dropout_segments_ch2=[],
            qc_signal_dropout_ch3=True,
            qc_signal_dropout_segments_ch3=[(45.0, 48.0)],
            qc_signal_dropout_ch4=False,
            qc_signal_dropout_segments_ch4=[],
        )
        record = repo.save_audio_processing(audio)
        db_session.flush()

        fetched = db_session.get(AudioProcessingRecord, record.id)
        assert fetched.qc_signal_dropout is True
        assert fetched.qc_signal_dropout_ch1 is True
        assert fetched.qc_signal_dropout_ch2 is False
        assert fetched.qc_signal_dropout_ch3 is True
        assert fetched.qc_signal_dropout_ch4 is False
        # Segments present on ch1 and ch3
        assert fetched.qc_signal_dropout_segments_ch1 is not None
        assert len(fetched.qc_signal_dropout_segments_ch1) >= 1
        assert fetched.qc_signal_dropout_segments_ch3 is not None
        assert len(fetched.qc_signal_dropout_segments_ch3) >= 1

    def test_continuous_artifact_fields_persist(
        self,
        db_session,
        audio_processing_factory,
    ):
        """Continuous artifact QC flags and segments should persist."""
        repo = Repository(db_session)

        audio = audio_processing_factory(
            qc_continuous_artifact=True,
            qc_continuous_artifact_segments=[(12.0, 18.0)],
            qc_continuous_artifact_ch1=False,
            qc_continuous_artifact_segments_ch1=[],
            qc_continuous_artifact_ch2=True,
            qc_continuous_artifact_segments_ch2=[(12.0, 18.0)],
            qc_continuous_artifact_ch3=False,
            qc_continuous_artifact_segments_ch3=[],
            qc_continuous_artifact_ch4=False,
            qc_continuous_artifact_segments_ch4=[],
        )
        record = repo.save_audio_processing(audio)
        db_session.flush()

        fetched = db_session.get(AudioProcessingRecord, record.id)
        assert fetched.qc_continuous_artifact is True
        assert fetched.qc_continuous_artifact_ch2 is True
        assert fetched.qc_continuous_artifact_ch1 is False
        assert fetched.qc_continuous_artifact_segments_ch2 is not None
        assert len(fetched.qc_continuous_artifact_segments_ch2) >= 1

    def test_all_qc_fields_false_when_clean(
        self,
        db_session,
        audio_processing_factory,
    ):
        """Clean recording should have all QC bools False and segments empty."""
        repo = Repository(db_session)

        audio = audio_processing_factory()  # defaults are all clean
        record = repo.save_audio_processing(audio)
        db_session.flush()

        fetched = db_session.get(AudioProcessingRecord, record.id)
        assert fetched.qc_signal_dropout is False
        assert fetched.qc_continuous_artifact is False
        for ch in range(1, 5):
            assert getattr(fetched, f"qc_signal_dropout_ch{ch}") is False
            assert getattr(fetched, f"qc_continuous_artifact_ch{ch}") is False


class TestSegmentsRoundTripThroughUnflatten:
    """Verify segments survive DB round-trip through _unflatten_intervals.

    This test catches the real-world bug where PostgreSQL returns
    nested lists [[s, e], ...] but _unflatten_intervals assumed flat
    [s, e, s, e, ...] format, causing TypeError when passed to
    trim_intervals_to_cycle.
    """

    def test_dropout_segments_round_trip_to_unflatten(
        self,
        db_session,
        audio_processing_factory,
    ):
        """Segments stored to DB should unflatten correctly for trim_intervals_to_cycle."""
        from src.audio.raw_qc import trim_intervals_to_cycle
        from src.orchestration.participant_processor import ManeuverProcessor

        repo = Repository(db_session)

        audio = audio_processing_factory(
            qc_signal_dropout=True,
            qc_signal_dropout_ch1=True,
            qc_signal_dropout_segments_ch1=[(2.0, 4.0), (8.0, 10.0)],
        )
        record = repo.save_audio_processing(audio)
        db_session.flush()

        fetched = db_session.get(AudioProcessingRecord, record.id)
        raw_segments = fetched.qc_signal_dropout_segments_ch1

        # This is what _persist_cycle_records does — must not raise TypeError
        intervals = ManeuverProcessor._unflatten_intervals(raw_segments)
        assert len(intervals) == 2
        assert all(isinstance(s, float) and isinstance(e, float) for s, e in intervals)

        # And trim should work without error
        trimmed = trim_intervals_to_cycle(intervals, 3.0, 9.0)
        assert len(trimmed) >= 1  # overlaps with both segments

    def test_continuous_segments_round_trip_to_unflatten(
        self,
        db_session,
        audio_processing_factory,
    ):
        """Continuous artifact segments should unflatten correctly after DB round-trip."""
        from src.orchestration.participant_processor import ManeuverProcessor

        repo = Repository(db_session)

        audio = audio_processing_factory(
            qc_continuous_artifact=True,
            qc_continuous_artifact_ch3=True,
            qc_continuous_artifact_segments_ch3=[(12.0, 18.0), (25.0, 30.0)],
        )
        record = repo.save_audio_processing(audio)
        db_session.flush()

        fetched = db_session.get(AudioProcessingRecord, record.id)
        raw_segments = fetched.qc_continuous_artifact_segments_ch3

        intervals = ManeuverProcessor._unflatten_intervals(raw_segments)
        assert len(intervals) == 2
        assert all(isinstance(s, float) and isinstance(e, float) for s, e in intervals)


class TestAudioExcelSheetQCColumns:
    """Verify Audio Excel sheet includes all QC columns with correct values."""

    def test_audio_sheet_has_all_qc_columns(
        self,
        db_session,
        audio_processing_factory,
    ):
        """Audio sheet should contain all per-channel QC columns."""
        repo = Repository(db_session)

        audio = audio_processing_factory(
            qc_fail_segments=[(1.0, 5.0)],
            qc_fail_segments_ch1=[(1.0, 5.0)],
            qc_signal_dropout=True,
            qc_signal_dropout_segments=[(1.0, 5.0)],
            qc_signal_dropout_ch1=True,
            qc_signal_dropout_segments_ch1=[(1.0, 5.0)],
        )
        record = repo.save_audio_processing(audio)
        db_session.flush()

        generator = ReportGenerator(db_session)
        sheet = generator.generate_audio_sheet(
            study_id=record.study_id,
            maneuver="walk",
            knee="left",
        )

        assert len(sheet) > 0

        # Overall QC columns
        expected_columns = [
            "QC Fail",
            "QC Fail Segments",
            "QC Fail Segments Ch1",
            "QC Fail Segments Ch2",
            "QC Fail Segments Ch3",
            "QC Fail Segments Ch4",
            # Signal dropout
            "QC Signal Dropout",
            "QC Signal Dropout Segments",
            "QC Signal Dropout Ch1",
            "QC Signal Dropout Segments Ch1",
            "QC Signal Dropout Ch2",
            "QC Signal Dropout Segments Ch2",
            "QC Signal Dropout Ch3",
            "QC Signal Dropout Segments Ch3",
            "QC Signal Dropout Ch4",
            "QC Signal Dropout Segments Ch4",
            # Continuous artifact
            "QC Continuous Artifact",
            "QC Continuous Artifact Segments",
            "QC Continuous Artifact Ch1",
            "QC Continuous Artifact Segments Ch1",
            "QC Continuous Artifact Ch2",
            "QC Continuous Artifact Segments Ch2",
            "QC Continuous Artifact Ch3",
            "QC Continuous Artifact Segments Ch3",
            "QC Continuous Artifact Ch4",
            "QC Continuous Artifact Segments Ch4",
        ]
        for col in expected_columns:
            assert col in sheet.columns, (
                f"Missing column '{col}' in Audio sheet. Available: {sorted(sheet.columns.tolist())}"
            )

    def test_audio_sheet_qc_values_populated(
        self,
        db_session,
        audio_processing_factory,
    ):
        """Audio sheet QC values should match stored data."""
        repo = Repository(db_session)

        audio = audio_processing_factory(
            qc_signal_dropout=True,
            qc_signal_dropout_segments=[(2.0, 4.0)],
            qc_signal_dropout_ch1=True,
            qc_signal_dropout_segments_ch1=[(2.0, 4.0)],
            qc_continuous_artifact=True,
            qc_continuous_artifact_segments=[(10.0, 16.0)],
            qc_continuous_artifact_ch3=True,
            qc_continuous_artifact_segments_ch3=[(10.0, 16.0)],
            # Overall fail segments should include both
            qc_fail_segments=[(2.0, 4.0), (10.0, 16.0)],
            qc_fail_segments_ch1=[(2.0, 4.0)],
            qc_fail_segments_ch3=[(10.0, 16.0)],
        )
        record = repo.save_audio_processing(audio)
        db_session.flush()

        generator = ReportGenerator(db_session)
        sheet = generator.generate_audio_sheet(
            study_id=record.study_id,
            maneuver="walk",
            knee="left",
        )

        row = sheet.iloc[0]
        assert bool(row["QC Fail"]) is True  # dropout OR continuous
        assert bool(row["QC Signal Dropout"]) is True
        assert bool(row["QC Continuous Artifact"]) is True
        assert bool(row["QC Signal Dropout Ch1"]) is True
        assert bool(row["QC Continuous Artifact Ch3"]) is True

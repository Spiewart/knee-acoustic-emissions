from typing import Annotated, Literal, Optional, Self

import pandas as pd
from pydantic import BaseModel, field_validator, model_validator
from pydantic_core import CoreSchema, PydanticCustomError, core_schema


class MicrophonePosition(BaseModel):
    """Metadata describing a microphone position on the knee."""

    patellar_position: Literal["Infrapatellar", "Suprapatellar"]
    laterality: Literal["Medial", "Lateral"]


class AcousticsMetadata(BaseModel):
    """Metadata for an acoustics recording."""

    scripted_maneuver: Literal["walk", "sit_to_stand", "flexion_extension"]
    speed: Optional[Literal["slow", "medium", "fast"]] = None
    knee: Literal["left", "right"]
    file_name: str
    microphones: Annotated[
        dict[Literal[1, 2, 3, 4], MicrophonePosition],
        "Microphone index (1-4) mapped to position metadata",
    ]
    notes: Optional[str] = None
    microphone_notes: Optional[dict[Literal[1, 2, 3, 4], str]] = None

    @field_validator("microphones")
    @classmethod
    def validate_microphone_keys(cls, value: dict) -> dict:
        """Ensure microphones dict has exactly keys {1,2,3,4}."""
        required_keys: set[int] = {1, 2, 3, 4}
        if set(value.keys()) != required_keys:
            raise ValueError(
                "microphones must contain exactly keys "
                f"{required_keys}, got {set(value.keys())}"
            )
        return value

    @field_validator("microphone_notes")
    @classmethod
    def validate_microphone_notes_keys(
        cls, value: Optional[dict]
    ) -> Optional[dict]:
        """Validate microphone_notes keys are within {1,2,3,4}."""
        if value is None:
            return value
        allowed_keys: set[int] = {1, 2, 3, 4}
        if not set(value.keys()).issubset(allowed_keys):
            raise ValueError(
                "microphone_notes keys must be in {1, 2, 3, 4}, "
                f"got {set(value.keys())}"
            )
        return value


class AcousticsRecording(pd.DataFrame):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type, _handler
    ) -> CoreSchema:
        def validate_dataframe(value: pd.DataFrame) -> Self:
            if not isinstance(value, pd.DataFrame):
                raise PydanticCustomError(
                    "not_dataframe", "Value is not a pandas DataFrame"
                )
            required_columns: list[str] = [
                "tt",
                "ch1",
                "ch2",
                "ch3",
                "ch4",
                "f_ch1",
                "f_ch2",
                "f_ch3",
                "f_ch4",
            ]
            if not all(col in value.columns for col in required_columns):
                missing_cols: set[str] = (
                    set(required_columns) - set(value.columns)
                )
                msg = (
                    "DataFrame is missing required columns: "
                    f"{', '.join(missing_cols)}"
                )
                raise PydanticCustomError("missing_column", msg)
            return cls(value)

        return core_schema.no_info_after_validator_function(
            validate_dataframe,
            core_schema.any_schema(),
        )


class AcousticsCycle(AcousticsMetadata):
    """Acoustics data paired with its metadata."""

    data: AcousticsRecording


class BiomechanicsMetadata(BaseModel):
    """Metadata for a biomechanics recording."""

    maneuver: Literal["walk", "sit_to_stand", "flexion_extension"]
    speed: Optional[Literal["slow", "normal", "fast"]] = None
    pass_number: Optional[int] = None

    @field_validator("speed")
    @classmethod
    def validate_speed_for_maneuver(
        cls, value: Optional[str], info
    ) -> Optional[str]:
        if info.data.get("maneuver") == "walk":
            if value is None:
                raise ValueError("speed is required when maneuver is 'walk'")
        elif value is not None:
            raise ValueError(
                "speed must be None when maneuver is "
                f"'{info.data.get('maneuver')}'"
            )
        return value

    @field_validator("pass_number")
    @classmethod
    def validate_pass_number_for_maneuver(
        cls, value: Optional[int], info
    ) -> Optional[int]:
        if info.data.get("maneuver") == "walk":
            if value is None:
                raise ValueError(
                    "pass_number is required when maneuver is 'walk'"
                )
        elif value is not None:
            raise ValueError(
                "pass_number must be None when maneuver is "
                f"'{info.data.get('maneuver')}'"
            )
        return value


class PassMetadata(pd.DataFrame):
    """Pass event metadata with required columns."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type, _handler
    ) -> CoreSchema:
        def validate_dataframe(value: pd.DataFrame) -> Self:
            if not isinstance(value, pd.DataFrame):
                raise PydanticCustomError(
                    "not_dataframe", "Value is not a pandas DataFrame"
                )

            required_columns: list[str] = ["Event Info", "Time (sec)"]
            if not all(col in value.columns for col in required_columns):
                missing: set[str] = set(required_columns) - set(value.columns)
                raise PydanticCustomError(
                    "missing_column",
                    "DataFrame is missing required columns: "
                    f"{', '.join(missing)}",
                )
            return cls(value)

        return core_schema.no_info_after_validator_function(
            validate_dataframe,
            core_schema.any_schema(),
        )


class BiomechanicsRecording(pd.DataFrame):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type, _handler
    ) -> CoreSchema:
        def validate_dataframe(value: pd.DataFrame) -> Self:
            if not isinstance(value, pd.DataFrame):
                raise PydanticCustomError(
                    "not_dataframe", "Value is not a pandas DataFrame"
                )
            if "TIME" not in value.columns:
                raise PydanticCustomError(
                    "missing_column", "DataFrame must contain 'required_col'"
                )
            return cls(value)

        return core_schema.no_info_after_validator_function(
            validate_dataframe,
            core_schema.any_schema(),
        )


class BiomechanicsCycle(BiomechanicsMetadata):
    """Biomechanics data paired with metadata and sync details."""

    data: BiomechanicsRecording
    sync_left_time: Optional[float] = None
    sync_right_time: Optional[float] = None
    pass_metadata: Optional[PassMetadata] = None


class SynchronizedRecording(pd.DataFrame):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type, _handler
    ) -> CoreSchema:
        def validate_dataframe(value: pd.DataFrame) -> Self:
            if not isinstance(value, pd.DataFrame):
                raise PydanticCustomError(
                    "not_dataframe", "Value is not a pandas DataFrame"
                )
            required_columns: list[str] = [
                "tt",
                "ch1",
                "ch2",
                "ch3",
                "ch4",
                "f_ch1",
                "f_ch2",
                "f_ch3",
                "f_ch4",
                "TIME",
            ]
            if not all(col in value.columns for col in required_columns):
                missing_cols: set[str] = (
                    set(required_columns) - set(value.columns)
                )
                msg = (
                    "DataFrame is missing required columns: "
                    f"{', '.join(missing_cols)}"
                )
                raise PydanticCustomError("missing_column", msg)
            return cls(value)

        return core_schema.no_info_after_validator_function(
            validate_dataframe,
            core_schema.any_schema(),
        )


class SynchronizedCycle(AcousticsMetadata, BiomechanicsMetadata):
    """Synchronized acoustics and biomechanics data."""

    data: SynchronizedRecording


class MovementCycleMetadata(BaseModel):
    """Metadata for a movement cycle from a synchronized recording."""

    maneuver: Literal["walk", "sit_to_stand", "flexion_extension"]
    speed: Optional[Literal["slow", "medium", "fast"]] = None
    pass_number: Optional[int] = None
    cycle_index: int
    knee: Literal["left", "right"]
    participant_id: Optional[str] = None
    acoustic_energy: float
    is_outlier: bool = False
    notes: Optional[str] = None

    @model_validator(mode="after")
    def validate_walk_parameters(self) -> "MovementCycleMetadata":
        if self.maneuver == "walk":
            missing: list[str] = []
            if self.speed is None:
                missing.append("speed")
            if self.pass_number is None:
                missing.append("pass_number")
            if missing:
                raise ValueError(
                    "Walk maneuvers require both speed and pass_number. "
                    f"Missing: {', '.join(missing)}"
                )
            if not isinstance(self.pass_number, int) or self.pass_number < 0:
                raise ValueError(
                    "pass_number must be a non-negative integer, "
                    f"got {self.pass_number}"
                )
        else:
            if self.speed is not None or self.pass_number is not None:
                invalid: list[str] = []
                if self.speed is not None:
                    invalid.append(f"speed={self.speed}")
                if self.pass_number is not None:
                    invalid.append(f"pass_number={self.pass_number}")
                raise ValueError(
                    f"{self.maneuver.replace('_', ' ').title()} maneuvers do not "
                    "support speed and pass_number parameters. "
                    f"Got: {', '.join(invalid)}"
                )

        return self


class MovementCycle(BaseModel):
    """Single movement cycle with synchronized data."""

    metadata: MovementCycleMetadata
    data: SynchronizedRecording

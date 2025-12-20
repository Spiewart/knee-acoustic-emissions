from typing import Annotated, Literal, Optional

import pandas as pd
from pydantic import BaseModel, field_validator, model_validator
from pydantic_core import CoreSchema, PydanticCustomError, core_schema


class MicrophonePosition(BaseModel):
    """Pydantic Model for microphone position metadata."""

    patellar_position: Literal["Infrapatellar", "Suprapatellar"]
    laterality: Literal["Medial", "Lateral"]


class AcousticsMetadata(BaseModel):
    """Pydantic Model for acoustics metadata.

    Associated with a scripted maneuver and knee.
    """

    scripted_maneuver: Literal["walk", "sit_to_stand", "flexion_extension"]
    speed: Optional[Literal["slow", "medium", "fast"]] = None
    knee: Literal["left", "right"]
    file_name: str
    microphones: Annotated[
        dict[Literal[1, 2, 3, 4], MicrophonePosition],
        "Dictionary mapping microphone numbers (1, 2, 3, 4) to positions"
    ]
    notes: Optional[str] = None
    microphone_notes: Optional[dict[Literal[1, 2, 3, 4], str]] = None

    @field_validator('microphones')
    @classmethod
    def validate_microphone_keys(cls, v: dict) -> dict:
        """Validate microphones dict contains exactly keys 1, 2, 3, 4."""
        required_keys = {1, 2, 3, 4}
        if set(v.keys()) != required_keys:
            raise ValueError(
                f"microphones must contain exactly keys {required_keys}, "
                f"got {set(v.keys())}"
            )
        return v

    @field_validator('microphone_notes')
    @classmethod
    def validate_microphone_notes_keys(cls, v: Optional[dict]) -> Optional[dict]:
        """Validate microphone_notes keys are subset of 1, 2, 3, 4."""
        if v is None:
            return v
        allowed_keys = {1, 2, 3, 4}
        if not set(v.keys()).issubset(allowed_keys):
            raise ValueError(
                f"microphone_notes keys must be in {allowed_keys}, "
                f"got {set(v.keys())}"
            )
        return v


class AcousticsRecording(pd.DataFrame):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler) -> CoreSchema:
        # Define a custom validation logic for MyDataFrame
        def validate_dataframe(value):
            if not isinstance(value, pd.DataFrame):
                raise PydanticCustomError("not_dataframe", "Value is not a pandas DataFrame")
            required_columns = ["tt", "ch1", "ch2", "ch3", "ch4", "f_ch1", "f_ch2", "f_ch3", "f_ch4"]
            if not all(col in value.columns for col in required_columns):
                msg = f"DataFrame is missing required columns: {', '.join(set(required_columns) - set(value.columns))}"
                raise PydanticCustomError(
                    "missing_column",
                    msg,
                )
            return cls(value)  # Ensure it's an instance of MyDataFrame

        return core_schema.no_info_after_validator_function(
            validate_dataframe,
            core_schema.any_schema()  # Accepts any input before validation
        )


class AcousticsCycle(AcousticsMetadata):
    """Pydantic Model for acoustics data and associated metadata."""

    data: AcousticsRecording


class BiomechanicsMetadata(BaseModel):
    """Pydantic Model for biomechanics metadata associated with a recording.

    Validation rules:
    - When maneuver is "walk": pass_number is required, speed is required
    - When maneuver is "sit_to_stand" or "flexion_extension": pass_number must be None, speed must be None
    """

    maneuver: Literal["walk", "sit_to_stand", "flexion_extension"]
    speed: Optional[Literal["slow", "normal", "fast"]] = None
    pass_number: Optional[int] = None

    @field_validator('speed')
    @classmethod
    def validate_speed_for_maneuver(cls, v: Optional[str], info) -> Optional[str]:
        """Validate speed based on maneuver type."""
        if info.data.get('maneuver') == 'walk':
            if v is None:
                raise ValueError("speed is required when maneuver is 'walk'")
        else:  # sit_to_stand or flexion_extension
            if v is not None:
                raise ValueError(f"speed must be None when maneuver is '{info.data.get('maneuver')}'")
        return v

    @field_validator('pass_number')
    @classmethod
    def validate_pass_number_for_maneuver(cls, v: Optional[int], info) -> Optional[int]:
        """Validate pass_number based on maneuver type."""
        if info.data.get('maneuver') == 'walk':
            if v is None:
                raise ValueError("pass_number is required when maneuver is 'walk'")
        else:  # sit_to_stand or flexion_extension
            if v is not None:
                raise ValueError(f"pass_number must be None when maneuver is '{info.data.get('maneuver')}'")
        return v


class PassMetadata(pd.DataFrame):
    """DataFrame subclass for biomechanics pass event metadata.

    Required columns:
    - Event Info: String describing the event (e.g., "SS Pass 1 Start", "Movement Start")
    - Time (sec): Float timestamp in seconds
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler) -> CoreSchema:
        def validate_dataframe(value):
            if not isinstance(value, pd.DataFrame):
                raise PydanticCustomError("not_dataframe", "Value is not a pandas DataFrame")

            required_columns = ["Event Info", "Time (sec)"]
            if not all(col in value.columns for col in required_columns):
                missing = set(required_columns) - set(value.columns)
                raise PydanticCustomError(
                    "missing_column",
                    f"DataFrame is missing required columns: {', '.join(missing)}"
                )
            return cls(value)

        return core_schema.no_info_after_validator_function(
            validate_dataframe,
            core_schema.any_schema()
        )


class BiomechanicsRecording(pd.DataFrame):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler) -> CoreSchema:
        # Define a custom validation logic for MyDataFrame
        def validate_dataframe(value):
            if not isinstance(value, pd.DataFrame):
                raise PydanticCustomError("not_dataframe", "Value is not a pandas DataFrame")
            if "TIME" not in value.columns:
                raise PydanticCustomError("missing_column", "DataFrame must contain 'required_col'")
            return cls(value)  # Ensure it's an instance of MyDataFrame

        return core_schema.no_info_after_validator_function(
            validate_dataframe,
            core_schema.any_schema()  # Accepts any input before validation
        )


class BiomechanicsCycle(BiomechanicsMetadata):
    """Pydantic Model for biomechanics data and associated metadata.

    Additional fields for walking maneuvers:
    - sync_left_time: Time of left foot stomp sync event (seconds)
    - sync_right_time: Time of right foot stomp sync event (seconds)
    - pass_metadata: DataFrame containing event metadata for the pass
    """

    data: BiomechanicsRecording
    sync_left_time: Optional[float] = None
    sync_right_time: Optional[float] = None
    pass_metadata: Optional[PassMetadata] = None


class SynchronizedRecording(pd.DataFrame):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler) -> CoreSchema:
        # Define a custom validation logic for MyDataFrame
        def validate_dataframe(value):
            if not isinstance(value, pd.DataFrame):
                raise PydanticCustomError("not_dataframe", "Value is not a pandas DataFrame")
            required_columns = ["tt", "ch1", "ch2", "ch3", "ch4", "f_ch1", "f_ch2", "f_ch3", "f_ch4", "TIME"]
            if not all(col in value.columns for col in required_columns):
                msg = f"DataFrame is missing required columns: {', '.join(set(required_columns) - set(value.columns))}"
                raise PydanticCustomError(
                    "missing_column",
                    msg,
                )
            return cls(value)  # Ensure it's an instance of MyDataFrame

        return core_schema.no_info_after_validator_function(
            validate_dataframe,
            core_schema.any_schema()  # Accepts any input before validation
        )


class SynchronizedCycle(AcousticsMetadata, BiomechanicsMetadata):
    """Pydantic Model for synchronized acoustics and biomechanics data."""

    data: SynchronizedRecording


class MovementCycleMetadata(BaseModel):
    """Metadata for a single movement cycle extracted from a synchronized recording."""

    maneuver: Literal["walk", "sit_to_stand", "flexion_extension"]
    speed: Optional[Literal["slow", "medium", "fast"]] = None
    pass_number: Optional[int] = None
    cycle_index: int
    knee: Literal["left", "right"]
    participant_id: Optional[str] = None
    acoustic_energy: float
    is_outlier: bool = False
    notes: Optional[str] = None

    @model_validator(mode='after')
    def validate_walk_parameters(self) -> "MovementCycleMetadata":
        """Validate speed and pass_number based on maneuver type.

        For walk maneuvers: both speed and pass_number must be provided.
        For other maneuvers: both speed and pass_number must be None.

        Raises:
            ValueError: If parameters don't match maneuver type.
        """
        if self.maneuver == "walk":
            # For walk, both speed and pass_number are required
            if self.speed is None or self.pass_number is None:
                missing = []
                if self.speed is None:
                    missing.append("speed")
                if self.pass_number is None:
                    missing.append("pass_number")
                raise ValueError(
                    f"Walk maneuvers require both speed and pass_number. "
                    f"Missing: {', '.join(missing)}"
                )
            # Validate pass_number is non-negative integer
            if not isinstance(self.pass_number, int) or self.pass_number < 0:
                raise ValueError(
                    f"pass_number must be a non-negative integer, got {self.pass_number}"
                )
        else:
            # For non-walk maneuvers, both must be None
            if self.speed is not None or self.pass_number is not None:
                invalid = []
                if self.speed is not None:
                    invalid.append(f"speed={self.speed}")
                if self.pass_number is not None:
                    invalid.append(f"pass_number={self.pass_number}")
                raise ValueError(
                    f"{self.maneuver.replace('_', ' ').title()} maneuvers do not support "
                    f"speed and pass_number parameters. Got: {', '.join(invalid)}"
                )

        return self



class MovementCycle(BaseModel):
    """Pydantic Model for a single movement cycle with audio and biomechanics."""

    metadata: MovementCycleMetadata
    data: SynchronizedRecording  # Cycle data with audio and biomechanics

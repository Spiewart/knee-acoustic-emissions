from typing import Annotated, Literal

import pandas as pd
from pydantic import BaseModel, field_validator
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
    speed: Literal["slow", "medium", "fast"] | None = None
    knee: Literal["left", "right"]
    file_name: str
    microphones: Annotated[
        dict[Literal[1, 2, 3, 4], MicrophonePosition],
        "Dictionary mapping microphone numbers (1, 2, 3, 4) to positions"
    ]
    notes: str | None = None
    microphone_notes: dict[Literal[1, 2, 3, 4], str] | None = None

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
    def validate_microphone_notes_keys(cls, v: dict | None) -> dict | None:
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
            # TODO: Add more specific DataFrame structure validation here
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
    speed: Literal["slow", "normal", "fast"] | None = None
    pass_number: int | None = None

    @field_validator('speed')
    @classmethod
    def validate_speed_for_maneuver(cls, v: str | None, info) -> str | None:
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
    def validate_pass_number_for_maneuver(cls, v: int | None, info) -> int | None:
        """Validate pass_number based on maneuver type."""
        if info.data.get('maneuver') == 'walk':
            if v is None:
                raise ValueError("pass_number is required when maneuver is 'walk'")
        else:  # sit_to_stand or flexion_extension
            if v is not None:
                raise ValueError(f"pass_number must be None when maneuver is '{info.data.get('maneuver')}'")
        return v


class BiomechanicsRecording(pd.DataFrame):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler) -> CoreSchema:
        # Define a custom validation logic for MyDataFrame
        def validate_dataframe(value):
            if not isinstance(value, pd.DataFrame):
                raise PydanticCustomError("not_dataframe", "Value is not a pandas DataFrame")
            # TODO: Add more specific DataFrame structure validation here
            if "TIME" not in value.columns:
                raise PydanticCustomError("missing_column", "DataFrame must contain 'required_col'")
            return cls(value)  # Ensure it's an instance of MyDataFrame

        return core_schema.no_info_after_validator_function(
            validate_dataframe,
            core_schema.any_schema()  # Accepts any input before validation
        )


class BiomechanicsCycle(BiomechanicsMetadata):
    """Pydantic Model for biomechanics data and associated metadata."""

    data: BiomechanicsRecording


class SynchronizedRecording(pd.DataFrame):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler) -> CoreSchema:
        # Define a custom validation logic for MyDataFrame
        def validate_dataframe(value):
            if not isinstance(value, pd.DataFrame):
                raise PydanticCustomError("not_dataframe", "Value is not a pandas DataFrame")
            # TODO: Add more specific DataFrame structure validation here
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

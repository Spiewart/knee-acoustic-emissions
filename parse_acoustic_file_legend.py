from typing import Literal

import pandas as pd

from models import AcousticsMetadata, MicrophonePosition


def get_acoustics_metadata(
    metadata_file_path: str,
    scripted_maneuver: Literal["walk", "sit_to_stand", "flexion_extension"],
    knee: Literal["left", "right"],
) -> AcousticsMetadata:
    """Gets acoustics metadata for a given acoustics file legend. Extracts the
    relevant metadata based on the scripted maneuver and knee laterality and returns
    a Pydantic Model instance that validates the metadata.

    File legend is two Excel tables on top of one another, separated by a blank row,
    where the first row in each table is either "R Knee" or "L Knee", and the second row
    is divided into columns:
        "Maneuvers": single row of Literal["Walk (slow,medium, fast)", "Flexion - Extension", "Sit - to - Stand"]
        "File Name": single row of str
        "Microphone": 4 rows of Literal[1, 2, 3, 4]
        "Patellar Position": 4 rows of Literal["Infrapatellar", "Suprapatellar"]
        "Laterality": 4 rows of Literal["Medial", "Lateral"]
        "Notes": 4 rows of str

    Args:
        metadata_file_path (str): Path to the Excel file containing the acoustics file legend.
        scripted_maneuver (Literal): The scripted maneuver to filter by.
        knee (Literal): The knee laterality to filter by.

    Returns:
        AcousticsMetadata: A Pydantic model instance containing the validated acoustics metadata.
    """
    metadata_df = pd.read_excel(metadata_file_path, sheet_name="Acoustic Notes", header=None)
    # Determine the starting row for the desired knee laterality
    # by getting the index of the first column that matches the knee laterality
    table_header_index = metadata_df.index[
        metadata_df.iloc[:, 0].str.contains(f"{knee[0].capitalize()} Knee", na=False)
    ].tolist()
    if not table_header_index:
        raise ValueError(f"No data found for {knee[0].capitalize()} Knee in metadata file.")
    table_columns_row = table_header_index[0] + 1  # Data starts after the table header row

    # Get the sub-DataFrame for the desired knee laterality and scripted maneuver
    # but only take the rows for the relevant knee, otherwise duplicate data will be included
    knee_metadata_df = metadata_df.iloc[
        table_columns_row + 0: table_columns_row + 13, :
    ].reset_index(drop=True)
    # Rename the columns based on the values in the first row
    knee_metadata_df.columns = knee_metadata_df.iloc[0]
    knee_metadata_df = knee_metadata_df.drop(knee_metadata_df.index[0]).reset_index(drop=True)
    # Strip any "-" characters from the "Maneuvers" column for easier matching
    knee_metadata_df["Maneuvers"] = knee_metadata_df["Maneuvers"].str.replace(
        "-", " ", regex=False,
    )
    # Eliminate anything more than a single space in the "Maneuvers" column
    knee_metadata_df["Maneuvers"] = knee_metadata_df["Maneuvers"].str.replace(
        r"\s+", " ", regex=True,
    ).str.strip()
    # Fill in the Maneuvers column downwards to associate each maneuver with the subsequent
    # three empty rows for microphones 2-4
    knee_metadata_df["Maneuvers"] = knee_metadata_df["Maneuvers"].ffill()

    maneuver_metadata_df = knee_metadata_df.loc[
        knee_metadata_df["Maneuvers"].str.contains(
            scripted_maneuver.replace("_", " "), case=False, na=False
        )
    ]

    file_name = maneuver_metadata_df["File Name"].values[0]

    microphones = {}
    microphone_notes = {}

    for _, row in maneuver_metadata_df.iterrows():
        mic_number = int(row["Microphone"])
        microphones[mic_number] = MicrophonePosition(
            patellar_position=row["Patellar Position"],
            laterality=row["Laterality"],
        )
        if pd.notna(row["Notes"]):
            microphone_notes[mic_number] = row["Notes"]

    notes = None

    if pd.notna(maneuver_metadata_df["Notes"].values[0]):
        notes = maneuver_metadata_df["Notes"].values[0]

    acoustics_metadata = AcousticsMetadata(
        scripted_maneuver=scripted_maneuver,
        knee=knee,
        file_name=file_name,
        microphones=microphones,
        notes=notes,
        microphone_notes=microphone_notes if microphone_notes else None,
    )

    return acoustics_metadata
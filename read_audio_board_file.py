"""read_audio_board_file.py

Converted from read_audio_board_file.m (MATLAB -> Python).

This module reads the binary file output from the audio board and
converts it to a MATLAB .mat file containing the parsed header and
audio channel data.

Notes:
- This is a near-direct translation focused on keeping the numeric
  behaviour consistent with the original MATLAB implementation.
"""

from __future__ import annotations

import json
import logging
import math
import os
import warnings
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def read_audio_board_file(fname: str, output_folder: Optional[str] = None) -> None:
    """Read binary file produced by the audio board and save a .mat file.

    Parameters
    ----------
    fname : str
        Path to the .bin file produced by the HeartPulse app.
    output_folder : Optional[str]
        Folder where the .mat output will be written. If None, the
        output is saved next to the input file.
    """
    if output_folder is None:
        output_folder = os.path.dirname(fname)

    if not os.path.isdir(output_folder):
        raise ValueError("outputFolder is not a valid path!")

    if not os.path.isfile(fname):
        raise ValueError("fname file does not exist!")

    # Ensure trailing separator
    output_folder = os.path.abspath(output_folder) + os.sep

    # Read header (512 bytes)
    with open(fname, "rb") as f:
        header = f.read(512)

    if len(header) < 512:
        raise ValueError("File header is too short")

    header_arr = np.frombuffer(header, dtype=np.uint8)

    # Parse header fields (MATLAB indices are 1-based)
    deviceSerial = np.frombuffer(header_arr[0:16].tobytes(), dtype="<u4")
    projectNum = np.frombuffer(header_arr[16:18].tobytes(), dtype="<u2")
    hwRevMajor = np.frombuffer(header_arr[18:20].tobytes(), dtype="<u2")
    hwRevMinor = np.frombuffer(header_arr[20:22].tobytes(), dtype="<u2")
    HP_Serial = np.frombuffer(header_arr[22:24].tobytes(), dtype="<u2")
    devFirmwareVersion = np.frombuffer(header_arr[24:28].tobytes(), dtype="<u4")
    numSDBlocks = np.frombuffer(header_arr[61:65].tobytes(), dtype="<u4")
    fileTime_bytes = header_arr[65:73].copy()

    # Convert arrays to Python numeric types where appropriate
    deviceSerial = deviceSerial.astype(np.uint64)
    projectNum = int(projectNum[0]) if projectNum.size > 0 else 0
    hwRevMajor = int(hwRevMajor[0]) if hwRevMajor.size > 0 else 0
    hwRevMinor = int(hwRevMinor[0]) if hwRevMinor.size > 0 else 0
    HP_Serial = int(HP_Serial[0]) if HP_Serial.size > 0 else 0
    devFirmwareVersion = int(devFirmwareVersion[0]) if devFirmwareVersion.size > 0 else 0
    numSDBlocks = int(numSDBlocks[0]) if numSDBlocks.size > 0 else 0

    # Convert the filetime following the MATLAB logic
    # The MATLAB code flips the byte order first
    fileTime_uint64 = int.from_bytes(fileTime_bytes[::-1].tobytes(), byteorder="little", signed=False)

    # Constants (from MATLAB code)
    numTicks_1601_01_01_to_1900_01_01 = 94354848000000000
    fileTime_1900 = fileTime_uint64 - numTicks_1601_01_01_to_1900_01_01

    # Timezone offset in hours (matching MATLAB approach using Java Date)
    # Use local timezone offset in hours
    try:
        utco = datetime.now(timezone.utc).astimezone().utcoffset()
        tz_offset_hours = (utco.total_seconds() / 3600.0) if utco is not None else 0.0
    except (OSError, ValueError, OverflowError) as e:
        # Rare environment/timezone errors; log and fall back to zero offset
        logging.exception("Could not determine local timezone offset; defaulting to 0: %s", e)
        tz_offset_hours = 0.0

    # The MATLAB code multiplies by 60*60/100e-9 to convert to ticks units
    fileTime_1900 = fileTime_1900 + tz_offset_hours * 60 * 60 / 100e-9
    numTicksPerDay = 24 * 60 * 60 / 100e-9
    excelTime = fileTime_1900 / numTicksPerDay
    excelTime = excelTime + 2

    # Convert Excel serial date (MATLAB's datetime with 'ConvertFrom','excel')
    # Excel serial 0 corresponds to 1899-12-30 in MATLAB/Python conversion
    excel_base = datetime(1899, 12, 30)
    fileTime_dt = excel_base + timedelta(days=float(excelTime))

    # Build the output structure (dict)
    file_audio: Dict = {}
    file_audio["deviceSerial"] = deviceSerial.tolist()
    file_audio["projectNum"] = projectNum
    file_audio["hwRevMajor"] = hwRevMajor
    file_audio["hwRevMinor"] = hwRevMinor
    file_audio["HP_Serial"] = HP_Serial
    file_audio["devFirmwareVersion"] = devFirmwareVersion
    file_audio["numSDBlocks"] = numSDBlocks
    file_audio["fileTime"] = fileTime_dt
    file_audio["matlabVersion"] = 1.0
    file_audio["fs_ast"] = 4096

    # Firmware-dependent constants
    if devFirmwareVersion == 1:
        num_bits_audio = 16
        fs_audio = 46.875e3
    elif devFirmwareVersion == 2:
        num_bits_audio = 16
        fs_audio = 46.875e3
    else:
        warnings.warn("Unrecognized firmware version! Using default number of bits and sample rate")
        num_bits_audio = 16
        fs_audio = 46.875e3

    file_audio["numBits"] = int(num_bits_audio)
    file_audio["fs"] = float(fs_audio)

    # Read and parse the audio file data
    file_data = get_audio_board_file_helper(fname, file_audio["fs"], file_audio["numBits"])

    if file_data is None:
        # No data found in the file
        pass
    else:
        file_audio.update(
            {
                "tt": file_data["tt"],
                "ch1": file_data["ch1"],
                "ch2": file_data["ch2"],
                "ch3": file_data["ch3"],
                "ch4": file_data["ch4"],
                "tt_blocks": file_data["tt_blocks"],
                "startTime": file_data["startTime"],
                "stopTime": file_data["stopTime"],
                "firstSampTime": file_data["firstSampTime"],
                "firstSampRaw": file_data["firstSampRaw"],
                "secondSampRaw": file_data["secondSampRaw"],
            }
        )

    # Save channel data as a pandas DataFrame (pickle) and write metadata to JSON
    outname, _ = os.path.splitext(os.path.basename(fname))
    fname_out = os.path.join(output_folder, outname + ".pkl")

    # Create DataFrame from time and channel arrays if available
    if all(k in file_audio for k in ("tt", "ch1", "ch2", "ch3", "ch4")):
        df = pd.DataFrame(
            {
                "tt": np.asarray(file_audio["tt"]),
                "ch1": np.asarray(file_audio["ch1"]),
                "ch2": np.asarray(file_audio["ch2"]),
                "ch3": np.asarray(file_audio["ch3"]),
                "ch4": np.asarray(file_audio["ch4"]),
            }
        )
    else:
        # If channel data missing, save an empty DataFrame and still write metadata
        df = pd.DataFrame()

    # Save DataFrame to a pickle file
    df.to_pickle(fname_out)

    # Save metadata (non-array values) to JSON for easy access
    meta = {}
    for k, v in file_audio.items():
        if k in ("tt", "ch1", "ch2", "ch3", "ch4", "tt_blocks"):
            continue
        # Convert numpy scalars to native types where possible
        if isinstance(v, (np.generic,)):
            try:
                meta[k] = v.item()
                continue
            except (AttributeError, TypeError, ValueError) as e:
                # Not convertible via .item(); skip to next serialization attempt
                logging.debug("Failed to convert numpy scalar for key %s: %s", k, e)
        # Convert datetimes to ISO strings
        if isinstance(v, (datetime,)):
            meta[k] = v.isoformat()
            continue
        # Lists, ints, strings are JSON serializable; otherwise try cast
        try:
            json.dumps(v)
            meta[k] = v
        except (TypeError, OverflowError) as e:
            logging.debug("Value for key %s not JSON serializable: %s", k, e)
            try:
                meta[k] = str(v)
            except (TypeError, ValueError) as e2:
                logging.exception("Failed to cast value to string for key %s: %s", k, e2)
                meta[k] = None

    meta_fname = os.path.join(output_folder, outname + "_meta.json")
    try:
        with open(meta_fname, "w", encoding="utf-8") as mf:
            json.dump(meta, mf, ensure_ascii=False, indent=2)
    except (OSError, TypeError) as e:
        # If metadata can't be written, continue without failing the whole process but log details
        logging.exception("Unable to write metadata JSON file: %s", e)
        warnings.warn("Unable to write metadata JSON file")

    # Aggregate 4-channel waveform plot removed (not useful). Per-channel
    # plotting is available via `plot_per_channel.py` and spectrograms via
    # `compute_spectrogram.py`.


def get_audio_board_file_helper(fname: str, fs: float, numBits: int) -> Optional[Dict]:
    """Helper to read the body of the audio file and parse audio channels."""
    B_DEBUG_PLOT = False

    headerLength = 512
    fs_ast = 4096
    adc_half_buffer = 5504
    tt_block_length = 128

    # Read raw uint32 data after header
    with open(fname, "rb") as f:
        f.seek(headerLength)
        raw_bytes = f.read()

    if len(raw_bytes) == 0:
        return None

    raw = np.frombuffer(raw_bytes, dtype="<u4").astype(np.uint32)

    # Remove padded blocks
    raw = remove_padded_data_helper(raw)
    if raw.size == 0:
        return None

    # Get sync time values (last 512 bytes as uint32s)
    sync_len = headerLength // 4
    syncTime = raw[-sync_len:]
    raw = raw[:-sync_len]

    startAST = int(syncTime[0])
    stopAST = int(syncTime[1])
    firstBuff = int(syncTime[2])
    secondBuff = int(syncTime[3])

    delay_time = 0.0
    if numBits == 16:
        numSamps = adc_half_buffer // 2
    else:
        numSamps = adc_half_buffer // 4

    buffTime = numSamps / fs
    totTime = buffTime + delay_time

    file_audio: Dict = {}
    file_audio["startTime"] = startAST / fs_ast
    file_audio["stopTime"] = stopAST / fs_ast
    file_audio["firstSampTime"] = firstBuff / fs_ast - totTime
    file_audio["firstSampRaw"] = firstBuff / fs_ast
    file_audio["secondSampRaw"] = secondBuff / fs_ast

    # Organize into packets
    num_uint32_per_write = adc_half_buffer + tt_block_length
    if raw.size % num_uint32_per_write != 0:
        # If not an integer number of packets, truncate the tail
        raw = raw[: (raw.size // num_uint32_per_write) * num_uint32_per_write]

    numPackets = raw.size // num_uint32_per_write
    if numPackets == 0:
        return None

    packets = raw.reshape((num_uint32_per_write, numPackets), order="F")

    raw_data = packets[:adc_half_buffer, :].reshape(-1, order="F")
    time_blocks_packets = packets[adc_half_buffer:, :].reshape(-1, order="F")
    time_blocks = time_blocks_packets[time_blocks_packets != 0]

    time_blocks = unwrap_count_vector_helper(time_blocks)
    tt_blocks = convert_ast_count_to_time_helper(time_blocks, fs_ast)

    # Convert ADC values to decimal channel values
    ch1, ch2, ch3, ch4 = convert_sd_raw_audio_to_decimal_helper(raw_data, numBits)

    ch1_volt = convert_audio_decimal_to_voltage_helper(ch1, numBits)
    ch2_volt = convert_audio_decimal_to_voltage_helper(ch2, numBits)
    ch3_volt = convert_audio_decimal_to_voltage_helper(ch3, numBits)
    ch4_volt = convert_audio_decimal_to_voltage_helper(ch4, numBits)

    # Construct time vector
    tt = np.arange(0, ch1_volt.size) / fs
    tt = tt.astype(np.float64)
    tt_corrected = tt + (file_audio["firstSampTime"] - file_audio["startTime"])

    file_audio["tt_blocks"] = tt_blocks
    file_audio["tt"] = tt_corrected
    file_audio["ch1"] = ch1_volt
    file_audio["ch2"] = ch2_volt
    file_audio["ch3"] = ch3_volt
    file_audio["ch4"] = ch4_volt
    file_audio["startTime"] = file_audio["startTime"]
    file_audio["stopTime"] = file_audio["stopTime"]
    file_audio["firstSampTime"] = file_audio["firstSampTime"]
    file_audio["firstSampRaw"] = file_audio["firstSampRaw"]
    file_audio["secondSampRaw"] = file_audio["secondSampRaw"]

    if B_DEBUG_PLOT:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(np.diff(time_blocks))

    return file_audio


def convert_sd_raw_audio_to_decimal_helper(
    raw: np.ndarray, numBits: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert packed uint32 raw values into four channel integer arrays."""
    # For the common 16-bit case, data is packed as two 16-bit words per uint32
    if numBits == 24:
        # 24-bit path: not commonly used in repository examples. Provide basic support.
        # Treat the 32-bit words, extract lower 24 bits and unpack in order
        # NOTE: this branch may need adjustments depending on actual hardware packing.
        # As a fallback, interpret raw as sequence of 32-bit values where every 4th
        # value corresponds to a channel.
        dec = raw.copy()
        ch1 = dec[0::4]
        ch2 = dec[1::4]
        ch3 = dec[2::4]
        ch4 = dec[3::4]
    else:
        upper = (raw >> 16) & 0xFFFF
        lower = raw & 0xFFFF
        ch1 = upper[0::2].astype(np.uint32)
        ch3 = upper[1::2].astype(np.uint32)
        ch2 = lower[0::2].astype(np.uint32)
        ch4 = lower[1::2].astype(np.uint32)

    return ch1, ch2, ch3, ch4


def convert_audio_decimal_to_voltage_helper(data_dec: np.ndarray, numBitsData: int) -> np.ndarray:
    """Convert integer ADC values (two's complement) to voltages.

    Formula taken from the MATLAB code:
      voltage = (double(data) .* 2 .* 4.5 .* sqrt(2) ./ (2.^numBitsData)) + 1.5;
    """
    if numBitsData == 24:
        # Sign-extend 24-bit values to int32
        data = data_dec.astype(np.int32)
        mask = 1 << 23
        neg = (data & mask) != 0
        data[neg] = data[neg] - (1 << 24)
        data_signed = data.astype(np.int32)
    else:
        # 16-bit case
        data_signed = data_dec.astype(np.int16).astype(np.int32)

    voltage = (data_signed.astype(np.float64) * 2.0 * 4.5 * math.sqrt(2.0) / (2**numBitsData)) + 1.5
    return voltage


def remove_padded_data_helper(raw: np.ndarray) -> np.ndarray:
    """Remove zero-padded blocks at the end of the file."""
    block_length = 512 // 4
    if raw.size % block_length != 0:
        # If not an integer number of blocks, truncate the tail
        raw = raw[: (raw.size // block_length) * block_length]

    if raw.size == 0:
        return raw

    blocks = raw.reshape((-1, block_length)).T

    padded_block = np.zeros((block_length,), dtype=raw.dtype)
    # Find columns that match padded_block
    matches = np.all(blocks == padded_block[:, None], axis=0)
    padded_cols = np.where(matches)[0]

    if padded_cols.size > 0 and padded_cols[-1] == blocks.shape[1] - 1:
        # Only consider contiguous padded blocks at the end
        if padded_cols.size > 1:
            diffs = np.diff(padded_cols)
            large = np.where(diffs > 1)[0]
            if large.size > 0:
                idx = large[-1] + 1
                padded_cols = padded_cols[idx:]

        blocks_rem = np.delete(blocks, padded_cols, axis=1)
        act_data = blocks_rem.T.reshape(-1)
        return act_data
    else:
        return raw


def unwrap_count_vector_helper(count: np.ndarray) -> np.ndarray:
    """Unwrap counter values that roll over at 2^32-1."""
    maxCount = 2**32 - 1
    dx = np.diff(count)
    inds = np.where(dx < -maxCount / 2)[0]
    new_count = count.astype(np.uint64).copy()
    for ii, ind in enumerate(inds):
        if ii < len(inds) - 1:
            next_ind = inds[ii + 1]
            dTick = maxCount - count[ind] + count[ind + 1] + 1
            new_count[ind + 1 : next_ind + 1] = new_count[ind] + dTick + new_count[ind + 1 : next_ind + 1]
        else:
            dTick = maxCount - count[ind] + count[ind + 1] + 1
            new_count[ind + 1 :] = new_count[ind] + dTick + new_count[ind + 1 :]

    return new_count


def convert_ast_count_to_time_helper(count: np.ndarray, fs_ast: float) -> np.ndarray:
    """Convert AST counter ticks to seconds."""
    return count.astype(np.float64) / float(fs_ast)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("fname")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    read_audio_board_file(args.fname, args.out)

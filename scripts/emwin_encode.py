#!/usr/bin/env python3
"""Encode a file (any file: PNG, GIF, TXT, ZIP, anything) into a legacy
GOES EMWIN-N style BPSK audio WAV file.

Usage:
    python3 emwin_encode.py INPUT_FILE OUTPUT.wav [options]

Options:
    --baud {1200,19200}   Bit rate (default: 19200)
    --freq HZ             Audio carrier frequency (default: 1500)
    -q, --quiet           Suppress progress messages

Round-trip with emwin_decode.py is byte-exact.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from emwin_common import (
    SUPPORTED_BAUD_RATES,
    bpsk_modulate,
    bytes_to_bits,
    defaults_for_baud,
    differential_encode,
    make_preamble_bits,
    packetize,
    write_wav,
)


def encode_file(
    input_path: str,
    output_path: str,
    baud: int = 19_200,
    carrier_hz: float | None = None,
    sample_rate: int | None = None,
    quiet: bool = False,
) -> None:
    if baud not in SUPPORTED_BAUD_RATES:
        raise SystemExit(f"--baud must be one of {SUPPORTED_BAUD_RATES}, got {baud}")
    d = defaults_for_baud(baud)
    if carrier_hz is None:
        carrier_hz = d["carrier_hz"]
    if sample_rate is None:
        sample_rate = d["sample_rate"]

    data = Path(input_path).read_bytes()
    filename = os.path.basename(input_path)
    packets = packetize(filename, data)

    if not quiet:
        print(
            f"Encoding {len(data):,} bytes from {filename!r} "
            f"into {len(packets)} packet(s) @ {baud} bps BPSK on "
            f"{carrier_hz:.0f} Hz carrier ({sample_rate} Hz sample rate)..."
        )

    preamble = make_preamble_bits()
    all_bits_chunks = []
    for p in packets:
        # Each packet gets its own preamble+sync so the decoder can re-acquire
        # if it loses lock partway through a long transmission.
        all_bits_chunks.append(preamble)
        all_bits_chunks.append(bytes_to_bits(p.to_bytes()))
    # A short tail of zeros so the matched filter has somewhere to settle.
    all_bits_chunks.append(np.zeros(32, dtype=np.int8))

    raw_bits = np.concatenate(all_bits_chunks)
    diff_bits = differential_encode(raw_bits)
    waveform = bpsk_modulate(diff_bits, baud=baud, carrier_hz=carrier_hz, sample_rate=sample_rate)
    write_wav(output_path, waveform, sample_rate=sample_rate)

    if not quiet:
        duration = len(waveform) / sample_rate
        print(f"Wrote {output_path} ({duration:.2f} s, {len(waveform):,} samples).")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Encode any file as a legacy GOES EMWIN-N style BPSK WAV."
    )
    p.add_argument("input", help="path to the file to encode")
    p.add_argument("output", help="path for the output .wav file")
    p.add_argument(
        "--baud",
        type=int,
        default=19_200,
        choices=SUPPORTED_BAUD_RATES,
        help="bit rate (default: 19200)",
    )
    p.add_argument(
        "--freq",
        type=float,
        default=None,
        help="carrier frequency in Hz (default: 1500 for 1200 bps, 14000 for 19200 bps)",
    )
    p.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="WAV sample rate in Hz (default: 44100 for 1200 bps, 96000 for 19200 bps)",
    )
    p.add_argument("-q", "--quiet", action="store_true", help="suppress progress messages")
    args = p.parse_args(argv)
    encode_file(
        args.input,
        args.output,
        baud=args.baud,
        carrier_hz=args.freq,
        sample_rate=args.sample_rate,
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

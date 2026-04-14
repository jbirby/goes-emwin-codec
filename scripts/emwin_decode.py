#!/usr/bin/env python3
"""Decode a legacy GOES EMWIN-N style BPSK audio WAV file back into the
original file.

Usage:
    python3 emwin_decode.py INPUT.wav [OUTPUT_FILE] [options]

Options:
    --baud {1200,19200}   Bit rate used during encoding (default: 19200)
    --freq HZ             Audio carrier frequency (default: 1500)
    --out-dir DIR         Directory to write the recovered file into
                          (uses the filename embedded in the EMWIN headers)
    -q, --quiet           Suppress progress messages
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import numpy as np

from emwin_common import (
    PACKET_SYNC,
    PACKET_TOTAL_BYTES,
    SUPPORTED_BAUD_RATES,
    Packet,
    bits_to_bytes,
    bpsk_demodulate,
    bytes_to_bits,
    defaults_for_baud,
    differential_decode,
    find_sync,
    read_wav,
    reassemble,
)


def _decode_bitstream(bits: np.ndarray, quiet: bool) -> List[Packet]:
    """Walk the bit stream looking for sync words and parsing packets behind each."""
    sync_bits = bytes_to_bits(PACKET_SYNC)
    packet_bit_len = PACKET_TOTAL_BYTES * 8
    packets: List[Packet] = []
    cursor = 0
    while cursor < len(bits):
        idx = find_sync(bits, sync_bits, start=cursor)
        if idx < 0:
            break
        end = idx + packet_bit_len
        if end > len(bits):
            break
        packet_bits = bits[idx:end]
        packet_bytes = bits_to_bytes(packet_bits)
        try:
            pkt = Packet.from_bytes(packet_bytes)
            packets.append(pkt)
            if not quiet:
                print(
                    f"  recovered packet {pkt.part}/{pkt.total} of {pkt.filename!r} "
                    f"({pkt.payload_len} bytes)"
                )
            cursor = end
        except ValueError as exc:
            if not quiet:
                print(f"  sync hit at bit {idx} but packet rejected: {exc}")
            cursor = idx + 1   # skip past this false sync and keep looking
    return packets


def decode_wav(
    input_path: str,
    output_path: str | None = None,
    out_dir: str | None = None,
    baud: int = 19_200,
    carrier_hz: float | None = None,
    quiet: bool = False,
) -> str:
    if baud not in SUPPORTED_BAUD_RATES:
        raise SystemExit(f"--baud must be one of {SUPPORTED_BAUD_RATES}, got {baud}")
    if carrier_hz is None:
        carrier_hz = defaults_for_baud(baud)["carrier_hz"]

    if not quiet:
        print(f"Reading {input_path}...")
    waveform, sr = read_wav(input_path)
    if not quiet:
        print(
            f"  {len(waveform):,} samples @ {sr} Hz, "
            f"{len(waveform) / sr:.2f} s, demodulating @ {baud} bps on {carrier_hz:.0f} Hz..."
        )
    raw_bits = bpsk_demodulate(waveform, baud=baud, carrier_hz=carrier_hz, sample_rate=sr)
    decoded_bits = differential_decode(raw_bits)

    packets = _decode_bitstream(decoded_bits, quiet=quiet)
    if not packets:
        # Try the inverted polarity — BPSK has a 180-degree phase ambiguity
        # that the differential decoder normally resolves, but if we picked
        # the wrong arm at sample-time the bits come out flipped.
        if not quiet:
            print("  no packets found, trying inverted polarity...")
        decoded_bits_inv = differential_decode(1 - raw_bits)
        packets = _decode_bitstream(decoded_bits_inv, quiet=quiet)

    if not packets:
        raise SystemExit("No EMWIN packets recovered from the WAV file.")

    filename, data = reassemble(packets)

    if output_path is None:
        directory = out_dir if out_dir else "."
        os.makedirs(directory, exist_ok=True)
        # Sanitize the embedded filename — it came from an untrusted source.
        safe_name = os.path.basename(filename) or "decoded.bin"
        output_path = os.path.join(directory, safe_name)

    Path(output_path).write_bytes(data)
    if not quiet:
        print(f"Wrote {output_path} ({len(data):,} bytes, original name: {filename!r}).")
    return output_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Decode a legacy GOES EMWIN-N style BPSK WAV back into the original file."
    )
    p.add_argument("input", help="path to the input .wav file")
    p.add_argument(
        "output",
        nargs="?",
        default=None,
        help="optional output file path (otherwise the filename embedded in the WAV is used)",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="if no explicit OUTPUT given, write the recovered file into this directory",
    )
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
    p.add_argument("-q", "--quiet", action="store_true", help="suppress progress messages")
    args = p.parse_args(argv)
    decode_wav(
        args.input,
        output_path=args.output,
        out_dir=args.out_dir,
        baud=args.baud,
        carrier_hz=args.freq,
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

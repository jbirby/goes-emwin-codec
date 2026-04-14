#!/usr/bin/env python3
"""Round-trip test: encode a file, decode it, verify byte-for-byte match."""
import os
import sys
import tempfile
import hashlib
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from emwin_encode import encode_file
from emwin_decode import decode_wav


def _hash(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def run_case(name: str, payload: bytes, baud: int, carrier_hz: float = 1500.0) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, name)
        wav_path = os.path.join(tmp, "out.wav")
        out_dir = os.path.join(tmp, "decoded")
        Path(in_path).write_bytes(payload)
        encode_file(in_path, wav_path, baud=baud, carrier_hz=carrier_hz, quiet=True)
        recovered_path = decode_wav(wav_path, out_dir=out_dir, baud=baud, carrier_hz=carrier_hz, quiet=True)
        recovered = Path(recovered_path).read_bytes()
        assert recovered == payload, (
            f"[{name} @ {baud}bps] mismatch: "
            f"orig sha256={_hash(payload)[:16]}, got sha256={_hash(recovered)[:16]}, "
            f"orig len={len(payload)}, got len={len(recovered)}"
        )
        wav_size = os.path.getsize(wav_path)
        print(
            f"  PASS  {name:<24} baud={baud:>5}  payload={len(payload):>6}B  "
            f"wav={wav_size:>9,}B  sha={_hash(payload)[:12]}"
        )


def main() -> int:
    print("Round-trip tests")
    # Small text payload.
    run_case("hello.txt", b"Hello, EMWIN! This is a test.\n", baud=19_200)
    run_case("hello.txt", b"Hello, EMWIN! This is a test.\n", baud=1_200)

    # Empty file.
    run_case("empty.bin", b"", baud=19_200)

    # Multi-packet payload (forces packet boundaries).
    payload = bytes((i * 31 + 7) & 0xFF for i in range(5_000))
    run_case("medium.bin", payload, baud=19_200)

    # Larger payload at slow baud (still finishes quickly because everything is in-memory).
    payload2 = bytes((i ^ (i >> 3)) & 0xFF for i in range(2_500))
    run_case("multi.bin", payload2, baud=1_200)

    # NWS-style filename and a fake bulletin.
    bulletin = (
        b"WFUS54 KFWD 141822\n"
        b"TORFWD\n"
        b"BULLETIN - IMMEDIATE BROADCAST REQUESTED\n"
        b"Tornado Warning\n"
        b"National Weather Service Fort Worth TX\n"
        b"122 PM CDT TUE APR 14 2026\n"
    ) * 10
    run_case("WFUS54KFWD.TXT", bulletin, baud=19_200)
    print("All round-trip tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

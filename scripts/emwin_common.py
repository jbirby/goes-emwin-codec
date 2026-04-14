"""
Common library for the legacy GOES EMWIN-N (1692.7 MHz) BPSK codec.

This implements a faithful-enough version of the EMWIN-N over-the-air format:
- File framing into fixed-size packets with ASCII header, payload, CRC trailer
- BPSK modulation at either 1200 bps (original) or 19200 bps (later upgrade)
- Differential (NRZ-M) bit encoding so the receiver doesn't need an absolute
  phase reference
- Raised-cosine pulse shaping to keep the spectrum tight
- Sync preamble (alternating bit pattern) for symbol-timing recovery

The goal is round-trip fidelity: an arbitrary binary file goes in, a WAV
comes out; that WAV plays back through the decoder and yields the original
file byte-for-byte. The framing follows the spirit of the published EMWIN
"Block Format" (filename / part N of M / checksum / timestamp) without
claiming bit-exact compatibility with NWS receivers from the GOES-N era,
which are no longer in service.
"""
from __future__ import annotations

import math
import struct
import wave
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_BAUD_RATES = (1_200, 19_200)

# At 19200 bps the data bandwidth is wider than what fits comfortably in
# the lower audio band, so we bump the sample rate and the carrier to give
# the signal room to breathe. Real EMWIN-N rides on a 1692.7 MHz RF
# carrier; we're simulating it in audio purely so the resulting WAV file
# can be played, transmitted over a cable, or analyzed on a laptop.
BAUD_DEFAULTS = {
    1_200:  {"sample_rate": 44_100, "carrier_hz": 1_500.0},
    19_200: {"sample_rate": 96_000, "carrier_hz": 14_000.0},
}

# For backward compatibility / convenience.
SAMPLE_RATE = 44_100
DEFAULT_CARRIER_HZ = 1_500.0


def defaults_for_baud(baud: int) -> dict:
    if baud not in BAUD_DEFAULTS:
        raise ValueError(f"baud {baud} not in {SUPPORTED_BAUD_RATES}")
    return dict(BAUD_DEFAULTS[baud])

PACKET_PAYLOAD_BYTES = 1024       # data bytes per on-air packet
HEADER_BYTES = 80                 # fixed-width ASCII header
TRAILER_BYTES = 4                 # CRC-32 over header + payload
PACKET_TOTAL_BYTES = HEADER_BYTES + PACKET_PAYLOAD_BYTES + TRAILER_BYTES

PACKET_SYNC = b"\xAA\xAA\xAA\xAA\x2D\xD4"   # preamble + frame sync word
PREAMBLE_BITS = 64                 # alternating bits before each packet
ROLLOFF = 0.35                     # raised-cosine roll-off factor


# ---------------------------------------------------------------------------
# CRC-32 (IEEE 802.3 polynomial, the same one zlib uses)
# ---------------------------------------------------------------------------

def crc32(data: bytes) -> int:
    import zlib
    return zlib.crc32(data) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Packet framing
# ---------------------------------------------------------------------------

@dataclass
class Packet:
    filename: str        # original filename (max 12 chars on the wire, but we allow up to 32)
    part: int            # 1-based part number
    total: int           # total parts in the file
    payload: bytes       # up to PACKET_PAYLOAD_BYTES bytes (zero-padded if short)
    payload_len: int     # actual payload byte count (so we can trim padding on the last packet)

    def to_bytes(self) -> bytes:
        # Header layout (fixed 80 bytes, ASCII, space-padded):
        #   /PF<filename:32>/PN<part:04>/PT<total:04>/LN<len:04>/
        # then padded with spaces out to HEADER_BYTES.
        header = (
            f"/PF{self.filename:<32.32}"
            f"/PN{self.part:04d}"
            f"/PT{self.total:04d}"
            f"/LN{self.payload_len:04d}/"
        ).encode("ascii")
        header = header.ljust(HEADER_BYTES, b" ")
        if len(header) != HEADER_BYTES:
            raise ValueError(f"header length {len(header)} != {HEADER_BYTES}")
        payload = self.payload.ljust(PACKET_PAYLOAD_BYTES, b"\x00")
        body = header + payload
        trailer = struct.pack(">I", crc32(body))
        return body + trailer

    @classmethod
    def from_bytes(cls, raw: bytes) -> "Packet":
        if len(raw) != PACKET_TOTAL_BYTES:
            raise ValueError(f"packet length {len(raw)} != {PACKET_TOTAL_BYTES}")
        body = raw[: HEADER_BYTES + PACKET_PAYLOAD_BYTES]
        trailer = raw[HEADER_BYTES + PACKET_PAYLOAD_BYTES :]
        expected_crc = struct.unpack(">I", trailer)[0]
        actual_crc = crc32(body)
        if expected_crc != actual_crc:
            raise ValueError(
                f"CRC mismatch: header says {expected_crc:08x}, computed {actual_crc:08x}"
            )
        header = body[:HEADER_BYTES].decode("ascii", errors="replace")
        payload = body[HEADER_BYTES:]
        # Parse the simple key/value header.
        fields = {}
        for chunk in header.strip().split("/"):
            if not chunk:
                continue
            if len(chunk) < 2:
                continue
            key, value = chunk[:2], chunk[2:]
            fields[key] = value.strip()
        try:
            filename = fields.get("PF", "").strip()
            part = int(fields["PN"])
            total = int(fields["PT"])
            payload_len = int(fields["LN"])
        except KeyError as exc:
            raise ValueError(f"missing required header field {exc}") from None
        if not (0 <= payload_len <= PACKET_PAYLOAD_BYTES):
            raise ValueError(f"payload_len out of range: {payload_len}")
        return cls(
            filename=filename,
            part=part,
            total=total,
            payload=payload[:payload_len],
            payload_len=payload_len,
        )


def packetize(filename: str, data: bytes) -> List[Packet]:
    """Split a file's bytes into a list of Packet objects."""
    if len(data) == 0:
        # Always emit at least one packet so the decoder has something to anchor on.
        return [Packet(filename=filename, part=1, total=1, payload=b"", payload_len=0)]
    total = math.ceil(len(data) / PACKET_PAYLOAD_BYTES)
    packets: List[Packet] = []
    for i in range(total):
        chunk = data[i * PACKET_PAYLOAD_BYTES : (i + 1) * PACKET_PAYLOAD_BYTES]
        packets.append(
            Packet(
                filename=filename,
                part=i + 1,
                total=total,
                payload=chunk,
                payload_len=len(chunk),
            )
        )
    return packets


def reassemble(packets: Iterable[Packet]) -> Tuple[str, bytes]:
    """Stitch decoded packets back into (filename, bytes)."""
    by_part = {p.part: p for p in packets}
    if not by_part:
        raise ValueError("no packets to reassemble")
    first = next(iter(by_part.values()))
    total = first.total
    if any(p.total != total for p in by_part.values()):
        raise ValueError("inconsistent total-part count across packets")
    missing = [i for i in range(1, total + 1) if i not in by_part]
    if missing:
        raise ValueError(f"missing packet parts: {missing}")
    filename = first.filename
    out = bytearray()
    for i in range(1, total + 1):
        out.extend(by_part[i].payload)
    return filename, bytes(out)


# ---------------------------------------------------------------------------
# Bit / byte helpers
# ---------------------------------------------------------------------------

def bytes_to_bits(data: bytes) -> np.ndarray:
    """MSB-first bit unpacking, returns a uint8 array of 0/1."""
    arr = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(arr)
    return bits.astype(np.int8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Pack MSB-first 0/1 bits into bytes. Truncates to a multiple of 8."""
    n = (len(bits) // 8) * 8
    if n == 0:
        return b""
    packed = np.packbits(bits[:n].astype(np.uint8))
    return packed.tobytes()


def differential_encode(bits: np.ndarray) -> np.ndarray:
    """NRZ-M: a '1' inverts the previous output, a '0' keeps it.

    Differential encoding lets the receiver recover bits without knowing
    the absolute carrier phase — only relative phase changes matter.
    """
    out = np.empty_like(bits)
    state = 0
    for i, b in enumerate(bits):
        if b == 1:
            state ^= 1
        out[i] = state
    return out


def differential_decode(bits: np.ndarray) -> np.ndarray:
    """Inverse of differential_encode."""
    out = np.empty_like(bits)
    prev = 0
    for i, b in enumerate(bits):
        out[i] = b ^ prev
        prev = b
    return out


# ---------------------------------------------------------------------------
# Pulse shaping
# ---------------------------------------------------------------------------

def raised_cosine_filter(samples_per_symbol: int, rolloff: float = ROLLOFF, span: int = 6) -> np.ndarray:
    """Raised-cosine pulse, normalized so passing impulses through it preserves amplitude."""
    n = np.arange(-span * samples_per_symbol, span * samples_per_symbol + 1)
    t = n / samples_per_symbol
    pi = np.pi
    with np.errstate(divide="ignore", invalid="ignore"):
        # Standard raised-cosine impulse response.
        sinc = np.sinc(t)
        denom = 1.0 - (2.0 * rolloff * t) ** 2
        cosine = np.where(np.abs(denom) < 1e-9, pi / 4.0, np.cos(pi * rolloff * t) / denom)
        h = sinc * cosine
    h = h / np.sum(h)
    return h


# ---------------------------------------------------------------------------
# BPSK modulator
# ---------------------------------------------------------------------------

def bpsk_modulate(
    bits: np.ndarray,
    baud: int,
    carrier_hz: float = DEFAULT_CARRIER_HZ,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """BPSK modulate a bit stream into a real audio waveform.

    Each bit becomes a symbol of amplitude +1 (bit=1) or -1 (bit=0).
    Symbols are upsampled, pulse-shaped with a raised cosine, and
    multiplied by a sine carrier.
    """
    if baud not in SUPPORTED_BAUD_RATES:
        raise ValueError(f"baud {baud} not in {SUPPORTED_BAUD_RATES}")
    samples_per_symbol = sample_rate / baud
    # We want an integer number of samples per symbol for clean filtering.
    # Pick the nearest integer; the small drift over a packet is well below
    # what the timing-recovery margin of BPSK can absorb.
    sps = max(2, int(round(samples_per_symbol)))
    symbols = (bits.astype(np.float32) * 2.0) - 1.0  # 0 -> -1, 1 -> +1
    # Upsample by inserting zeros between symbols.
    upsampled = np.zeros(len(symbols) * sps, dtype=np.float32)
    upsampled[::sps] = symbols
    # Pulse shape.
    h = raised_cosine_filter(sps, rolloff=ROLLOFF, span=6).astype(np.float32)
    shaped = np.convolve(upsampled, h, mode="same")
    # Mix to carrier.
    t = np.arange(len(shaped)) / sample_rate
    carrier = np.sin(2.0 * np.pi * carrier_hz * t).astype(np.float32)
    waveform = shaped * carrier
    # Normalize to ~0.8 full-scale to leave headroom.
    peak = float(np.max(np.abs(waveform)))
    if peak > 0:
        waveform = waveform * (0.8 / peak)
    return waveform


def make_preamble_bits() -> np.ndarray:
    """Alternating 1010… preamble plus a recognizable sync pattern."""
    pre = np.tile(np.array([1, 0], dtype=np.int8), PREAMBLE_BITS // 2)
    sync = bytes_to_bits(PACKET_SYNC)
    return np.concatenate([pre, sync])


# ---------------------------------------------------------------------------
# BPSK demodulator
# ---------------------------------------------------------------------------

def bpsk_demodulate(
    waveform: np.ndarray,
    baud: int,
    carrier_hz: float = DEFAULT_CARRIER_HZ,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Coherent BPSK demodulation that produces a stream of 0/1 bits.

    Pipeline:
      1. Mix to baseband with both I (sin) and Q (cos) carriers so we don't
         have to know the absolute phase. The encoder uses sin, so in a
         noiseless round-trip the data lands cleanly on the I arm — but
         being robust to a 90-degree offset costs us nothing.
      2. Low-pass filter each arm with a boxcar of length `sps`. This both
         kills the 2*fc image left over from the mix-down AND acts as a
         (rectangular) matched filter for our shaped symbols. For a
         noiseless channel this is more than enough; the raised cosine at
         the transmitter only matters for spectral compactness, not for
         recoverability.
      3. Sample at symbol centers, picking the timing offset and the I/Q
         arm that maximize the eye opening. Slice on sign to get bits.
         A 180-degree phase ambiguity is left for the differential
         decoder upstream to resolve.
    """
    if baud not in SUPPORTED_BAUD_RATES:
        raise ValueError(f"baud {baud} not in {SUPPORTED_BAUD_RATES}")
    sps = max(2, int(round(sample_rate / baud)))
    t = np.arange(len(waveform)) / sample_rate
    i_mix = waveform * np.sin(2.0 * np.pi * carrier_hz * t)
    q_mix = waveform * np.cos(2.0 * np.pi * carrier_hz * t)
    # Boxcar of length `sps` — integrate-and-dump matched filter, also
    # kills the 2*fc image.
    boxcar = np.ones(sps, dtype=np.float32) / float(sps)
    i_lp = np.convolve(i_mix, boxcar, mode="same")
    q_lp = np.convolve(q_mix, boxcar, mode="same")
    # Pick the better arm + best symbol-timing offset by maximizing the sum
    # of squared sampled values (|eye opening|^2). Squared so polarity
    # doesn't matter.
    best = None  # (score, arm_idx, offset)
    for arm_idx, arm in enumerate((i_lp, q_lp)):
        for off in range(sps):
            samples = arm[off::sps]
            if len(samples) == 0:
                continue
            score = float(np.sum(samples * samples))
            if best is None or score > best[0]:
                best = (score, arm_idx, off)
    if best is None:
        return np.zeros(0, dtype=np.int8)
    _, arm_idx, off = best
    samples = (i_lp if arm_idx == 0 else q_lp)[off::sps]
    bits = (samples > 0).astype(np.int8)
    return bits


# ---------------------------------------------------------------------------
# Frame finder: locate the sync pattern in a noisy bit stream
# ---------------------------------------------------------------------------

def find_sync(bits: np.ndarray, sync_bits: np.ndarray, start: int = 0) -> int:
    """Return the index in `bits` immediately after the first occurrence of
    `sync_bits` at or after `start`. Returns -1 if not found.

    Tolerates up to 2 bit errors in the 48-bit sync word.
    """
    n = len(sync_bits)
    if len(bits) < n:
        return -1
    sync = sync_bits.astype(np.int8)
    threshold = 2  # max Hamming distance
    for i in range(start, len(bits) - n + 1):
        diff = int(np.sum(bits[i : i + n] ^ sync))
        if diff <= threshold:
            return i + n
    return -1


# ---------------------------------------------------------------------------
# WAV I/O
# ---------------------------------------------------------------------------

def write_wav(path: str, waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    samples = np.clip(waveform, -1.0, 1.0)
    pcm = (samples * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def read_wav(path: str) -> Tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        sr = w.getframerate()
        sw = w.getsampwidth()
        ch = w.getnchannels()
        raw = w.readframes(n)
    if sw == 2:
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 1:
        pcm = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported sample width: {sw} bytes")
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)
    return pcm, sr

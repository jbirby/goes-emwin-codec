# goes-emwin-codec

A pure-Python encoder/decoder for the legacy **GOES EMWIN-N** (1692.7 MHz)
direct-broadcast format, packaged as a Claude Skill.

EMWIN-N was the NWS's pre-GOES-R direct-to-receiver weather data downlink:
text bulletins, warnings, forecasts, and imagery pushed from the GOES-N/O/P
satellites at 1200 bps (and later 19200 bps) BPSK. The satellites are
retired; this codec is a faithful-in-spirit recreation, useful for
preservation, education, and the more general "I just want to ship a file
as audio" use case.

## What it does

- **Encode any file** — PNG, GIF, TXT, ZIP, .docx, anything — into a WAV
  file modulated as BPSK at 1200 or 19200 bps.
- **Decode a WAV** back into the original file, byte-for-byte. Filename
  and part numbering are carried in the EMWIN packet headers.
- **One-shot** — no flags required for the common case. Just `encode in.png out.wav`
  and `decode out.wav`.

## Install (standalone)

```bash
pip install numpy
python3 scripts/emwin_encode.py somefile.png signal.wav
python3 scripts/emwin_decode.py signal.wav
```

## Install (as a Claude Skill)

Drop the `goes-emwin.skill` zip into your Claude Skills directory or use
the "Save skill" install button when Claude offers it. Then just ask:

> Encode this PNG as an EMWIN signal at 1200 bps.
> Decode this WAV — it's an EMWIN broadcast.

## Round-trip self-test

```bash
python3 scripts/test_roundtrip.py
```

Verifies byte-exact recovery for several payloads at both bit rates.

## License

MIT.

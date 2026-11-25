# AI Movie Recast Studio (V1)

Replace manually selected faces in authorized video clips with consenting people's faces. Everything runs locally on your Android phone using Termux. GitHub only stores the source code.

## Responsible use
- Use only footage you have the right to edit.
- Use only face photos of people who have consented.
- Exported videos must be labelled as AI-generated.
- No celebrity impersonation or deceptive use.

## Status
V1 is under construction, built file by file. Target: clips of 1-3 minutes, processed slowly in resumable batches. Speed and quality depend on your phone. Cinema-grade output is not claimed.

## Planned files
- `README.md`
- `requirements.txt`
- `server.py` (local web server)
- `detect_track.py` (face detection and tracking)
- `mapping.py` (character to reference photo mapping)
- `swap.py` (face swap, model plug-in)
- `export.py` (FFmpeg re-encode and audio merge)
- `index.html`, `style.css`, `app.js` (interface)
- `LICENSES.md`

## Setup (Android)
Instructions will be added after the code files are done.

## Privacy
Videos and photos never leave your phone. No cloud services are used.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Raspberry Pi-based access control system combining RFID (MFRC522), face recognition (MobileFaceNet + BlazeFace), and liveness detection. The system controls a door servo and logs all access attempts to SQLite with AES-128-GCM encrypted face embeddings.

**Hardware**: Raspberry Pi, MFRC522 RFID (SPI), USB Camera, Servo (GPIO18/PWM0), Optional LCD (I2C).

## Commands

```bash
# Activate virtual environment
source env/Scripts/activate  # Windows
# or
source env/bin/activate      # Linux

# Run main CLI application
python main.py

# Run AES-GCM tests (Wycheproof vectors)
python test_aes_gcm.py
```

No build system, linter, or test framework beyond the standalone test file. The web interface starts automatically when running `main.py`.

## Architecture

### Face Recognition Pipeline
Frame → Resize to 128px → **BlazeFace** (PyTorch, detection) → Crop face → Resize to 112px → **MobileFaceNet** (TFLite, embedding) → 512-d vector → Cosine similarity against stored embeddings.

Fallback mode uses OpenCV LBPH histogram if MobileFaceNet TFLite fails to load.

### Core Modules (`core/`)
- **database.py** - SQLite (WAL mode) with `users` and `access_logs` tables. Face embeddings stored as AES-GCM encrypted base64 blobs. Runs AES-GCM self-test with Wycheproof vectors on init.
- **face_engine.py** - `FaceEngine` class wrapping BlazeFace (detection) and MobileFaceNet (recognition). Loads models from `BlazeFace-PyTorch/` and `MobileFaceNet_TF/`.
- **liveness.py** - `LivenessDetector` with 3 voting methods: LBP texture analysis, Farneback optical flow, Haar cascade eye blink detection.
- **crypto.py** - AES-128-GCM encrypt/decrypt for face embeddings. Key from `config.AES_KEY_HEX`. Format: `[12-byte IV | ciphertext | 16-byte tag]`.
- **rfid_reader.py** - MFRC522 SPI interface (`spidev`).
- **servo.py** - Door controller using RPi.GPIO PWM on GPIO18.
- **camera_stream.py** - Threaded USB camera capture.

### Menu System (`menus/`)
CLI menus imported by `main.py`:
- **access.py** - Continuous/kontinu mode, single access, liveness toggle, liveness test
- **enrollment.py** - New user registration (RFID + face), face data update
- **admin.py** - User management, access logs, statistics, configuration

### Web Interface (`web/`)
Flask app (`app.py`) running in a daemon thread. Provides:
- `/stream` - MJPEG camera stream
- `/` or `/display` - HDMI/browser status display (embedded HTML)
- `/api/state` - JSON state (step, user, similarity)
- `/api/logs` - Recent access logs
- `/api/stats` - System statistics

State updates flow from `menus/*.py` → `web/app.py::update_state()` → browser via polling.

## Configuration

All hardware pins, thresholds, and paths in `config.py`:
- `AES_KEY_HEX` - 128-bit key for embedding encryption (generate with `secrets.token_hex(16)`)
- `FACE_MATCH_THRESH` - Cosine similarity threshold (default 0.55)
- `LIVENESS_ENABLED`, `LIVENESS_MIN_SCORE`, `LIVENESS_MIN_VOTES`
- `SERVO_PIN`, `SERVO_OPEN`, `SERVO_CLOSED`, `DOOR_OPEN_SEC`
- `CAMERA_INDEX`, `CAMERA_WIDTH`, `CAMERA_HEIGHT`

## Key Dependencies
- `tflite-runtime` or `tensorflow` (MobileFaceNet)
- `torch` + BlazeFace model (face detection)
- `opencv-python` (image processing, liveness)
- `cryptography` (AES-GCM)
- `Flask` (web interface)

## Claude Code Guidelines (Token & Workflow Optimization)

To simplify workflows and minimize token usage, adhere to the following rules when working in this repository:

1. **Be Concise**: Keep explanations to an absolute minimum. Provide only the necessary code changes, diffs, or direct answers. Skip pleasantries.
2. **Search Before Reading**: Use `grep` or file search tools to find specific functions, classes, or variables instead of viewing entire files.
   - Example: `grep -rn "class FaceEngine" core/`
3. **Targeted Edits**: Only replace the specific lines of code that need modification. Do not output or rewrite entire files unless completely necessary.
4. **Read Line Ranges**: When viewing files, always restrict your view to the relevant line ranges.
5. **Self-Verification**: If modifying cryptography code, autonomously run `python test_aes_gcm.py` to verify the changes before completing the task.
6. **Assume Raspberry Pi**: Code runs on resource-constrained hardware. Avoid adding heavy dependencies or memory-intensive operations.

## Global Efficiency Rules

- Always prioritize minimal context
- Never scan entire project unless explicitly asked
- Prefer summaries over full file reads
- Limit output verbosity
- Avoid duplicate analysis across agents

## Ignore Rules
Do not load or analyze:
- *.db
- *.onnx
- *.pt
- *.h5
- MobileFaceNet_TF/*
- large binary files

When analyzing dependencies:
- Prefer summaries over full file reads
- Only expand full file if necessary

## Context Rules

- Only load files that are directly related to the task
- Avoid scanning entire project unless explicitly requested
- Prioritize:
  - changed files
  - directly imported modules

### Quick Workflow Commands

- **Check Database Schema**: `grep -rn "CREATE TABLE" core/database.py`
- **Check Hardware Pins**: `cat config.py` (File is small enough for a full read)
- **Test Crypto Logic**: `python test_aes_gcm.py`
- **Run System**: `python main.py`

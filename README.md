<div align="center">

# 📋 Rangkumin

### AI-Powered Meeting Transcription & Summarization Tool

*Offline-first, privacy-focused audio transcription and intelligent summarization — optimized for Bahasa Indonesia*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Ollama](https://img.shields.io/badge/Ollama-0.17+-000000?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com)
[![Whisper](https://img.shields.io/badge/Whisper-large--v3--turbo-412991?style=for-the-badge&logo=openai&logoColor=white)](https://github.com/SYSTRAN/faster-whisper)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

**3-minute audio → Full transcript + Summary in ~10 minutes**
<br/>
*100% offline · No API costs · Your data stays on your machine*

<br/>

[Getting Started](#-getting-started) · [Features](#-features) · [Architecture](#-architecture) · [Benchmarks](#-benchmarks) · [Hardware](#-hardware-setup)

</div>

---

## 📌 Overview

**Rangkumin** (Indonesian slang for *"summarize it!"*) is an end-to-end, offline meeting transcription and summarization pipeline built for professionals who need accurate notetaking — especially in **Bahasa Indonesia** with English code-switching.

In many Indonesian workplaces, meetings are conducted in a mix of Bahasa Indonesia and English. Existing transcription tools either don't support Indonesian well, or require expensive cloud APIs. **Rangkumin** solves this by combining state-of-the-art open-source models running entirely on local hardware.

### The Problem

- Manual meeting notetaking is time-consuming and error-prone
- Cloud-based transcription services raise privacy concerns for sensitive corporate meetings
- Most ASR tools have poor accuracy for Bahasa Indonesia
- Meetings often mix Indonesian and English (*code-switching*), which breaks most single-language tools

### The Solution

Rangkumin provides a **one-click pipeline** that:

1. **Transcribes** audio using OpenAI's Whisper (via `faster-whisper`) — optimized for Indonesian
2. **Corrects** transcription errors using a local LLM (Qwen3 via Ollama)
3. **Summarizes** the entire meeting in structured Bahasa Indonesia

All processing happens **100% offline** on a single mini PC — no cloud, no API keys, no recurring costs.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎧 **Accurate Transcription** | Whisper `large-v3-turbo` with `int8` quantization — 8× faster than `large-v3` with comparable accuracy |
| 🧠 **Intelligent Correction** | Qwen3 LLM fixes mishearing, adds punctuation, and preserves code-switched English terms |
| 📋 **Structured Summaries** | Auto-generates summaries with Key Points, Decisions, and Action Items — all in Bahasa Indonesia |
| 🔒 **100% Offline** | No internet required after setup. Sensitive meeting data never leaves your machine |
| 🌐 **Bilingual Support** | Handles Indonesian–English code-switching seamlessly |
| ⚡ **One-Click Pipeline** | Upload audio → Raw transcript → Corrected transcript → Summary. Fully automated |
| 📥 **Export Ready** | Download raw transcript, corrected transcript, and summary as `.txt` files |
| 🖥️ **Web UI** | Clean Streamlit interface accessible via browser |

---

## 🏗️ Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Audio File  │────▶│  faster-whisper   │────▶│   Ollama (Qwen3.5) │────▶│   Ollama     │
│  .mp3/.wav   │     │  large-v3-turbo   │     │   Fix Transcript │     │   Summarize  │
│  .m4a/.ogg   │     │  (int8, beam=1)   │     │   (think=True)  │     │   (Bahasa ID)│
└──────────────┘     └──────────────────┘     └──────────────────┘     └──────────────┘
                              │                         │                       │
                              ▼                         ▼                       ▼
                     📄 Raw Transcript         📝 Corrected Text         📋 Summary
                     (displayed immediately)   (displayed immediately)   (displayed immediately)
```

### Tech Stack

| Component | Technology | Role |
|---|---|---|
| **Speech-to-Text** | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) `large-v3-turbo` | Audio → raw transcript |
| **LLM Engine** | [Ollama](https://ollama.com) + [qwen3.5:9b-q8_0](https://ollama.com/library/qwen3.5:9b-q8_0) | Transcript correction + summarization |
| **Frontend** | [Streamlit](https://streamlit.io) | Web-based user interface |
| **Audio Processing** | [pydub](https://github.com/jiaaro/pydub) + [FFmpeg](https://ffmpeg.org) | Audio format conversion |

### Why These Models?

- **Whisper `large-v3-turbo`** — A distilled version of `large-v3` with only 4 decoder layers (vs 32). Achieves **8× faster inference** with accuracy comparable to `large-v2`. Research from Institut Teknologi Sumatera (Feb 2026) found it to be the **most accurate Whisper variant for Indonesian conversational speech** at 7.97% WER.

---

## 📊 Benchmarks

Tested on **GEEKOM A6 Mini PC** (specs below):

| Audio Duration | Words Detected | Whisper Time | LLM Fix Time | Summary Time | **Total Time** |
|---|---|---|---|---|---|
| 3.2 minutes | 229 words | ~2 min | ~5 min | ~5 min | **~12 min** |

### Performance Characteristics

| Metric | Value |
|---|---|
| CPU utilization | 71–81% |
| RAM usage | ~16 GB total |
---

## 💻 Hardware Setup

This project was developed and tested on a **GEEKOM A6 Mini PC** — a compact, fanless-capable mini desktop that fits on any desk.

### GEEKOM A6 Specifications

| Component | Specification |
|---|---|
| **CPU** | AMD Ryzen 7 6800H (8 cores / 16 threads, 3.2–4.7 GHz, Zen 3+) |
| **RAM** | 64 GB DDR5-4800 Dual Channel |
| **Storage** | NVMe SSD |
| **GPU** | AMD Radeon 680M (integrated — not used for inference) |
| **OS** | Windows 11 |
| **Form Factor** | Ultra-compact mini PC |

---

## 🚀 Getting Started

### Prerequisites

- **Python** 3.10 or higher
- **FFmpeg** (required by pydub for audio conversion)
- **Ollama** (local LLM runtime)
- **~20 GB disk space** (for models)

### Step 1: Install Python

Download and install Python 3.10+ from [python.org](https://www.python.org/downloads/).

Verify installation:

```bash
python --version
# Python 3.10.x or higher
```

### Step 2: Install FFmpeg

**Windows (via winget):**
```bash
winget install FFmpeg
```

**Windows (via Chocolatey):**
```bash
choco install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update && sudo apt install ffmpeg
```

Verify:
```bash
ffmpeg -version
```

### Step 3: Install Ollama

Download from [ollama.com/download](https://ollama.com/download) and install.

Verify Ollama is running:
```bash
ollama --version
```

### Step 4: Pull the LLM Model

```bash
# Pull qwen3.5:9b-q8_0
ollama pull qwen3.5:9b-q8_0
```
### Step 5: Clone the Repository

```bash
git clone https://github.com/yourusername/rangkumin.git
cd rangkumin
```

### Step 6: Install Python Dependencies

```bash
pip install streamlit faster-whisper ollama pydub
```

### Step 7: Run Rangkumin

```bash
streamlit run audio.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📖 Usage

1. **Upload** an audio file (MP3, WAV, OGG, M4A, FLAC, WMA, AAC)
2. **Click** "▶️ Mulai Transkripsi"
3. **Watch** the pipeline execute automatically:
   - 🎧 **Step 1:** Whisper transcribes → raw transcript displayed immediately
   - 🧠 **Step 2:** LLM corrects wording → corrected transcript displayed
   - 📋 **Step 3:** LLM generates summary → summary displayed in Bahasa Indonesia
4. **Download** all results as `.txt` files

> The entire pipeline is **fully automated** — one click, zero intervention.

---

## 📁 Project Structure

```
rangkumin/
├── audio.py              # Main Streamlit application
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── LICENSE               # MIT License
```

---

## 🔧 Configuration

Key parameters can be adjusted at the top of `audio.py`:

```python
WHISPER_MODEL_SIZE = "large-v3-turbo"  # Whisper model variant
OLLAMA_MODEL = "qwen3.5:9b-q8_0"             # LLM for correction & summary
CPU_THREADS = 12                       # CPU threads (adjust to your hardware)
```

### Thread Tuning Guide

| CPU | Recommended Threads |
|---|---|
| 4 cores / 8 threads | 6 |
| 6 cores / 12 threads | 8–10 |
| 8 cores / 16 threads | 12 |
| 12+ cores | cores × 1.5 |

---

## ⚠️ Known Limitations

- **CPU-only inference** — No GPU acceleration (Radeon 680M iGPU lacks ROCm support). An NVIDIA GPU would significantly improve speed.
- **Whisper language** — Hardcoded to `language="id"` (Indonesian). Change this parameter for other languages.
- **Long audio** — Files over 2 hours may require significant processing time (>6 hour) But supported.
- **Qwen3.5 incompatibility** — Qwen3.5 thinking mode cannot be disabled via Ollama, causing severe slowdowns. Stick with Qwen3.

---

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) — Speech recognition model
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2-based Whisper implementation
- [Ollama](https://ollama.com) — Local LLM runtime
- [Qwen3.5](https://ollama.com/library/qwen3.5)
- [Streamlit](https://streamlit.io) — Web application framework

---

<div align="center">

**Built with ❤️ for Indonesian professionals who deserve better meeting tools.**

*Rangkumin — Rangkum meeting-mu, otomatis!*

</div>

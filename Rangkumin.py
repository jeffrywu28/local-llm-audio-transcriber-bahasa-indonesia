import streamlit as st
from faster_whisper import WhisperModel
import ollama
import os
import re
import warnings
from pydub import AudioSegment
import math
import time

# ── Suppress warnings ──
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["OMP_NUM_THREADS"] = "12"
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
TEMPORARY_FILE = "temp.wav"
WHISPER_MODEL_SIZE = "large-v3-turbo"
OLLAMA_MODEL = "qwen3.5:9b-q8_0"
CPU_THREADS = 12


@st.cache_resource
def load_whisper_model():
    return WhisperModel(
        WHISPER_MODEL_SIZE,
        device="cpu",
        compute_type="int8",
        cpu_threads=CPU_THREADS,
    )


def convert_to_wav(uploaded_file):
    audio = AudioSegment.from_file(uploaded_file)
    audio.export(TEMPORARY_FILE, format="wav")
    return TEMPORARY_FILE


def transcribe_audio(audio_path, progress_callback=None):
    model = load_whisper_model()
    segments, info = model.transcribe(
        audio_path,
        beam_size=1,
        language="id",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    all_segments = []
    for seg in segments:
        all_segments.append(seg.text)
        if progress_callback:
            progress_callback(seg.end / max(info.duration, 1))

    return " ".join(all_segments), info.duration / 60


def strip_think_tags(text):
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def chunk_text(text, max_words=2000):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def call_ollama(prompt, text, temperature=0.3, ctx=4096):
    response = ollama.generate(
        model=OLLAMA_MODEL,
        prompt=f"{prompt}\n\n{text}",
        think=False,
        options={
            "num_thread": CPU_THREADS,
            "num_ctx": ctx,
            "temperature": temperature,
        },
    )
    return strip_think_tags(response["response"])


def fix_transcript_chunked(text, progress_bar=None):
    prompt = (
        "INSTRUKSI: Output WAJIB Bahasa Indonesia. "
        "JANGAN terjemahkan istilah Inggris — biarkan apa adanya. "
        "Jika pembicara bicara bahasa Inggris, tulis apa adanya.\n\n"
        "TUGAS:\n"
        "1. Perbaiki kata yang salah dengar\n"
        "2. Tambahkan tanda baca yang benar\n"
        "3. Pisahkan setiap kalimat ke baris baru\n"
        "4. Jangan ubah makna\n\n"
        "Output HANYA transkrip yang diperbaiki."
    )

    chunks = chunk_text(text, max_words=2000)
    fixed_parts = []

    for i, chunk in enumerate(chunks):
        try:
            result = call_ollama(prompt, chunk, temperature=0.2)
            fixed_parts.append(result)
        except Exception as e:
            st.error(f"❌ Chunk {i+1} error: {e}")
            fixed_parts.append(chunk)

        if progress_bar:
            progress_bar.progress((i + 1) / len(chunks))

    return "\n\n".join(fixed_parts)


def summarize_ollama(text):
    prompt = (
        "INSTRUKSI KETAT:\n"
        "1. Seluruh output WAJIB Bahasa Indonesia.\n"
        "2. DILARANG menulis dalam bahasa Inggris.\n"
        "3. Istilah teknis boleh tetap bahasa Inggris.\n"
        "4. Gunakan perbendaharaan kata yang kaya.\n\n"
        "Buatkan ringkasan komprehensif:\n\n"
        "## Ringkasan Umum\n"
        "(isi keseluruhan percakapan)\n\n"
        "## Poin-Poin Utama\n"
        "- (poin penting)\n\n"
        "## Keputusan Penting\n"
        "- (jika ada)\n\n"
        "## Tindak Lanjut\n"
        "- (jika ada)\n\n"
        "SEMUA dalam Bahasa Indonesia."
    )

    try:
        return call_ollama(prompt, text, temperature=0.4, ctx=8192)
    except Exception as e:
        st.error(f"❌ Summarize error: {e}")
        return None


def fmt_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f} detik"
    m, s = divmod(int(seconds), 60)
    return f"{m} menit {s} detik" if m < 60 else f"{m//60} jam {m%60} menit"


# ── UI ──
st.set_page_config(page_title="Audio Transcriber", page_icon="🎙️", layout="wide")
st.title("🎙️ Audio Transcriber")
st.caption(
    f"Whisper `{WHISPER_MODEL_SIZE}` → Ollama `{OLLAMA_MODEL}` "
    f"(think=False) | {CPU_THREADS} threads"
)

uploaded_file = st.file_uploader(
    "📁 Upload audio",
    type=["mp3", "wav", "ogg", "m4a", "flac", "wma", "aac"],
)

if uploaded_file:
    st.audio(uploaded_file)

    if st.button("▶️ Mulai Transkripsi", type="primary", use_container_width=True):
        total_start = time.time()

        try:
            with st.spinner("🔄 Converting..."):
                audio_path = convert_to_wav(uploaded_file)

            # ── STEP 1: WHISPER ──
            st.subheader("🎧 Step 1: Whisper Transcription")
            t1 = time.time()
            wp = st.progress(0, "Transcribing...")
            raw, dur = transcribe_audio(
                audio_path,
                lambda p: wp.progress(min(p, 1.0), f"Transcribing... {p*100:.0f}%"),
            )
            t1 = time.time() - t1
            wp.progress(1.0, f"✅ {fmt_time(t1)}")
            wc = len(raw.split())
            st.success(
                f"⏱️ Durasi audio: {dur:.1f} menit | "
                f"📝 Kata: {wc:,} | ⚡ Whisper: {fmt_time(t1)}"
            )

            st.markdown("### 📄 Raw Transcript")
            st.container(border=True).text(raw)
            st.session_state["raw_transcript"] = raw

            if os.path.exists(TEMPORARY_FILE):
                os.remove(TEMPORARY_FILE)

            # ── STEP 2: LLM FIX ──
            st.subheader(f"🧠 Step 2: Perbaiki Wording ({OLLAMA_MODEL})")
            t2 = time.time()
            nc = math.ceil(wc / 2000)
            st.info(f"Processing {nc} chunk(s)...")
            lp = st.progress(0, "Fixing transcript...")
            fixed = fix_transcript_chunked(raw, lp)
            t2 = time.time() - t2
            lp.progress(1.0, f"✅ {fmt_time(t2)}")

            st.markdown("### 📝 Hasil Transkripsi (Diperbaiki)")
            st.container(border=True).write(fixed)
            st.session_state["fixed_transcript"] = fixed

            # ── STEP 3: RINGKASAN ──
            st.subheader("📋 Step 3: Ringkasan (Bahasa Indonesia)")
            t3 = time.time()
            with st.spinner("Meringkas dalam Bahasa Indonesia..."):
                summary = summarize_ollama(fixed)
            t3 = time.time() - t3

            if summary:
                st.markdown("### 📋 Ringkasan")
                st.container(border=True).write(summary)
                st.success(f"⏱️ Ringkasan selesai dalam {fmt_time(t3)}")
                st.session_state["summary"] = summary

            # ── TOTAL ──
            tt = time.time() - total_start
            st.divider()
            st.success(
                f"🎉 **Selesai!** Total waktu: **{fmt_time(tt)}**\n\n"
                f"- Whisper: {fmt_time(t1)}\n"
                f"- LLM Fix: {fmt_time(t2)}\n"
                f"- Ringkasan: {fmt_time(t3)}"
            )

        except Exception as e:
            st.error(f"❌ Error: {e}")
            if os.path.exists(TEMPORARY_FILE):
                os.remove(TEMPORARY_FILE)

    # ── DOWNLOAD ──
    if "fixed_transcript" in st.session_state or "summary" in st.session_state:
        st.divider()
        st.subheader("💾 Download Hasil")
        col1, col2, col3 = st.columns(3)

        with col1:
            if "raw_transcript" in st.session_state:
                st.download_button(
                    "📄 Raw Transcript", st.session_state["raw_transcript"],
                    "raw_transcript.txt", "text/plain", use_container_width=True,
                )
        with col2:
            if "fixed_transcript" in st.session_state:
                st.download_button(
                    "📝 Transkripsi Diperbaiki", st.session_state["fixed_transcript"],
                    "transkripsi.txt", "text/plain", use_container_width=True,
                )
        with col3:
            if "summary" in st.session_state:
                st.download_button(
                    "📋 Ringkasan", st.session_state["summary"],
                    "ringkasan.txt", "text/plain", use_container_width=True,
                )

else:
    st.info("📁 Upload file audio untuk memulai.")

with st.sidebar:
    st.markdown("## ⚙️ Config")
    st.markdown(f"- 🎧 **`{WHISPER_MODEL_SIZE}`** int8, beam=1")
    st.markdown(f"- 🧠 **`{OLLAMA_MODEL}`** + `think=False`")
    st.markdown(f"- 💻 **{CPU_THREADS}** threads")
    st.divider()
    st.markdown("## 🔄 Flow Otomatis")
    st.markdown("1. ▶️ Klik **Mulai**")
    st.markdown("2. 🎧 Whisper → **tampil raw**")
    st.markdown("3. 🧠 LLM fix → **tampil diperbaiki**")
    st.markdown("4. 📋 Ringkasan → **tampil ringkasan**")
    st.markdown("5. 💾 Download semua hasil")
    st.markdown("")
    st.markdown("*Semua otomatis, 1x klik!*")

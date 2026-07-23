"""Local transcription via faster-whisper.

Reads a local media file and returns the same shape the highlight generator
expects: {duration, segments[start, end, text, words[start, end, word]]}.
"""
import os
import json
from pathlib import Path
from typing import Dict, Optional

from ..config import LOCAL_OUTPUT_DIR, LOCAL_WHISPER_DEVICE, LOCAL_WHISPER_MODEL


def _transcript_cache_path(media_path: str) -> Path:
    """Return the .json cache path for a media file."""
    cache_dir = Path(LOCAL_OUTPUT_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / (Path(media_path).stem + ".json")


def _write_json_cache(media_path: str, transcript: Dict) -> Path:
    cache_path = _transcript_cache_path(media_path)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    return cache_path


def _load_json_cache(cache_path: Path) -> Dict:
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_device() -> str:
    if LOCAL_WHISPER_DEVICE != "auto":
        return LOCAL_WHISPER_DEVICE
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            # Test that CUDA actually works (catches missing cuBLAS/cuDNN libs)
            torch.zeros(1, device="cuda")
            return "cuda"
    except (ImportError, OSError, RuntimeError):
        pass
    return "cpu"


def transcribe_local(media_path: str, language: Optional[str] = None) -> Dict:
    """Run faster-whisper on a local file path, caching the result as .json."""
    cache_path = _transcript_cache_path(media_path)
    
    # Fallback: check if old .srt cache exists and load/delete it to upgrade to json
    srt_cache = cache_path.with_suffix(".srt")
    if srt_cache.exists() and not cache_path.exists():
        srt_cache.unlink(missing_ok=True)

    if cache_path.exists():
        source_mtime = os.path.getmtime(media_path)
        cache_mtime = cache_path.stat().st_mtime
        if cache_mtime >= source_mtime:
            print(f"\033[93m[transcribe/local] Reusing cached transcript:\033[0m {cache_path}", flush=True)
            cached = _load_json_cache(cache_path)
            if not cached.get("segments") or cached.get("duration", 0.0) <= 0.0:
                print(f"\033[91m[transcribe/local] Cache is empty/invalid, deleting:\033[0m {cache_path}", flush=True)
                cache_path.unlink(missing_ok=True)
            else:
                print(
                    f"\033[93m[transcribe/local]\033[0m \033[92mLoaded {len(cached['segments'])} cached segments, "
                    f"{cached['duration']:.0f}s of audio\033[0m",
                    flush=True,
                )
                return cached

    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "faster-whisper is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    device = _resolve_device()
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"\033[93m[transcribe/local]\033[0m \033[1mRunning faster-whisper model={LOCAL_WHISPER_MODEL} device={device}\033[0m", flush=True)

    from ..config import LOCAL_WHISPER_VAD_FILTER, LOCAL_WHISPER_VAD_PARAMETERS

    model = WhisperModel(LOCAL_WHISPER_MODEL, device=device, compute_type=compute_type)

    transcribe_kwargs = {
        "audio": media_path,
        "language": language,
        "beam_size": 5,
        "condition_on_previous_text": False,
        "word_timestamps": True,  # Generate precise word-level timestamps
    }
    if LOCAL_WHISPER_VAD_FILTER:
        transcribe_kwargs["vad_filter"] = True
        transcribe_kwargs["vad_parameters"] = LOCAL_WHISPER_VAD_PARAMETERS
    else:
        transcribe_kwargs["vad_filter"] = False

    segments_iter, info = model.transcribe(**transcribe_kwargs)

    segments = []
    for s in segments_iter:
        words_list = []
        if getattr(s, "words", None):
            for w in s.words:
                words_list.append({
                    "start": float(w.start),
                    "end": float(w.end),
                    "word": str(w.word).strip(),
                })
        segments.append({
            "start": float(s.start),
            "end": float(s.end),
            "text": (s.text or "").strip(),
            "words": words_list
        })

    duration = float(getattr(info, "duration", 0.0)) or (segments[-1]["end"] if segments else 0.0)
    print(f"\033[93m[transcribe/local]\033[0m \033[92mCompleted {len(segments)} segments, {duration:.0f}s of audio\033[0m", flush=True)
    transcript = {"duration": duration, "segments": segments}
    cache_path = _write_json_cache(media_path, transcript)
    print(f"\033[93m[transcribe/local]\033[0m Wrote cache: {cache_path}", flush=True)
    return transcript

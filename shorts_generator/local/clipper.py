"""Local clipping: ffmpeg subclip + OpenCV face-aware vertical crop.

Two stages per highlight:
  1. Cut the source video to [start, end] with ffmpeg (re-encoded, audio kept).
  2. Reframe the cut to the target aspect ratio. For 9:16 we slide a vertical
     window horizontally across the frame to keep faces centred (Haar
     cascade — same approach as the original repo, no external models).
"""
import os
import subprocess
from typing import Dict, List, Optional, Tuple

from ..config import LOCAL_OUTPUT_DIR


def _ratio(aspect_ratio: str) -> float:
    """Parse '9:16' → 9/16, '1:1' → 1.0."""
    try:
        w, h = aspect_ratio.split(":")
        return float(w) / float(h)
    except (ValueError, ZeroDivisionError):
        return 9.0 / 16.0


def _cut_subclip(source_path: str, start: float, end: float, out_path: str) -> str:
    """ffmpeg -ss start -to end → re-encoded mp4 with audio."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", source_path,
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


def _get_subtitle_text(transcript: Optional[Dict], current_time: float) -> str:
    if not transcript:
        return ""
    for seg in transcript.get("segments", []):
        if seg["start"] <= current_time <= seg["end"]:
            return seg.get("text", "").strip()
    return ""


def _wrap_text(text: str, max_chars: int = 18) -> List[str]:
    words = text.split()
    lines = []
    current_line = []
    current_len = 0
    for w in words:
        if current_len + len(w) + 1 > max_chars:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [w]
            current_len = len(w)
        else:
            current_line.append(w)
            current_len += len(w) + 1
    if current_line:
        lines.append(" ".join(current_line))
    return lines


def _draw_styled_text(img, text: str, org: Tuple[int, int], font_face, font_scale, color, thickness, outline_thickness):
    # Outline (black border)
    import cv2  # type: ignore
    cv2.putText(
        img, text, org, font_face, font_scale,
        (0, 0, 0), thickness + outline_thickness * 2,
        lineType=cv2.LINE_AA
    )
    # Primary text
    cv2.putText(
        img, text, org, font_face, font_scale,
        color, thickness,
        lineType=cv2.LINE_AA
    )


def _reframe_vertical(
    in_path: str,
    out_path: str,
    aspect_ratio: str,
    transcript: Optional[Dict] = None,
    start_time: float = 0.0,
) -> str:
    """Crop the cut clip to the target aspect ratio, tracking faces if possible."""
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "opencv-python is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    target_ratio = _ratio(aspect_ratio)
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {in_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Compute the largest crop that fits inside the frame at the target ratio.
    if target_ratio < src_w / src_h:
        crop_h = src_h
        crop_w = int(crop_h * target_ratio)
    else:
        crop_w = src_w
        crop_h = int(crop_w / target_ratio)
    crop_w = max(2, crop_w - (crop_w % 2))
    crop_h = max(2, crop_h - (crop_h % 2))

    import torch
    use_gpu = torch.cuda.is_available()
    mtcnn = None
    face_cascade = None

    if use_gpu:
        try:
            from facenet_pytorch import MTCNN
            device = torch.device('cuda')
            mtcnn = MTCNN(keep_all=False, device=device, select_largest=True)
            print("[clip/local] using PyTorch MTCNN on GPU (CUDA) for face tracking", flush=True)
        except ImportError:
            print("[clip/local] facenet-pytorch not installed, falling back to CPU face tracking", flush=True)
            use_gpu = False

    if not use_gpu:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        print("[clip/local] using OpenCV CascadeClassifier on CPU for face tracking", flush=True)

    silent_path = out_path + ".silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_path, fourcc, fps, (crop_w, crop_h))

    last_center: Optional[Tuple[int, int]] = None
    target_center: Optional[Tuple[int, int]] = None
    smoothing = 0.05  # lower = smoother tracking, higher = faster tracking
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 5 == 0:
            cx, cy = None, None
            if use_gpu and mtcnn is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes, _ = mtcnn.detect(rgb_frame)
                if boxes is not None and len(boxes) > 0:
                    x1, y1, x2, y2 = boxes[0]
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
            elif face_cascade is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
                if len(faces) > 0:
                    # Pick the largest face — usually the speaker.
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    cx = x + w // 2
                    cy = y + h // 2
            
            if cx is not None and cy is not None:
                target_center = (cx, cy)

        if target_center is None:
            target_center = (src_w // 2, src_h // 2)

        if last_center is None:
            last_center = target_center
        else:
            lx, ly = last_center
            tx, ty = target_center
            last_center = (
                int(lx + (tx - lx) * smoothing),
                int(ly + (ty - ly) * smoothing),
            )

        cx, cy = last_center
        x0 = max(0, min(src_w - crop_w, cx - crop_w // 2))
        y0 = max(0, min(src_h - crop_h, cy - crop_h // 2))
        cropped = frame[y0:y0 + crop_h, x0:x0 + crop_w]

        # Draw dynamic subtitles if transcript is available
        if transcript:
            current_time = start_time + (frame_count / fps)
            sub_text = _get_subtitle_text(transcript, current_time)
            if sub_text:
                wrapped = _wrap_text(sub_text, max_chars=18)
                font_face = cv2.FONT_HERSHEY_DUPLEX
                # Scale dynamically based on resolution
                font_scale = crop_w / 360.0 * 0.8
                thickness = max(1, int(font_scale * 2))
                outline_thickness = max(1, int(font_scale * 3))
                
                # Render captions centered horizontally and placed at 70% height
                base_y = int(crop_h * 0.70)
                line_height = int(35 * font_scale)
                for line_idx, line in enumerate(wrapped):
                    text_size, _ = cv2.getTextSize(line, font_face, font_scale, thickness)
                    text_w, text_h = text_size
                    x_org = max(10, (crop_w - text_w) // 2)
                    y_org = base_y + line_idx * line_height
                    # Use vibrant yellow color (BGR: 0, 255, 255)
                    _draw_styled_text(
                        cropped, line, (x_org, y_org),
                        font_face, font_scale, (0, 255, 255),
                        thickness, outline_thickness
                    )

        writer.write(cropped)
        frame_count += 1

    cap.release()
    writer.release()

    # Mux audio from the cut clip back onto the silent reframed video.
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", silent_path,
        "-i", in_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    os.remove(silent_path)
    return out_path


def crop_clip_local(
    source_path: str,
    start_time: float,
    end_time: float,
    aspect_ratio: str,
    out_path: str,
    transcript: Optional[Dict] = None,
) -> str:
    """Cut + reframe one highlight, returning the local mp4 path."""
    cut_path = out_path + ".cut.mp4"
    try:
        _cut_subclip(source_path, start_time, end_time, cut_path)
        _reframe_vertical(cut_path, out_path, aspect_ratio, transcript=transcript, start_time=start_time)
    finally:
        if os.path.exists(cut_path):
            os.remove(cut_path)
    return out_path


def crop_highlights_local(
    source_path: str,
    highlights: List[Dict],
    aspect_ratio: str = "9:16",
    out_dir: Optional[str] = None,
    transcript: Optional[Dict] = None,
) -> List[Dict]:
    out_dir = out_dir or LOCAL_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    results: List[Dict] = []
    for i, h in enumerate(highlights, 1):
        out_path = os.path.join(out_dir, f"short_{i:02d}.mp4")
        print(f"[clip/local] {i}/{len(highlights)}: {h.get('title', '(untitled)')}", flush=True)
        try:
            crop_clip_local(
                source_path,
                float(h["start_time"]),
                float(h["end_time"]),
                aspect_ratio,
                out_path,
                transcript=transcript,
            )
            results.append({**h, "clip_url": out_path})
        except Exception as e:
            print(f"[clip/local] {i} failed: {e}", flush=True)
            results.append({**h, "clip_url": None, "error": str(e)})
    return results

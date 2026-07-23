"""Local clipping: ffmpeg subclip + OpenCV face-aware vertical crop.

Two stages per highlight:
  1. Cut the source video to [start, end] with ffmpeg (re-encoded, audio kept).
  2. Reframe the cut to the target aspect ratio. For 9:16 we slide a vertical
     window horizontally across the frame to keep faces centred (Haar
     cascade — same approach as the original repo, no external models).
"""
import os
import subprocess
import re
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


def _draw_dynamic_captions(canvas, words: List[Dict], current_time: float, out_w: int, out_h: int):
    import cv2  # type: ignore
    # Find the active word index
    active_idx = -1
    for idx, w in enumerate(words):
        if w["start"] <= current_time <= w["end"]:
            active_idx = idx
            break
            
    # If no word is currently active, fallback to the word closest to current_time
    if active_idx == -1 and words:
        active_idx = min(range(len(words)), key=lambda i: abs(words[i]["start"] - current_time))
        
    if active_idx == -1 or not words:
        return
        
    # Create a 3-word window centered around the active word
    start_win = max(0, active_idx - 1)
    end_win = min(len(words), start_win + 3)
    if end_win - start_win < 3:
        start_win = max(0, end_win - 3)
    window_words = words[start_win:end_win]
    
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = out_w / 360.0 * 0.70
    thickness = max(1, int(font_scale * 2.2))
    outline_thickness = max(1, int(font_scale * 3.2))
    
    word_sizes = []
    space_size, _ = cv2.getTextSize(" ", font_face, font_scale, thickness)
    space_w = space_size[0]
    
    total_w = 0
    for idx, w in enumerate(window_words):
        t_size, _ = cv2.getTextSize(w["word"].upper(), font_face, font_scale, thickness)
        word_sizes.append(t_size[0])
        total_w += t_size[0]
        if idx < len(window_words) - 1:
            total_w += space_w
            
    start_x = max(10, (out_w - total_w) // 2)
    y_org = int(out_h * 0.80)
    
    current_x = start_x
    for idx, w in enumerate(window_words):
        word_text = w["word"].upper()
        is_active = (start_win + idx == active_idx)
        color = (0, 255, 255) if is_active else (255, 255, 255)
        
        _draw_styled_text(
            canvas, word_text, (current_x, y_org),
            font_face, font_scale, color,
            thickness, outline_thickness
        )
        current_x += word_sizes[idx] + space_w


def _reframe_vertical(
    in_path: str,
    out_path: str,
    aspect_ratio: str,
    transcript: Optional[Dict] = None,
    start_time: float = 0.0,
    top_bar_hook: str = "",
) -> str:
    """Crop the cut clip to the target aspect ratio, tracking faces if possible."""
    try:
        import cv2  # type: ignore
        import numpy as np
    except ImportError as e:
        raise RuntimeError(
            "opencv-python/numpy is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {in_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Cinematic Crop Sizing:
    # We crop the original frame to a 1:1 square that tracks the speaker.
    crop_w = min(src_w, src_h)
    crop_h = crop_w

    # The final output frame is 9:16 (black bars on top and bottom of the 1:1 square)
    out_w = crop_w
    out_h = int(out_w / (9.0 / 16.0))
    out_w = max(2, out_w - (out_w % 2))
    out_h = max(2, out_h - (out_h % 2))

    import torch
    use_gpu = torch.cuda.is_available()
    mtcnn = None
    face_cascade = None

    if use_gpu:
        try:
            from facenet_pytorch import MTCNN
            device = torch.device('cuda')
            mtcnn = MTCNN(keep_all=False, device=device, select_largest=True)
            print("\033[95m[clip/local]\033[0m \033[92m\033[1mUsing PyTorch MTCNN on GPU (CUDA) for face tracking\033[0m", flush=True)
        except ImportError:
            print("\033[95m[clip/local]\033[0m \033[93mfacenet-pytorch not installed, falling back to CPU face tracking\033[0m", flush=True)
            use_gpu = False

    if not use_gpu:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        print("\033[95m[clip/local]\033[0m \033[93mUsing OpenCV CascadeClassifier on CPU for face tracking\033[0m", flush=True)

    silent_path = out_path + ".silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_path, fourcc, fps, (out_w, out_h))

    last_center: Optional[Tuple[int, int]] = None
    target_center: Optional[Tuple[int, int]] = None
    smoothing = 0.05  # lower = smoother tracking, higher = faster tracking
    frame_count = 0

    # Cooldown setup to prevent rapid camera cutting (hold camera for 2.0s minimum)
    cooldown_frames = int(fps * 2.0)
    frames_since_cut = cooldown_frames

    # Pending face tracking to prevent switching on short "ok"/"yeah" sounds
    pending_center: Optional[Tuple[int, int]] = None
    pending_count = 0
    required_consecutive_detections = 5  # 5 detections * 5 frame step = 25 frames (~0.8 seconds of speaking)

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
                if last_center is not None:
                    lx, ly = last_center
                    distance = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
                    
                    if distance > (crop_w // 3):
                        if pending_center is not None:
                            plx, ply = pending_center
                            p_dist = ((cx - plx) ** 2 + (cy - ply) ** 2) ** 0.5
                            if p_dist < (crop_w // 4):
                                pending_count += 1
                            else:
                                pending_center = (cx, cy)
                                pending_count = 1
                        else:
                            pending_center = (cx, cy)
                            pending_count = 1
                        
                        if pending_count >= required_consecutive_detections:
                            if frames_since_cut >= cooldown_frames:
                                target_center = pending_center
                                last_center = target_center
                                frames_since_cut = 0
                            pending_center = None
                            pending_count = 0
                    else:
                        target_center = (cx, cy)
                        pending_center = None
                        pending_count = 0
                else:
                    target_center = (cx, cy)
            else:
                pending_count = max(0, pending_count - 1)

        frame_count += 1
        frames_since_cut += 1

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

        # Create a black 9:16 canvas and paste the square crop centered vertically
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        y_offset = (out_h - crop_h) // 2
        canvas[y_offset:y_offset + crop_h, 0:out_w] = cropped

        # Draw dynamic subtitles if transcript is available
        if transcript:
            # Use millisecond container position to maintain perfect sync
            current_time = start_time + (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            active_seg = None
            for seg in transcript.get("segments", []):
                if seg["start"] <= current_time <= seg["end"]:
                    active_seg = seg
                    break
            
            if active_seg:
                words = active_seg.get("words", [])
                if words:
                    _draw_dynamic_captions(canvas, words, current_time, out_w, out_h)
                else:
                    sub_text = active_seg.get("text", "").strip()
                    if sub_text:
                        wrapped = _wrap_text(sub_text, max_chars=18)
                        font_face = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = out_w / 360.0 * 0.70
                        thickness = max(1, int(font_scale * 2.2))
                        outline_thickness = max(1, int(font_scale * 3.2))
                        
                        base_y = int(out_h * 0.80)
                        line_height = int(35 * font_scale)
                        for line_idx, line in enumerate(wrapped):
                            text_size, _ = cv2.getTextSize(line, font_face, font_scale, thickness)
                            text_w, text_h = text_size
                            x_org = max(10, (out_w - text_w) // 2)
                            y_org = base_y + line_idx * line_height
                            _draw_styled_text(
                                canvas, line, (x_org, y_org),
                                font_face, font_scale, (0, 255, 255),
                                thickness, outline_thickness
                            )

        # Draw static top bar hook (white color: 255, 255, 255)
        if top_bar_hook:
            wrapped_top = _wrap_text(top_bar_hook.upper(), max_chars=18)
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = out_w / 360.0 * 0.70
            thickness = max(1, int(font_scale * 2.2))
            outline_thickness = max(1, int(font_scale * 3.2))
            
            top_bar_height = y_offset
            line_height = int(40 * font_scale)
            total_text_h = len(wrapped_top) * line_height
            start_y = max(15, (top_bar_height - total_text_h) // 2 + line_height)
            
            for line_idx, line in enumerate(wrapped_top):
                text_size, _ = cv2.getTextSize(line, font_face, font_scale, thickness)
                text_w, text_h = text_size
                x_org = max(10, (out_w - text_w) // 2)
                y_org = start_y + line_idx * line_height
                _draw_styled_text(
                    canvas, line, (x_org, y_org),
                    font_face, font_scale, (255, 255, 255),
                    thickness, outline_thickness
                )

        writer.write(canvas)

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


def _upload_to_youtube(file_path: str, title: str, description: str):
    import os
    client_id = os.environ.get("YOUTUBE_CLIENT_ID")
    client_secret = os.environ.get("YOUTUBE_CLIENT_SECRET")
    refresh_token = os.environ.get("YOUTUBE_REFRESH_TOKEN")
    rules = os.environ.get("YOUTUBE_RULES", "")

    if not client_id or not client_secret or not refresh_token:
        print("\033[93m[clip/local] YouTube credentials not set. Skipping upload.\033[0m", flush=True)
        return

    try:
        from google.oauth2.credentials import Credentials  # type: ignore
        from googleapiclient.discovery import build  # type: ignore
        from googleapiclient.http import MediaFileUpload  # type: ignore
    except ImportError:
        print("\033[91m[clip/local] Warning: google-api-python-client or google-auth not installed. Skipping upload.\033[0m", flush=True)
        return

    final_title = title
    final_description = description

    # Extract hashtags and tags from rules
    hashtags = re.findall(r"#[a-zA-Z0-9_]+", rules)
    tags = re.findall(r"@[a-zA-Z0-9_]+", rules)

    clean_rules = rules
    for ht in hashtags:
        clean_rules = clean_rules.replace(ht, "")
    for tg in tags:
        clean_rules = clean_rules.replace(tg, "")
    clean_rules = clean_rules.strip()

    if "#shorts" not in final_title.lower():
        final_title += " #shorts"
    for ht in hashtags:
        if ht.lower() not in final_title.lower():
            final_title += f" {ht}"
            
    if len(final_title) > 100:
        final_title = final_title[:97] + "..."

    desc_lines = [final_description]
    if tags:
        desc_lines.append(" ".join(tags))
    final_description = "\n\n".join(desc_lines)

    print(f"\033[92m[clip/local] Uploading to YouTube channel (@ghclip1) as Short...\033[0m", flush=True)
    print(f"  Title: {final_title}", flush=True)
    
    try:
        creds = Credentials(
            token=None,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=client_id,
            client_secret=client_secret
        )
        youtube = build("youtube", "v3", credentials=creds)
        
        body = {
            "snippet": {
                "title": final_title,
                "description": final_description,
                "categoryId": "22"
            },
            "status": {
                "privacyStatus": "public",
                "selfDeclaredMadeForKids": False
            }
        }
        
        media = MediaFileUpload(file_path, chunksize=-1, resumable=True, mimetype="video/mp4")
        request = youtube.videos().insert(
            part="snippet,status",
            body=body,
            media_body=media
        )
        
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                print(f"  Upload progress: {int(status.progress() * 100)}%", flush=True)
                
        video_id = response.get("id")
        print(f"\033[92m\033[1m[clip/local] Success! Uploaded as Short. Video Link: https://youtu.be/{video_id}\033[0m", flush=True)
    except Exception as e:
        print(f"\033[91m[clip/local] YouTube Upload Failed: {e}\033[0m", flush=True)


def crop_clip_local(
    source_path: str,
    start_time: float,
    end_time: float,
    aspect_ratio: str,
    out_path: str,
    transcript: Optional[Dict] = None,
    top_bar_hook: str = "",
) -> str:
    """Cut + reframe one highlight, returning the local mp4 path."""
    cut_path = out_path + ".cut.mp4"
    try:
        _cut_subclip(source_path, start_time, end_time, cut_path)
        _reframe_vertical(cut_path, out_path, aspect_ratio, transcript=transcript, start_time=start_time, top_bar_hook=top_bar_hook)
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
        print(f"\033[95m[clip/local] {i}/{len(highlights)}:\033[0m \033[1m{h.get('title', '(untitled)')}\033[0m", flush=True)
        try:
            crop_clip_local(
                source_path,
                float(h["start_time"]),
                float(h["end_time"]),
                aspect_ratio,
                out_path,
                transcript=transcript,
                top_bar_hook=h.get("top_bar_hook", "WAIT FOR IT..."),
            )
            # Upload clip to YouTube immediately
            _upload_to_youtube(
                file_path=out_path,
                title=h.get("title", "YouTube Short"),
                description=h.get("virality_reason", "Awesome Short Video")
            )
            results.append({**h, "clip_url": out_path})
        except Exception as e:
            print(f"\033[95m[clip/local] {i} failed:\033[0m \033[91m{e}\033[0m", flush=True)
            results.append({**h, "clip_url": None, "error": str(e)})
    return results

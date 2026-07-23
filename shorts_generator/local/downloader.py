"""Local YouTube download via yt-dlp.

Returns a local mp4 path so the rest of the local pipeline can read it
directly off disk.
"""
import os
import re
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
from typing import Optional

from ..config import LOCAL_OUTPUT_DIR


def _import_ytdlp():
    try:
        import yt_dlp  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "yt-dlp is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e
    return yt_dlp


def _format_for(fmt: str) -> str:
    """Map our '720' / '1080' shorthand to a yt-dlp format selector."""
    try:
        height = int(fmt)
    except ValueError:
        height = 720
    return (
        f"bestvideo[height<={height}]+bestaudio/"
        f"best[height<={height}]/"
        f"best"
    )


def _extract_youtube_video_id(source: str) -> Optional[str]:
    """Best-effort extraction of a YouTube video id from a URL."""
    parsed = urlparse(source)
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]

    if host in ("youtu.be", "www.youtu.be"):
        video_id = parsed.path.lstrip("/").split("/", 1)[0]
        return video_id or None

    if "youtube.com" in host:
        if parsed.path.startswith("/watch"):
            qs = parse_qs(parsed.query)
            video_id = qs.get("v", [""])[0]
            return video_id or None
        match = re.search(r"/(?:shorts|embed|live)/([^/?#&]+)", parsed.path)
        if match:
            return match.group(1)

    return None


def _resolve_local_path(source: str) -> Optional[str]:
    """Return a local filesystem path if the input already points at one."""
    parsed = urlparse(source)
    if parsed.scheme == "file":
        raw_path = unquote(parsed.path)
        if parsed.netloc and parsed.netloc not in ("", "localhost"):
            raw_path = f"//{parsed.netloc}{raw_path}"
        candidate = Path(raw_path).expanduser()
        if candidate.exists() and candidate.is_file():
            return str(candidate.resolve())
        raise RuntimeError(f"Local file URL does not exist: {source}")

    if parsed.scheme in ("http", "https"):
        return None

    candidate = Path(source).expanduser()
    if candidate.exists() and candidate.is_file():
        return str(candidate.resolve())

    if any(sep in source for sep in (os.sep, "/")) or source.startswith("~") or source.startswith("."):
        raise RuntimeError(f"Local file path does not exist: {source}")

    return None


def _existing_download(out_dir: str, video_id: str) -> Optional[str]:
    """Return a cached download path if we already have this YouTube id."""
    for ext in (".mp4", ".mkv", ".webm"):
        candidate = os.path.join(out_dir, f"source_{video_id}{ext}")
        if os.path.exists(candidate):
            return candidate
    return None


def download_youtube_local(video_url: str, fmt: str = "720", out_dir: Optional[str] = None) -> str:
    """Download a remote URL or return a local file path unchanged."""
    # Ensure Deno/Node path is visible to yt-dlp inside python
    deno_path = os.path.expanduser("~/.deno/bin")
    if deno_path not in os.environ.get("PATH", ""):
        os.environ["PATH"] = deno_path + os.pathsep + os.environ.get("PATH", "")

    local_path = _resolve_local_path(video_url)
    if local_path:
        print(f"[download/local] using local file: {local_path}", flush=True)
        return local_path

    yt_dlp = _import_ytdlp()
    out_dir = out_dir or LOCAL_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    video_id = _extract_youtube_video_id(video_url)
    if video_id:
        cached = _existing_download(out_dir, video_id)
        if cached:
            print(f"[download/local] reusing cached download: {cached}", flush=True)
            return cached

    import subprocess
    cmd = [
        "python3", "-m", "yt_dlp",
        "--format", _format_for(fmt),
        "--merge-output-format", "mp4",
        "--output", os.path.join(out_dir, "source_%(id)s.%(ext)s"),
        "--remote-components", "ejs:github",
    ]
    
    cookies_to_use = None
    cookies_path = os.path.join(os.getcwd(), "cookies.txt")
    cookies_json_path = os.path.join(os.getcwd(), "cookies.json")
    if os.path.exists(cookies_json_path):
        try:
            import json
            with open(cookies_json_path, "r", encoding="utf-8") as f:
                cookies = json.load(f)
            temp_cookies = os.path.join(out_dir, "cookies_converted.txt")
            if os.path.exists(temp_cookies):
                try:
                    os.chmod(temp_cookies, 0o666)
                    os.remove(temp_cookies)
                except Exception:
                    pass
            netscape = "# Netscape HTTP Cookie File\n"
            for c in cookies:
                domain = c.get("domain", "")
                flag = "TRUE"
                path = c.get("path", "/")
                secure = "TRUE" if c.get("secure") else "FALSE"
                exp = str(int(c.get("expirationDate", 0)))
                name = c.get("name", "")
                value = c.get("value", "")
                netscape += f"{domain}\t{flag}\t{path}\t{secure}\t{exp}\t{name}\t{value}\n"
            with open(temp_cookies, "w", encoding="utf-8") as f:
                f.write(netscape)
            os.chmod(temp_cookies, 0o444)
            cookies_to_use = temp_cookies
            print("[download/local] loaded and converted cookies.json", flush=True)
        except Exception as e:
            print(f"[download/local] warning: failed to parse cookies.json: {e}", flush=True)
            if os.path.exists(cookies_path):
                cookies_to_use = cookies_path
    elif os.path.exists(cookies_path):
        cookies_to_use = cookies_path

    if cookies_to_use:
        cmd.extend(["--cookies", cookies_to_use])

    cmd.append(video_url)
    
    print(f"[download/local] downloading with command: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    
    # Locate the downloaded file
    path = os.path.join(out_dir, f"source_{video_id}.mp4")
    if not os.path.exists(path):
        stem = os.path.join(out_dir, f"source_{video_id}")
        for ext in (".mp4", ".mkv", ".webm"):
            if os.path.exists(stem + ext):
                path = stem + ext
                break

    print(f"[download/local] ready: {path}", flush=True)
    return path

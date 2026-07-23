"""CLI entry point.

Usage:
    python main.py "https://www.youtube.com/watch?v=..." \
        --num-clips 3 --aspect-ratio 9:16
"""
import argparse
import json
import sys

# Windows uses 'charmap' by default, which can't encode Unicode characters
# like →. Reconfigure stdout/stderr to UTF-8 so output works on all platforms.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from shorts_generator import generate_shorts


def main() -> int:
    parser = argparse.ArgumentParser(description="AI YouTube Shorts Generator")
    parser.add_argument("url", help="YouTube URL, file:// URL, or local file path")
    parser.add_argument(
        "--mode",
        choices=["api", "local"],
        default="api",
        help="api (default, MuAPI) or local (remote URL, file://, or local path + faster-whisper + LLM provider + ffmpeg).",
    )
    parser.add_argument("--num-clips", type=int, default=3, help="How many shorts to render (default: 3)")
    parser.add_argument("--aspect-ratio", default="9:16", help="Output aspect ratio (default: 9:16)")
    parser.add_argument("--format", default="720", help="Source download resolution: 360 / 480 / 720 / 1080 (default: 720)")
    parser.add_argument("--language", default=None, help="Force Whisper language code, e.g. 'en' (default: auto-detect)")
    parser.add_argument("--output-json", default=None, help="Write the full result JSON to this path")
    args = parser.parse_args()

    try:
        result = generate_shorts(
            youtube_url=args.url,
            num_clips=args.num_clips,
            aspect_ratio=args.aspect_ratio,
            download_format=args.format,
            language=args.language,
            mode=args.mode,
        )
    except Exception as e:
        print(f"\n\033[91m\033[1m[ERROR] FAILED:\033[0m \033[91m{e}\033[0m", file=sys.stderr)
        return 1

    print("\n\033[96m\033[1m" + "=" * 72 + "\033[0m")
    print(f"\033[96mMode:          \033[0m\033[1m{result.get('mode', args.mode).upper()}\033[0m")
    print(f"\033[96mSource video:  \033[0m{result['source_video_url']}")
    print(f"\033[96mHighlights:    \033[0m{len(result['highlights'])} candidates \033[92m\033[1m→\033[0m kept top {len(result['shorts'])}")
    print("\033[96m\033[1m" + "=" * 72 + "\033[0m")
    for i, s in enumerate(result["shorts"], 1):
        print(f"\n\033[92m\033[1m#{i}\033[0m  \033[93m\033[1mscore={s.get('score')}\033[0m  \033[96m{s.get('start_time'):.1f}s → {s.get('end_time'):.1f}s\033[0m")
        print(f"     \033[1mtitle:\033[0m  {s.get('title')}")
        print(f"     \033[1mhook:\033[0m   \"{s.get('hook_sentence')}\"")
        if s.get("clip_url"):
            print(f"     \033[92m\033[1mclip:\033[0m   {s['clip_url']}")
        else:
            print(f"     \033[91m\033[1mclip:\033[0m   FAILED ({s.get('error')})")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nFull JSON written to {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

import os, sys, subprocess

os.environ["LLM_PROVIDER"] = "openai"
os.environ["OPENAI_BASE_URL"] = "https://opencode.ai/zen/v1"
os.environ["OPENAI_MODEL"] = "nemotron-3-ultra-free"
os.environ["OPENAI_API_KEY"] = "sk-v60mFLUTTwfHOzuU32URqARD6VzWjYb6jod7ao1b1G6Yv0zZi35feoy60ZIrEjL5"
os.environ["LOCAL_WHISPER_DEVICE"] = "cuda"
os.environ["LOCAL_WHISPER_MODEL"] = "base"
os.environ["LOCAL_OUTPUT_DIR"] = "/content/drive/MyDrive/YoutubeShortsOutput"

url = sys.argv[1] if len(sys.argv) > 1 else "https://www.youtube.com/watch?v=0-OOW3TCBRA"
clips = sys.argv[2] if len(sys.argv) > 2 else "3"

result = subprocess.run(
    ["python3", "main.py", url, "--mode", "local", "--num-clips", clips],
    cwd="/content/AI-Youtube-Shorts-Generator",
    capture_output=False,
    text=True
)
sys.exit(result.returncode)

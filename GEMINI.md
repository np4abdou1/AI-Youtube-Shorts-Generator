# AI YouTube Shorts Generator - Google Colab CLI Guide

This guide describes how to run this repository using the **Google Colab CLI** (`colab`), which provisions a remote T4 GPU runtime and executes your code — all from the terminal, no browser needed.

Install the CLI once:
```bash
pip install google-colab-cli
colab new --gpu T4   # Authenticate on first use
```

---

## One-Shot Run (Full Pipeline)

Provision a T4 GPU, clone the repo, install deps, generate 3 shorts, download them, and tear down:

```bash
# 1. Provision T4 GPU
colab new --gpu T4

# 2. Clone the repository
echo "!git clone https://github.com/np4abdou1/AI-Youtube-Shorts-Generator.git" | colab exec

# 3. Install Deno + Python dependencies
echo "!curl -fsSL https://deno.land/install.sh | sh && pip install yt-dlp faster-whisper openai google-genai opencv-python-headless socksio httpx[socks] google-api-python-client google-auth-oauthlib facenet-pytorch --no-deps onnxruntime ctranslate2 av" | colab exec --timeout 300

# 4. Generate shorts (replace VIDEO_URL)
printf 'import subprocess, os\nos.chdir("/content/AI-Youtube-Shorts-Generator")\nsubprocess.run(["python3", "main.py", "VIDEO_URL", "--mode", "local", "--num-clips", "3"], env={**os.environ, "OPENAI_MODEL": "nemotron-3-ultra-free", "OPENAI_API_KEY": "sk-v60mFLUTTwfHOzuU32URqARD6VzWjYb6jod7ao1b1G6Yv0zZi35feoy60ZIrEjL5", "OPENAI_BASE_URL": "https://opencode.ai/zen/v1", "LLM_PROVIDER": "openai", "LOCAL_WHISPER_DEVICE": "cuda"})\n' | colab exec --timeout 600

# 5. Download output clips
colab download /content/drive/MyDrive/YoutubeShortsOutput/short_01.mp4 ./
colab download /content/drive/MyDrive/YoutubeShortsOutput/short_02.mp4 ./
colab download /content/drive/MyDrive/YoutubeShortsOutput/short_03.mp4 ./

# 6. Tear down
colab stop
```

---

## Quick Reference (Colab CLI)

| Command | Description |
|---------|-------------|
| `colab new --gpu T4` | Provision a T4 GPU runtime |
| `echo "CMD" \| colab exec` | Run a shell command on the remote VM |
| `colab exec -f script.py` | Run a local Python script on the remote VM |
| `colab upload LOCAL REMOTE` | Upload a file to the remote VM |
| `colab download REMOTE LOCAL` | Download a file from the remote VM |
| `colab install PKG` | Install a Python package on the remote VM |
| `colab status` | Check session hardware and status |
| `colab stop` | Terminate the VM (free up resources) |

---

## One-Time Setup (First Use Only)

On your **first use**, authenticate the CLI by running:
```bash
colab new --gpu T4
# Follow the browser URL to authorize
```

After that, authentication is cached and you can skip this step on subsequent runs.

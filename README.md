# AI YouTube Shorts Generator (Patched & Enhanced Fork)

An optimized, GPU-accelerated fork of the open-source YouTube Shorts generator. Transforms long-form YouTube videos or local clips into dynamic, viral-ready 9:16 Shorts with **GPU face-tracking, dynamic word-highlighted captions, silent parts removal, and automatic YouTube upload integration**.

Optimized for **Google Colab (T4 GPU)** and self-hosted environments.

---

## 🌟 Key Updates in this Fork

1. **⚡ GPU-Accelerated Face Tracking (MTCNN)**: Replaced CPU-only OpenCV Haar cascades with a PyTorch GPU-accelerated Multi-task Cascaded Convolutional Network (`facenet-pytorch`). High-accuracy face tracking runs directly on the T4 GPU.
2. **🎬 Smooth Camera Cuts**: Built-in speaking time filter detects real conversation. Ignores short sounds (like *"ok"*, *"yeah"*) and has a 2-second cooldown to prevent jumpy back-and-forth camera switching.
3. **🔥 Dynamic CapCut-Style Captions**: Generates word-by-word timestamped subtitles. Highlights the currently spoken word in vibrant yellow while adjacent words remain in clean white, sliding fluidly across a 3-word window.
4. **🎞️ Cinematic Split Layout**: Automatically crops the tracking box into a `1:1` square, centered vertically on a black `9:16` frame to create a clean, modern aesthetic with top and bottom black bars.
5. **🏷️ Catchy Top Hooks**: Dynamically generates clickbait/curiosity-gap top hooks (like *"WAIT FOR IT..."*, *"Watch until the end!"*) and renders them in bold white in the top black bar.
6. **🔇 Automatic Silence Cutter (Jump-Cuts)**: Word gaps exceeding 3.0 seconds are automatically cut out, splitting the video into multiple speech-active segments and stitching them together seamlessly using FFmpeg.
7. **🚀 YouTube Auto-Upload Integration**: Completed clips automatically upload directly to your selected YouTube Brand Account as Public Shorts immediately on completion.
8. **💬 Custom Rules Support**: Configures environment rules (e.g. `put #joy hashtag in the video captions, tag @channel in the description`) to customize titles and descriptions automatically.

---

## 🚀 Colab Setup (T4 GPU)

For a step-by-step setup in Google Colab, refer directly to the detailed [GEMINI.md](./GEMINI.md) guide inside this repository.

### Quick Setup Cell:
```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone the repository
!git clone https://github.com/np4abdou1/AI-Youtube-Shorts-Generator.git
%cd /content/AI-Youtube-Shorts-Generator

# 3. Install Deno (for YouTube signature decryption)
!curl -fsSL https://deno.land/install.sh | sh
import os
os.environ["PATH"] += ":/root/.deno/bin"

# 4. Install CUDA-enabled PyTorch and dependencies
!pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
!pip install facenet-pytorch --no-deps
!pip install yt-dlp faster-whisper openai google-genai opencv-python-headless socksio httpx[socks] google-api-python-client google-auth-oauthlib
```

---

## 🔧 Configuration (.env)

Create a `.env` file in the root of the cloned directory:

```env
# LLM API configuration (OpenAI-compatible)
LLM_PROVIDER=openai
OPENAI_API_KEY=your_opencode_or_openai_key
OPENAI_BASE_URL=https://opencode.ai/zen/v1
OPENAI_MODEL=nemotron-3-ultra-free

# Local transcription & rendering configurations
LOCAL_WHISPER_MODEL=base
LOCAL_WHISPER_DEVICE=cuda
LOCAL_OUTPUT_DIR=/content/drive/MyDrive/YoutubeShortsOutput

# YouTube API Credentials (Optional, for auto-uploading)
YOUTUBE_CLIENT_ID=your_gcp_oauth_client_id
YOUTUBE_CLIENT_SECRET=your_gcp_oauth_client_secret
YOUTUBE_REFRESH_TOKEN=your_google_auth_refresh_token
YOUTUBE_RULES="put #joy hashtag in the video captions, tag @ghclip1 in the description"
```

---

## 🏃 Run

Execute the pipeline in local mode:

```bash
python3 main.py "https://www.youtube.com/watch?v=VIDEO_ID" --mode local --num-clips 3
```

*Final output clips and their transcription JSONs will land in the output folder. If YouTube credentials are set, the video uploads as a Short immediately.*

---

## 📝 License

This project is licensed under the MIT License.

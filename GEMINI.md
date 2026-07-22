# AI YouTube Shorts Generator - Google Colab Deployment Guide

This guide describes how to run this repository in Google Colab using a T4 GPU. Running on Colab runs Whisper transcription and video rendering significantly faster than a CPU-only VPS, and allows you to bypass YouTube rate-limiting / bot checks by using browser cookies.

---

## Notebook 1: Setup and Environment Installation

Create a new Google Colab notebook, set the runtime to **T4 GPU**, and run the following cells.

### Cell 1: Mount Google Drive
Mount your Google Drive to save the output video clips permanently:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2: Clone the Repository
```bash
!git clone https://github.com/SamurAIGPT/AI-Youtube-Shorts-Generator.git
%cd /content/AI-Youtube-Shorts-Generator
```

### Cell 3: Install GPU-Accelerated & Local Dependencies
```bash
!pip install yt-dlp faster-whisper openai google-genai opencv-python-headless socksio httpx[socks]
```

### Cell 4: Create Configuration (`.env`)
Create a `.env` file containing your OpenCode Zen API key and configuration settings:
```python
env_content = """LLM_PROVIDER=openai
OPENAI_API_KEY=sk-v60mFLUTTwfHOzuU32URqARD6VzWjYb6jod7ao1b1G6Yv0zZi35feoy60ZIrEjL5
OPENAI_BASE_URL=https://opencode.ai/zen/v1
OPENAI_MODEL=nemotron-3-ultra-free
LOCAL_WHISPER_MODEL=base
LOCAL_WHISPER_DEVICE=cuda
LOCAL_OUTPUT_DIR=/content/drive/MyDrive/YoutubeShortsOutput
"""

with open("/content/AI-Youtube-Shorts-Generator/.env", "w") as f:
    f.write(env_content)
print("Created .env configuration file.")
```

### Cell 5: Create Cookie File (`cookies.json`)
Export your YouTube cookies from your browser using an extension (such as "EditThisCookie" or "Get cookies.txt LOCALLY") as a JSON array, paste them here, and save:
```python
import json

# Paste your JSON cookies list inside the brackets
cookies = [
    {
        "domain": ".youtube.com",
        "expirationDate": 1784759771,
        "hostOnly": False,
        "httpOnly": False,
        "name": "ST-3opvp5",
        "path": "/",
        "secure": False,
        "value": "session_logininfo=..."
    }
    # ... paste the rest of your cookies here
]

with open("/content/AI-Youtube-Shorts-Generator/cookies.json", "w") as f:
    json.dump(cookies, f, indent=4)
print("Saved cookies.json file.")
```

### Cell 6: Verify CUDA Device Availability
Verify that PyTorch can utilize the T4 GPU:
```python
import torch
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device Name:", torch.cuda.get_device_name(0))
```

---

## Notebook 2: Run and Generate Shorts

Create a second Colab notebook or add these cells at the bottom of the first notebook.

### Cell 1: Parameters Form
Create form fields to easily customize the parameters for each run:
```python
#@markdown ### Configure YouTube Shorts Generator
VIDEO_URL = "https://www.youtube.com/watch?v=QqvfdGTtFpw" #@param {type:"string"}
NUM_CLIPS = 3 #@param {type:"integer"}
ASPECT_RATIO = "9:16" #@param {type:"string"}
```

### Cell 2: Run Generator
Run the generator using Cloudflare WARP proxy (optional, to avoid rate-limiting on OpenCode Zen API) or directly:
```bash
# Set proxy if WARP is running on port 40000
import os
# os.environ["ALL_PROXY"] = "socks5h://127.0.0.1:40000"

%cd /content/AI-Youtube-Shorts-Generator
!python main.py "{VIDEO_URL}" --mode local --num-clips {NUM_CLIPS} --aspect-ratio "{ASPECT_RATIO}"
```

### Cell 3: Display Output Videos
Locate the rendered `.mp4` outputs inside `/content/drive/MyDrive/YoutubeShortsOutput` and display them directly in the notebook:
```python
import glob
from IPython.display import HTML
from base64 import b64encode

video_files = glob.glob('/content/drive/MyDrive/YoutubeShortsOutput/short_*.mp4')
for video_file in sorted(video_files):
    print(f"Displaying: {video_file}")
    mp4 = open(video_file, 'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    display(HTML(f"""
    <video width="320" height="560" controls>
          <source src="{data_url}" type="video/mp4">
    </video>
    """))
```

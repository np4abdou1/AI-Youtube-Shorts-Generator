#!/bin/bash
# Colab T4 GPU setup script — run once per session
# Usage on remote VM: bash setup_colab.sh

set -e

echo "=== Installing Deno ==="
curl -fsSL https://deno.land/install.sh | sh
export PATH="$HOME/.deno/bin:$PATH"

echo "=== Installing Python deps ==="
pip install -q yt-dlp faster-whisper openai google-genai opencv-python-headless socksio httpx[socks] google-api-python-client google-auth-oauthlib facenet-pytorch --no-deps onnxruntime ctranslate2 av

echo "=== Setup complete ==="

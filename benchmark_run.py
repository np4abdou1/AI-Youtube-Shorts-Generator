import os
import time
import sys

os.chdir("/content/AI-Youtube-Shorts-Generator")

os.environ["OPENAI_MODEL"] = "nemotron-3-ultra-free"
os.environ["OPENAI_API_KEY"] = "sk-v60mFLUTTwfHOzuU32URqARD6VzWjYb6jod7ao1b1G6Yv0zZi35feoy60ZIrEjL5"
os.environ["OPENAI_BASE_URL"] = "https://opencode.ai/zen/v1"
os.environ["LLM_PROVIDER"] = "openai"
os.environ["LOCAL_WHISPER_DEVICE"] = "cuda"
os.environ["LOCAL_OUTPUT_DIR"] = "/content/output_test"

from shorts_generator.local.downloader import download_youtube_local
from shorts_generator.local.transcriber import transcribe_local
from shorts_generator.local.llm import call_local_llm
from shorts_generator.highlights import get_highlights
from shorts_generator.local.clipper import crop_highlights_local

url = "https://www.youtube.com/watch?v=sjn4XaWXgPA&t=153s"
num_clips = 2

print("=== STARTING PIPELINE BENCHMARK ===", flush=True)
t0 = time.time()

# Step 1: Download / Load source video
t_dl_start = time.time()
source_path = "/content/source_sjn4XaWXgPA.mp4"
t_dl_end = time.time()
download_time = t_dl_end - t_dl_start
print(f"⏱️ Step 1 (Source Video Loaded): {download_time:.2f}s", flush=True)

# Step 2: Transcribe
t_tr_start = time.time()
transcript = transcribe_local(source_path)
t_tr_end = time.time()
transcribe_time = t_tr_end - t_tr_start
print(f"⏱️ Step 2 (Transcribe): {transcribe_time:.2f}s", flush=True)

# Step 3: LLM Highlights
t_llm_start = time.time()
highlights_result = get_highlights(transcript, num_clips=num_clips, llm_fn=call_local_llm)
all_highlights = highlights_result.get("highlights", [])
top = sorted(all_highlights, key=lambda h: int(h.get("score", 0)), reverse=True)[:num_clips]
t_llm_end = time.time()
llm_time = t_llm_end - t_llm_start
print(f"⏱️ Step 3 (LLM Highlights): {llm_time:.2f}s", flush=True)

# Step 4: Clipper Rendering
t_clip_start = time.time()
shorts = crop_highlights_local(source_path, top, aspect_ratio="9:16", transcript=transcript, out_dir="/content/output_test")
t_clip_end = time.time()
clip_time = t_clip_end - t_clip_start
print(f"⏱️ Step 4 (GPU Clipping/Rendering): {clip_time:.2f}s", flush=True)

t_total = time.time() - t0
print("\n==========================================", flush=True)
print(f"🏁 TOTAL TIME: {t_total:.2f}s", flush=True)
print(f"📥 Download:  {download_time:.2f}s", flush=True)
print(f"🎙️ Transcribe: {transcribe_time:.2f}s", flush=True)
print(f"🧠 LLM Rank:   {llm_time:.2f}s", flush=True)
print(f"✂️ GPU Render: {clip_time:.2f}s ({clip_time/num_clips:.2f}s per clip)", flush=True)
print("==========================================", flush=True)

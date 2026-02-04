import os
import sys
import tempfile
import json
import argparse
from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm
import whisper
import torch
import yt_dlp

MODEL_SIZE = "small"
CHUNK_MINUTES = 60
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def download_audio(youtube_url):
    print(f"[Download] {youtube_url}")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{tempfile.gettempdir()}/%(id)s.%(ext)s",
        "nocheckcertificate": True,
        "quiet": True,
        "no_warnings": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"]
            }
        }
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        video_id = info.get("id")
        filename = ydl.prepare_filename(info)
    return Path(filename), info.get("title", video_id)

def split_audio(input_file, minutes=60):
    print("[Split] Splitting audio...")
    audio = AudioSegment.from_file(input_file)
    chunk_ms = minutes * 60 * 1000
    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        end = min(start + chunk_ms, len(audio))
        chunk = audio[start:end]
        out_path = Path(tempfile.gettempdir()) / f"chunk_{i}.mp3"
        chunk.export(out_path, format="mp3")
        chunks.append(out_path)
    print(f"[Split] Created {len(chunks)} chunks.")
    return chunks

def transcribe_chunks(chunks, model_size="base", use_gpu=False):
    if use_gpu and not torch.cuda.is_available():
        print("Error: --gpu was set but CUDA is not available on this system.")
        sys.exit(1)
    print(f"[Model] Loading Whisper ({model_size}) ...")
    device = "cuda" if use_gpu else "cpu"
    model = whisper.load_model(model_size, device=device)
    full_text = []
    for i, chunk_path in enumerate(tqdm(chunks, desc="Transcribing chunks")):
        result = model.transcribe(str(chunk_path))
        segment_text = result["text"].strip()
        full_text.append(f"[Part {i+1}] {segment_text}\n")
    return "\n".join(full_text)

def save_transcript(text, title):
    out_path = OUTPUT_DIR / f"{title[:50].replace(' ', '_')}_transcript.txt"
    out_path.write_text(text, encoding="utf-8")
    print(f"[Done] Transcript saved to {out_path.resolve()}")

def main(youtube_url, split_minutes=None, use_gpu=False):
    audio_path, title = download_audio(youtube_url)
    if split_minutes == 0:
        chunks = [audio_path]
    else:
        chunks = split_audio(audio_path, split_minutes or CHUNK_MINUTES)
    transcript_text = transcribe_chunks(chunks, MODEL_SIZE, use_gpu=use_gpu)
    save_transcript(transcript_text, title)

def load_urls_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("download", [])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe YouTube videos using Whisper")
    parser.add_argument("url", nargs="?", help="Single YouTube URL to transcribe")
    parser.add_argument("--src", help="Path to JSON file containing list of YouTube URLs")
    parser.add_argument("--split", type=int, default=0, help="Minutes per audio chunk (0 to disable splitting)")
    parser.add_argument("--gpu", action="store_true", help="Force GPU (CUDA) for transcription")
    
    args = parser.parse_args()
    
    if args.src:
        urls = load_urls_from_json(args.src)
        if not urls:
            print("Error: No URLs found in the 'download' key of the JSON file")
            sys.exit(1)
        print(f"[Batch] Processing {len(urls)} videos from {args.src}")
        for idx, url in enumerate(urls, 1):
            print(f"\n{'='*60}")
            print(f"[{idx}/{len(urls)}] Processing: {url}")
            print(f"{'='*60}")
            main(url, args.split, use_gpu=args.gpu)
    elif args.url:
        main(args.url, args.split, use_gpu=args.gpu)
    else:
        parser.print_help()
        sys.exit(1)

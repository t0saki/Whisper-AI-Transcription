# Whisper AI Video/Audio Transcription Tool

[中文README](README.md) 

To speed up pre-exam review, I developed a tool to quickly convert video/audio into text materials, helping to save time from repeatedly watching videos and improve review efficiency.

## Main Features

- 🎥 Supports various media formats (MP4/AVI/MKV/MOV/MP3/WAV, etc.)
- ⏱️ Automatically generates timestamped transcripts for quick focus on key points
- 📝 One-click generation of plain text transcription files
- 📜 Option to generate SRT subtitle files (with precise timestamps)
- 🚀 Supports GPU-accelerated transcription (automatically detects available devices)

## Quick Start

1. Install dependencies: Install [PyTorch](https://pytorch.org/get-started/locally/), [transformers](https://huggingface.co/docs/transformers/installation), [ffmpeg](https://ffmpeg.org/download.html), [flash-attention](https://github.com/Dao-AILab/flash-attention)

2. Run the command:
```bash
python whisper-v3.py [path to video folder] [options]

# Example - Generate transcript and subtitles
python whisper-v3.py ./lectures -l english -s
```

## Use Cases

✅ Quickly organize lecture highlights before exams  
✅ Create searchable course notes  
✅ Create timestamped study materials  
✅ Extract core content from audio courses

## Notes

- It is recommended to use an NVIDIA graphics card or Apple M series chip for optimal performance
- The AI model will be automatically downloaded on the first run
- Ensure FFmpeg is installed and added to the system path
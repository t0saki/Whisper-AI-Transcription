import torch
import os
import argparse
import subprocess
import tempfile
import datetime
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def format_timestamp(seconds):
    """Convert seconds to SRT format timestamp (HH:MM:SS,mmm)"""
    td = datetime.timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def generate_srt(chunks, output_path):
    """Generate SRT subtitle file from transcription chunks with accumulated timestamps"""
    with open(output_path, "w", encoding="utf-8") as f:
        valid_chunks = []

        # First pass: collect all chunks with valid start times
        for chunk in chunks:
            start_time = chunk.get("timestamp")[0] if type(chunk.get("timestamp")) in [
                list, tuple] else chunk.get("start")
            if start_time is not None:
                valid_chunks.append(chunk)

        # Calculate accumulated time (in case timestamps are relative)
        current_accumulated_time = 0
        accumulated_starts = []
        accumulated_ends = []

        # Second pass: generate SRT entries with proper end times and accumulated timestamps
        for i, chunk in enumerate(valid_chunks):
            # Get original start time
            start_time = chunk.get("timestamp")[0] if type(chunk.get("timestamp")) in [
                list, tuple] else chunk.get("start")

            # Handle end time logic
            if type(chunk.get("timestamp")) in [list, tuple] and len(chunk.get("timestamp")) > 1:
                end_time = chunk.get("timestamp")[1]
            else:
                end_time = chunk.get("end")

            # If end time is missing, set it to the start of the next chunk or start_time + 5s
            if end_time is None:
                if i < len(valid_chunks) - 1:
                    # Use the start time of the next chunk
                    next_start = valid_chunks[i+1].get("timestamp")[0] if type(valid_chunks[i+1].get(
                        "timestamp")) in [list, tuple] else valid_chunks[i+1].get("start")
                    # But limit to maximum 5 seconds from current start
                    end_time = min(next_start, start_time + 5.0)
                else:
                    # For the last chunk, add 5 seconds
                    end_time = start_time + 5.0
            end_time = min(end_time, start_time + 5.0)

            # Calculate accumulated timestamps
            if i == 0:
                accumulated_start = start_time
                accumulated_end = end_time
            else:
                # Check if this chunk starts before the end of the previous chunk
                if start_time < accumulated_ends[-1]:
                    # Overlap - start from where the previous chunk ended
                    accumulated_start = accumulated_ends[-1]
                else:
                    # Gap - maintain the gap from the previous chunk
                    gap = start_time - \
                        accumulated_ends[-1] + accumulated_starts[-1]
                    accumulated_start = accumulated_ends[-1] + \
                        (start_time - gap)

                # Calculate the duration of this chunk
                duration = end_time - start_time
                accumulated_end = accumulated_start + duration

            accumulated_starts.append(accumulated_start)
            accumulated_ends.append(accumulated_end)

            # Format for SRT file
            f.write(f"{i+1}\n")
            f.write(
                f"{format_timestamp(accumulated_start)} --> {format_timestamp(accumulated_end)}\n")
            f.write(f"{chunk['text'].strip()}\n\n")

def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description="Convert videos/audio to transcription")
    parser.add_argument("input_folder", help="Folder containing media files to process")
    parser.add_argument("--language", "-l", default="english", help="Language for transcription (default: english)")
    parser.add_argument("--subtitles", "-s", action="store_true", help="Generate timestamped subtitles in SRT format")
    parser.add_argument("--formats", default=".mp4,.avi,.mkv,.mov,.wmv,.flv,.webm,.mp3,.wav,.ogg,.flac,.m4a", 
                        help="Comma-separated list of file extensions to process (default: common video/audio formats)")
    args = parser.parse_args()

    # 检查 GPU 可用性
    if torch.cuda.is_available():
        device = "cuda:0"
        torch_dtype = torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float32
    else:
        device = "cpu"
        torch_dtype = torch.float32
    
    print(f"Using device: {device}")
    
    # 加载模型
    model_id = "openai/whisper-large-v3-turbo"
    print(f"Loading model {model_id}...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)#, attn_implementation="flash_attention_2")
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=True,
        # return_timestamps="word",
        # chunk_length_s=30,
    )
    
    # 遍历文件夹中的所有支持的媒体文件
    input_folder = args.input_folder
    supported_formats = args.formats.lower().split(",")
    media_files = [f for f in os.listdir(input_folder) 
                  if any(f.lower().endswith(ext) for ext in supported_formats)]
    
    print(f"Found {len(media_files)} media files to process.")
    print(f"Using language: {args.language}")
    
    for i, media_file in enumerate(media_files):
        media_path = os.path.join(input_folder, media_file)
        base_name = os.path.splitext(media_file)[0]
        txt_path = os.path.join(input_folder, f"{base_name}.txt")
        srt_path = os.path.join(input_folder, f"{base_name}.srt") if args.subtitles else None
        
        print(f"[{i+1}/{len(media_files)}] Processing {media_file}...")
        
        # 创建临时WAV文件
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            temp_wav_path = temp_wav.name
        
        try:
            # 提取音频
            print(f"  Extracting audio...")
            subprocess.run([
                'ffmpeg', '-i', media_path, '-vn', temp_wav_path, 
                '-y', '-loglevel', 'error'
            ], check=True)
            
            # 转录音频
            print(f"  Transcribing audio...")
            result = pipe(temp_wav_path, generate_kwargs={"language": args.language})
            
            # 保存结果到TXT文件
            with open(txt_path, "w", encoding="utf-8") as f:
                if "chunks" in result:
                    for chunk in result["chunks"]:
                        f.write(chunk["text"].strip() + "\n")
                else:
                    for line in result["text"].splitlines():
                        f.write(line + "\n")
            
            print(f"  Transcription saved to {txt_path}")
            
            # 如果需要，生成SRT字幕文件
            if args.subtitles and "chunks" in result:
                generate_srt(result["chunks"], srt_path)
                print(f"  SRT subtitles saved to {srt_path}")
            
        except Exception as e:
            print(f"  Error processing {media_file}: {e}")
        
        finally:
            # 删除临时WAV文件
            if os.path.exists(temp_wav_path):
                os.unlink(temp_wav_path)
    
    print("All processing complete.")

if __name__ == "__main__":
    main()
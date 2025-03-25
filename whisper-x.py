import os
import argparse
import time
import logging
from datetime import datetime, timedelta
import whisperx
import torch
import json


def setup_logger():
    """Set up logger to output to both console and log file"""
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Create a logger
    logger = logging.getLogger("whisperx_transcribe")
    logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Create console handler for stdout output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Create file handler with timestamp in filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(
        logs_dir, f"whisperx_cli_{timestamp}.log")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f"Log file: {log_file_path}")

    return logger


def get_media_duration(media_path, logger):
    """Get the duration of a media file in seconds using ffprobe"""
    try:
        import subprocess
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            media_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        duration = float(result.stdout.strip())
        return duration
    except (ValueError, IndexError, subprocess.SubprocessError):
        logger.warning(f"  Warning: Couldn't determine media duration")
        return None


def process_with_whisperx(media_path, output_dir, model_name, language=None, batch_size=16, compute_type="float16", logger=None):
    """Process media using WhisperX API"""
    if logger:
        logger.info(f"  Using WhisperX API to transcribe media")
    
    # Determine device (use CUDA if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if logger:
        logger.info(f"  Using device: {device}")
    
    # Load audio
    audio = whisperx.load_audio(media_path)
    
    # Load model
    model = whisperx.load_model(model_name, device, compute_type=compute_type)
    
    # Transcribe with WhisperX
    result = model.transcribe(audio, batch_size=batch_size, language=language)
    detected_language = result["language"]
    if logger and language is None:
        logger.info(f"  Detected language: {detected_language}")
    
    logger.info(f"  Aligning segments")
    # Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=detected_language, device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
    # Clean up to free memory
    del model
    del model_a
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # Save results
    base_name = os.path.splitext(os.path.basename(media_path))[0]
    
    # Save to JSON file
    json_path = os.path.join(output_dir, f"{base_name}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
    # Create TSV file
    tsv_path = os.path.join(output_dir, f"{base_name}.tsv")
    with open(tsv_path, 'w', encoding='utf-8') as f:
        f.write("start\tend\ttext\n")
        for segment in result["segments"]:
            start_ms = int(segment["start"] * 1000)
            end_ms = int(segment["end"] * 1000)
            text = segment["text"].strip()
            f.write(f"{start_ms}\t{end_ms}\t{text}\n")
    
    # Create TXT file
    txt_path = os.path.join(output_dir, f"{base_name}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        for segment in result["segments"]:
            f.write(f"{segment['text'].strip()}\n")
    
    # Create VTT file
    vtt_path = os.path.join(output_dir, f"{base_name}.vtt")
    with open(vtt_path, 'w', encoding='utf-8') as f:
        f.write("WEBVTT\n\n")
        for i, segment in enumerate(result["segments"]):
            start_time = format_time_vtt(segment["start"])
            end_time = format_time_vtt(segment["end"])
            f.write(f"{i+1}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{segment['text'].strip()}\n\n")
    
    return {
        "result": result,
        "detected_language": detected_language,
        "tsv_path": tsv_path,
        "vtt_path": vtt_path,
        "txt_path": txt_path
    }


def format_time_vtt(seconds):
    """Format seconds to VTT time format (HH:MM:SS.mmm)"""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}".replace('.', ',')


def convert_tsv_to_srt(tsv_path, srt_path, logger):
    """Convert WhisperX TSV output to SRT format"""
    logger.info(f"  Converting TSV to SRT format")
    
    try:
        with open(tsv_path, 'r', encoding='utf-8') as tsv_file:
            lines = tsv_file.readlines()
        
        # Skip header line if it exists
        if lines and lines[0].startswith('start\t'):
            lines = lines[1:]
        
        with open(srt_path, 'w', encoding='utf-8') as srt_file:
            for i, line in enumerate(lines):
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    start_ms = int(parts[0])
                    end_ms = int(parts[1])
                    text = parts[2]
                    
                    # Convert milliseconds to SRT format (HH:MM:SS,mmm)
                    start_time = format_srt_time(start_ms)
                    end_time = format_srt_time(end_ms)
                    
                    # Write SRT entry
                    srt_file.write(f"{i+1}\n")
                    srt_file.write(f"{start_time} --> {end_time}\n")
                    srt_file.write(f"{text}\n\n")
        
        logger.info(f"  SRT file saved to {srt_path}")
        return True
    
    except Exception as e:
        logger.error(f"  Error converting TSV to SRT: {e}")
        return False


def format_srt_time(milliseconds):
    """Format milliseconds to SRT time format (HH:MM:SS,mmm)"""
    seconds, milliseconds = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def format_time(seconds):
    """Format seconds into a human-readable time string"""
    return str(timedelta(seconds=int(seconds)))


def main():
    # Set up logger
    logger = setup_logger()

    # Command line argument parsing
    parser = argparse.ArgumentParser(
        description="Convert videos/audio to transcription using WhisperX")
    parser.add_argument(
        "input_folder", help="Folder containing media files to process")
    parser.add_argument("--language", "-l", default=None,
                        help="Language for transcription (default: auto-detect)")
    parser.add_argument("--auto-detect", "-a", action="store_true",
                        help="Auto-detect language (default behavior if --language not specified)")
    parser.add_argument("--batch-size", "-b", type=int, default=16,
                        help="Batch size for processing (default: 16, reduce if low on GPU memory)")
    parser.add_argument("--compute-type", "-c", default="int8",
                        help="Compute type (default: int8, use int8 if low on GPU memory)")
    parser.add_argument("--formats", default=".mp4,.avi,.mkv,.mov,.wmv,.flv,.webm,.mp3,.wav,.ogg,.flac,.m4a",
                        help="Comma-separated list of file extensions to process (default: common video/audio formats)")
    parser.add_argument("--model", default="large-v2",
                        help="WhisperX model to use (default: large-v2)")
    args = parser.parse_args()

    # Traverse folder for supported media files
    input_folder = args.input_folder
    supported_formats = args.formats.lower().split(",")
    media_files = [f for f in os.listdir(input_folder)
                   if any(f.lower().endswith(ext) for ext in supported_formats)]
    logger.info(f"Found {len(media_files)} media files to process.")

    # Language setting
    if args.language and args.auto_detect:
        logger.info(
            "Both language and auto-detect specified. Auto-detect will be ignored.")
        language_mode = f"Specified language: {args.language}"
    elif args.language:
        language_mode = f"Specified language: {args.language}"
    else:
        language_mode = "Automatic language detection"
    logger.info(f"Language mode: {language_mode}")
    logger.info(f"Using model: {args.model}")
    logger.info(f"Compute type: {args.compute_type}")
    logger.info(f"Batch size: {args.batch_size}")

    # Track total times
    total_media_duration = 0
    total_processing_time = 0

    for i, media_file in enumerate(media_files):
        media_path = os.path.join(input_folder, media_file)
        base_name = os.path.splitext(media_file)[0]
        logger.info(f"[{i+1}/{len(media_files)}] Processing {media_file}...")

        # Get media duration
        media_duration = get_media_duration(media_path, logger)
        if media_duration:
            logger.info(
                f"  Media duration: {format_time(media_duration)} ({media_duration:.2f} seconds)")
            total_media_duration += media_duration

        # Start timing the processing
        start_time = time.time()

        # Create output directory if it doesn't exist
        output_dir = os.path.join(input_folder, "whisperx_output")
        os.makedirs(output_dir, exist_ok=True)

        # Process the media with WhisperX
        logger.info(f"  Transcribing with WhisperX...")
        try:
            # Process with WhisperX API
            result = process_with_whisperx(
                media_path,
                output_dir,
                args.model,
                args.language,
                args.batch_size,
                args.compute_type,
                logger
            )
            
            # Convert TSV to SRT
            tsv_path = result["tsv_path"]
            srt_path = os.path.join(input_folder, f"{base_name}.srt")
            
            if os.path.exists(tsv_path):
                convert_tsv_to_srt(tsv_path, srt_path, logger)
            else:
                logger.warning(f"  TSV file not found at {tsv_path}")

            # Copy files to main folder if needed
            for file_type in ["txt", "vtt"]:
                source_path = result[f"{file_type}_path"]
                dest_path = os.path.join(input_folder, f"{base_name}.{file_type}")
                
                if os.path.exists(source_path):
                    with open(source_path, 'r', encoding='utf-8') as src:
                        with open(dest_path, 'w', encoding='utf-8') as dst:
                            dst.write(src.read())
                    logger.info(f"  {file_type.upper()} file saved to {dest_path}")

        except Exception as e:
            logger.error(f"  Error processing {media_file}: {e}")

        # End timing and report
        end_time = time.time()
        processing_time = end_time - start_time
        total_processing_time += processing_time

        logger.info(
            f"  Processing completed in {format_time(processing_time)} ({processing_time:.2f} seconds)")

        if media_duration:
            speed_ratio = media_duration / processing_time
            logger.info(f"  Processing speed: {speed_ratio:.2f}x real-time")

    # Report totals
    if total_media_duration > 0 and total_processing_time > 0:
        logger.info("\nOverall statistics:")
        logger.info(
            f"Total media duration: {format_time(total_media_duration)} ({total_media_duration:.2f} seconds)")
        logger.info(
            f"Total processing time: {format_time(total_processing_time)} ({total_processing_time:.2f} seconds)")
        overall_speed = total_media_duration / total_processing_time
        logger.info(
            f"Average processing speed: {overall_speed:.2f}x real-time")

    logger.info("All processing complete.")


if __name__ == "__main__":
    main()
import os
import argparse
import subprocess
import tempfile
import shutil
import time
import logging
from datetime import datetime, timedelta


def setup_logger():
    """Set up logger to output to both console and log file"""
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Create a logger
    logger = logging.getLogger("whisper_transcribe")
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
        logs_dir, f"whisper_cli_{timestamp}.log")
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
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        media_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    try:
        duration = float(result.stdout.strip())
        return duration
    except (ValueError, IndexError):
        logger.warning(f"  Warning: Couldn't determine media duration")
        return None


def run_whisper(audio_path, output_dir, model, language=None, output_format="txt", verbose=False):
    """Run whisper command with the given parameters and return the process result"""
    whisper_cmd = [
        'whisper', audio_path,
        '--model', model,
        '--output_dir', output_dir,
        '--output_format', output_format,
        '--verbose', str(verbose)
    ]
    if language:
        whisper_cmd.extend(['--language', language])
    return subprocess.run(
        whisper_cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )


def copy_output_file(temp_dir, temp_basename, dest_dir, dest_basename, ext, logger):
    """Copy output file from temp directory to destination"""
    temp_path = os.path.join(temp_dir, f"{temp_basename}{ext}")
    dest_path = os.path.join(dest_dir, f"{dest_basename}{ext}")
    if os.path.exists(temp_path):
        shutil.copy2(temp_path, dest_path)
        logger.info(f"  {ext.upper()[1:]} file saved to {dest_path}")
        return True
    return False


def format_time(seconds):
    """Format seconds into a human-readable time string"""
    return str(timedelta(seconds=int(seconds)))


def main():
    # Set up logger
    logger = setup_logger()

    # Command line argument parsing
    parser = argparse.ArgumentParser(
        description="Convert videos/audio to transcription")
    parser.add_argument(
        "input_folder", help="Folder containing media files to process")
    parser.add_argument("--language", "-l", default=None,
                        help="Language for transcription (default: auto-detect)")
    parser.add_argument("--auto-detect", "-a", action="store_true",
                        help="Auto-detect language (default behavior if --language not specified)")
    parser.add_argument("--subtitles", "-s", action="store_true",
                        help="Generate timestamped subtitles in SRT format")
    parser.add_argument("--formats", default=".mp4,.avi,.mkv,.mov,.wmv,.flv,.webm,.mp3,.wav,.ogg,.flac,.m4a",
                        help="Comma-separated list of file extensions to process (default: common video/audio formats)")
    parser.add_argument("--model", default="large-v3-turbo",
                        help="Whisper model to use (default: large-v3-turbo)")
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

    audio_formats = ['.mp3', '.wav', '.ogg', '.flac', '.m4a']

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

        # Create temporary directory for whisper output
        with tempfile.TemporaryDirectory() as temp_output_dir:
            # For video files, extract audio first
            audio_path = media_path
            is_audio_file = any(media_file.lower().endswith(ext)
                                for ext in audio_formats)
            if not is_audio_file:
                logger.info(f"  Extracting audio from video...")
                temp_wav_path = os.path.join(temp_output_dir, "temp_audio.wav")
                try:
                    subprocess.run([
                        'ffmpeg', '-i', media_path, '-vn', temp_wav_path,
                        '-y', '-loglevel', 'error'
                    ], check=True)
                    audio_path = temp_wav_path
                except subprocess.CalledProcessError as e:
                    logger.error(
                        f"  Error extracting audio from {media_file}: {e}")
                    continue

            # Process the audio with whisper
            logger.info(f"  Transcribing audio...")
            output_format = "all" if args.subtitles else "txt"
            audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
            try:
                # Run whisper command
                result = run_whisper(
                    audio_path,
                    temp_output_dir,
                    args.model,
                    args.language,
                    output_format
                )

                # If using auto-detect, log detected language
                if not args.language:
                    for line in result.stdout.splitlines():
                        if "Detected language:" in line:
                            logger.info(f"  {line.strip()}")
                            break

                # Copy output files to destination
                copy_output_file(temp_output_dir, audio_basename,
                                 input_folder, base_name, ".txt", logger)

                # Copy SRT file if needed
                if args.subtitles and not copy_output_file(temp_output_dir, audio_basename, input_folder, base_name, ".srt", logger):
                    logger.warning(
                        f"  WARNING: SRT file was not generated by whisper")

            except subprocess.CalledProcessError as e:
                logger.error(f"  Error processing {media_file}:")
                logger.error(f"  Error output: {e.stderr}")

                # Try again with txt format only if 'all' failed
                if args.subtitles and output_format == "all":
                    logger.info(f"  Retrying with txt format only...")
                    try:
                        run_whisper(audio_path, temp_output_dir,
                                    args.model, args.language, "txt")
                        copy_output_file(
                            temp_output_dir, audio_basename, input_folder, base_name, ".txt", logger)
                    except subprocess.CalledProcessError as e2:
                        logger.error(
                            f"  Failed to process file even with txt format only: {e2}")

            except Exception as e:
                logger.error(
                    f"  Unexpected error processing {media_file}: {e}")

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

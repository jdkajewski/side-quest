import whisper
import os
import tempfile
import subprocess

# === CONFIGURATION ===
WHISPER_MODEL = "base"  # Whisper model size (tiny, base, small, medium, large)
OUTPUT_DIR = "transcriptions"  # Folder for transcriptions

def extract_audio_with_ffmpeg(video_path):
    """Extract audio from video file using ffmpeg and return path to temporary WAV file"""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        temp_path = temp_audio.name
    
    # Use ffmpeg to extract audio
    command = [
        'ffmpeg',
        '-i', video_path,
        '-ac', '1',
        '-ar', '16000',
        '-acodec', 'pcm_s16le',
        '-y',  # Overwrite without asking
        temp_path
    ]
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        os.remove(temp_path)
        raise RuntimeError(f"FFmpeg error: {e.stderr.decode()}") from e
    
    return temp_path

def transcribe_video(video_path, output_dir):
    """Transcribe a video file and save results to text file"""
    # Get relative path and base filename without extension
    rel_path = os.path.relpath(video_path)
    base_name = os.path.splitext(rel_path)[0]
    
    # Create output path maintaining directory structure
    txt_filename = os.path.join(output_dir, f"{base_name}.txt")
    os.makedirs(os.path.dirname(txt_filename), exist_ok=True)
    
    print(f"Processing: {video_path}")
    
    # Extract audio to temporary file
    audio_path = extract_audio_with_ffmpeg(video_path)
    
    try:
        # Load Whisper model
        print("Loading Whisper model...")
        model = whisper.load_model(WHISPER_MODEL, device="cpu")
        
        # Transcribe
        print("Transcribing...")
        result = model.transcribe(audio_path, fp16=False, language="en")
        
        # Save transcription
        with open(txt_filename, "w") as f:
            f.write(result["text"])
        
        print(f"Saved transcription to {txt_filename}")
    finally:
        # Clean up temporary audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)

def find_and_process_mov_files(root_dir=".", output_dir=OUTPUT_DIR):
    """Recursively find and process all .mov files in directory tree"""
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.mov'):
                video_path = os.path.join(dirpath, filename)
                transcribe_video(video_path, output_dir)

# === MAIN ===
if __name__ == "__main__":
    # Create output folder if needed
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Looking for .mov files in current directory and subdirectories...")
    find_and_process_mov_files()
    print("Done processing all .mov files")
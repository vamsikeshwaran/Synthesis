import subprocess
import os
import shutil
from moviepy import VideoFileClip, AudioFileClip
from pyannote.audio import Pipeline
import warnings
from google import genai
import argparse
from gradio_client import Client, handle_file
import time


warnings.filterwarnings('ignore')


def separate_vocals_and_other(input_file, output_dir="separated_audio"):
    try:
        subprocess.run(["demucs", "--help"], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise EnvironmentError(
            "Demucs is not installed. Install it with 'pip install demucs'.")

    try:
        subprocess.run(["demucs", "-o", output_dir, input_file], check=True)
        print("Audio separation complete.")
    except subprocess.CalledProcessError as e:
        print("Error during audio separation:", e)
        return None

    separated_folder = os.path.join(
        output_dir, "htdemucs", os.path.splitext(
            os.path.basename(input_file))[0]
    )

    if not os.path.exists(separated_folder):
        print(f"Separation folder not found: {separated_folder}")
        return None

    vocal_file = "vocals.wav"
    src_vocal_path = os.path.join(separated_folder, vocal_file)
    refined_vocal_path = os.path.join(
        output_dir, f"{os.path.splitext(os.path.basename(input_file))[0]}_refined.wav")

    if os.path.exists(src_vocal_path):
        shutil.move(src_vocal_path, refined_vocal_path)
        print(f"Refined vocal file saved: {refined_vocal_path}")
    else:
        print(f"File not found: {vocal_file}")
        refined_vocal_path = None

    other_file = "other.wav"
    src_other_path = os.path.join(separated_folder, other_file)
    refined_other_path = os.path.join(
        output_dir, f"{os.path.splitext(os.path.basename(input_file))[0]}_back_music.wav")

    if os.path.exists(src_other_path):
        shutil.move(src_other_path, refined_other_path)
        print(f"Refined other file saved: {refined_other_path}")
    else:
        print(f"File not found: {other_file}")
        refined_other_path = None

    htdemucs_dir = os.path.join(output_dir, "htdemucs")
    if os.path.exists(htdemucs_dir):
        shutil.rmtree(htdemucs_dir)
        print(f"Deleted htdemucs directory: {htdemucs_dir}")

    return refined_vocal_path, refined_other_path


def extract_audio(video_path, output_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_path)
    video.close()
    audio.close()

def upload_with_retry(client, file_path, retries=3, delay=5):
    for attempt in range(retries):
        try:
            return client.files.upload(file=file_path)
        except Exception as e:
            print(f"Upload attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (2 ** attempt))
            else:
                raise

def generate_audio_from_tanglish(refined_vocal_path, tanglish_text, english_text, output_folder):
    print(tanglish_text)
    output_filename = f"{os.path.basename(refined_vocal_path).split('.')[0]}_clonedvoice.wav"

    command = [
        "f5-tts_infer-cli",
        "--model", "F5TTS_v1_Base",
        "--ref_audio", refined_vocal_path,
        "--ref_text", tanglish_text,
        "--gen_text", english_text,
        "--output_dir", output_folder,
        "--output_file", output_filename
    ]

    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("TTS Output:\n", result.stdout)
        if result.stderr:
            print("TTS Warnings/Errors:\n", result.stderr)
        print(f"Generated TTS file: {os.path.join(output_folder, output_filename)}")
    except subprocess.CalledProcessError as e:
        print("An error occurred while generating TTS audio:")
        print(e.stderr)
    
def generate_transcript(refined_vocal_path, output_folder, lang):
    client = genai.Client(api_key="AIzaSyBgBZBciAWvwVnggYJkVPQv0dCMKGx872Y")
    tanglish = genai.Client(api_key="AIzaSyBgBZBciAWvwVnggYJkVPQv0dCMKGx872Y")
    # try:
    #     myfile = upload_with_retry(client, refined_vocal_path)
    # except Exception as e:
    #     print(f"Error uploading file: {e}")
    #     return
    try:
        cli = Client("ai4bharat/indic-conformer")
        result = cli.predict(
            input_audio=handle_file(refined_vocal_path),
            target_language=lang,
            api_name="/asr"
        )
        response = result.strip()

        response1 = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"translate this to english {response} only 1 response and nothing else",
        )
        tanglish_text = tanglish.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"translate this to tanglish {response} only 1 response and nothing else",
        )
        transcript_text = f"Tamil :{response}\nEnglish :{response1.text}\nTanglish: {tanglish_text.text}"

        transcript_filename = os.path.join(
            output_folder, f"{os.path.basename(refined_vocal_path).split('.')[0]}_transcript.txt")

        os.makedirs(output_folder, exist_ok=True)

        with open(transcript_filename, 'w') as f:
            f.write(transcript_text)

        print(f"Transcript saved: {transcript_filename}")

        generate_audio_from_tanglish(refined_vocal_path, tanglish_text.text, response1.text, output_folder)
    except Exception as e:
        print(f"Error generating transcript: {e}")

def perform_diarization(audio_path):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token="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")
    diarization = pipeline(audio_path)
    speaker_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segment = {
            'speaker': speaker,
            'start': turn.start,
            'end': turn.end
        }
        speaker_segments.append(segment)

    return speaker_segments


def cut_segments(video_path, audio_path, speaker_segments, output_folder="segments", lang=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)

    try:
        combined_segments = []
        current_segment = speaker_segments[0]

        for i in range(1, len(speaker_segments)):
            if speaker_segments[i]['speaker'] == current_segment['speaker']:
                current_segment['end'] = speaker_segments[i]['end']
            else:
                combined_segments.append(current_segment)
                current_segment = speaker_segments[i]
        combined_segments.append(current_segment)

        for i, segment in enumerate(combined_segments):
            start_time = segment['start']
            end_time = segment['end']
            speaker = segment['speaker']

            video_segment = video.subclipped(start_time, end_time)
            video_output_path = os.path.join(
                output_folder, f"{speaker}_segment_{i}.mp4")
            video_segment.write_videofile(video_output_path)

            audio_segment = audio.subclipped(start_time, end_time)
            audio_output_path = os.path.join(
                output_folder, f"{speaker}_segment_{i}.wav")
            audio_segment.write_audiofile(audio_output_path)

            refined_vocal_path, refined_other_path = separate_vocals_and_other(
                audio_output_path, output_dir=output_folder)
            if refined_vocal_path:
                print(f"Refined vocal file saved: {refined_vocal_path}")
                generate_transcript(refined_vocal_path, output_folder, lang)
            if refined_other_path:
                print(f"Refined back music file saved: {refined_other_path}")
    finally:
        video.close()
        audio.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video files for speaker diarization.")
    parser.add_argument("--video_file", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--lang", type=str, required=True, help="Language of the video.")
    args = parser.parse_args()
    video_file = args.video_file
    lang = args.lang
    audio_file = "output_audio.wav"
    try:
        extract_audio(video_file, audio_file)
        speaker_segments = perform_diarization(audio_file)
        print("Speaker segments:", speaker_segments)
        cut_segments(video_file, audio_file, speaker_segments,
                     output_folder="segments", lang=lang)
    except Exception as e:
        print(f"Error: {e}")

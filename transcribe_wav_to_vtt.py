import os
import json
import sys
import torch
import reazonspeech as rs
import pydub
import os
import argparse

from dataclasses import dataclass


if not torch.cuda.is_available():
    print("Error: CUDA is not available. Please ensure you have a compatible GPU and the appropriate drivers installed.")
    sys.exit(1)


@dataclass
class Caption:
    start_seconds: float
    end_seconds: float
    text: str
    wav_file: str = None


class VTTWriter:
    ext = 'vtt'

    @staticmethod
    def _format_time(seconds):
        h = int(seconds / 3600)
        m = int(seconds / 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return "%02i:%02i:%02i.%03i" % (h, m, s, ms)

    def header(self, file):
        file.write("WEBVTT\n\n")

    def caption(self, file, caption):
        start = self._format_time(caption.start_seconds)
        end = self._format_time(caption.end_seconds)
        file.write("%s --> %s\n%s\n\n" % (start, end, caption.text))
        print("%s --> %s\n%s\n\n" % (start, end, caption.text))

def split_audio_file(input_file, output_dir, chunk_length_seconds):
    audio = pydub.AudioSegment.from_wav(input_file)
    audio_length_ms = len(audio)
    chunk_length_ms = chunk_length_seconds * 1000
    output_files = []

    for i, start_ms in enumerate(range(0, audio_length_ms, chunk_length_ms)):
        end_ms = start_ms + chunk_length_ms
        chunk = audio[start_ms:end_ms]
        output_file = os.path.join(output_dir, f"speech-{i + 1:03d}.wav")
        chunk.export(output_file, format="wav")
        output_files.append(output_file)

    return output_files


def get_wav_files(input_file, output_dir, chunk_length_seconds):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    wav_files = split_audio_file(input_file, output_dir, chunk_length_seconds)
    wav_files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".wav")])

    return wav_files


def show_progress(current, total):
    progress = (current / total) * 100
    print(f"Processing file {current}/{total}: {progress:.2f}% complete")


def transcribe_and_save_vtt(wav_files, output_vtt_path, resume_file):
    all_captions = []
    total_seconds = 0

    if os.path.exists(resume_file):
        with open(resume_file, "r") as f:
            resume_data = json.load(f)
            total_seconds = resume_data["total_seconds"]
            all_captions = [Caption(**caption) for caption in resume_data["captions"]]

    for idx, wav_file in enumerate(wav_files):
        if any(caption.wav_file == wav_file for caption in all_captions):
            continue

        # Show progress
        show_progress(idx + 1, len(wav_files))

        captions = list(rs.transcribe(wav_file))

        for caption in captions:
            caption.start_seconds += total_seconds
            caption.end_seconds += total_seconds
            caption.wav_file = wav_file

        total_seconds += captions[-1].end_seconds
        all_captions.extend(captions)

        # Save resume data
        resume_data = {
            "total_seconds": total_seconds,
            "captions": [caption.__dict__ for caption in all_captions]
        }
        with open(resume_file, "w") as f:
            json.dump(resume_data, f)

    vtt_writer = VTTWriter()

    with open(output_vtt_path, 'w', encoding='utf-8') as f:
        vtt_writer.header(f)

        for caption in all_captions:
            vtt_writer.caption(f, caption)

    # Remove resume file
    if os.path.exists(resume_file):
        os.remove(resume_file)

def main():
    original_wav_file = "dist.wav"
    output_dir = "/tmp/split_wavs"
    chunk_length_seconds = 600

    wav_files = get_wav_files(original_wav_file, output_dir, chunk_length_seconds)

    output_vtt_path = "output.vtt"
    resume_file = "./tmp/resume.json"

    transcribe_and_save_vtt(wav_files, output_vtt_path, resume_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe WAV to VTT")
    parser.add_argument("--input", required=True, help="Path to the input WAV file")
    parser.add_argument("--output", required=True, help="Path to the output VTT file")
    parser.add_argument("--resume", default="resume.json", help="Path to the resume file (default: resume.json)")
    parser.add_argument("--chunk_length", type=int, default=600, help="Length of chunks to split WAV file (in seconds, default: 600)")

    args = parser.parse_args()

    wav_files = get_wav_files(args.input, "split_wavs", args.chunk_length)

    transcribe_and_save_vtt(wav_files, args.output, args.resume)

import unittest
import os
from unittest.mock import patch, MagicMock
from transcribe_wav_to_vtt import get_wav_files, Caption, transcribe_and_save_vtt, split_audio_file

class TestTranscribeWavToVtt(unittest.TestCase):
    @patch("transcribe_wav_to_vtt.split_audio_file")
    def test_get_wav_files(self, mock_split_audio_file):
        original_wav_file = "test_input.wav"
        output_dir = "/tmp/test_split_wavs"
        chunk_length_seconds = 2

        mock_split_audio_file.return_value = ["fake_wav_file1.wav", "fake_wav_file2.wav"]
        wav_files = get_wav_files(original_wav_file, output_dir, chunk_length_seconds)

        mock_split_audio_file.assert_called_once_with(original_wav_file, output_dir, chunk_length_seconds)

    @patch("reazonspeech.transcribe", return_value=[Caption(0.0, 1.0, "This is a test.")])
    def test_transcribe_and_save_vtt(self, mock_transcribe):
        wav_files = ["test_input.wav"]
        output_vtt_path = "test_output.vtt"
        resume_file = "test_resume.json"

        transcribe_and_save_vtt(wav_files, output_vtt_path, resume_file)

        self.assertTrue(os.path.exists(output_vtt_path))

        with open(output_vtt_path, "r") as f:
            content = f.read()
            self.assertIn("WEBVTT", content)
            self.assertIn("This is a test.", content)

        os.remove(output_vtt_path)

        if os.path.exists(resume_file):
            os.remove(resume_file)

if __name__ == "__main__":
    unittest.main()

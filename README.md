# docker-reazon-speech

```
# python3 transcribe_wav_to_vtt.py -h
usage: transcribe_wav_to_vtt.py [-h] --input INPUT --output OUTPUT [--resume RESUME] [--chunk_length CHUNK_LENGTH]

Transcribe WAV to VTT

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         Path to the input WAV file
  --output OUTPUT       Path to the output VTT file
  --resume RESUME       Path to the resume file (default: resume.json)
  --chunk_length CHUNK_LENGTH
                        Length of chunks to split WAV file (in seconds, default: 600)
```
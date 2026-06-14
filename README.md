# Music Identification Using Spectrogram Matching

A Shazam-inspired music identification system implemented in Python.

The project identifies unknown audio clips by comparing their spectrogram representations against a database of known songs.

## Features

- Short-Time Fourier Transform (STFT)
- Logarithmic spectrogram representation
- Frequency band filtering (300–3400 Hz)
- Spectrogram normalization
- Sliding-window matching
- Correlation-based similarity scoring
- Top-1 and Top-5 retrieval evaluation

## Method

1. Compute STFT from the audio signal.
2. Convert amplitudes to logarithmic scale.
3. Keep only the 300–3400 Hz frequency range.
4. Normalize spectrogram channels.
5. Slide a 5-second window through reference songs.
6. Compute similarity using correlation.
7. Select the highest scoring match.

## Results

The final implementation was evaluated on the provided validation dataset.

| Metric | Accuracy |
|----------|----------|
| Top-1 | 94% |
| Top-5 | 100% |

## Technologies

- Python
- NumPy
- SciPy
- Matplotlib

## Example Pipeline

Audio Signal

→ STFT

→ Spectrogram

→ Normalization

→ Sliding Window Matching

→ Song Identification

## Project Structure

```text
protocol.pdf
shazaam-audio-fingerprinting.py
README.md
```

## Dataset

The dataset was provided as part of the ISS 2025/26 assignment at FIT VUT.

It consists of:

- 706 known 10-second audio recordings
- 50 validation 5-second clips
- 50 evaluation 5-second clips

The dataset is not included in this repository.

## Report

Detailed analysis and evaluation are available in the project report.

## Author

Boris Dráb

FIT VUT Brno
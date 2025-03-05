import os
import librosa
import numpy as np
from tqdm import tqdm



# ====== MODEL SETTINGS FROM VOCODERMODEL.PY ======
SAMPLING_RATE = 48000  # Ensure 48kHz to match your model
NUM_MELS = 100  # Your model's mel-channel count
N_FFT = 2048  # FFT window size
HOP_SIZE = 600  # Derived from upsampling structure
WIN_SIZE = 2048  # Window size for STFT
FMIN = 0  # Lower frequency limit
FMAX = 11025  # Upper frequency limit (half of 48kHz)


INPUT_WAV_DIR = "C:/Users/Phoen/source/repos/StreamSpeak/New folder/dev-clean/wavs"  # Change this to your dataset path
OUTPUT_MEL_DIR = "./mels"

# Create output directory if not exists
os.makedirs(OUTPUT_MEL_DIR, exist_ok=True)

def extract_mel(wav_path):
    """Load an audio file and extract its log-mel spectrogram."""
    wav, sr = librosa.load(wav_path, sr=SAMPLING_RATE)

    # Convert to mel spectrogram (raw power, no dB conversion)
    mel_spec = librosa.feature.melspectrogram(
        y=wav, sr=SAMPLING_RATE, n_fft=N_FFT, hop_length=HOP_SIZE, win_length=WIN_SIZE,
        fmin=FMIN, fmax=FMAX, n_mels=NUM_MELS, power=1.0
    )

    # Convert power to decibels and normalize to [0, 1]
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to decibels
    mel_spec_db = (mel_spec_db + 100) / 100  # Normalize to [0, 1]

    # Ensure mel spectrogram has the correct shape
    target_length = (wav.shape[0] // HOP_SIZE)  # Match hop size
    if mel_spec_db.shape[1] < target_length:
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, target_length - mel_spec_db.shape[1])), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :target_length]


    return mel_spec_db

# Process all WAV files recursively
wav_files = [os.path.join(root, file) for root, _, files in os.walk(INPUT_WAV_DIR) for file in files if file.endswith(".wav")]

for wav_file in tqdm(wav_files, desc="Processing WAV files"):

    mel = extract_mel(wav_file)

    mel = (mel - mel.min()) / (np.clip(mel.max() - mel.min(), a_min=1e-5, a_max=None)) #normalize mel spectogram for training same as infering.

    relative_path = os.path.relpath(wav_file, INPUT_WAV_DIR)
    mel_output_path = os.path.join(OUTPUT_MEL_DIR, relative_path).replace('.wav', '.npy')
    
    os.makedirs(os.path.dirname(mel_output_path), exist_ok=True)
    np.save(mel_output_path, mel)

print(f"Preprocessing complete. Mel spectrograms saved under '{OUTPUT_MEL_DIR}' following the original directory structure.")
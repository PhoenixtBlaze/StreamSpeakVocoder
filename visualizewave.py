import torch
import numpy as np
import matplotlib.pyplot as plt
import torchaudio
from VocoderInfer import infer  # Ensure this is your correct inference function

# Load a test mel spectrogram (replace with an actual mel file)
test_mel_path = "p374_236.npy"  # Ensure this file exists and is valid
mel_spectrogram = np.load(test_mel_path)

# Generate audio using the inference model
output_path = "generated_audio.wav"
audio = infer(mel_spectrogram, output_path)

# Ensure audio is in a correct format
if isinstance(audio, torch.Tensor):
    audio = audio.cpu().numpy()  # Convert from PyTorch tensor to NumPy
elif not isinstance(audio, np.ndarray):
    raise ValueError(f" Expected numpy array but got {type(audio)}")

# Check if the audio is empty or has unusual values
if len(audio) == 0:
    raise ValueError(" Generated audio is empty! Check VocoderInfer.py")
if np.all(audio == 0):
    raise ValueError(" Warning: Generated audio is completely silent!")

# Save & Load the audio for visualization
torchaudio.save("temp_waveform_check.wav", torch.tensor(audio).unsqueeze(0), 48000)
wav, sr = torchaudio.load("temp_waveform_check.wav")

# Plot the waveform
plt.figure(figsize=(10, 4))
plt.plot(wav.numpy().squeeze(), alpha=0.7)
plt.title(f"Generated Waveform - {sr}Hz")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()
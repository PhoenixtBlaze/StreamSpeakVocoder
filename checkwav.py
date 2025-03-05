import torchaudio
import matplotlib.pyplot as plt
import numpy as np

#Path to the test audio file (Modify this to your actual test file)
test_audio_path = "p374_236.wav"  # Update path

#Load the audio file
wav, sr = torchaudio.load(test_audio_path)

#Ensure the audio is valid
if wav.shape[1] == 0:
    raise ValueError("The audio file is empty! Check the test data.")

#Convert to NumPy for visualization
wav_np = wav.numpy().squeeze()

# Plot the waveform
plt.figure(figsize=(10, 4))
plt.plot(wav_np, alpha=0.7)
plt.title(f"Waveform - {sr}Hz")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()

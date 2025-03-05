import torch
import os
import numpy as np
import soundfile as sf
import torchaudio
from VocoderModel import Generator  # Ensure this matches your model file name

# Define paths
CHECKPOINT_DIR = "vocoder_checkpoints"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = Generator().to(device)
ema_generator_ckpt = os.path.join(CHECKPOINT_DIR, "generator_epoch6.pt")  # Default to latest checkpoint

if os.path.exists(ema_generator_ckpt):
    model.load_state_dict(torch.load(ema_generator_ckpt, map_location=device, weights_only=True))
    print(f"Loaded EMA generator for inference from checkpoint: {ema_generator_ckpt}")
else:
    generator_ckpt = os.path.join(CHECKPOINT_DIR, "generator_epoch6.pt")
    if os.path.exists(generator_ckpt):
        model.load_state_dict(torch.load(generator_ckpt, map_location=device, weights_only=True))
        print(f"Warning: No EMA generator checkpoint found, using standard generator from {generator_ckpt}")
    else:
        raise FileNotFoundError("No valid generator checkpoint found!")

model.eval()

# Inference function
def infer(mel, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Convert mel spectrogram to tensor
    mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        print("Debug: Running inference...")
        try:
            audio = model(mel)  # Run inference
            if audio is None:
                raise ValueError(" Generator returned None!")
            
            audio = audio.squeeze().cpu().numpy()
            print(f" Generated audio shape: {audio.shape}, min: {audio.min()}, max: {audio.max()}")

            # Save audio to file
            torchaudio.save(output_path, torch.tensor(audio).unsqueeze(0), 48000)
            return audio
        except Exception as e:
            print(f" Error during inference: {e}")
            return None

# Example usage
if __name__ == "__main__":
    test_mel = np.load("test_mel.npy")  # Replace with actual mel spectrogram file
    infer(test_mel, "generated_audio.wav")

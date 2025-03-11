import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import json
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from VocoderModel import Generator, Discriminator, MultiScaleDiscriminator, MultiPeriodDiscriminator
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel

# GPU memory optimizations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

CHECKPOINT_DIR = "vocoder_checkpoints"
LOSS_FILE = "loss.json"
SETTINGS_FILE = "settings.json"
GRADIENT_VALUES_FILE="Gradient_value.json"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # Optimizes CUDA kernel selection

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            return json.load(f)
    return {"processed_subdirs": [], "epochs": 0}

def save_settings(processed_subdirs, epochs):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump({"processed_subdirs": processed_subdirs, "epochs": epochs}, f)

def save_loss(epoch, g_loss, d_loss):
    loss_data = []
    if os.path.exists(LOSS_FILE):
        with open(LOSS_FILE, "r") as f:
            loss_data = json.load(f)
    loss_data.append({"Epoch": epoch, "Generator Loss": g_loss, "Discriminator Loss": d_loss})
    with open(LOSS_FILE, "w") as f:
        json.dump(loss_data, f, indent=4)

def save_gradient(grname,grsum,epoch):
    gradient_data_values=[]
    if os.path.exists(GRADIENT_VALUES_FILE):
        with open (GRADIENT_VALUES_FILE,"r") as f:
            gradient_data_values=json.load(f)
    gradient_data_values.append({"Epoch":epoch,"Name": grname,"Value":grsum})
    with open(GRADIENT_VALUES_FILE,"w") as f:
        json.dump(gradient_data_values, f, indent=4)


class MelDataset(Dataset):
    def __init__(self, data_dir):
        self.mel_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
        if len(self.mel_files) == 0:
            print(f"Error: No `.npy` files found in {data_dir} - Check preprocessing!")
        
    def __len__(self):
        return len(self.mel_files)


    
    def __getitem__(self, idx):
        mel_path = self.mel_files[idx]
        wav_path = mel_path.replace('.npy', '.wav')

         # Ensure both files exist
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Missing corresponding `.wav` file for {mel_path}")

        # Load Mel Spectrogram
        mel = np.load(mel_path).astype(np.float32)  # Ensure correct dtype
        mel = torch.tensor(mel, dtype=torch.float32)

        # Load Audio File
        wav, sr = torchaudio.load(wav_path)
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=48000)(wav).squeeze(0)

        # Ensure Fixed Length
        max_samples = 11 * 48000  # 11 seconds at 48kHz
        wav = torch.cat([wav, torch.zeros(max_samples - wav.shape[0])]) if wav.shape[0] < max_samples else wav[:max_samples]

        # Ensure Mel Spectrogram has Correct Shape
        target_mel_length = max_samples // 600  # Match the hop size of 600
        mel = mel[:, :target_mel_length] if mel.shape[1] >= target_mel_length else \
              torch.cat([mel, torch.zeros((100, target_mel_length - mel.shape[1]))], dim=1)

        return mel, wav.unsqueeze(0)

def find_latest_epoch():
    max_epoch = 0
    if os.path.exists(CHECKPOINT_DIR):
        for file in os.listdir(CHECKPOINT_DIR):
            if file.startswith("generator_epoch") and file.endswith(".pt"):
                try:
                    max_epoch = max(max_epoch, int(file.split("epoch")[1].split(".pt")[0]))
                except ValueError:
                    continue
    return max_epoch


def collate_fn(batch):
        mel_tensors, wav_tensors = zip(*batch)

        # Convert to tensors and stack them properly
        mel_tensors = torch.stack(mel_tensors)
        wav_tensors = torch.stack(wav_tensors)

        return mel_tensors, wav_tensors

def feature_loss(fmap_r, fmap_g):
    """Ensures that feature maps are the same size before computing L1 loss."""
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            if rl.shape != gl.shape:
                gl = torch.nn.functional.interpolate(gl, size=rl.shape[-1], mode="nearest")  # Resize to match
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2  # Keep loss scale consistent


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """Computes the adversarial loss for the discriminator."""
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)  # Real should be close to 1
        g_loss = torch.mean(dg ** 2)  # Fake should be close to 0
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    """Computes the adversarial loss for the generator."""
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)  # Fake should be close to 1
        gen_losses.append(l)
        loss += l
    return loss, gen_losses


def lr_lambda(epoch):
    if epoch < 10:
        return (epoch + 1) / 10  # Warmup over 10 epochs
    return 0.5 * (1 + np.cos((epoch - 10) / (epoch - 10) * np.pi))  # Cosine decay



def train(epochs=500, batch_size=2, data_dir='Training_data/train', checkpoint_interval=1, ACCUMULATION_STEPS = 4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings = load_settings()
    processed_subdirs = settings["processed_subdirs"]
    current_epoch = find_latest_epoch()

    subdirs=[]
    for root, dirs, files in os.walk(data_dir):
        if any(f.endswith(".npy") for f in files):  #Only add directories that contain `.npy` files
            subdirs.append(root)
    
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    ema_generator = AveragedModel(generator)  # EMA model for smoother updates

    learning_rate = 1e-4 * (batch_size / 4)

    optimizer_g = optim.AdamW(generator.parameters(), lr=learning_rate, betas=(0.8, 0.99))
    optimizer_d = optim.AdamW(list(mpd.parameters()) + list(msd.parameters()), lr=1e-4* (0.98 ** current_epoch), betas=(0.8, 0.99))

    scheduler_g = LambdaLR(optimizer_g, lr_lambda=lambda epoch: min(1.0, epoch / 10) if epoch < 10 else 0.999 ** epoch)
    scheduler_d = LambdaLR(optimizer_d, lr_lambda=lambda epoch: min(1.0, epoch / 10) if epoch < 10 else 0.999 ** epoch)

    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0, growth_factor=1.5, backoff_factor=0.5)
    
    generator_ckpt = os.path.join(CHECKPOINT_DIR, f"generator_epoch{current_epoch}.pt")
    discriminator_ckpt = os.path.join(CHECKPOINT_DIR, f"discriminator_epoch{current_epoch}.pt")
    optimizer_g_ckpt = os.path.join(CHECKPOINT_DIR, f'optimizer_g_epoch{current_epoch}.pt')
    optimizer_d_ckpt = os.path.join(CHECKPOINT_DIR, f'optimizer_d_epoch{current_epoch}.pt')
    scheduler_d_ckpt = os.path.join(CHECKPOINT_DIR, f'scheduler_d_epoch{current_epoch}.pt')
    scheduler_g_ckpt = os.path.join(CHECKPOINT_DIR, f'scheduler_g_epoch{current_epoch}.pt')

    # Load Generator and Optimizer G
    if os.path.exists(generator_ckpt):
        generator.load_state_dict(torch.load(generator_ckpt, weights_only=True))  
        if os.path.exists(optimizer_g_ckpt): 
            optimizer_g.load_state_dict(torch.load(optimizer_g_ckpt)) 
            scheduler_g.load_state_dict(torch.load(scheduler_g_ckpt))
            print(f"Loaded generator, optimizer_g and scheduler_g from checkpoints: {generator_ckpt}, {optimizer_g_ckpt}, {scheduler_g_ckpt}")
        else:
            print(f"Warning: Optimizer or Scheduler G checkpoint not found, starting fresh.")

    # Load Discriminator and Optimizer D
    if os.path.exists(discriminator_ckpt):
        discriminator.load_state_dict(torch.load(discriminator_ckpt, weights_only=True))  
        if os.path.exists(optimizer_d_ckpt): 
            optimizer_d.load_state_dict(torch.load(optimizer_d_ckpt)) 
            scheduler_d.load_state_dict(torch.load(scheduler_d_ckpt))
            print(f"Loaded discriminator, optimizer_d and scheduler_d from checkpoints: {discriminator_ckpt}, {optimizer_d_ckpt}, {scheduler_d_ckpt}")
        else:
            print(f"Warning: Optimizer or Scheduler D checkpoint not found, starting fresh.")

   
    for param in generator.parameters():
        param.requires_grad = True  
    for param in discriminator.parameters():
        param.requires_grad = True

    while current_epoch < epochs:

        for subdir in subdirs:
            if subdir in processed_subdirs:
                continue

            dataset = MelDataset(subdir)

            if len(dataset) == 0:
                raise ValueError(f"Dataset is empty! No training data found in {data_dir}")

            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=False, collate_fn=collate_fn)
            
            print(f"Starting epoch {current_epoch + 1} in {subdir}...")

            g_loss_total, d_loss_total = 0.0, 0.0
            
            generator.train()
            mpd.train()
            msd.train()

            torch.autograd.set_detect_anomaly(True)

            for i, (mel, wav) in enumerate(tqdm(dataloader)):
                mel, wav = mel.to(device), wav.to(device)
                mel = mel + torch.randn_like(mel) * 0.01  # Add small Gaussian noise

                optimizer_g.zero_grad()
                optimizer_d.zero_grad()
                
                with torch.amp.autocast("cuda"):
                    fake_wav = generator(mel)
                    if fake_wav.shape[-1] != wav.shape[-1]:  # Ensure output length matches input
                        fake_wav = torch.nn.functional.pad(fake_wav, (0, wav.shape[-1] - fake_wav.shape[-1]))
                                
                    y_d_r_mpd, y_d_g_mpd, fmap_r_mpd, fmap_g_mpd = mpd(wav, fake_wav)
                    y_d_r_msd, fmap_r_msd = msd(wav, return_intermediate=True)
                    y_d_g_msd, fmap_g_msd = msd(fake_wav, return_intermediate=True)
                
                    loss_disc_mpd, _, _ = discriminator_loss(y_d_r_mpd, y_d_g_mpd)
                    loss_disc_msd, _, _ = discriminator_loss(y_d_r_msd, y_d_g_msd)
                    loss_disc_all = 0.75 * loss_disc_mpd + 0.25 * loss_disc_msd  #Ensures discriminator does not overpower generator

                    loss_fm_mpd = feature_loss(fmap_r_mpd, fmap_g_mpd)
                    loss_fm_msd = feature_loss(fmap_r_msd, fmap_g_msd)
                    loss_fm = loss_fm_mpd + loss_fm_msd
                    loss_gen_mpd, _ = generator_loss(y_d_g_mpd)
                    loss_gen_msd, _ = generator_loss(y_d_g_msd)
                    fm_weight = max(1.2, 2.0 * (0.999 ** current_epoch))  # Reduce feature loss weight over time
                    loss_gen_all = loss_gen_mpd + loss_gen_msd + fm_weight * loss_fm


                loss_gen_all = loss_gen_all / ACCUMULATION_STEPS  # Normalize loss
                loss_disc_all = loss_disc_all / ACCUMULATION_STEPS

                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                scaler.scale(loss_gen_all).backward(retain_graph=True)
                

                if (i + 1) % ACCUMULATION_STEPS == 0:  # Update every ACCUMULATION_STEPS
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                    scaler.step(optimizer_g)
                    scaler.update()
                    optimizer_g.zero_grad()
                    ema_generator.update_parameters(generator)

                scaler.scale(loss_disc_all).backward()

                if (i + 1) % ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(mpd.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(msd.parameters(), max_norm=1.0)
                    scaler.step(optimizer_d)
                    scaler.update()
                    optimizer_d.zero_grad()
            
                g_loss_total += loss_gen_all.item()
                d_loss_total += loss_disc_all.item()

            
            scheduler_g.step()
            scheduler_d.step()

            print(f"Epoch [{current_epoch + 1}/{epochs}] - Generator Loss: {g_loss_total:.4f}, Discriminator Loss: {d_loss_total:.4f}")
            save_loss(current_epoch, g_loss_total, d_loss_total)
            
            if (current_epoch + 1) % checkpoint_interval == 0:
                generator.remove_weight_norm()
                torch.save(generator.state_dict(), os.path.join(CHECKPOINT_DIR, f'generator_epoch{current_epoch + 1}.pt'))
                torch.save(ema_generator.state_dict(), os.path.join(CHECKPOINT_DIR, f'ema_generator_epoch{current_epoch + 1}.pt'))
                torch.save(discriminator.state_dict(), os.path.join(CHECKPOINT_DIR, f'discriminator_epoch{current_epoch + 1}.pt'))
                torch.save(optimizer_d.state_dict(), os.path.join(CHECKPOINT_DIR, f'optimizer_d_epoch{current_epoch + 1}.pt'))
                torch.save(optimizer_g.state_dict(), os.path.join(CHECKPOINT_DIR, f'optimizer_g_epoch{current_epoch + 1}.pt'))
                torch.save(scheduler_g.state_dict(), os.path.join(CHECKPOINT_DIR, f'scheduler_g_epoch{current_epoch + 1}.pt'))
                torch.save(scheduler_d.state_dict(), os.path.join(CHECKPOINT_DIR, f'scheduler_d_epoch{current_epoch + 1}.pt'))
                print(f"Saved checkpoints for epoch {current_epoch + 1}")
                
            
            processed_subdirs.append(subdir)
            save_settings(processed_subdirs, current_epoch + 1)
            current_epoch += 1
            
if __name__ == "__main__":
    train()

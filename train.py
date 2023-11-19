from dataloader import ActivationsDataset
from config import Config
from model import AutoEncoder
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import wandb
from utils import get_freqs, re_init
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import argparse

parser = argparse.ArgumentParser(description='AutoEncoder Training Script')

# Training parameters
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--batch_size', type=int, default=4096, help='Batch size for training')
parser.add_argument('--act_size', type=int, default=784, help='Size of the activation vectors')
parser.add_argument('--buffer_mult', type=int, default=384, help='Multiplier for buffer size')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--num_tokens', type=int, default=int(2e9), help='Number of tokens')
parser.add_argument('--l1_coeff', type=float, default=0.00001, help='L1 loss coefficient')
parser.add_argument('--beta1', type=float, default=0.9, help='Adam optimizer beta1')
parser.add_argument('--beta2', type=float, default=0.99, help='Adam optimizer beta2')
parser.add_argument('--dict_mult', type=int, default=32, help='Multiplier for dictionary size')
parser.add_argument('--seq_len', type=int, default=128, help='Sequence length')
parser.add_argument('--enc_dtype', type=str, default='fp32', help='Encoder data type')
parser.add_argument('--remove_rare_dir', action='store_true', help='Remove rare directory')
parser.add_argument('--model_name', type=str, default='gelu-2l', help='Model name')
parser.add_argument('--site', type=str, default='mlp_out', help='Site')
parser.add_argument('--layer', type=int, default=15, help='Layer number')
parser.add_argument('--device', type=str, default='cuda:0', help='Device for training')
parser.add_argument('--latent_dim', type=int, default=32, help='Latent dimension dict multiplier')

args = parser.parse_args()

cfg = vars(args)
print(cfg)



encoder = AutoEncoder(cfg).to(cfg["device"])
activations = pickle.load(open('./data/auto_encoder_activationsencoder.layers.15.attn.self_attention.fc_out.pkl', 'rb'))
X = activations['activations']
from sklearn.model_selection import train_test_split

# Assuming you have X_train, X_val as your training and validation sets
X_train, X_val = train_test_split(X, test_size=0.1, random_state=42)

dataset_train = ActivationsDataset(X_train)
dataset_val = ActivationsDataset(X_val)

batch_size = 128

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

dataloader_train = tqdm(dataloader_train, desc="Training", unit="batch")
dataloader_val = tqdm(dataloader_val, desc="Validation", unit="batch")

# The rest of your code remains unchanged
wandb.init(project="autoencoder3", config=cfg)
encoder_optim = torch.optim.Adam(encoder.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
recons_scores = []
act_freq_scores_list = []

EPOCHS = 5

for epoch in range(EPOCHS):
    print("Epoch", epoch)
    
    # Training loop
    encoder.train()
    for i, activations in enumerate(dataloader_train):
        loss, x_reconstruct, mid_acts, l2_loss, l1_loss = encoder(activations.to(cfg["device"]))
        loss.backward()
        encoder_optim.step()
        encoder_optim.zero_grad()
        loss_dict = {"loss": loss.item(), "l2_loss": l2_loss.item(), "l1_loss": l1_loss.item()}
        del loss, x_reconstruct, mid_acts, l2_loss, l1_loss, activations

        if (i) % 10 == 0:
            wandb.log(loss_dict)
            print(loss_dict)

        if (i) % 1000 == 0:
            freqs = get_freqs(dataloader_train, cfg, local_encoder=encoder)
            act_freq_scores_list.append(freqs)
            wandb.log({
                "dead": (freqs==0).float().mean().item(),
                "below_1e-6": (freqs<1e-6).float().mean().item(),
                "below_1e-5": (freqs<1e-5).float().mean().item(),
            })

        if (i+1) % 300 == 0:
            encoder.save()
            wandb.log({"reset_neurons": 0.0})
            freqs = get_freqs(dataloader_train, cfg, local_encoder=encoder)
            to_be_reset = (freqs<10**(-5.5))
            print("Resetting neurons!", to_be_reset.sum())
            re_init(to_be_reset, encoder)

    # Validation loop
    encoder.eval()
    with torch.no_grad():
        val_loss_sum = 0.0
        num_batches_val = 0
        for i, activations_val in enumerate(dataloader_val):
            loss_val, _, _, l2_loss, _ = encoder(activations_val.to(cfg["device"]))
            
            val_loss_sum += l2_loss.item()
            num_batches_val += 1
        avg_val_loss = val_loss_sum / num_batches_val

        print(f"Validation Loss: {avg_val_loss}")
        wandb.log({"val_loss": avg_val_loss})

activations = next(iter(dataloader_val))
print(f'activations shape: {activations.shape}')
loss, x_reconstruct, mid_acts, l2_loss, l1_loss = encoder(activations[12].to(cfg["device"]))
plt.figure()
sns.heatmap(activations[12].reshape(28, 28), center=0)
plt.title(f"Original image")
wandb.log({f"Original image": wandb.Image(plt)})
plt.close()

print(f'reconstruction shape: {x_reconstruct.shape}')
plt.figure()
sns.heatmap(x_reconstruct.detach().cpu().reshape(28, 28), center=0)
plt.title(f"Original image")
wandb.log({f"Reconstructed image": wandb.Image(plt)})
plt.close()


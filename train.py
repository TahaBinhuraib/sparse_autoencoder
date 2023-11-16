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


cfg = Config().to_dict()
print(cfg)


encoder = AutoEncoder(cfg).to(cfg["device"])
activations = pickle.load(open('./data/auto_encoder_activationsencoder.layers.15.attn.self_attention.fc_out.pkl', 'rb'))
X = activations['activations']
scaler = StandardScaler().fit(X)
X = scaler.transform(X)

print("Loaded activations!")

dataset = ActivationsDataset(X)

batch_size = 128

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
dataloader = tqdm(dataloader, desc="Processing", unit="batch")

wandb.init(project="autoencoder", config=cfg)
encoder_optim = torch.optim.Adam(encoder.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
recons_scores = []
act_freq_scores_list = []

EPOCHS = 20

# for e in range(EPOCHS):
for i, (activations) in enumerate(dataloader):
    loss, x_reconstruct, mid_acts, l2_loss, l1_loss = encoder(activations.to(cfg["device"]))
    loss.backward()
    encoder.make_decoder_weights_and_grad_unit_norm()
    encoder_optim.step()
    encoder_optim.zero_grad()
    loss_dict = {"loss": loss.item(), "l2_loss": l2_loss.item(), "l1_loss": l1_loss.item()}
    del loss, x_reconstruct, mid_acts, l2_loss, l1_loss, activations
    if (i) % 10 == 0:
        wandb.log(loss_dict)
        print(loss_dict)
    if (i) % 1000 == 0:
        freqs = get_freqs(dataloader, cfg, local_encoder=encoder)
        act_freq_scores_list.append(freqs)
        # histogram(freqs.log10(), marginal="box", histnorm="percent", title="Frequencies")
        wandb.log({
            "dead": (freqs==0).float().mean().item(),
            "below_1e-6": (freqs<1e-6).float().mean().item(),
            "below_1e-5": (freqs<1e-5).float().mean().item(),
        })
    if (i+1) % 3000 == 0:
        encoder.save()
        wandb.log({"reset_neurons": 0.0})
        freqs = get_freqs(dataloader, cfg, local_encoder=encoder)
        to_be_reset = (freqs<10**(-5.5))
        print("Resetting neurons!", to_be_reset.sum())
        re_init(to_be_reset, encoder)

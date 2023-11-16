import torch
import tqdm

@torch.no_grad()
def get_freqs(dataloader, cfg, local_encoder=None):
    act_freq_scores = torch.zeros(local_encoder.d_hidden, dtype=torch.float32).to(cfg["device"])
    total = 0
    for i, activations in enumerate(dataloader):
        acts = activations.to(cfg["device"]) 
        hidden = local_encoder(acts)[2]
        
        act_freq_scores += (hidden > 0).sum(0)
        total+=hidden.shape[0]
    act_freq_scores /= total
    num_dead = (act_freq_scores==0).float().mean()
    print("Num dead", num_dead)
    return act_freq_scores



@torch.no_grad()
def re_init(indices, encoder):
    new_W_enc = (torch.nn.init.kaiming_uniform_(torch.zeros_like(encoder.W_enc)))
    new_W_dec = (torch.nn.init.kaiming_uniform_(torch.zeros_like(encoder.W_dec)))
    new_b_enc = (torch.zeros_like(encoder.b_enc))
    print(new_W_dec.shape, new_W_enc.shape, new_b_enc.shape)
    encoder.W_enc.data[:, indices] = new_W_enc[:, indices]
    encoder.W_dec.data[indices, :] = new_W_dec[indices, :]
    encoder.b_enc.data[indices] = new_b_enc[indices]
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Config:
    seed: int = 42
    batch_size: int = 4096
    act_size: int = 784
    buffer_mult: int = 384
    lr: float = 1e-5
    num_tokens: int = int(2e9)
    l1_coeff: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.99
    dict_mult: int = 32
    seq_len: int = 128
    enc_dtype: str = "fp32"
    remove_rare_dir: bool = False
    model_name: str = "gelu-2l"
    site: str = "mlp_out"
    layer: int = 15
    device: str = "cuda:0"
    latent_dim: int = act_size*16
    

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from sae_scorer import importance_scores, select_top_dims
from sae_train import SparseAutoencoder, sae_loss, train_sae

def train_all_saes(
    filtered: dict,
    save_dir: str,
    device: str = 'cpu',
) -> dict:
    """
    Train one SAE per activation vector and save each to disk.

    Args:
        filtered:  dict of filtered activation tensors (from select_top_dims)
        save_dir:  directory to save trained SAE state dicts
        device:    'cuda' or 'cpu'

    Returns:
        saes: dict mapping vector name -> trained SparseAutoencoder
    """
    sae_configs = {
        'z_mu':     {'hidden_dim': 512,  'lam': 1e-3},
        'z_var':    {'hidden_dim': 512,  'lam': 1e-3},
        'z':        {'hidden_dim': 512,  'lam': 1e-3},
        'cond_emb': {'hidden_dim': 256,  'lam': 1e-3},
        'z_cond':   {'hidden_dim': 1024, 'lam': 1e-3},
    }

    saes = {}

    for name, cfg in sae_configs.items():
        activations = filtered[name].float()
        input_dim   = activations.shape[1]

        print(f"── {name}  (in={input_dim} → hidden={cfg['hidden_dim']}) ──")

        sae = train_sae(
            activations=activations,
            input_dim=input_dim,
            hidden_dim=cfg['hidden_dim'],
            lam=cfg['lam'],
            device=device,
            verbose=True,
        )

        saes[name] = sae
        save_path  = f"{save_dir}/sae_{name}.pt"
        torch.save(sae.state_dict(), save_path)
        print(f"  saved → {save_path}\n")

    return saes

if __name__ == "__main__":
    BASE = '/afs/inf.ed.ac.uk/user/s23/s2305255/Desktop/MLP/Con-CDVAE/data/mp_20'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # score and filter
    data     = torch.load(f'{BASE}/activation_dataset.pt')
    filtered, top_dims = select_top_dims(data, K=150, save_path_filtered=f'{BASE}/activation_dataset_filtered.pt')

    # train all five SAEs
    saes = train_all_saes(filtered, save_dir=BASE, device=device)
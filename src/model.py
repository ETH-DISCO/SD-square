import torch
import torch.nn as nn


class PrepLatentDeltas(nn.Module):
    """
    Maps verifier guidance embeddings (B,[S],d_embd) → latent deltas
    with shape (n_layers,B,[S],out_size) after an efficient, layer-wise RMSNorm.
    """

    def __init__(self, d_embd: int, out_size: int, n_layers: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(d_embd)
        self.w_guide = nn.Linear(d_embd, out_size * n_layers, bias=False)
        self.w_guide.weight.data.zero_()  # start with “no-op” deltas
        self.n_layers = n_layers
        self.out_size = out_size

    def forward(self, guidance_embeds: torch.Tensor) -> torch.Tensor:
        # (B,[S],d_embd) → (B,[S],n_layers*out_size)
        latent = self.w_guide(self.norm(guidance_embeds))

        # → (B,[S],n_layers,out_size)
        shapes = guidance_embeds.shape
        latent = latent.view(*shapes[:-1], self.n_layers, self.out_size)

        # match the original (n_layers, B, …) layout expected downstream
        if len(shapes) == 2:  # B,DE  →  nL,B,O
            latent = latent.transpose(0, 1)
        else:  # B,S,DE →  nL,B,S,O
            latent = latent.transpose(1, 2).transpose(0, 1)
        return latent


class WithKwargs(nn.Module):
    def __init__(self, orig):
        super().__init__()
        self.orig = orig

    def forward(self, x, **kwargs):
        return self.orig(x)

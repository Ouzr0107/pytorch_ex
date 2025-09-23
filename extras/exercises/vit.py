import torch
from torch import nn

class ViT(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_classes: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        channels: int = 3,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ):
        super(ViT, self).__init__()
        assert (
            image_size % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size**2

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                activation="gelu",
            ),
            num_layers=depth,
        )

        self.classifier = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x):
        batch_size = x.shape(0)

        # Add class token to the beginning
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Linear embedding of patches
        x = self.patch_to_embedding(x)

        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embedding
        x += self.pos_embedding
        x = self.dropout(x)

        # Pass through transformer
        x = self.transformer(x)

        # Take the class token output
        x = self.classifier(x[:, 0])

        return x

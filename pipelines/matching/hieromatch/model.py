"""Encoder + cosine classifier head for symbol matching."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class HieroEncoder(nn.Module):
    """Grayscale-image -> embedding. Any torchvision resnet works as backbone
    (resnet18 default; resnet34/50 for the HPC run)."""

    def __init__(self, arch="resnet18", embed_dim=256, pretrained=True):
        super().__init__()
        self.arch, self.embed_dim = arch, embed_dim
        net = getattr(torchvision.models, arch)(weights="DEFAULT" if pretrained else None)
        if not hasattr(net, "fc"):
            raise ValueError(f"unsupported arch {arch} (use a resnet)")
        w = net.conv1.weight.data
        net.conv1 = nn.Conv2d(1, w.shape[0], kernel_size=7, stride=2, padding=3, bias=False)
        net.conv1.weight.data = w.mean(1, keepdim=True)      # ImageNet RGB -> gray init
        net.fc = nn.Linear(net.fc.in_features, embed_dim)
        self.backbone = net

    def forward(self, x):                                    # raw (unnormalized) embedding
        return self.backbone(x)

    def embed(self, x):                                      # L2-normalized embedding
        return F.normalize(self.forward(x), dim=-1)


class CosineHead(nn.Module):
    """Normalized-weight classifier (NormFace-style): logits = s * cos(emb, W).
    Keeps the embedding space metric so nearest-prototype retrieval works."""

    def __init__(self, embed_dim, n_classes, scale=20.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_classes, embed_dim) * 0.01)
        self.scale = scale

    def forward(self, emb):
        return self.scale * F.normalize(emb, dim=-1) @ F.normalize(self.weight, dim=-1).T


def save_encoder(path, encoder, classes, size, extra=None):
    torch.save({"state_dict": encoder.state_dict(), "arch": encoder.arch,
                "embed_dim": encoder.embed_dim, "size": size, "classes": classes,
                **(extra or {})}, path)


def load_encoder(path, device="cpu"):
    """Returns (encoder.eval(), meta dict with size/classes/arch/embed_dim)."""
    # weights_only=True: checkpoint is tensors + str/int/float/list only, so the
    # safe loader suffices and a swapped checkpoint can't execute code (F8).
    ck = torch.load(path, map_location="cpu", weights_only=True)
    enc = HieroEncoder(arch=ck["arch"], embed_dim=ck["embed_dim"], pretrained=False)
    enc.load_state_dict(ck["state_dict"])
    enc.to(device).eval()
    return enc, {k: ck[k] for k in ("arch", "embed_dim", "size", "classes")}

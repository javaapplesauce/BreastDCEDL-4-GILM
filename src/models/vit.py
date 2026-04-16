"""
Vision Transformer for pCR prediction from DCE-MRI.

Wraps HuggingFace ViT/DINOv2 with:
  - Optional clinical feature concatenation before the classifier head
  - Configurable dropout
  - LLRD parameter group construction
"""

import torch
import torch.nn as nn
from transformers import ViTForImageClassification, Dinov2ForImageClassification


class BreastDCEViT(nn.Module):

    def __init__(
        self,
        backbone: str = "google/vit-base-patch16-224-in21k",
        num_classes: int = 2,
        dropout: float = 0.3,
        use_clinical: bool = False,
        clinical_dim: int = 0,
        pretrained_weights: str | None = None,
    ):
        super().__init__()
        self.use_clinical = use_clinical

        ModelClass = (Dinov2ForImageClassification if "dinov2" in backbone.lower()
                      else ViTForImageClassification)
        self.encoder = ModelClass.from_pretrained(
            backbone,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

        hidden = self.encoder.classifier.in_features
        head_input = hidden + clinical_dim if use_clinical else hidden

        self.encoder.classifier = nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(head_input, num_classes),
        )

        if pretrained_weights:
            self.load_pretrained(pretrained_weights)

    def load_pretrained(self, path: str):
        """Load pretrained weights (e.g. paper's checkpoint from Zenodo).
        Accepts either an already-wrapped state_dict (has `encoder.`/`head.`
        keys) or a bare HF ViTForImageClassification state_dict. Skips
        classifier-head mismatches gracefully."""
        import os
        if not os.path.isfile(path):
            print(f"  [pretrained] Not found: {path}")
            return
        sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]

        has_wrapper = any(k.startswith("encoder.") for k in sd)
        if has_wrapper:
            target_sd = sd
        else:
            # Remap bare HF state_dict -> self.encoder.*
            target_sd = {}
            for k, v in sd.items():
                if k.startswith("classifier."):
                    continue  # skip head weights (shape may differ)
                target_sd[f"encoder.{k}"] = v
        missing, unexpected = self.load_state_dict(target_sd, strict=False)
        print(f"  [pretrained] loaded {path}  missing={len(missing)} unexpected={len(unexpected)}")

    def forward(self, pixel_values: torch.Tensor, clinical: torch.Tensor = None):
        features = self.encoder(pixel_values).logits  # (B, hidden)
        if self.use_clinical and clinical is not None:
            features = torch.cat([features, clinical], dim=-1)
        return self.head(features)

    def freeze_backbone(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.encoder.parameters():
            param.requires_grad = True


def build_llrd_param_groups(
    model: BreastDCEViT,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
    llrd: float = 0.85,
) -> list[dict]:
    """
    Layer-wise learning rate decay: classifier head gets head_lr,
    each encoder layer below gets backbone_lr * llrd^(distance_from_top),
    embeddings get the lowest LR.
    """
    no_decay = {"bias", "LayerNorm.weight", "layernorm"}
    seen = set()
    groups = []

    def _add(named_params, lr):
        d = [p for n, p in named_params if n not in seen
             and not any(nd in n for nd in no_decay)]
        nd = [p for n, p in named_params if n not in seen
              and any(nd in n for nd in no_decay)]
        seen.update(n for n, _ in named_params)
        if d:
            groups.append({"params": d, "lr": lr, "weight_decay": weight_decay})
        if nd:
            groups.append({"params": nd, "lr": lr, "weight_decay": 0.0})

    # Head
    _add(list(model.head.named_parameters(prefix="head")), head_lr)

    # Encoder layers top-down
    num_layers = 12
    for i in range(num_layers - 1, -1, -1):
        depth = num_layers - 1 - i
        layer_lr = backbone_lr * (llrd ** depth)
        named = [(n, p) for n, p in model.encoder.named_parameters(prefix="encoder")
                 if f"layer.{i}." in n and n not in seen]
        _add(named, layer_lr)

    # Embeddings (lowest LR)
    rest = [(n, p) for n, p in model.named_parameters() if n not in seen]
    _add(rest, backbone_lr * (llrd ** num_layers))

    return groups

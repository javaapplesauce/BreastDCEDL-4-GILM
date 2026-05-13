"""
Vision Transformer for pCR prediction from DCE-MRI.

Wraps HuggingFace ViT/DINOv2 with:
  - Optional clinical feature concatenation before the classifier head
  - Configurable dropout
  - LLRD parameter group construction
"""

import math
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
        pos_class_prior: float = 0.5,
        debug_pretrained: bool = True,
        load_paper_classifier: bool = True,
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

        # Bias-init to log-prior breaks the all-positive symmetry that focal-
        # weighted loss has trouble escaping with a frozen backbone. If
        # load_pretrained subsequently warmstarts the head from the paper's
        # checkpoint, this is overwritten — which is what we want.
        if num_classes == 2 and 0.0 < pos_class_prior < 1.0:
            self._init_head_bias_to_prior(pos_class_prior)

        if pretrained_weights:
            self.load_pretrained(
                pretrained_weights,
                debug=debug_pretrained,
                load_classifier=load_paper_classifier,
            )
            # If we skipped the paper's classifier, the log-prior bias init
            # above is what the head starts with — keep it. If we loaded it,
            # the bias init was overwritten by the paper's bias, which is
            # what the warmstart intends.

    def _init_head_bias_to_prior(self, p: float):
        """Set head bias so softmax(bias) = [1-p, p] at init. Avoids the
        all-positive collapse seen when focal+class-weights+frozen-backbone
        train a randomly-initialised head."""
        delta = math.log(p / (1.0 - p))  # logit of class 1
        bias = torch.tensor([-delta / 2.0, delta / 2.0], dtype=torch.float32)
        with torch.no_grad():
            self.head[1].bias.copy_(bias)

    def load_pretrained(self, path: str, debug: bool = False, load_classifier: bool = True):
        """Load pretrained weights (e.g. paper's checkpoint from Zenodo).
        Accepts either an already-wrapped state_dict (has `encoder.`/`head.`
        keys) or a bare HF ViTForImageClassification state_dict. Tries to
        load the paper's classifier into our head.1 — the paper trained
        for 2-class pCR so shapes match unless use_clinical=True changes
        head_input. Classifier weight+bias are loaded as a pair (both or
        neither) so we never end up with a paper-bias on a paper-foreign
        weight.

        Note on the pooler: HF ViTForImageClassification.forward uses
        sequence_output[:, 0] (the CLS token) directly and never calls
        self.vit.pooler on the classification path. So any pooler.*
        weights in the paper checkpoint are dead — they map to keys we
        own but no forward pass reads. The debug listing below makes this
        visible by printing which keys were transferred and which were
        dropped.
        """
        import os
        if not os.path.isfile(path):
            print(f"  [pretrained] Not found: {path}")
            return
        sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]

        own = self.state_dict()
        target_sd = {}
        has_wrapper = any(k.startswith("encoder.") for k in sd)

        # Decide head load up front — we want weight+bias as a pair, or skip both.
        # When load_classifier=False we intentionally skip the warmstart so the
        # log-prior bias init in __init__ survives. Empirically the paper's
        # classifier load can lock phase 1 into an all-positive prediction
        # collapse that focal+class-weights at lr=5e-4 cannot escape in 5
        # epochs.
        classifier_loaded = False
        if not load_classifier:
            print(f"  [pretrained] skipping classifier warmstart (load_classifier=False); "
                  f"head uses log-prior bias init")
        elif not has_wrapper:
            cls_w, cls_b = sd.get("classifier.weight"), sd.get("classifier.bias")
            head_w, head_b = own.get("head.1.weight"), own.get("head.1.bias")
            if (cls_w is not None and cls_b is not None
                    and head_w is not None and head_b is not None
                    and head_w.shape == cls_w.shape and head_b.shape == cls_b.shape):
                target_sd["head.1.weight"] = cls_w
                target_sd["head.1.bias"] = cls_b
                classifier_loaded = True
            elif cls_w is not None or cls_b is not None:
                shape_ours = tuple(head_w.shape) if head_w is not None else None
                shape_theirs = tuple(cls_w.shape) if cls_w is not None else None
                print(f"  [pretrained] skipping classifier: ours={shape_ours} theirs={shape_theirs}")

        dropped_keys: list[tuple[str, str]] = []  # (key, reason)
        for k, v in sd.items():
            if has_wrapper:
                new_k = k
            elif k.startswith("classifier."):
                if not classifier_loaded:
                    reason = ("classifier warmstart disabled (load_classifier=False)"
                              if not load_classifier
                              else "classifier handled separately (shape mismatch)")
                    dropped_keys.append((k, reason))
                continue
            else:
                new_k = f"encoder.{k}"
            if new_k not in own:
                dropped_keys.append((k, f"no target key ({new_k!r})"))
                continue
            if own[new_k].shape != v.shape:
                dropped_keys.append(
                    (k, f"shape mismatch ours={tuple(own[new_k].shape)} theirs={tuple(v.shape)}")
                )
                continue
            target_sd[new_k] = v

        missing, unexpected = self.load_state_dict(target_sd, strict=False)
        head_missing = sum(1 for m in missing if m.startswith("head.1."))
        print(f"  [pretrained] loaded {path}")
        print(f"  [pretrained]   transferred={len(target_sd)} keys  "
              f"dropped={len(dropped_keys)}  missing={len(missing)}  unexpected={len(unexpected)}")
        if classifier_loaded:
            print(f"  [pretrained]   classifier warmstarted from paper checkpoint")
        elif head_missing > 0:
            print(f"  [pretrained]   head random-init (bias-init kept; classifier shape mismatch or absent)")

        if debug:
            print(f"  [pretrained-debug] checkpoint keys total: {len(sd)}")
            print(f"  [pretrained-debug] non-encoder top-level keys: "
                  f"{sorted(k.split('.')[0] for k in sd if '.' in k and not k.startswith('encoder.'))[:20]}")
            if target_sd:
                shown = list(target_sd.keys())
                print(f"  [pretrained-debug] transferred ({len(shown)}):")
                for k in shown[:30]:
                    print(f"  [pretrained-debug]   + {k}")
                if len(shown) > 30:
                    print(f"  [pretrained-debug]   + ... and {len(shown)-30} more")
            if dropped_keys:
                print(f"  [pretrained-debug] dropped ({len(dropped_keys)}):")
                for k, reason in dropped_keys[:30]:
                    print(f"  [pretrained-debug]   - {k}  [{reason}]")
                if len(dropped_keys) > 30:
                    print(f"  [pretrained-debug]   - ... and {len(dropped_keys)-30} more")
            if missing:
                print(f"  [pretrained-debug] keys we own but ckpt did not provide ({len(missing)}):")
                for k in missing[:30]:
                    print(f"  [pretrained-debug]   ? {k}")
                if len(missing) > 30:
                    print(f"  [pretrained-debug]   ? ... and {len(missing)-30} more")

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

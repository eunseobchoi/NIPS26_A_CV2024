"""
DINOv2-LoRA classifier for capsule endoscopy.
Base: DINOv2-ViT-L/14 (frozen backbone + LoRA adapters + linear head)
This is our source model trained on A100, deployed on edge-device for TTA.
"""
import torch
import torch.nn as nn
from typing import Optional


class DINOv2LoRAClassifier(nn.Module):
    """DINOv2 backbone with LoRA adapters and classification head.

    Architecture:
        DINOv2-ViT-L/14 (frozen) → LoRA on q,v projections → Linear head → num_classes

    This follows the winning approach from Capsule Vision 2024 (PuppyOps team).
    """

    def __init__(self, num_classes: int = 10, backbone: str = "dinov2_vitl14",
                 lora_r: int = 8, lora_alpha: int = 16, freeze_backbone: bool = True):
        super().__init__()
        self.num_classes = num_classes

        # Load DINOv2 backbone
        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone)
        embed_dim = self.backbone.embed_dim  # 1024 for ViT-L

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )

        # LoRA will be injected via PEFT after construction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)  # [B, embed_dim]
        logits = self.head(features)  # [B, num_classes]
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification head (for TTA objectives)."""
        return self.backbone(x)


def build_model(num_classes: int = 10, lora_r: int = 8,
                pretrained_path: Optional[str] = None) -> nn.Module:
    """Build DINOv2-LoRA classifier with optional pretrained weights."""
    model = DINOv2LoRAClassifier(num_classes=num_classes, lora_r=lora_r)

    if pretrained_path:
        state = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        print(f"Loaded pretrained weights from {pretrained_path}")

    return model


def add_lora_adapters(model: nn.Module, r: int = 8, alpha: int = 16):
    """Add LoRA adapters to DINOv2 attention layers via PEFT."""
    from peft import get_peft_model, LoraConfig

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["qkv"],  # DINOv2 uses fused qkv
        lora_dropout=0.0,
        bias="none",
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model

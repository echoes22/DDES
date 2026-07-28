import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from torchvision import models as torchvision_models
from torchvision.models import convnext_small, ConvNeXt_Small_Weights
from torchvision.models.feature_extraction import create_feature_extractor


class MODELS(Enum):
    convnext_unet = 5
    convnext_8 = 6
    convnext_2 = 7


def get_model(model_type, ckpt=None, out_features=None, train=True, scratch=False, freeze_backbone=False):
    # Accept both string and enum model_type
    if isinstance(model_type, str):
        model_type = MODELS[model_type]

    # Treat empty string as no checkpoint
    if not ckpt:
        ckpt = None

    use_imagenet = (ckpt is None and not scratch)

    if model_type == MODELS.convnext_unet:
        model = ConvNextUnet(
            output_grid_size_n=28,
            num_decoder_stages=2,
            use_pretrained=use_imagenet,
            freeze_backbone=freeze_backbone,
        )
        if ckpt is not None:
            state = _load_state(ckpt)
            model.load_state_dict(state)

    elif model_type == MODELS.convnext_8:
        n_out = out_features if out_features is not None else 8
        model = CustomConvNeXt(out_features=n_out, use_pretrained=use_imagenet, freeze_backbone=freeze_backbone)
        if ckpt is not None:
            state = _load_state(ckpt)
            model.load_state_dict(state)

    elif model_type == MODELS.convnext_2:
        model = CustomConvNeXt(out_features=2, use_pretrained=use_imagenet, freeze_backbone=freeze_backbone)
        if ckpt is not None:
            state = _load_state(ckpt)
            model.load_state_dict(state)

    else:
        raise NotImplementedError(f'Model {model_type} is not implemented')

    return model


def _load_state(path):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(obj, nn.Module):
        return obj.state_dict()
    if hasattr(obj, 'state_dict'):
        return obj.state_dict()
    return obj


class CustomConvNeXt(nn.Module):
    def __init__(self, out_features=8, use_pretrained=True, freeze_backbone=False):
        super().__init__()
        weights = ConvNeXt_Small_Weights.DEFAULT if use_pretrained else None
        self.base_model = torchvision_models.convnext_small(weights=weights)

        in_feats = self.base_model.classifier[-1].in_features
        bias = self.base_model.classifier[-1].bias is not None
        self.base_model.classifier[-1] = nn.Linear(in_feats, out_features, bias=bias)
        if out_features != 2:
            self.base_model.classifier.add_module('log_softmax', nn.LogSoftmax(dim=1))

        if freeze_backbone:
            for param in self.base_model.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.base_model(x)


class ConvNeXtDecoderBlock(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return residual + x


class DecoderStage(nn.Module):
    def __init__(self, input_dim, output_dim, num_blocks=1):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_adjust = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        self.blocks = nn.Sequential(*[ConvNeXtDecoderBlock(output_dim) for _ in range(num_blocks)])

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv_adjust(x)
        return self.blocks(x)


class ConvNeXtDecoder(nn.Module):
    """2-stage decoder: 7×7 → 14×14 → 28×28, channel halving at each stage."""

    def __init__(self, input_channels=768, num_stages=2, blocks_per_stage=1):
        super().__init__()
        self.stages = nn.ModuleList()
        in_ch = input_channels
        out_ch = input_channels // 2

        for _ in range(num_stages):
            self.stages.append(DecoderStage(in_ch, out_ch, num_blocks=blocks_per_stage))
            in_ch = out_ch
            out_ch = max(16, out_ch // 2)

        self.final_conv = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        x = self.final_conv(x)
        B, C, H, W = x.shape
        x = F.log_softmax(x.view(B, -1), dim=1).view(B, C, H, W)
        return x


class ConvNextUnet(nn.Module):
    """DDES-Net: ConvNeXt-Small backbone + 2-stage decoder → 28×28 log-prob heatmap."""

    def __init__(self, output_grid_size_n=28, num_decoder_stages=2, use_pretrained=True, freeze_backbone=False):
        super().__init__()
        self.output_grid_size_n = output_grid_size_n

        weights = ConvNeXt_Small_Weights.DEFAULT if use_pretrained else None
        backbone = convnext_small(weights=weights)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        self.feature_extractor = create_feature_extractor(
            backbone, return_nodes={'features.7': 'deep_features'}
        )

        self.decoder = ConvNeXtDecoder(
            input_channels=768,
            num_stages=num_decoder_stages,
            blocks_per_stage=1,
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.decoder(features['deep_features'])


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

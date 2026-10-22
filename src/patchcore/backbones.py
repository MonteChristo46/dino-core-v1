import os

import timm  # noqa
import torch
import torchvision.models as models  # noqa

from patchcore.dinov3 import DinoV3Backbone

_BACKBONES = {
    "alexnet": "models.alexnet(pretrained=False)",
    "bninception": 'pretrainedmodels.__dict__["bninception"]'
    '(pretrained="imagenet", num_classes=1000)',
    "resnet50": "models.resnet50(pretrained=False)",
    "resnet101": "models.resnet101(pretrained=False)",
    "resnext101": "models.resnext101_32x8d(pretrained=False)",
    "resnet200": 'timm.create_model("resnet200", pretrained=False)',
    "resnest50": 'timm.create_model("resnest50d_4s2x40d", pretrained=False)',
    "resnetv2_50_bit": 'timm.create_model("resnetv2_50x3_bitm", pretrained=False)',
    "resnetv2_50_21k": 'timm.create_model("resnetv2_50x3_bitm_in21k", pretrained=False)',
    "resnetv2_101_bit": 'timm.create_model("resnetv2_101x3_bitm", pretrained=False)',
    "resnetv2_101_21k": 'timm.create_model("resnetv2_101x3_bitm_in21k", pretrained=False)',
    "resnetv2_152_bit": 'timm.create_model("resnetv2_152x4_bitm", pretrained=False)',
    "resnetv2_152_21k": 'timm.create_model("resnetv2_152x4_bitm_in21k", pretrained=False)',
    "resnetv2_152_384": 'timm.create_model("resnetv2_152x2_bit_teacher_384", pretrained=False)',
    "resnetv2_101": 'timm.create_model("resnetv2_101", pretrained=False)',
    "vgg11": "models.vgg11(pretrained=False)",
    "vgg19": "models.vgg19(pretrained=False)",
    "vgg19_bn": "models.vgg19_bn(pretrained=False)",
    "wideresnet50": "models.wide_resnet50_2(pretrained=False)",
    "wideresnet101": "models.wide_resnet101_2(pretrained=False)",
    "mnasnet_100": 'timm.create_model("mnasnet_100", pretrained=False)',
    "mnasnet_a1": 'timm.create_model("mnasnet_a1", pretrained=False)',
    "mnasnet_b1": 'timm.create_model("mnasnet_b1", pretrained=False)',
    "densenet121": 'timm.create_model("densenet121", pretrained=False)',
    "densenet201": 'timm.create_model("densenet201", pretrained=False)',
    "inception_v4": 'timm.create_model("inception_v4", pretrained=False)',
    "vit_small": 'timm.create_model("vit_small_patch16_224", pretrained=False)',
    "vit_base": 'timm.create_model("vit_base_patch16_224", pretrained=False)',
    "vit_large": 'timm.create_model("vit_large_patch16_224", pretrained=False)',
    "vit_r50": 'timm.create_model("vit_large_r50_s32_224", pretrained=False)',
    "vit_deit_base": 'timm.create_model("deit_base_patch16_224", pretrained=False)',
    "vit_deit_distilled": 'timm.create_model("deit_base_distilled_patch16_224", pretrained=False)',
    "vit_swin_base": 'timm.create_model("swin_base_patch4_window7_224", pretrained=False)',
    "vit_swin_large": 'timm.create_model("swin_large_patch4_window7_224", pretrained=False)',
    "efficientnet_b7": 'timm.create_model("tf_efficientnet_b7", pretrained=False)',
    "efficientnet_b5": 'timm.create_model("tf_efficientnet_b5", pretrained=False)',
    "efficientnet_b3": 'timm.create_model("tf_efficientnet_b3", pretrained=False)',
    "efficientnet_b1": 'timm.create_model("tf_efficientnet_b1", pretrained=False)',
    "efficientnetv2_m": 'timm.create_model("tf_efficientnetv2_m", pretrained=False)',
    "efficientnetv2_l": 'timm.create_model("tf_efficientnetv2_l", pretrained=False)',
    "efficientnet_b3a": 'timm.create_model("efficientnet_b3a", pretrained=False)',
}

_TORCHVISION_WEIGHTS = {
    "wideresnet50": "weights/wideresnet50/wide_resnet50_2.pth",
}


def _resolve_torchvision_weights(name):
    name = name.lower()
    if name in _TORCHVISION_WEIGHTS:
        return _TORCHVISION_WEIGHTS[name]
    return None


_DINOV3_WEIGHTS_DIRS = {
    "dinov3_vits": "weights/dinov3_vits",
    "dinov3_vitsplus": "weights/dinov3_vitsplus",
    "dinov3_vitb": "weights/dinov3_vitb",
    "dinov3_vitl": "weights/dinov3_vitl",
    "dinov3_vithplus": "weights/dinov3_vithplus",
    "dinov3": "weights/dinov3",
}


def _resolve_dinov3_weights_dir(name):
    name = name.lower()
    if name in _DINOV3_WEIGHTS_DIRS:
        return _DINOV3_WEIGHTS_DIRS[name]
    # Longest keys first so e.g. "dinov3_vitsplus" does not match "dinov3_vits".
    for key in sorted(_DINOV3_WEIGHTS_DIRS, key=len, reverse=True):
        if key in name:
            return _DINOV3_WEIGHTS_DIRS[key]
    if os.path.exists(_DINOV3_WEIGHTS_DIRS["dinov3"]):
        return _DINOV3_WEIGHTS_DIRS["dinov3"]
    raise FileNotFoundError(
        f"Unknown DINOv3 backbone name '{name}'. Known names: "
        f"{sorted(_DINOV3_WEIGHTS_DIRS)}. Place weights in the matching "
        f"weights/<name> directory."
    )


def load(name):
    if "dinov3" in name.lower():
        return DinoV3Backbone(weights_dir=_resolve_dinov3_weights_dir(name))

    weight_path = _resolve_torchvision_weights(name)
    if weight_path and os.path.exists(weight_path):
        model = eval(_BACKBONES[name])
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        return model

    return eval(_BACKBONES[name])

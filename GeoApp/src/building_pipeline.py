import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


# =========================
# Model (как в ноутбуке)
# =========================

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class UNetDecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class ResNet50UNet(nn.Module):
    """
    1:1 с ноутбуком. Возвращает {"out": logits}
    logits shape: [B,1,H,W]
    """
    def __init__(self, num_classes=1):
        super().__init__()
        self.encoder = resnet50(weights=None)
        self.encoder.fc = nn.Identity()

        self.enc0 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu)  # /2, 64
        self.pool = self.encoder.maxpool                                                  # /4
        self.enc1 = self.encoder.layer1                                                    # /4, 256
        self.enc2 = self.encoder.layer2                                                    # /8, 512
        self.enc3 = self.encoder.layer3                                                    # /16, 1024
        self.enc4 = self.encoder.layer4                                                    # /32, 2048

        self.dec4 = UNetDecoderBlock(in_ch=2048, skip_ch=1024, out_ch=512)  # /16
        self.dec3 = UNetDecoderBlock(in_ch=512,  skip_ch=512,  out_ch=256)  # /8
        self.dec2 = UNetDecoderBlock(in_ch=256,  skip_ch=256,  out_ch=128)  # /4
        self.dec1 = UNetDecoderBlock(in_ch=128,  skip_ch=64,   out_ch=64)   # /2

        self.head = nn.Sequential(
            ConvBNReLU(64, 32),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x0 = self.enc0(x)             # /2
        x1 = self.enc1(self.pool(x0)) # /4
        x2 = self.enc2(x1)            # /8
        x3 = self.enc3(x2)            # /16
        x4 = self.enc4(x3)            # /32

        d4 = self.dec4(x4, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)

        out = F.interpolate(d1, size=x.shape[-2:], mode="bilinear", align_corners=False)
        out = self.head(out)          # [B,1,H,W]
        return {"out": out}


# =========================
# Config
# =========================

@dataclass
class BuildingSegConfig:
    # положи сюда best_unet.pt
    weights_path: str = os.path.join("weights", "best_unet.pt")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True

    # В ноутбуке обучение было на patch_size=256
    # Можно использовать другие tile_size, но надежнее 256/512 и т.п. с паддингом.
    pad_mode: str = "reflect"  # "reflect" как наиболее “естественный” для краёв


_MODEL: Optional[nn.Module] = None
_MODEL_CFG: Optional[BuildingSegConfig] = None


def _get_model(cfg: BuildingSegConfig) -> nn.Module:
    """
    Загружаем модель один раз (в память + на GPU).
    Чекпойнт ожидаем в формате:
      {"model_state_dict": ... , ...}
    """
    global _MODEL, _MODEL_CFG

    if _MODEL is not None and _MODEL_CFG is not None:
        # если путь тот же — используем загруженную
        if os.path.abspath(_MODEL_CFG.weights_path) == os.path.abspath(cfg.weights_path):
            return _MODEL

    if not os.path.exists(cfg.weights_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {cfg.weights_path}\n"
            f"Ожидаю best_unet.pt рядом с проектом в папке weights/."
        )

    model = ResNet50UNet(num_classes=1)

    ckpt = torch.load(cfg.weights_path, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    # на случай module. префикса
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.to(cfg.device)
    model.eval()

    _MODEL = model
    _MODEL_CFG = cfg
    return model


def _preprocess_tile(tile_rgb: np.ndarray) -> torch.Tensor:
    """
    ВАЖНО: как в ноутбуке: просто float 0..1, без mean/std.
    tile_rgb: uint8 [h,w,3]
    return: float tensor [1,3,h,w]
    """
    x = tile_rgb.astype(np.float32)

    # поддержка uint16 (на всякий случай)
    if x.max() > 255.0:
        x = x / 65535.0
    else:
        x = x / 255.0

    x = np.transpose(x, (2, 0, 1))  # [3,h,w]
    x = torch.from_numpy(x).unsqueeze(0)  # [1,3,h,w]
    return x


@torch.no_grad()
def _predict_prob_mask_tiled(
    image_rgb: np.ndarray,
    model: nn.Module,
    cfg: BuildingSegConfig,
    tile_size: int,
    stride: int,
) -> Tuple[np.ndarray, Dict]:
    """
    Возвращает prob_mask float32 [H,W] в [0..1]
    Склейка: sum/count (правильно для overlap)
    """
    H, W, _ = image_rgb.shape
    sum_map = np.zeros((H, W), dtype=np.float32)
    cnt_map = np.zeros((H, W), dtype=np.float32)

    tiles_count = 0

    # защитимся от глупостей
    tile_size = int(tile_size)
    stride = int(stride)
    if tile_size <= 0 or stride <= 0:
        raise ValueError("tile_size and stride must be positive")

    amp_enabled = cfg.use_amp and cfg.device.startswith("cuda")

    for y0 in range(0, H, stride):
        for x0 in range(0, W, stride):
            y1 = min(y0 + tile_size, H)
            x1 = min(x0 + tile_size, W)

            tile = image_rgb[y0:y1, x0:x1]

            # pad to tile_size (модель спокойно примет tile_size x tile_size)
            pad_bottom = tile_size - (y1 - y0)
            pad_right = tile_size - (x1 - x0)

            if pad_bottom > 0 or pad_right > 0:
                tile = np.pad(
                    tile,
                    ((0, pad_bottom), (0, pad_right), (0, 0)),
                    mode=cfg.pad_mode,
                )

            x = _preprocess_tile(tile).to(cfg.device, non_blocking=True)

            # torch.amp.autocast (современный)
            autocast_ctx = (
                torch.amp.autocast("cuda", dtype=torch.float16, enabled=True)
                if amp_enabled else
                torch.amp.autocast("cpu", enabled=False)
            )

            with autocast_ctx:
                logits = model(x)["out"]          # [1,1,tile,tile]
                probs = torch.sigmoid(logits)[0, 0]  # [tile,tile]

            prob_np = probs.float().cpu().numpy()

            # crop back (если был padding)
            prob_np = prob_np[: (y1 - y0), : (x1 - x0)]

            sum_map[y0:y1, x0:x1] += prob_np
            cnt_map[y0:y1, x0:x1] += 1.0

            tiles_count += 1

    prob_mask = sum_map / np.maximum(cnt_map, 1e-6)

    debug = {
        "tiles_count": tiles_count,
        "image_hw": [int(H), int(W)],
        "tile_size": tile_size,
        "stride": stride,
        "device": cfg.device,
        "use_amp": bool(amp_enabled),
    }
    return prob_mask.astype(np.float32), debug


def predict_building_mask(
    image_rgb: np.ndarray,
    tile_size: int,
    stride: int,
    threshold: float,
) -> Tuple[np.ndarray, Dict]:
    """
    API для Streamlit.
    Возвращает бинарную маску (0/1) и debug-словарь.
    """
    cfg = BuildingSegConfig()
    model = _get_model(cfg)

    prob_mask, debug = _predict_prob_mask_tiled(
        image_rgb=image_rgb,
        model=model,
        cfg=cfg,
        tile_size=tile_size,
        stride=stride,
    )

    thr = float(threshold)
    bin_mask = (prob_mask >= thr).astype(np.uint8)

    debug.update({
        "threshold": thr,
        "prob_min": float(prob_mask.min()),
        "prob_max": float(prob_mask.max()),
        "prob_mean": float(prob_mask.mean()),
        "mask_area_px": int(bin_mask.sum()),
        "weights_path": cfg.weights_path,
    })

    return bin_mask, debug

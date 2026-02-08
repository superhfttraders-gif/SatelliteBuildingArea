import os
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import cv2
from ultralytics import YOLO


# =========================
# Config
# =========================

@dataclass
class CarScaleConfig:
    """
    model_ref:
      - локальный путь: "weights/yolo26x-obb.pt"
      - или просто имя: "yolo26x-obb.pt" (ultralytics скачает сам)
    """
    model_ref: str = "yolo26x-obb.pt"

    car_class_id: int = 10

    # adaptive thresholds
    conf_primary: float = 0.10
    conf_fallback: float = 0.01

    # minimum cars for accepting result
    min_cars_primary: int = 6
    min_cars_fallback: int = 4

    iou_thres_tile: float = 0.50
    merge_cell_px: int = 3

    # batching (tiles)
    tile_batch: int = 16

    half: bool = True
    device: Optional[str] = None  # None=auto, or "cuda:0", "cpu"

    # +2% upscale перед детекцией
    pre_upscale: float = 1.02

    # Filters (car-like boxes)
    aspect_ratio_min: float = 1.2
    aspect_ratio_max: float = 3.8
    iqr_k: float = 1.5  # for sqrt(L*W) outlier filter
    iqr_min_n: int = 6  # apply IQR only if >= 6 dets (как в ноутбуке)

    # S1 базовая формула
    target_sqrtlw_m: float = 3.212

    # log-calib coefficients
    a: float = 0.11032050
    b: float = 1.02631206
    c: float = -0.01989536
    d: float = -0.00519021

    # shift
    k_shift: float = 0.9986315218974585


# =========================
# Global cached model
# =========================

_MODEL: Optional[YOLO] = None
_MODEL_REF: Optional[str] = None


def _resolve_model_ref(cfg: CarScaleConfig) -> str:
    ref = cfg.model_ref

    # если передано просто имя, попробуем weights/имя
    if os.path.sep not in ref and "/" not in ref:
        local_candidate = os.path.join("weights", ref)
        if os.path.exists(local_candidate) and os.path.getsize(local_candidate) > 0:
            return local_candidate

    return ref


def _get_model(cfg: CarScaleConfig) -> YOLO:
    global _MODEL, _MODEL_REF

    ref = _resolve_model_ref(cfg)

    if _MODEL is not None and _MODEL_REF is not None and _MODEL_REF == ref:
        return _MODEL

    model = YOLO(ref)  # Ultralytics сам скачает/кэширует при необходимости
    _MODEL = model
    _MODEL_REF = ref
    return model


# =========================
# Tiling
# =========================

def _generate_tiles_with_stride(img_bgr: np.ndarray, tile_size: int, stride: int):
    H, W = img_bgr.shape[:2]
    tile_size = int(tile_size)
    stride = int(stride)

    if tile_size <= 0 or stride <= 0:
        raise ValueError("tile_size and stride must be positive")

    xs = list(range(0, W, stride))
    ys = list(range(0, H, stride))

    for y0 in ys:
        for x0 in xs:
            x1 = min(x0 + tile_size, W)
            y1 = min(y0 + tile_size, H)

            tile = img_bgr[y0:y1, x0:x1]
            pad_right = tile_size - (x1 - x0)
            pad_bottom = tile_size - (y1 - y0)

            if pad_right > 0 or pad_bottom > 0:
                tile = cv2.copyMakeBorder(
                    tile, 0, pad_bottom, 0, pad_right,
                    borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )

            yield (x0, y0, x1, y1, tile)


# =========================
# OBB extraction + merge
# =========================

def _extract_detections_xywhr(result, keep_class_id: Optional[int] = None) -> List[Dict]:
    dets = []
    if not hasattr(result, "obb") or result.obb is None:
        return dets

    obb = result.obb
    boxes = obb.xywhr.cpu().numpy()  # (N,5): cx,cy,w,h,angle
    conf = obb.conf.cpu().numpy() if hasattr(obb, "conf") else None
    cls  = obb.cls.cpu().numpy().astype(int) if hasattr(obb, "cls") else None

    n = boxes.shape[0]
    for i in range(n):
        c = int(cls[i]) if cls is not None else -1
        if keep_class_id is not None and c != keep_class_id:
            continue

        dets.append({
            "cx": float(boxes[i, 0]),
            "cy": float(boxes[i, 1]),
            "w":  float(boxes[i, 2]),
            "h":  float(boxes[i, 3]),
            "angle": float(boxes[i, 4]),
            "conf": float(conf[i]) if conf is not None else 0.0,
            "cls": c
        })

    return dets


def _merge_xywhr_by_center_hash(detections: List[Dict], cell: int = 12) -> List[Dict]:
    if not detections:
        return []

    best = {}  # (gx,gy) -> det

    for d in detections:
        gx = int(d["cx"] // cell)
        gy = int(d["cy"] // cell)

        winner_key = None
        winner_conf = -1.0

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                k = (gx + dx, gy + dy)
                if k in best and best[k]["conf"] > winner_conf:
                    winner_conf = best[k]["conf"]
                    winner_key = k

        if winner_key is None:
            best[(gx, gy)] = d
        else:
            if d["conf"] > best[winner_key]["conf"]:
                best[winner_key] = d

    return list(best.values())


# =========================
# Filters (aspect ratio + IQR on sqrtLW)
# =========================

def _filter_by_aspect_ratio(dets: List[Dict], rmin: float, rmax: float) -> List[Dict]:
    if not dets:
        return []

    out = []
    for d in dets:
        w = float(d.get("w", 0.0))
        h = float(d.get("h", 0.0))
        if w <= 0 or h <= 0:
            continue
        L = max(w, h)
        W = min(w, h)
        if W <= 0:
            continue
        ratio = L / W
        if rmin <= ratio <= rmax:
            out.append(d)
    return out


def _filter_by_sqrtLW_iqr(dets: List[Dict], k: float, min_n: int) -> List[Dict]:
    """
    IQR filter on sqrt(L*W) to remove outlier boxes.
    Keeps values within [Q1-k*IQR, Q3+k*IQR].
    """
    if not dets or len(dets) < int(min_n):
        return dets

    w = np.array([d["w"] for d in dets], dtype=np.float32)
    h = np.array([d["h"] for d in dets], dtype=np.float32)

    L = np.maximum(w, h)
    W = np.minimum(w, h)
    s = np.sqrt(L * W)

    q1 = np.percentile(s, 25)
    q3 = np.percentile(s, 75)
    iqr = q3 - q1

    lo = q1 - k * iqr
    hi = q3 + k * iqr

    out = []
    for d in dets:
        Ld = max(float(d["w"]), float(d["h"]))
        Wd = min(float(d["w"]), float(d["h"]))
        sd = math.sqrt(max(Ld * Wd, 0.0))
        if lo <= sd <= hi:
            out.append(d)

    return out


def _apply_filters(dets: List[Dict], cfg: CarScaleConfig) -> List[Dict]:
    dets2 = _filter_by_aspect_ratio(dets, cfg.aspect_ratio_min, cfg.aspect_ratio_max)
    dets3 = _filter_by_sqrtLW_iqr(dets2, cfg.iqr_k, cfg.iqr_min_n)
    return dets3


# =========================
# Stats + GSD formula (S1)
# =========================

def _compute_stats_pixels(dets: List[Dict]) -> Dict:
    if not dets:
        return {
            "median_length_px": np.nan,
            "median_width_px":  np.nan,
            "median_sqrtLW_px": np.nan,
            "count": 0,
        }

    w = np.array([d["w"] for d in dets], dtype=np.float32)
    h = np.array([d["h"] for d in dets], dtype=np.float32)

    lengths = np.maximum(w, h)
    widths  = np.minimum(w, h)
    sqrtLW  = np.sqrt(lengths * widths)

    return {
        "median_length_px": float(np.median(lengths)),
        "median_width_px":  float(np.median(widths)),
        "median_sqrtLW_px": float(np.median(sqrtLW)),
        "count": int(len(dets)),
    }


def _gsd_from_stats_s1(med_sqrtLW_px: float, count: int, cfg: CarScaleConfig) -> float:
    # base
    gsd_pred = cfg.target_sqrtlw_m / float(med_sqrtLW_px)

    # calibrated
    log_g = np.log(max(gsd_pred, 1e-9))
    log_c = np.log(max(int(count), 1))

    log_true = cfg.a + cfg.b * log_g + cfg.c * log_c + cfg.d * (log_g * log_c)
    gsd_cal = float(np.exp(log_true))

    # shift
    gsd_cal *= cfg.k_shift
    return gsd_cal


# =========================
# Detection core (batched tiles)
# =========================

def _detect_image_tiled_batched(
    model: YOLO,
    img_bgr: np.ndarray,
    tile_size: int,
    stride: int,
    conf_thres: float,
    cfg: CarScaleConfig
) -> List[Dict]:
    H, W = img_bgr.shape[:2]

    all_dets_global: List[Dict] = []

    batch_tiles_rgb: List[np.ndarray] = []
    batch_offsets: List[Tuple[int, int]] = []

    def run_batch():
        nonlocal batch_tiles_rgb, batch_offsets, all_dets_global
        if not batch_tiles_rgb:
            return

        results = model.predict(
            source=batch_tiles_rgb,  # list of tiles
            imgsz=int(tile_size),
            conf=float(conf_thres),
            iou=float(cfg.iou_thres_tile),
            device=cfg.device,
            half=bool(cfg.half),
            classes=[int(cfg.car_class_id)],
            verbose=False
        )

        for r0, (x0, y0) in zip(results, batch_offsets):
            dets_tile = _extract_detections_xywhr(r0, keep_class_id=int(cfg.car_class_id))

            for d in dets_tile:
                cx = d["cx"] + x0
                cy = d["cy"] + y0
                if cx < 0 or cy < 0 or cx >= W or cy >= H:
                    continue

                all_dets_global.append({
                    "cx": cx, "cy": cy,
                    "w": d["w"], "h": d["h"],
                    "angle": d["angle"],
                    "conf": d["conf"],
                    "cls": d["cls"],
                })

        batch_tiles_rgb = []
        batch_offsets = []

    # collect tiles
    for x0, y0, x1, y1, tile_bgr in _generate_tiles_with_stride(img_bgr, tile_size, stride):
        tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
        batch_tiles_rgb.append(tile_rgb)
        batch_offsets.append((x0, y0))

        if len(batch_tiles_rgb) >= int(cfg.tile_batch):
            run_batch()

    # flush remainder
    run_batch()

    merged = _merge_xywhr_by_center_hash(all_dets_global, cell=int(cfg.merge_cell_px))
    return merged


# =========================
# Public API
# =========================

def estimate_gsd_from_cars(
    image_rgb: np.ndarray,
    tile_size: int,
    stride: int,
) -> Tuple[Optional[float], Dict]:
    """
    Input: image_rgb uint8 [H,W,3]
    Output: (gsd_m_per_px | None, stats dict)
    """
    cfg = CarScaleConfig()
    model = _get_model(cfg)

    # --- pre-upscale +2% ---
    upscale = float(cfg.pre_upscale)
    if upscale <= 0:
        upscale = 1.0

    img_rgb = image_rgb
    if abs(upscale - 1.0) > 1e-9:
        H0, W0 = img_rgb.shape[:2]
        newW = max(1, int(round(W0 * upscale)))
        newH = max(1, int(round(H0 * upscale)))
        img_rgb = cv2.resize(img_rgb, (newW, newH), interpolation=cv2.INTER_CUBIC)

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    Hs, Ws = img_bgr.shape[:2]

    # ---- try PRIMARY ----
    merged_primary_raw = _detect_image_tiled_batched(
        model=model,
        img_bgr=img_bgr,
        tile_size=int(tile_size),
        stride=int(stride),
        conf_thres=float(cfg.conf_primary),
        cfg=cfg,
    )
    merged_primary = _apply_filters(merged_primary_raw, cfg)
    st_primary = _compute_stats_pixels(merged_primary)

    used = "primary"
    conf_used = float(cfg.conf_primary)

    # ---- fallback logic ----
    if int(st_primary["count"]) < int(cfg.min_cars_primary):
        merged_fallback_raw = _detect_image_tiled_batched(
            model=model,
            img_bgr=img_bgr,
            tile_size=int(tile_size),
            stride=int(stride),
            conf_thres=float(cfg.conf_fallback),
            cfg=cfg,
        )
        merged_fallback = _apply_filters(merged_fallback_raw, cfg)
        st_fallback = _compute_stats_pixels(merged_fallback)

        if int(st_fallback["count"]) >= int(cfg.min_cars_fallback):
            used = "fallback"
            conf_used = float(cfg.conf_fallback)
            merged_used = merged_fallback
            st_used = st_fallback
            raw_count_used = len(merged_fallback_raw)
        else:
            # not enough cars even in fallback
            raw_count_primary = len(merged_primary_raw)
            raw_count_fallback = len(merged_fallback_raw)

            stats = {
                "gsd_status": "not_enough_cars",
                "used_pass": "none",

                "tile_size": int(tile_size),
                "stride": int(stride),
                "tile_batch": int(cfg.tile_batch),

                "pre_upscale": float(cfg.pre_upscale),
                "image_hw_upscaled": [int(Hs), int(Ws)],

                "conf_primary": float(cfg.conf_primary),
                "min_cars_primary": int(cfg.min_cars_primary),
                "primary_raw_count": int(raw_count_primary),
                "primary_count_after_filters": int(st_primary["count"]),

                "conf_fallback": float(cfg.conf_fallback),
                "min_cars_fallback": int(cfg.min_cars_fallback),
                "fallback_raw_count": int(raw_count_fallback),
                "fallback_count_after_filters": int(st_fallback["count"]),

                "aspect_ratio_min": float(cfg.aspect_ratio_min),
                "aspect_ratio_max": float(cfg.aspect_ratio_max),
                "iqr_k": float(cfg.iqr_k),
                "iqr_min_n": int(cfg.iqr_min_n),

                "model_ref": cfg.model_ref,
                "model_ref_resolved": _resolve_model_ref(cfg),
            }

            # debuggable weights path
            try:
                stats["weights_used"] = getattr(model, "ckpt_path", None) or getattr(model, "weights", None)
            except Exception:
                stats["weights_used"] = None

            return None, stats

    else:
        merged_used = merged_primary
        st_used = st_primary
        raw_count_used = len(merged_primary_raw)

    # stats common
    stats: Dict = {
        "gsd_status": "ok",
        "used_pass": used,
        "conf_used": float(conf_used),

        "tile_size": int(tile_size),
        "stride": int(stride),
        "tile_batch": int(cfg.tile_batch),

        "pre_upscale": float(cfg.pre_upscale),
        "image_hw_upscaled": [int(Hs), int(Ws)],

        "cars_raw_count": int(raw_count_used),
        "cars_count": int(st_used["count"]),
        "median_length_px": st_used["median_length_px"],
        "median_width_px": st_used["median_width_px"],
        "median_sqrtLW_px": st_used["median_sqrtLW_px"],

        "merge_cell_px": int(cfg.merge_cell_px),
        "iou_thres_tile": float(cfg.iou_thres_tile),

        "aspect_ratio_min": float(cfg.aspect_ratio_min),
        "aspect_ratio_max": float(cfg.aspect_ratio_max),
        "iqr_k": float(cfg.iqr_k),
        "iqr_min_n": int(cfg.iqr_min_n),

        "model_ref": cfg.model_ref,
        "model_ref_resolved": _resolve_model_ref(cfg),
    }

    try:
        stats["weights_used"] = getattr(model, "ckpt_path", None) or getattr(model, "weights", None)
    except Exception:
        stats["weights_used"] = None

    # sanity check
    if st_used["count"] <= 0 or not np.isfinite(st_used["median_sqrtLW_px"]) or st_used["median_sqrtLW_px"] <= 0:
        stats["gsd_status"] = "no_cars_detected_after_filters"
        return None, stats

    # gsd on upscaled
    gsd_upscaled = _gsd_from_stats_s1(st_used["median_sqrtLW_px"], int(st_used["count"]), cfg)

    # correct back to original scale
    gsd_original = float(gsd_upscaled * upscale)

    stats.update({
        "gsd_pred_m_per_px_upscaled": float(gsd_upscaled),
        "gsd_m_per_px": float(gsd_original),
    })

    return gsd_original, stats

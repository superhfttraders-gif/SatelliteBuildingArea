import numpy as np

def area_px(mask01: np.ndarray) -> int:
    # mask01: [H,W] 0/1
    return int(np.sum(mask01.astype(np.uint8)))

def area_m2(area_px_value: int, gsd_m_per_px: float) -> float:
    # площадь = (число пикселей) * (м/пикс)^2
    return float(area_px_value) * float(gsd_m_per_px) ** 2

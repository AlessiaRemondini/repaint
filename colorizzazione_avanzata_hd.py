from __future__ import annotations
import os
import sys
import math
import time
import argparse
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import skimage.color as skc
    import skimage.exposure as ske
    HAS_SKI = True
except Exception:
    HAS_SKI = False

import matplotlib.pyplot as plt


# ================== PARAMETRI ==================
@dataclass
class Params:
    output_res: Tuple[int, int] = (1700, 1368)  # (H, W)
    resize_mode: str = "fit"                    # 'fit' | 'fill' | 'stretch'
    pad_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    dark_th: float = 0.30
    bright_th: float = 0.70

    grid_xy_bins: int = 32
    grid_L_bins: int = 16
    grid_sigma: float = 0.8

    bilat_r: int = 3
    bilat_sigma_s: float = 2.0
    bilat_sigma_r: float = 0.10

    unsharp_amt: float = 0.6

    alpha_blend: float = 0.6
    gamma: float = 0.8


# ================== UTIL ==================

def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def im2double_local(x: np.ndarray) -> np.ndarray:
    """Converte in float64 in [0,1]."""
    if np.issubdtype(x.dtype, np.floating):
        d = x.astype(np.float64, copy=False)
    elif np.issubdtype(x.dtype, np.integer):
        d = x.astype(np.float64) / np.iinfo(x.dtype).max
    else:
        d = x.astype(np.float64)
    return clip01(d)


def mat2gray_local(A: np.ndarray) -> np.ndarray:
    A = A.astype(np.float64)
    mn, mx = float(np.min(A)), float(np.max(A))
    if mx > mn:
        return (A - mn) / (mx - mn)
    return np.zeros_like(A)


def rgb2gray_fallback(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 3:
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        gimg = 0.2989360213 * r + 0.5870430745 * g + 0.1140209043 * b
    else:
        gimg = img
    return clip01(gimg)


def blend_images(a: np.ndarray, b: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    return clip01(alpha * clip01(a) + (1.0 - alpha) * clip01(b))


def gamma_adjust(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    return clip01(clip01(img) ** gamma)


def zone_mean_rgb(img: np.ndarray, mask: np.ndarray, fallback_rgb: np.ndarray) -> np.ndarray:
    m = np.zeros((1, 1, 3), dtype=np.float64)
    if np.any(mask):
        for c in range(3):
            vals = img[..., c][mask]
            m[0, 0, c] = float(np.mean(vals)) if vals.size > 0 else float(fallback_rgb[0, 0, c])
    else:
        m = fallback_rgb
    return m


# ================== RESIZE ==================

def _ensure_3c(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[2] == 1:
        return np.repeat(img, 3, axis=2)
    return img


def resize_img_local(img: np.ndarray, new_hw: Tuple[int, int], method: str = "bicubic") -> np.ndarray:
    h, w = int(new_hw[0]), int(new_hw[1])
    img = im2double_local(img)

    if Image is not None:
        pil_mode = "RGB" if (img.ndim == 3 and img.shape[2] == 3) else "L"
        # Conversione sicura per evitare errori su array float
        arr8 = (clip01(img) * 255.0).astype(np.uint8) 
        pim = Image.fromarray(arr8.squeeze(), mode=pil_mode)
        resample = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
        }.get(method.lower(), Image.BICUBIC)
        pim = pim.resize((w, h), resample=resample)
        out = np.asarray(pim).astype(np.float64) / 255.0
        if pil_mode == "L":
            out = out.reshape(h, w)
        else:
            out = out.reshape(h, w, 3)
        return clip01(out)

    return _bilinear_resize_numpy(img, h, w)


def _bilinear_resize_numpy(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    img = im2double_local(img)
    H, W = img.shape[:2]
    if img.ndim == 2:
        img = img[..., None]
    C = img.shape[2]
    y = np.linspace(0, H - 1, out_h)
    x = np.linspace(0, W - 1, out_w)
    x0 = np.floor(x).astype(int)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y0 = np.floor(y).astype(int)
    y1 = np.clip(y0 + 1, 0, H - 1)

    dx = (x - x0)[None, :]
    dy = (y - y0)[:, None]

    Ia = img[y0[:, None], x0[None, :]]
    Ib = img[y0[:, None], x1[None, :]]
    Ic = img[y1[:, None], x0[None, :]]
    Id = img[y1[:, None], x1[None, :]]

    out = (Ia * (1 - dx) * (1 - dy) +
           Ib * dx * (1 - dy) +
           Ic * (1 - dx) * dy +
           Id * dx * dy)
    out = out.reshape(out_h, out_w, C)
    return clip01(out if C == 3 else out[..., 0])


def resize_to_canvas_local(img: np.ndarray, outH: int, outW: int, mode: str = "fit",
                           pad_color: Tuple[float, float, float] = (0, 0, 0)) -> np.ndarray:
    img = im2double_local(img)
    h, w = img.shape[:2]
    img3 = _ensure_3c(img)

    mode_l = (mode or "fit").lower()
    if mode_l == "stretch":
        return resize_img_local(img3, (outH, outW), method="bicubic")

    if mode_l == "fill":
        scale = max(outW / w, outH / h)
        newW, newH = max(1, round(w * scale)), max(1, round(h * scale))
        tmp = resize_img_local(img3, (newH, newW), method="bicubic")
        y1 = (newH - outH) // 2
        x1 = (newW - outW) // 2
        return clip01(tmp[y1:y1 + outH, x1:x1 + outW, :])

    # fit (letterbox)
    scale = min(outW / w, outH / h)
    newW, newH = max(1, round(w * scale)), max(1, round(h * scale))
    tmp = resize_img_local(img3, (newH, newW), method="bicubic")
    out = np.zeros((outH, outW, 3), dtype=np.float64)
    out[:] = np.array(pad_color, dtype=np.float64).reshape(1, 1, 3)
    y1 = (outH - newH) // 2
    x1 = (outW - newW) // 2
    out[y1:y1 + newH, x1:x1 + newW, :] = tmp
    return clip01(out)


# ================== HSV CONVERSIONI ==================

def rgb2hsv_local(rgb: np.ndarray) -> np.ndarray:
    rgb = clip01(rgb)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = np.max(rgb, axis=-1)
    cmin = np.min(rgb, axis=-1)
    delta = cmax - cmin + 1e-12

    h = np.zeros_like(cmax)
    mask = delta > 0
    r_is_max = (cmax == r) & mask
    g_is_max = (cmax == g) & mask
    b_is_max = (cmax == b) & mask

    h[r_is_max] = ((g - b)[r_is_max] / delta[r_is_max]) % 6
    h[g_is_max] = ((b - r)[g_is_max] / delta[g_is_max]) + 2
    h[b_is_max] = ((r - g)[b_is_max] / delta[b_is_max]) + 4
    h = (h / 6.0) % 1.0

    s = np.where(cmax == 0, 0.0, delta / (cmax + 1e-12))
    v = cmax
    return np.stack([h, s, v], axis=-1)


def hsv2rgb_local(hsv: np.ndarray) -> np.ndarray:
    H = hsv[..., 0] * 6.0
    S = hsv[..., 1]
    V = hsv[..., 2]
    C = V * S
    X = C * (1 - np.abs((H % 2) - 1))
    m = V - C

    r = np.zeros_like(H)
    g = np.zeros_like(H)
    b = np.zeros_like(H)

    conds = [
        (0 <= H) & (H < 1),
        (1 <= H) & (H < 2),
        (2 <= H) & (H < 3),
        (3 <= H) & (H < 4),
        (4 <= H) & (H < 5),
        (5 <= H) & (H < 6),
    ]

    r = np.where(conds[0], C, r)
    g = np.where(conds[0], X, g)
    r = np.where(conds[1], X, r)
    g = np.where(conds[1], C, g)
    g = np.where(conds[2], C, g)
    b = np.where(conds[2], X, b)
    g = np.where(conds[3], X, g)
    b = np.where(conds[3], C, b)
    r = np.where(conds[4], X, r)
    b = np.where(conds[4], C, b)
    r = np.where(conds[5], C, r)
    b = np.where(conds[5], X, b)

    r += m
    g += m
    b += m
    return clip01(np.stack([r, g, b], axis=-1))


# ================== GAUSS/CONVOLUZIONI ==================

def gaussian1d_local(sigma: float, ksz: Optional[int] = None) -> np.ndarray:
    if ksz is None:
        ksz = max(3, 2 * int(math.ceil(2 * sigma)) + 1)
    x = np.arange(-(ksz - 1) / 2, (ksz - 1) / 2 + 1)
    g = np.exp(-(x ** 2) / (2 * sigma ** 2))
    g /= np.sum(g)
    return g.astype(np.float64)


def _pad_reflect(a: np.ndarray, pad: int, axis: int) -> np.ndarray:
    pad_width = [(0, 0)] * a.ndim
    pad_width[axis] = (pad, pad)
    return np.pad(a, pad_width, mode='reflect')


def conv1d_along_axis(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    k = kernel.reshape([-1])
    pad = len(k) // 2
    arr_pad = _pad_reflect(arr, pad, axis)
    arr_t = np.moveaxis(arr_pad, axis, -1)
    
    from numpy.lib.stride_tricks import sliding_window_view
    sw = sliding_window_view(arr_t, window_shape=(len(k),), axis=-1)
    out_t = np.tensordot(sw, k, axes=([-1], [0]))
    return np.moveaxis(out_t, -1, axis)


def gaussian_blur2d(img: np.ndarray, sigma: float = 1.0, ksz: int = 5) -> np.ndarray:
    g = gaussian1d_local(sigma, ksz)
    tmp = conv1d_along_axis(img, g, axis=0)
    out = conv1d_along_axis(tmp, g, axis=1)
    return out


def smooth3_separable(vol: np.ndarray, sigma: float) -> np.ndarray:
    g = gaussian1d_local(sigma)
    out = conv1d_along_axis(vol, g, axis=0)
    out = conv1d_along_axis(out, g, axis=1)
    out = conv1d_along_axis(out, g, axis=2)
    return out


# ================== BILATERAL GRID ==================

def _to_bins_ceiling(norm_vals: np.ndarray, nbins: int) -> np.ndarray:
    idx = np.ceil(np.clip(norm_vals, 0, 1) * nbins).astype(int) - 1
    return np.clip(idx, 0, nbins - 1)


def bilateral_grid_colorize_local(ref_rgb: np.ndarray, target_L: np.ndarray, params: Params):
    ref_rgb = clip01(ref_rgb)
    rows, cols = ref_rgb.shape[:2]

    ref_hsv = rgb2hsv_local(ref_rgb)
    H = ref_hsv[..., 0]
    S = ref_hsv[..., 1]
    Vref = ref_hsv[..., 2]

    xg, yg = np.meshgrid(np.arange(cols), np.arange(rows))
    xg = (xg + 0.5) / cols
    yg = (yg + 0.5) / rows

    GX = GY = int(params.grid_xy_bins)
    GL = int(params.grid_L_bins)

    ix = _to_bins_ceiling(xg, GX)
    iy = _to_bins_ceiling(yg, GY)
    iz_ref = _to_bins_ceiling(Vref, GL)

    theta = 2 * np.pi * H
    w = np.maximum(S, 1e-6)

    size = (GY, GX, GL)
    Wcos = np.zeros(size, dtype=np.float64)
    Wsin = np.zeros(size, dtype=np.float64)
    Wsat = np.zeros(size, dtype=np.float64)
    Wcnt = np.zeros(size, dtype=np.float64)

    lin = (iy * GX + ix) * GL + iz_ref
    lin = lin.ravel()

    np.add.at(Wcos.ravel(), lin, np.cos(theta).ravel() * w.ravel())
    np.add.at(Wsin.ravel(), lin, np.sin(theta).ravel() * w.ravel())
    np.add.at(Wsat.ravel(), lin, S.ravel())
    np.add.at(Wcnt.ravel(), lin, np.ones_like(S).ravel())

    Wcos = smooth3_separable(Wcos, params.grid_sigma)
    Wsin = smooth3_separable(Wsin, params.grid_sigma)
    Wsat = smooth3_separable(Wsat, params.grid_sigma)
    Wcnt = smooth3_separable(Wcnt, params.grid_sigma)

    epsc = 1e-8
    H_grid = np.mod(np.arctan2(Wsin, Wcos) / (2 * np.pi), 1.0)
    S_grid = clip01(Wsat / np.maximum(Wcnt, epsc))

    iz_t = _to_bins_ceiling(clip01(target_L), GL)
    ix_t = _to_bins_ceiling((np.arange(cols) + 0.5) / cols, GX)
    iy_t = _to_bins_ceiling((np.arange(rows) + 0.5) / rows, GY)

    ix_tg, iy_tg = np.meshgrid(ix_t, iy_t)
    lin_t = (iy_tg * GX + ix_tg) * GL + iz_t
    H_out = H_grid.ravel()[lin_t.ravel()].reshape(rows, cols)
    S_out = S_grid.ravel()[lin_t.ravel()].reshape(rows, cols)

    hsv_out = np.stack([H_out, S_out, target_L], axis=-1)
    rgb_out = hsv2rgb_local(hsv_out)
    return hsv_out, clip01(rgb_out)


# ================== JOINT BILATERAL ==================

def joint_bilateral_color_local(img_rgb: np.ndarray, guide_gray: np.ndarray,
                                r: int, sigma_s: float, sigma_r: float) -> np.ndarray:
    img_rgb = clip01(img_rgb)
    guide = clip01(guide_gray)
    rows, cols = guide.shape

    X, Y = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
    Gsp = np.exp(-((X ** 2 + Y ** 2) / (2 * sigma_s ** 2)))

    out = np.zeros_like(img_rgb)
    for y in range(rows):
        y1, y2 = max(0, y - r), min(rows - 1, y + r)
        for x in range(cols):
            x1, x2 = max(0, x - r), min(cols - 1, x + r)
            patchG = guide[y1:y2 + 1, x1:x2 + 1]
            patchI = img_rgb[y1:y2 + 1, x1:x2 + 1, :]
            gc = guide[y, x]
            Gr = np.exp(-((patchG - gc) ** 2) / (2 * sigma_r ** 2))
            Gs = Gsp[(y1 - (y - r)):(y2 - (y - r) + 1), (x1 - (x - r)):(x2 - (x - r) + 1)]
            W = Gs * Gr
            Wsum = np.sum(W) + 1e-12
            out[y, x, :] = np.sum(patchI * W[..., None], axis=(0, 1)) / Wsum
    return clip01(out)


# ================== UNSHARP ==================

def unsharp_luminance_local(V: np.ndarray, amount: float = 0.6) -> np.ndarray:
    V = clip01(V)
    Vb = gaussian_blur2d(V, sigma=1.0, ksz=5)
    hp = V - Vb
    return clip01(V + amount * hp)



def colorizzazione_avanzata_hd(bw_img: np.ndarray, ref_img: np.ndarray, params: Params = None) -> dict:
    """
    Funzione adattata per backend (FastAPI/Render).
    Input: Array Numpy [0,1]
    Output: Dizionario con risultati
    """
    if params is None:
        params = Params()

    # 1. Utilizza direttamente le immagini passate dal server
    # bw_img e ref_img sono già array normalizzati e caricati da main.py
    I_bw = bw_img
    I_ref = ref_img

    # 2. Resize e adattamento al canvas (secondo i params)
    bw_img_res = resize_to_canvas_local(I_bw, params.output_res[0], params.output_res[1], params.resize_mode, params.pad_color)
    col_img_res = resize_to_canvas_local(I_ref, params.output_res[0], params.output_res[1], params.resize_mode, params.pad_color)

    # 3. Conversione B&N (Luma fallback)
    if bw_img_res.ndim == 3 and bw_img_res.shape[2] == 3:
        bw_gray = rgb2gray_fallback(bw_img_res)
    else:
        bw_gray = bw_img_res if bw_img_res.ndim == 2 else bw_img_res[..., 0]

    rows, cols = bw_gray.shape

    # Contenitori risultati
    results: List[np.ndarray] = []
    names: List[str] = []

    # ===== Metodo 1 (opz.) — LAB transfer =====
    if HAS_SKI:
        try:
            col_lab = skc.rgb2lab(col_img_res)
            bw_lab = skc.rgb2lab(np.dstack([bw_gray, bw_gray, bw_gray]))
            out_lab = bw_lab.copy()
            out_lab[..., 1] = col_lab[..., 1]
            out_lab[..., 2] = col_lab[..., 2]
            result1 = clip01(skc.lab2rgb(out_lab))
            results.append(result1)
        except:
            results.append(bw_gray)
    else:
        results.append(bw_gray) # Fallback se manca scikit-image

    names.append('lab_transfer')

    # ===== Metodo 2 — Zone di luminosità =====
    dark_mask = bw_gray < params.dark_th
    mid_mask = (bw_gray >= params.dark_th) & (bw_gray <= params.bright_th)
    bright_mask = bw_gray > params.bright_th

    fallback_rgb = np.mean(col_img_res.reshape(-1, 3), axis=0).reshape(1, 1, 3)
    col_dark = zone_mean_rgb(col_img_res, dark_mask, fallback_rgb)
    col_mid = zone_mean_rgb(col_img_res, mid_mask, fallback_rgb)
    col_bright = zone_mean_rgb(col_img_res, bright_mask, fallback_rgb)

    result2 = np.zeros((rows, cols, 3), dtype=np.float64)
    for i in range(3):
        result2[..., i] = (dark_mask * col_dark[0, 0, i] +
                           mid_mask * col_mid[0, 0, i] +
                           bright_mask * col_bright[0, 0, i])
    mL = float(np.mean(bw_gray)) or 1.0
    intensity_factor = bw_gray / mL
    result2 = clip01(result2 * intensity_factor[..., None])
    results.append(result2); names.append('luminosity_zones')

    # ===== Metodo 3 — HSV mixing semplice =====
    col_hsv = rgb2hsv_local(col_img_res)
    hue_map = resize_img_local(col_hsv[..., 0], (rows, cols), method='bilinear')
    sat_map = resize_img_local(col_hsv[..., 1], (rows, cols), method='bilinear')
    sat_mod = sat_map * (0.3 + 0.7 * (1.0 - bw_gray))
    result3_hsv = np.stack([hue_map, sat_mod, bw_gray], axis=-1)
    result3 = hsv2rgb_local(result3_hsv)
    result3 = clip01(result3)
    results.append(result3); names.append('advanced_mixing')

    # ===== Metodo 3B — Bilateral grid + Joint bilateral + Unsharp =====
    # Nota: Bilateral grid potrebbe essere lento su risoluzioni alte, qui manteniamo la logica
    _, result5 = bilateral_grid_colorize_local(col_img_res, bw_gray, params)
    
    # Joint bilateral (Attenzione: O(r^2), lento se r è grande)
    result5_ref = joint_bilateral_color_local(result5, bw_gray,
                                              params.bilat_r, params.bilat_sigma_s, params.bilat_sigma_r)
    
    V = rgb2hsv_local(result5_ref)[..., 2]
    Vsh = unsharp_luminance_local(V, params.unsharp_amt)
    hsv_ref = rgb2hsv_local(result5_ref)
    hsv_ref[..., 2] = Vsh
    result5_sharp = hsv2rgb_local(hsv_ref)
    result5_sharp = clip01(result5_sharp)
    
    results.append(result5_ref);   names.append('bilateral_grid_refined')
    results.append(result5_sharp); names.append('bilateral_grid_sharp')

    # ===== Metodo 4 (opz.) — Equalizzazione + LAB =====
    if HAS_SKI:
        try:
            bw_eq = ske.equalize_hist(clip01(bw_gray))
            col_lab = skc.rgb2lab(col_img_res)
            bwq_lab = skc.lab2rgb(np.dstack([bw_eq, bw_eq, bw_eq]))
            out_lab = bwq_lab.copy()
            out_lab[..., 1] = col_lab[..., 1]
            out_lab[..., 2] = col_lab[..., 2]
            result4 = clip01(skc.lab2rgb(out_lab))
            results.append(result4)
        except:
            results.append(bw_gray)
    else:
        results.append(bw_gray)
    names.append('histogram_eq')

    # ===== Blend + Enhance =====
    # Usa il risultato del metodo 3 o 1 come base A
    idx_base = 0 if len(results) > 0 else 0
    baseA = results[idx_base] # Default
    if len(results) >= 3:
         baseA = results[2] # advanced_mixing

    baseB = result5_sharp
    result_blend = blend_images(baseA, baseB, params.alpha_blend)
    results.append(result_blend); names.append('blend')

    result_enhanced = gamma_adjust(baseB, params.gamma)
    results.append(result_enhanced); names.append('enhanced')

    # ===== OUTPUT FINALE PER MAIN.PY =====
    # Creiamo il dizionario che serve al server
    # Riempiamo con fallback se alcuni metodi sono falliti
    def get_res(idx):
        return results[idx] if len(results) > idx else bw_gray

    output_dict = {
        "method1": get_res(0), # Lab Transfer
        "method2": get_res(1), # Luminosity Zones
        "method3": get_res(2), # Advanced Mixing
        "method4": get_res(3), # Bilateral Refined
        "method5": get_res(4), # Bilateral Sharp
        "method6": get_res(5), # Histogram EQ
        "method7": get_res(6), # Blend
        "method8": get_res(7), # Enhanced
    }
    
    return output_dict


# ================== CLI (Per test locali) ==================
# Questo blocco if __name__ == "__main__" assicura che il server ignori questa parte
if __name__ == '__main__':
    # Esempio fittizio se esegui lo script da solo
    print("Questo script è ora configurato come modulo per il backend.")
    print("Per testarlo, usa main.py.")

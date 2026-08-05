"""
stego_ai.stego_algorithms
----------------------------

Practical steganography utilities.

Algorithms
----------
- **LSB (robust)**: bulletproof text hide/reveal (seeded permutation).
- **PVD**: parity-of-difference embedding (noise generation).
- **DFT / SVD**: perturbation-based noise (augmentation).
- **DCT / DWT**: transform-domain embedding with reliable extraction.

DCT/DWT reliability fixes
-------------------------
Transform-domain stego often fails after saving to PNG because the pipeline
(transform -> tweak coeffs -> inverse -> uint8 clamp/round -> transform again)
introduces coefficient drift. If you store bits as plain LSB of a coefficient,
those bits flip.

This module fixes that with:
1) QIM (Quantization Index Modulation): store the bit in the parity of the
   quantized coefficient index.
2) Redundancy: repeat each bit `rep` times, then majority vote.
3) Self-describing header embedded deterministically (no RNG required to read it).
   Header contains payload length, CRC32, seed, and parameters.

So DCT/DWT actually round-trip through PNG.

Notes
-----
- LSB is still the king for live demos.
- For text embedding, this module keeps the old behavior (safe mode) for all
  algorithms except when you explicitly ask for 'dct' or 'dwt'.
"""

from __future__ import annotations

import os
import random
import struct
import zlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import cv2

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

DELIMITER = "###END###"

HEADER_MAGIC = b"STG2"
HEADER_VERSION = 1

ALG_ID = {"dct": 1, "dwt": 2}
ID_ALG = {1: "dct", 2: "dwt"}

# Fixed 32-byte header.
# Layout:
#  4s magic
#  B  version
#  B  alg_id
#  B  flags (reserved)
#  B  payload_rep
#  H  payload_delta
#  I  payload_len_bytes
#  I  crc32
#  I  payload_seed
#  H  reserved
#  8s reserved
HEADER_FMT = ">4sBBBBHIIIH8s"
HEADER_BYTES = struct.calcsize(HEADER_FMT)  # 32

# Strong settings for the header (so header can always be recovered).
HEADER_REP = 7
HEADER_DELTA_DCT = 20
HEADER_DELTA_DWT = 12

# Default payload settings
DEFAULT_DCT_DELTA = 10
DEFAULT_DCT_REP = 5

DEFAULT_DWT_DELTA = 6
DEFAULT_DWT_REP = 5

# DCT coefficient coordinates used for embedding (avoid DC (0,0)).
DCT_COORDS: List[Tuple[int, int]] = [
    (1, 2), (2, 1), (2, 2),
    (1, 3), (3, 1), (3, 2),
    (2, 3), (4, 1), (1, 4),
    (3, 3), (4, 2), (2, 4),
]


def payload_capacity_bytes(height: int, width: int, algorithm: str) -> int:
    """Return usable text/noise bytes after transform headers and redundancy."""
    alg = algorithm.lower().strip()
    header_samples = HEADER_BYTES * 8 * HEADER_REP
    if alg == "dct":
        blocks = ((int(height) + 7) // 8) * ((int(width) + 7) // 8)
        total_slots = blocks * len(DCT_COORDS)
        usable_bits = max(0, (total_slots - header_samples) // DEFAULT_DCT_REP)
        return usable_bits // 8
    if alg == "dwt":
        band_height = (int(height) + 1) // 2
        band_width = (int(width) + 1) // 2
        total_slots = 2 * band_height * band_width
        usable_bits = max(0, (total_slots - header_samples) // DEFAULT_DWT_REP)
        return usable_bits // 8
    if alg == "lsb":
        return max(0, (int(height) * int(width) * 3) // 8)
    if alg == "pvd":
        return max(0, (int(height) * int(width) * 3) // 16)
    return 0

# -----------------------------------------------------------------------------
# BIT / BYTE HELPERS
# -----------------------------------------------------------------------------

def bytes_to_bits(b: bytes) -> np.ndarray:
    """Bytes -> bits (uint8 array of 0/1), MSB-first."""
    out: List[int] = []
    for byte in b:
        for i in range(8):
            out.append((byte >> (7 - i)) & 1)
    return np.array(out, dtype=np.uint8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Bits -> bytes (truncates incomplete final byte)."""
    if bits.size == 0:
        return b""
    n_full = (bits.size // 8) * 8
    bits = bits[:n_full]
    out = bytearray()
    for i in range(0, n_full, 8):
        v = 0
        for b in bits[i:i + 8]:
            v = (v << 1) | int(b)
        out.append(v)
    return bytes(out)


def str_to_bits(s: str) -> np.ndarray:
    """String -> bits with delimiter (used for LSB safe mode)."""
    return bytes_to_bits((s + DELIMITER).encode("utf-8"))


def bits_to_str(bits: np.ndarray) -> str:
    """Bits -> string, stopping at delimiter (used for LSB safe mode)."""
    data = bits_to_bytes(bits)
    try:
        full = data.decode("utf-8", errors="ignore")
    except Exception:
        return "<decoding_error>"
    if DELIMITER in full:
        return full.split(DELIMITER)[0]
    return full


def _get_random_bits(count: int, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    return np.array([rng.getrandbits(1) for _ in range(count)], dtype=np.uint8)


def _get_random_bytes(nbytes: int, seed: int) -> bytes:
    rg = np.random.default_rng(seed)
    return rg.integers(0, 256, size=max(1, int(nbytes)), dtype=np.uint8).tobytes()

# -----------------------------------------------------------------------------
# HEADER PACK/UNPACK
# -----------------------------------------------------------------------------

@dataclass
class StegoHeader:
    alg_id: int
    payload_rep: int
    payload_delta: int
    payload_len: int
    crc32: int
    payload_seed: int
    flags: int = 0


def _pack_header(h: StegoHeader) -> bytes:
    return struct.pack(
        HEADER_FMT,
        HEADER_MAGIC,
        HEADER_VERSION,
        int(h.alg_id) & 0xFF,
        int(h.flags) & 0xFF,
        int(h.payload_rep) & 0xFF,
        int(h.payload_delta) & 0xFFFF,
        int(h.payload_len) & 0xFFFFFFFF,
        int(h.crc32) & 0xFFFFFFFF,
        int(h.payload_seed) & 0xFFFFFFFF,
        0,
        b"\x00" * 8,
    )


def _unpack_header(b: bytes) -> StegoHeader:
    if len(b) != HEADER_BYTES:
        raise ValueError(f"Header must be {HEADER_BYTES} bytes, got {len(b)}")
    magic, ver, alg_id, flags, rep, delta, plen, crc, seed, _, _ = struct.unpack(HEADER_FMT, b)
    if magic != HEADER_MAGIC or ver != HEADER_VERSION:
        raise ValueError("Invalid stego header (magic/version mismatch)")
    if alg_id not in ID_ALG:
        raise ValueError(f"Unknown alg_id in header: {alg_id}")
    return StegoHeader(
        alg_id=int(alg_id),
        payload_rep=int(rep),
        payload_delta=int(delta),
        payload_len=int(plen),
        crc32=int(crc),
        payload_seed=int(seed),
        flags=int(flags),
    )

# -----------------------------------------------------------------------------
# QIM + MAJORITY VOTE
# -----------------------------------------------------------------------------

def _qim_set(val: float, bit: int, delta: int) -> float:
    """Set bit using parity of quantization index."""
    if delta <= 0:
        raise ValueError("delta must be > 0")
    bit = 1 if bit else 0
    q = int(np.round(val / float(delta)))
    q_even = q - (q & 1)
    q_odd = q_even + 1
    target_q = q_even if bit == 0 else q_odd
    candidates = [target_q, target_q + 2, target_q - 2]
    best_q = min(candidates, key=lambda qq: abs((qq * delta) - val))
    return float(best_q * delta)


def _qim_get(val: float, delta: int) -> int:
    if delta <= 0:
        return 0
    q = int(np.round(val / float(delta)))
    return q & 1


def _majority(bits: Sequence[int]) -> int:
    ones = sum(1 for b in bits if b)
    return 1 if ones > (len(bits) // 2) else 0

# -----------------------------------------------------------------------------
# IMAGE PADDING HELPERS
# -----------------------------------------------------------------------------

def _pad_to_multiple(mat: np.ndarray, mult: int, mode: str = "edge") -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = mat.shape[:2]
    pad_h = (mult - (h % mult)) % mult
    pad_w = (mult - (w % mult)) % mult
    if pad_h == 0 and pad_w == 0:
        return mat, (h, w)
    padded = np.pad(mat, ((0, pad_h), (0, pad_w)), mode=mode)
    return padded, (h, w)


def _crop_to_shape(mat: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    return mat[:h, :w]

# -----------------------------------------------------------------------------
# ROBUST LSB (TEXT DEMO KING)
# -----------------------------------------------------------------------------

def _handle_lsb_robust(img: np.ndarray, bits: Optional[np.ndarray], rng: random.Random, action: str) -> Any:
    flat = img.reshape(-1)
    indices = list(range(len(flat)))
    rng.shuffle(indices)

    if action == "embed":
        if bits is None:
            raise ValueError("Bits required for embedding")
        if len(bits) > len(flat):
            raise ValueError("Message too big for cover image (LSB)")
        for i, b in enumerate(bits):
            idx = indices[i]
            flat[idx] = (flat[idx] & 0xFE) | int(b)
        return flat.reshape(img.shape)

    read_len = min(len(flat), 50000)
    extracted = [flat[indices[i]] & 1 for i in range(read_len)]
    return bits_to_str(np.array(extracted, dtype=np.uint8))

# -----------------------------------------------------------------------------
# PVD (USED FOR NOISE)
# -----------------------------------------------------------------------------

def _handle_pvd_real(img: np.ndarray, bits: Optional[np.ndarray], rng: random.Random, action: str) -> Any:
    flat = img.reshape(-1).astype(np.int16)
    if action != "embed":
        return ""
    if bits is None or len(bits) == 0:
        return img

    indices = list(range(len(flat)))
    rng.shuffle(indices)
    if len(indices) % 2 != 0:
        indices.pop()

    pairs = [(indices[i], indices[i + 1]) for i in range(0, len(indices), 2)]
    limit = min(len(bits), len(pairs))

    for i in range(limit):
        p1, p2 = pairs[i]
        v1, v2 = int(flat[p1]), int(flat[p2])
        if (abs(v1 - v2) & 1) != int(bits[i]):
            flat[p2] = v2 + 1 if v2 < 255 else v2 - 1

    return flat.reshape(img.shape).astype(np.uint8)

# -----------------------------------------------------------------------------
# DFT / SVD NOISE (AUGMENTATION)
# -----------------------------------------------------------------------------

def _handle_dft_noise(img: np.ndarray, bits: Optional[np.ndarray], rng: random.Random, action: str) -> Any:
    if action != "embed":
        return ""
    img_float = img.astype(np.float32)
    out = np.zeros_like(img_float)

    for ch in range(img.shape[2]):
        channel = img_float[:, :, ch]
        f = np.fft.fft2(channel)
        fshift = np.fft.fftshift(f)

        avg_mag = float(np.mean(np.abs(fshift)))
        scale = max(avg_mag * 0.05, 1e-3)

        noise_real = np.array([rng.uniform(-1.0, 1.0) for _ in range(channel.size)], dtype=np.float32).reshape(channel.shape)
        noise_imag = np.array([rng.uniform(-1.0, 1.0) for _ in range(channel.size)], dtype=np.float32).reshape(channel.shape)

        fshift_noisy = fshift + (noise_real + 1j * noise_imag) * scale
        recon = np.fft.ifft2(np.fft.ifftshift(fshift_noisy))
        out[:, :, ch] = np.real(recon)

    return np.clip(out, 0, 255).astype(np.uint8)


def _handle_svd_noise(img: np.ndarray, bits: Optional[np.ndarray], rng: random.Random, action: str) -> Any:
    if action != "embed":
        return ""
    img_float = img.astype(np.float32)
    out = np.zeros_like(img_float)

    for ch in range(img.shape[2]):
        A = img_float[:, :, ch]
        U, s, Vt = np.linalg.svd(A, full_matrices=False)

        mean_s = float(np.mean(s))
        perturb = np.array([rng.uniform(-1.0, 1.0) for _ in range(len(s))], dtype=np.float32)

        s2 = s + perturb * mean_s * 0.02
        s2 = np.maximum(s2, 0.0)

        out[:, :, ch] = U @ np.diag(s2) @ Vt

    return np.clip(out, 0, 255).astype(np.uint8)

# -----------------------------------------------------------------------------
# DCT STEGO (REAL EMBED/EXTRACT)
# -----------------------------------------------------------------------------

def _dct_all_slots(h_blocks: int, w_blocks: int, coords: Sequence[Tuple[int, int]]) -> List[Tuple[int, int, int, int]]:
    slots: List[Tuple[int, int, int, int]] = []
    for br in range(h_blocks):
        for bc in range(w_blocks):
            for (cy, cx) in coords:
                slots.append((br, bc, cy, cx))
    return slots


def _dct_compute(y_pad: np.ndarray, h_blocks: int, w_blocks: int) -> np.ndarray:
    dcts = np.zeros((h_blocks, w_blocks, 8, 8), dtype=np.float32)
    for br in range(h_blocks):
        for bc in range(w_blocks):
            block = y_pad[br * 8:(br + 1) * 8, bc * 8:(bc + 1) * 8].astype(np.float32)
            dcts[br, bc] = cv2.dct(block)
    return dcts


def _dct_reconstruct(dcts: np.ndarray, h_blocks: int, w_blocks: int) -> np.ndarray:
    y_pad = np.zeros((h_blocks * 8, w_blocks * 8), dtype=np.float32)
    for br in range(h_blocks):
        for bc in range(w_blocks):
            y_pad[br * 8:(br + 1) * 8, bc * 8:(bc + 1) * 8] = cv2.idct(dcts[br, bc])
    return y_pad


def _dct_prepare(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int], int, int, List[Tuple[int, int, int, int]]]:
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float32)
    cr = ycrcb[:, :, 1].copy()
    cb = ycrcb[:, :, 2].copy()

    y_pad, orig_hw = _pad_to_multiple(y, 8, mode="edge")
    h_pad, w_pad = y_pad.shape
    h_blocks, w_blocks = h_pad // 8, w_pad // 8
    slots_all = _dct_all_slots(h_blocks, w_blocks, DCT_COORDS)
    return y_pad, cr, cb, orig_hw, h_blocks, w_blocks, slots_all


def _dct_embed(
    img_bgr: np.ndarray,
    payload_bytes: bytes,
    seed: int,
    payload_delta: int = DEFAULT_DCT_DELTA,
    payload_rep: int = DEFAULT_DCT_REP,
    return_embed_map: bool = False,
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    crc = zlib.crc32(payload_bytes) & 0xFFFFFFFF
    header = StegoHeader(
        alg_id=ALG_ID["dct"],
        payload_rep=int(payload_rep),
        payload_delta=int(payload_delta),
        payload_len=int(len(payload_bytes)),
        crc32=int(crc),
        payload_seed=int(seed),
    )
    header_bits = bytes_to_bits(_pack_header(header))
    payload_bits = bytes_to_bits(payload_bytes)

    y_pad, cr, cb, orig_hw, h_blocks, w_blocks, slots_all = _dct_prepare(img_bgr)

    header_samples = header_bits.size * HEADER_REP
    rep = max(1, int(payload_rep))
    payload_samples = payload_bits.size * rep
    needed = header_samples + payload_samples

    if needed > len(slots_all):
        raise ValueError(f"DCT capacity too small: need {needed} slots, have {len(slots_all)}")

    header_slots = slots_all[:header_samples]
    payload_pool = slots_all[header_samples:]
    rng = random.Random(seed)
    rng.shuffle(payload_pool)
    payload_slots = payload_pool[:payload_samples]

    dcts = _dct_compute(y_pad, h_blocks, w_blocks)

    # embed header deterministically
    for i in range(header_bits.size):
        bit = int(header_bits[i])
        start = i * HEADER_REP
        for r in range(HEADER_REP):
            br, bc, cy, cx = header_slots[start + r]
            dcts[br, bc, cy, cx] = _qim_set(float(dcts[br, bc, cy, cx]), bit, HEADER_DELTA_DCT)

    # embed payload shuffled
    for i in range(payload_bits.size):
        bit = int(payload_bits[i])
        start = i * rep
        for r in range(rep):
            br, bc, cy, cx = payload_slots[start + r]
            dcts[br, bc, cy, cx] = _qim_set(float(dcts[br, bc, cy, cx]), bit, int(payload_delta))

    y_pad2 = _dct_reconstruct(dcts, h_blocks, w_blocks)
    y2 = _crop_to_shape(y_pad2, orig_hw)
    y2_u8 = np.clip(np.round(y2), 0, 255).astype(np.uint8)

    ycrcb = np.dstack([y2_u8, cr, cb])
    out_bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    emap = None
    if return_embed_map:
        emap = {
            "alg": "dct",
            "header_bits": int(header_bits.size),
            "payload_bits": int(payload_bits.size),
            "header_rep": int(HEADER_REP),
            "payload_rep": int(rep),
            "header_delta": int(HEADER_DELTA_DCT),
            "payload_delta": int(payload_delta),
            "header_slots_sample": header_slots[:20],
            "payload_slots_sample": payload_slots[:20],
            "total_coeff_slots": int(len(slots_all)),
        }
    return out_bgr, emap


def _dct_extract(
    img_bgr: np.ndarray,
    force_seed: Optional[int] = None,
    return_embed_map: bool = False,
) -> Tuple[bytes, Dict[str, Any]]:
    y_pad, cr, cb, orig_hw, h_blocks, w_blocks, slots_all = _dct_prepare(img_bgr)
    dcts = _dct_compute(y_pad, h_blocks, w_blocks)

    header_bits_count = HEADER_BYTES * 8
    header_samples = header_bits_count * HEADER_REP
    if header_samples > len(slots_all):
        raise ValueError("Image too small to hold DCT header")

    header_slots = slots_all[:header_samples]
    hb = np.zeros(header_bits_count, dtype=np.uint8)

    for i in range(header_bits_count):
        start = i * HEADER_REP
        votes = []
        for r in range(HEADER_REP):
            br, bc, cy, cx = header_slots[start + r]
            votes.append(_qim_get(float(dcts[br, bc, cy, cx]), HEADER_DELTA_DCT))
        hb[i] = _majority(votes)

    header = _unpack_header(bits_to_bytes(hb))
    seed = int(force_seed) if force_seed is not None else int(header.payload_seed)

    payload_bits_count = int(header.payload_len) * 8
    rep = max(1, int(header.payload_rep))
    payload_samples = payload_bits_count * rep

    payload_pool = slots_all[header_samples:]
    rng = random.Random(seed)
    rng.shuffle(payload_pool)
    if payload_samples > len(payload_pool):
        raise ValueError("DCT payload capacity too small")

    payload_slots = payload_pool[:payload_samples]
    pb = np.zeros(payload_bits_count, dtype=np.uint8)

    for i in range(payload_bits_count):
        start = i * rep
        votes = []
        for r in range(rep):
            br, bc, cy, cx = payload_slots[start + r]
            votes.append(_qim_get(float(dcts[br, bc, cy, cx]), int(header.payload_delta)))
        pb[i] = _majority(votes)

    payload_bytes = bits_to_bytes(pb)
    crc = zlib.crc32(payload_bytes) & 0xFFFFFFFF
    ok = (crc == int(header.crc32))

    info = {
        "alg": "dct",
        "seed_used": seed,
        "seed_in_header": int(header.payload_seed),
        "payload_len": int(header.payload_len),
        "crc_ok": bool(ok),
        "payload_rep": int(rep),
        "payload_delta": int(header.payload_delta),
        "header_rep": int(HEADER_REP),
        "header_delta": int(HEADER_DELTA_DCT),
    }
    if not ok:
        info["crc_extracted"] = int(crc)
        info["crc_expected"] = int(header.crc32)

    if return_embed_map:
        info["header_slots_sample"] = header_slots[:20]
        info["payload_slots_sample"] = payload_slots[:20]

    return payload_bytes, info

# -----------------------------------------------------------------------------
# HAAR DWT (1-LEVEL) + DWT STEGO
# -----------------------------------------------------------------------------

def _haar_dwt2(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    """1-level 2D Haar DWT. Pads by edge replicate if needed."""
    h, w = mat.shape
    orig = (h, w)
    if h % 2 == 1:
        mat = np.pad(mat, ((0, 1), (0, 0)), mode="edge")
    if w % 2 == 1:
        mat = np.pad(mat, ((0, 0), (0, 1)), mode="edge")

    low_r = (mat[0::2, :] + mat[1::2, :]) / 2.0
    high_r = (mat[0::2, :] - mat[1::2, :]) / 2.0

    LL = (low_r[:, 0::2] + low_r[:, 1::2]) / 2.0
    LH = (low_r[:, 0::2] - low_r[:, 1::2]) / 2.0
    HL = (high_r[:, 0::2] + high_r[:, 1::2]) / 2.0
    HH = (high_r[:, 0::2] - high_r[:, 1::2]) / 2.0
    return LL, LH, HL, HH, orig


def _haar_idwt2(LL: np.ndarray, LH: np.ndarray, HL: np.ndarray, HH: np.ndarray, orig_shape: Tuple[int, int]) -> np.ndarray:
    """Inverse of _haar_dwt2."""
    # Rebuild low/high rows
    low_r = np.zeros((LL.shape[0], LL.shape[1] * 2), dtype=np.float32)
    high_r = np.zeros_like(low_r)

    low_r[:, 0::2] = LL + LH
    low_r[:, 1::2] = LL - LH

    high_r[:, 0::2] = HL + HH
    high_r[:, 1::2] = HL - HH

    mat = np.zeros((low_r.shape[0] * 2, low_r.shape[1]), dtype=np.float32)
    mat[0::2, :] = low_r + high_r
    mat[1::2, :] = low_r - high_r

    return mat[:orig_shape[0], :orig_shape[1]]


def _dwt_prepare(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float32)
    cr = ycrcb[:, :, 1].copy()
    cb = ycrcb[:, :, 2].copy()
    return y, cr, cb


def _dwt_embed(
    img_bgr: np.ndarray,
    payload_bytes: bytes,
    seed: int,
    payload_delta: int = DEFAULT_DWT_DELTA,
    payload_rep: int = DEFAULT_DWT_REP,
    return_embed_map: bool = False,
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    crc = zlib.crc32(payload_bytes) & 0xFFFFFFFF
    header = StegoHeader(
        alg_id=ALG_ID["dwt"],
        payload_rep=int(payload_rep),
        payload_delta=int(payload_delta),
        payload_len=int(len(payload_bytes)),
        crc32=int(crc),
        payload_seed=int(seed),
    )
    header_bits = bytes_to_bits(_pack_header(header))
    payload_bits = bytes_to_bits(payload_bytes)

    y, cr, cb = _dwt_prepare(img_bgr)
    LL, LH, HL, HH, orig_shape = _haar_dwt2(y)

    # Use LH and HL detail bands for embedding
    detail = np.concatenate([LH.flatten(), HL.flatten()]).astype(np.float32)
    total_slots = detail.size

    header_samples = header_bits.size * HEADER_REP
    rep = max(1, int(payload_rep))
    payload_samples = payload_bits.size * rep
    needed = header_samples + payload_samples
    if needed > total_slots:
        raise ValueError(f"DWT capacity too small: need {needed}, have {total_slots}")

    # Deterministic header indices, then shuffled payload indices
    header_idx = np.arange(header_samples, dtype=np.int64)
    payload_pool = np.arange(header_samples, total_slots, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(payload_pool)
    payload_idx = payload_pool[:payload_samples]

    # embed header
    for i in range(header_bits.size):
        bit = int(header_bits[i])
        start = i * HEADER_REP
        for r in range(HEADER_REP):
            idx = int(header_idx[start + r])
            detail[idx] = _qim_set(float(detail[idx]), bit, HEADER_DELTA_DWT)

    # embed payload
    for i in range(payload_bits.size):
        bit = int(payload_bits[i])
        start = i * rep
        for r in range(rep):
            idx = int(payload_idx[start + r])
            detail[idx] = _qim_set(float(detail[idx]), bit, int(payload_delta))

    # write back into LH, HL
    lh_len = LH.size
    LH2 = detail[:lh_len].reshape(LH.shape)
    HL2 = detail[lh_len:].reshape(HL.shape)

    y2 = _haar_idwt2(LL, LH2, HL2, HH, orig_shape)
    y2_u8 = np.clip(np.round(y2), 0, 255).astype(np.uint8)
    ycrcb = np.dstack([y2_u8, cr, cb])
    out_bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    emap = None
    if return_embed_map:
        emap = {
            "alg": "dwt",
            "header_bits": int(header_bits.size),
            "payload_bits": int(payload_bits.size),
            "header_rep": int(HEADER_REP),
            "payload_rep": int(rep),
            "header_delta": int(HEADER_DELTA_DWT),
            "payload_delta": int(payload_delta),
            "header_idx_sample": header_idx[:20].tolist(),
            "payload_idx_sample": payload_idx[:20].tolist(),
            "total_coeff_slots": int(total_slots),
        }
    return out_bgr, emap


def _dwt_extract(
    img_bgr: np.ndarray,
    force_seed: Optional[int] = None,
    return_embed_map: bool = False,
) -> Tuple[bytes, Dict[str, Any]]:
    y, cr, cb = _dwt_prepare(img_bgr)
    LL, LH, HL, HH, orig_shape = _haar_dwt2(y)
    detail = np.concatenate([LH.flatten(), HL.flatten()]).astype(np.float32)
    total_slots = detail.size

    header_bits_count = HEADER_BYTES * 8
    header_samples = header_bits_count * HEADER_REP
    if header_samples > total_slots:
        raise ValueError("Image too small to hold DWT header")

    header_idx = np.arange(header_samples, dtype=np.int64)

    hb = np.zeros(header_bits_count, dtype=np.uint8)
    for i in range(header_bits_count):
        start = i * HEADER_REP
        votes = []
        for r in range(HEADER_REP):
            idx = int(header_idx[start + r])
            votes.append(_qim_get(float(detail[idx]), HEADER_DELTA_DWT))
        hb[i] = _majority(votes)

    header = _unpack_header(bits_to_bytes(hb))
    seed = int(force_seed) if force_seed is not None else int(header.payload_seed)

    payload_bits_count = int(header.payload_len) * 8
    rep = max(1, int(header.payload_rep))
    payload_samples = payload_bits_count * rep

    payload_pool = np.arange(header_samples, total_slots, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(payload_pool)
    if payload_samples > payload_pool.size:
        raise ValueError("DWT payload capacity too small")
    payload_idx = payload_pool[:payload_samples]

    pb = np.zeros(payload_bits_count, dtype=np.uint8)
    for i in range(payload_bits_count):
        start = i * rep
        votes = []
        for r in range(rep):
            idx = int(payload_idx[start + r])
            votes.append(_qim_get(float(detail[idx]), int(header.payload_delta)))
        pb[i] = _majority(votes)

    payload_bytes = bits_to_bytes(pb)
    crc = zlib.crc32(payload_bytes) & 0xFFFFFFFF
    ok = (crc == int(header.crc32))

    info = {
        "alg": "dwt",
        "seed_used": seed,
        "seed_in_header": int(header.payload_seed),
        "payload_len": int(header.payload_len),
        "crc_ok": bool(ok),
        "payload_rep": int(rep),
        "payload_delta": int(header.payload_delta),
        "header_rep": int(HEADER_REP),
        "header_delta": int(HEADER_DELTA_DWT),
    }
    if not ok:
        info["crc_extracted"] = int(crc)
        info["crc_expected"] = int(header.crc32)

    if return_embed_map:
        info["header_idx_sample"] = header_idx[:20].tolist()
        info["payload_idx_sample"] = payload_idx[:20].tolist()

    return payload_bytes, info

# -----------------------------------------------------------------------------
# PUBLIC API
# -----------------------------------------------------------------------------

def embed_image(cover_path: str, out_path: str, algorithm: str, bpp: float, rng: Optional[random.Random] = None) -> None:
    seed = rng.randint(0, 1000000) if rng else 42
    process_image(cover_path, algorithm, "embed", "", seed, out_path, bpp)


def process_image(
    path: str,
    algorithm: str,
    action: str = "embed",
    secret_message: Union[str, bytes] = "",
    seed: Optional[int] = None,
    out_path: Optional[str] = None,
    bpp: float = 0.4,
    *,
    return_bytes: bool = False,
    return_embed_map: bool = False,
    force_seed: bool = False,
) -> Any:
    """
    Embed/extract a message or generate noise.

    - For algorithm in {'dct','dwt'} with secret_message: real transform stego.
    - For other algorithms with secret_message (or extraction): robust LSB safe mode.

    force_seed:
        Only relevant for DCT/DWT extraction.
        If False (default), extraction uses seed stored in the header.
        If True, extraction uses provided seed.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read {path}")

    alg = algorithm.lower().strip()
    if seed is None:
        seed = 42
    rng = random.Random(seed)
    h, w, _ = img.shape

    # ------------------------------
    # DCT / DWT: real stego path
    # ------------------------------
    if alg in ("dct", "dwt"):
        if action == "embed":
            if isinstance(secret_message, (bytes, bytearray)):
                payload_bytes = bytes(secret_message)
            elif isinstance(secret_message, str) and secret_message != "":
                payload_bytes = secret_message.encode("utf-8")
            else:
                requested_bytes = max(1, int((h * w * float(bpp)) / 8))
                capacity = payload_capacity_bytes(h, w, alg)
                if capacity < 1:
                    raise ValueError(f"Image is too small for the {alg.upper()} transform header")
                nbytes = min(requested_bytes, capacity)
                payload_bytes = _get_random_bytes(nbytes, seed)

            if alg == "dct":
                out_img, emap = _dct_embed(
                    img,
                    payload_bytes,
                    seed=seed,
                    payload_delta=DEFAULT_DCT_DELTA,
                    payload_rep=DEFAULT_DCT_REP,
                    return_embed_map=return_embed_map,
                )
            else:
                out_img, emap = _dwt_embed(
                    img,
                    payload_bytes,
                    seed=seed,
                    payload_delta=DEFAULT_DWT_DELTA,
                    payload_rep=DEFAULT_DWT_REP,
                    return_embed_map=return_embed_map,
                )

            if out_path:
                out_dir = os.path.dirname(out_path)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                cv2.imwrite(out_path, out_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                return "Success" if not return_embed_map else {"status": "Success", "embed_map": emap}

            return out_img if not return_embed_map else (out_img, emap)

        # extract
        forced = seed if (force_seed and seed is not None) else None
        if alg == "dct":
            payload, info = _dct_extract(img, force_seed=forced, return_embed_map=return_embed_map)
        else:
            payload, info = _dwt_extract(img, force_seed=forced, return_embed_map=return_embed_map)

        if return_bytes:
            return payload
        return payload.decode("utf-8", errors="replace")

    # ------------------------------
    # Non-transform algorithms
    # ------------------------------

    use_safe_mode = (isinstance(secret_message, str) and secret_message != "") or action == "extract"

    bits: Optional[np.ndarray] = None
    if action == "embed":
        if isinstance(secret_message, str) and secret_message != "":
            bits = str_to_bits(secret_message)
        else:
            req_bits = int(h * w * float(bpp))
            max_cap = h * w * 3
            bits = _get_random_bits(min(req_bits, max_cap), seed)
            use_safe_mode = False

    if use_safe_mode:
        res = _handle_lsb_robust(img, bits, rng, action)
    else:
        if alg == "lsb":
            res = _handle_lsb_robust(img, bits, rng, action)
        elif alg == "pvd":
            res = _handle_pvd_real(img, bits, rng, action)
        elif alg == "dft":
            res = _handle_dft_noise(img, bits, rng, action)
        elif alg == "svd":
            res = _handle_svd_noise(img, bits, rng, action)
        else:
            raise ValueError(f"Unknown algo: {alg}")

    if action == "embed":
        if out_path:
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(out_path, res, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            return "Success"
        return res

    return res

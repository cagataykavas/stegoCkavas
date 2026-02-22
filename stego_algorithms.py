"""
stego_ai.stego_algorithms
-------------------------
Algorithms for hiding text (with a key/seed) or noise (for AI training).
Implements LSB, PVD, DCT, and DWT with random scattering.
"""
import random
import os
import numpy as np
import cv2

try:
    import pywt
except ImportError:
    pywt = None

# --- Helpers: Text <-> Bits ---

def str_to_bits(s: str) -> np.ndarray:
    """Converts string to a numpy array of bits (utf-8)."""
    b = s.encode("utf-8")
    bits = []
    for byte in b:
        for i in range(8):
            bits.append((byte >> (7 - i)) & 1)
    return np.array(bits, dtype=np.uint8)

def bits_to_str(bits: np.ndarray) -> str:
    """Converts bit array back to string."""
    chars = []
    for i in range(0, len(bits), 8):
        byte_chunk = bits[i:i+8]
        if len(byte_chunk) < 8: break
        val = 0
        for b in byte_chunk:
            val = (val << 1) | int(b)
        if val == 0: break 
        chars.append(val)
    
    try:
        return bytes(chars).decode("utf-8", errors="ignore")
    except:
        return "<decoding_error>"

def _get_random_bits(count: int, seed: int) -> np.ndarray:
    """Generates random noise bits for AI training sets."""
    rng = random.Random(seed)
    return np.array([rng.getrandbits(1) for _ in range(count)], dtype=np.uint8)

# --- BACKWARDS COMPATIBILITY WRAPPER ---

def embed_image(cover_path: str, out_path: str, algorithm: str, bpp: float, rng=None) -> None:
    """
    Wrapper for the pipeline. Converts the old 'embed_image' call 
    into the new 'process_image' logic.
    """
    seed = rng.randint(0, 1000000) if rng else 42
    
    process_image(
        path=cover_path,
        algorithm=algorithm,
        action="embed",
        secret_message="", # Empty message triggers 'bpp' noise mode
        seed=seed,
        out_path=out_path,
        bpp=bpp # <--- Passed correctly now!
    )

# --- Main Entry Point ---

def process_image(path: str, algorithm: str, action: str = "embed", secret_message: str = "", seed: int = None, out_path: str = None, bpp: float = 0.4):
    """
    Main spell for the GUI and API.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None: 
        raise FileNotFoundError(f"Cannot read {path}")
    
    alg = algorithm.lower()
    h, w, _ = img.shape
    
    if seed is None: seed = 42
    rng = random.Random(seed)

    # --- PREPARE PAYLOAD (With Safety Scissors ✂️) ---
    bits = None
    if action == "embed":
        if secret_message:
            # TEXT MODE: Must fit or crash (User needs to know their text is too long)
            bits = str_to_bits(secret_message)
        else:
            # BATCH NOISE MODE: Calculate max capacity to prevent crashing
            # 1. Calculate requested bits
            req_bits = int(h * w * bpp)
            
            # 2. Calculate REAL capacity based on algorithm
            max_cap = req_bits # Default to high
            if alg == "dct":
                # DCT capacity is roughly (H/8)*(W/8)
                max_cap = (h // 8) * (w // 8)
            elif alg == "dwt":
                # DWT capacity is (H/2)*(W/2)
                max_cap = (h // 2) * (w // 2)
            elif alg == "pvd":
                # PVD capacity is approx 50% of pixels (pairs)
                max_cap = (h * w) // 2
            elif alg == "lsb":
                max_cap = h * w * 3 # 3 channels
                
            # 3. Trim request to fit capacity
            final_bits = min(req_bits, max_cap)
            bits = _get_random_bits(final_bits, seed)
    
    # --- DISPATCH ---
    if alg == "lsb": res = _handle_lsb(img, bits, rng, action)
    elif alg == "pvd": res = _handle_pvd(img, bits, rng, action)
    elif alg == "dct": res = _handle_dct(img, bits, rng, action)
    elif alg == "dwt": res = _handle_dwt(img, bits, rng, action)
    else: raise ValueError(f"Unknown algo: {alg}")

    # --- SAVE ---
    if action == "embed":
        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, res)
            return "Success"
        return res 
    else:
        return res 

# --- Algorithm Implementations ---

def _handle_lsb(img, bits, rng, action):
    flat = img.reshape(-1)
    total_slots = len(flat)
    indices = list(range(total_slots))
    rng.shuffle(indices) 

    if action == "embed":
        if len(bits) > total_slots: 
            raise ValueError("Message too huge for this image!")
        for i, b in enumerate(bits):
            idx = indices[i]
            flat[idx] = (flat[idx] & 0xFE) | b
        return flat.reshape(img.shape)
    else: 
        read_len = min(total_slots, 8000) 
        extracted_bits = []
        for i in range(read_len):
            idx = indices[i]
            extracted_bits.append(flat[idx] & 1)
        return bits_to_str(np.array(extracted_bits, dtype=np.uint8))

def _handle_pvd(img, bits, rng, action):
    flat = img.reshape(-1)
    total_pixels = len(flat)
    all_indices = list(range(total_pixels))
    rng.shuffle(all_indices)
    if len(all_indices) % 2 != 0: all_indices.pop()
    pairs = [(all_indices[i], all_indices[i+1]) for i in range(0, len(all_indices), 2)]

    if action == "embed":
        if len(bits) > len(pairs): raise ValueError("Message too huge for PVD!")
        for i, b in enumerate(bits):
            p1_idx, p2_idx = pairs[i]
            val1 = int(flat[p1_idx])
            val2 = int(flat[p2_idx])
            diff = abs(val1 - val2)
            if (diff % 2) != b:
                if val2 < 255: val2 += 1
                else: val2 -= 1
            flat[p2_idx] = val2 
        return flat.reshape(img.shape)
    else:
        read_len = min(len(pairs), 8000)
        extracted_bits = []
        for i in range(read_len):
            p1_idx, p2_idx = pairs[i]
            val1 = int(flat[p1_idx])
            val2 = int(flat[p2_idx])
            diff = abs(val1 - val2)
            extracted_bits.append(diff % 2)
        return bits_to_str(np.array(extracted_bits, dtype=np.uint8))

def _handle_dct(img, bits, rng, action):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:,:,0].astype(np.float32)
    h, w = Y.shape
    blocks = []
    for r in range(0, h - 7, 8):
        for c in range(0, w - 7, 8):
            blocks.append((r, c))
    rng.shuffle(blocks) 
    
    if action == "embed":
        if len(bits) > len(blocks): raise ValueError("Message too huge for DCT!")
        for i, b in enumerate(bits):
            r, c = blocks[i]
            block = Y[r:r+8, c:c+8]
            dct_block = cv2.dct(block)
            coeff = dct_block[4, 1]
            coeff_int = int(round(coeff))
            if (coeff_int % 2) != b:
                if coeff_int < 0: coeff_int -= 1
                else: coeff_int += 1
            dct_block[4, 1] = float(coeff_int)
            idct_block = cv2.idct(dct_block)
            Y[r:r+8, c:c+8] = idct_block
        ycrcb[:,:,0] = np.clip(Y, 0, 255).astype(np.uint8)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else: 
        read_len = min(len(blocks), 8000)
        extracted_bits = []
        for i in range(read_len):
            r, c = blocks[i]
            block = Y[r:r+8, c:c+8]
            dct_block = cv2.dct(block)
            coeff_int = int(round(dct_block[4, 1]))
            extracted_bits.append(coeff_int % 2)
        return bits_to_str(np.array(extracted_bits, dtype=np.uint8))

def _handle_dwt(img, bits, rng, action):
    if pywt is None: 
        print("Warning: PyWavelets not found, using LSB fallback for DWT.")
        return _handle_lsb(img, bits, rng, action)

    B = img[:,:,0].astype(np.float32)
    coeffs = pywt.dwt2(B, 'haar')
    cA, (cH, cV, cD) = coeffs
    flat_cD = cD.ravel()
    indices = list(range(len(flat_cD)))
    rng.shuffle(indices)

    if action == "embed":
        if len(bits) > len(indices): raise ValueError("Message too huge for DWT!")
        for i, b in enumerate(bits):
            idx = indices[i]
            val = int(round(flat_cD[idx]))
            if (val % 2) != b: val += 1
            flat_cD[idx] = val
        cD_new = flat_cD.reshape(cD.shape)
        rec_B = pywt.idwt2((cA, (cH, cV, cD_new)), 'haar')
        img[:,:,0] = np.clip(rec_B, 0, 255).astype(np.uint8)
        return img
    else: 
        read_len = min(len(indices), 8000)
        extracted_bits = []
        for i in range(read_len):
            idx = indices[i]
            val = int(round(flat_cD[idx]))
            extracted_bits.append(val % 2)
        return bits_to_str(np.array(extracted_bits, dtype=np.uint8))
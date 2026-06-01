import numpy as np
from scipy.fftpack import dct, idct
from PIL import Image
import os

QUANT_TABLE = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
])

def dct2d(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2d(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def load_jpeg(path):
    npy_path = path.replace('.jpg', '_dct.npy')
    npy_shape_path = path.replace('.jpg', '_shape.npy')
    
    if os.path.exists(npy_path) and os.path.exists(npy_shape_path):
        coeff_array = np.load(npy_path)
        original_shape = tuple(np.load(npy_shape_path))
        h, w = coeff_array.shape
        dct_blocks = []
        block_positions = []
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                block = coeff_array[i:i+8, j:j+8].copy().astype(np.int32)
                dct_blocks.append(block)
                block_positions.append((i, j))
        print(f"[load_jpeg] Chargé depuis DCT bruts : {npy_path}")
        print(f"[load_jpeg] Blocs DCT 8x8 : {len(dct_blocks)}")
        return None, dct_blocks, block_positions, original_shape
    
    # Chargement normal depuis JPEG
    img = Image.open(path).convert('L')
    image_array = np.array(img, dtype=np.float64)
    h, w = image_array.shape
    h_pad = (8 - h % 8) % 8
    w_pad = (8 - w % 8) % 8
    if h_pad > 0 or w_pad > 0:
        image_array = np.pad(image_array, ((0, h_pad), (0, w_pad)), mode='edge')
    h_new, w_new = image_array.shape
    dct_blocks = []
    block_positions = []
    for i in range(0, h_new, 8):
        for j in range(0, w_new, 8):
            block = image_array[i:i+8, j:j+8] - 128
            quant_block = np.round(dct2d(block) / QUANT_TABLE).astype(np.int32)
            dct_blocks.append(quant_block)
            block_positions.append((i, j))
    print(f"[load_jpeg] Image chargée : {path}")
    print(f"[load_jpeg] Dimensions    : {h}x{w} → paddée à {h_new}x{w_new}")
    print(f"[load_jpeg] Blocs DCT 8x8 : {len(dct_blocks)}")
    return image_array, dct_blocks, block_positions, (h, w)


def save_jpeg(path, dct_blocks, block_positions, original_shape):
    npy_path = path.replace('.jpg', '_dct.npy')
    npy_shape_path = path.replace('.jpg', '_shape.npy')
    
    h_orig, w_orig = original_shape
    h_pad = (8 - h_orig % 8) % 8
    w_pad = (8 - w_orig % 8) % 8
    h_new = h_orig + h_pad
    w_new = w_orig + w_pad
    
    coeff_array = np.zeros((h_new, w_new), dtype=np.int32)
    for block, (i, j) in zip(dct_blocks, block_positions):
        coeff_array[i:i+8, j:j+8] = block
    
    np.save(npy_path, coeff_array)
    np.save(npy_shape_path, np.array(original_shape))
    print(f"[save_jpeg] Coefficients DCT sauvegardés : {npy_path}")


def get_ac_coefficients(dct_blocks):
    ac_values = []
    ac_positions = []
    nonzero_count = 0
    for block_idx, block in enumerate(dct_blocks):
        for r in range(8):
            for c in range(8):
                if r == 0 and c == 0:
                    continue
                val = block[r, c]
                ac_values.append(int(val))
                ac_positions.append((block_idx, r, c))
                if val != 0:
                    nonzero_count += 1
    print(f"[get_ac] Coefficients AC extraits (tous) : {len(ac_values)}")
    print(f"[get_ac] Coefficients AC non-nuls extraits : {nonzero_count}")
    return ac_values, ac_positions, nonzero_count


def get_lsb(value):
    return abs(value) % 2


def set_lsb(value, bit):
    if get_lsb(value) == bit:
        return value
    if value > 0:
        new_value = value - 1
    elif value < 0:
        new_value = value + 1
    else:
        return None
    if new_value == 0:
        return None
    return new_value
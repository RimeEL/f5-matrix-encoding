import numpy as np
from scipy.fftpack import dct, idct
from PIL import Image

def dct2d(block):
    """DCT 2D sur un bloc 8x8"""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2d(block):
    """DCT inverse 2D sur un bloc 8x8"""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# Table de quantification JPEG standard (qualité 50)
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

def load_jpeg(path):
    """
    Lit une image JPEG, la convertit en niveaux de gris,
    découpe en blocs 8x8, applique la DCT et quantifie.
    
    Retourne :
    - image_array  : image originale en numpy array
    - dct_blocks   : liste de blocs DCT quantifiés (chaque bloc = array 8x8)
    - block_positions : liste des (row, col) de chaque bloc dans l'image
    """
    img = Image.open(path).convert('L')  # Convertit en niveaux de gris
    image_array = np.array(img, dtype=np.float64)
    
    h, w = image_array.shape
    
    # Ajuste les dimensions pour être multiples de 8
    h_pad = (8 - h % 8) % 8
    w_pad = (8 - w % 8) % 8
    if h_pad > 0 or w_pad > 0:
        image_array = np.pad(image_array, ((0, h_pad), (0, w_pad)), mode='edge')
    
    h_new, w_new = image_array.shape
    
    dct_blocks = []
    block_positions = []
    
    for i in range(0, h_new, 8):
        for j in range(0, w_new, 8):
            block = image_array[i:i+8, j:j+8] - 128  # Centrage autour de 0
            dct_block = dct2d(block)
            quant_block = np.round(dct_block / QUANT_TABLE).astype(np.int32)
            dct_blocks.append(quant_block)
            block_positions.append((i, j))
    
    print(f"[load_jpeg] Image chargée : {path}")
    print(f"[load_jpeg] Dimensions    : {h}x{w} → paddée à {h_new}x{w_new}")
    print(f"[load_jpeg] Blocs DCT 8x8 : {len(dct_blocks)}")
    
    return image_array, dct_blocks, block_positions, (h, w)


def get_ac_coefficients(dct_blocks):
    """
    Extrait tous les coefficients AC non-nuls de tous les blocs.
    
    Dans un bloc DCT 8x8 :
    - Position (0,0) = coefficient DC → ignoré
    - Toutes les autres positions = coefficients AC → on les utilise
    
    Retourne :
    - ac_values   : liste des valeurs AC non-nulles
    - ac_positions: liste des (bloc_index, row_in_block, col_in_block)
    """
    ac_values = []
    ac_positions = []
    nonzero_count = 0

    for block_idx, block in enumerate(dct_blocks):
        for r in range(8):
            for c in range(8):
                if r == 0 and c == 0:
                    continue  # Saute le coefficient DC
                val = block[r, c]
                # On enregistre toutes les positions AC (y compris les zéros)
                ac_values.append(int(val))
                ac_positions.append((block_idx, r, c))
                if val != 0:
                    nonzero_count += 1

    print(f"[get_ac] Coefficients AC extraits (tous) : {len(ac_values)}")
    print(f"[get_ac] Coefficients AC non-nuls extraits : {nonzero_count}")
    return ac_values, ac_positions, nonzero_count


def save_jpeg(path, dct_blocks, block_positions, original_shape):
    """
    Reconstruit l'image depuis les blocs DCT modifiés et la sauvegarde.
    
    - path           : chemin de sortie
    - dct_blocks     : liste des blocs DCT (potentiellement modifiés)
    - block_positions: liste des (row, col) de chaque bloc
    - original_shape : (h, w) dimensions originales pour recadrer
    """
    # Calcule les dimensions paddées
    h_orig, w_orig = original_shape
    h_pad = (8 - h_orig % 8) % 8
    w_pad = (8 - w_orig % 8) % 8
    h_new = h_orig + h_pad
    w_new = w_orig + w_pad
    
    reconstructed = np.zeros((h_new, w_new), dtype=np.float64)
    
    for block, (i, j) in zip(dct_blocks, block_positions):
        # Dequantification
        dequant_block = block.astype(np.float64) * QUANT_TABLE
        # DCT inverse
        spatial_block = idct2d(dequant_block) + 128
        # Clip pour rester dans [0, 255]
        spatial_block = np.clip(spatial_block, 0, 255)
        reconstructed[i:i+8, j:j+8] = spatial_block
    
    # Recadre aux dimensions originales
    reconstructed = reconstructed[:h_orig, :w_orig]
    
    # Sauvegarde
    img_out = Image.fromarray(reconstructed.astype(np.uint8))
    img_out.save(path, 'JPEG', quality=95)
    
    print(f"[save_jpeg] Image stégo sauvegardée : {path}")


def get_lsb(value):
    """Retourne le LSB de la valeur absolue d'un coefficient."""
    return abs(value) % 2


def set_lsb(value, bit):
    """
    Modifie le LSB d'un coefficient selon la règle F5.
    Retourne None si shrinkage (coefficient devient 0).
    """
    if get_lsb(value) == bit:
        return value  # Déjà correct
    
    if value > 0:
        new_value = value - 1
    elif value < 0:
        new_value = value + 1
    else:
        return None  # Zéro → on ne touche pas
    
    if new_value == 0:
        return None  # Shrinkage !
    
    return new_value
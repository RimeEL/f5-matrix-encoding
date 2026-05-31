import numpy as np
from jpeg_io import load_jpeg, save_jpeg, get_ac_coefficients, get_lsb, set_lsb
from payload import file_to_bitstream, bitstream_to_file
from shuffling import shuffle_coefficients, deshuffle_coefficients
from capacity import compute_w
from matrix_encoding import build_hamming_matrix, encode_block, decode_block

def embed(cover_image_path, payload_path, output_path, password):
    """
    Fonction principale d'embedding F5.
    Cache un fichier (image ou audio) dans une image JPEG.
    
    - cover_image_path : chemin de l'image couverture (.jpg)
    - payload_path     : chemin du fichier à cacher (image ou audio)
    - output_path      : chemin de l'image stégo de sortie (.jpg)
    - password         : mot de passe pour le shuffling
    """
    
    print("\n" + "="*60)
    print("EMBEDDING F5 — DÉBUT")
    print("="*60)
    
    # ─── ÉTAPE 1 : Charger l'image JPEG ───────────────────────
    print("\n[F5] Étape 1 : Chargement de l'image JPEG...")
    image_array, dct_blocks, block_positions, original_shape = load_jpeg(cover_image_path)
    ac_values, ac_positions, ac_nonzero = get_ac_coefficients(dct_blocks)
    
    # ─── ÉTAPE 2 : Préparer le payload ────────────────────────
    print("\n[F5] Étape 2 : Préparation du payload...")
    payload_bits = file_to_bitstream(payload_path)
    total_bits = len(payload_bits)
    
    # ─── ÉTAPE 3 : Calculer w ─────────────────────────────────
    print("\n[F5] Étape 3 : Calcul du paramètre w...")
    w = compute_w(ac_nonzero, total_bits)
    n = (2 ** w) - 1  # taille d'un bloc
    
    # Vérification de capacité
    max_capacity = (ac_nonzero // n) * w
    if total_bits > max_capacity:
        print(f"[F5] ERREUR : Payload trop grand !")
        print(f"[F5] Capacité max : {max_capacity} bits")
        print(f"[F5] Payload      : {total_bits} bits")
        print(f"[F5] Utilise une image couverture plus grande ou un payload plus petit.")
        return False
    
    # ─── ÉTAPE 4 : Shuffling ──────────────────────────────────
    print("\n[F5] Étape 4 : Shuffling des coefficients...")
    shuffled_vals, shuffled_pos, mapping = shuffle_coefficients(
        ac_values, ac_positions, password
    )
    
    # ─── ÉTAPE 5 : Embedding du header ────────────────────────
    # On cache w (8 bits) et la longueur du payload (32 bits)
    # en LSB simple sur les 40 premiers coefficients
    print("\n[F5] Étape 5 : Embedding du header (w + longueur)...")
    
    header_bits = []
    # 8 bits pour w
    for i in range(7, -1, -1):
        header_bits.append((w >> i) & 1)
    # 32 bits pour la longueur
    for i in range(31, -1, -1):
        header_bits.append((total_bits >> i) & 1)
    
    # Embedding du header en LSB simple (pas de matrix encoding)
    header_idx = 0
    coeff_idx = 0
    
    while header_idx < len(header_bits) and coeff_idx < len(shuffled_vals):
        val = shuffled_vals[coeff_idx]
        bit = header_bits[header_idx]
        
        new_val = set_lsb(val, bit)
        if new_val is not None:
            shuffled_vals[coeff_idx] = new_val
            header_idx += 1
        # Si shrinkage, on saute ce coefficient
        coeff_idx += 1
    
    header_end_idx = coeff_idx  # Index où commence le payload réel
    print(f"[F5] Header embedé sur les {header_end_idx} premiers coefficients")
    
    # ─── ÉTAPE 6 : Embedding du message avec Matrix Encoding ──
    print(f"\n[F5] Étape 6 : Embedding du message avec Matrix Encoding (w={w})...")
    
    payload_idx = 0
    coeff_idx = header_end_idx
    nb_blocks = 0
    nb_modifications = 0
    nb_shrinkages = 0
    
    while payload_idx < total_bits and coeff_idx + n <= len(shuffled_vals):
        # Prendre w bits du message
        msg_block = payload_bits[payload_idx:payload_idx + w]
        if len(msg_block) < w:
            # Padding si nécessaire
            msg_block = msg_block + [0] * (w - len(msg_block))
        
        # Prendre n coefficients
        coeffs_block = shuffled_vals[coeff_idx:coeff_idx + n]
        
        # Encoder avec Matrix Encoding
        new_coeffs, changes, shrinkage = encode_block(coeffs_block, msg_block, w)
        
        if shrinkage:
            nb_shrinkages += 1
            coeff_idx += 1  # Saute 1 coefficient et réessaie
            continue
        
        # Mettre à jour les coefficients
        for k in range(n):
            shuffled_vals[coeff_idx + k] = new_coeffs[k]
        
        nb_modifications += changes
        nb_blocks += 1
        payload_idx += w
        coeff_idx += n
    
    print(f"[F5] Blocs encodés      : {nb_blocks}")
    print(f"[F5] Modifications      : {nb_modifications}")
    print(f"[F5] Shrinkages         : {nb_shrinkages}")
    print(f"[F5] Taux modification  : {nb_modifications/(nb_blocks*n)*100:.2f}% " if nb_blocks > 0 else "")
    
    # ─── ÉTAPE 7 : Déshuffling ────────────────────────────────
    print("\n[F5] Étape 7 : Déshuffling...")
    restored_vals, restored_pos = deshuffle_coefficients(
        shuffled_vals, shuffled_pos, mapping
    )
    
    # ─── ÉTAPE 8 : Reconstruire les blocs DCT modifiés ────────
    print("\n[F5] Étape 8 : Reconstruction des blocs DCT...")
    
    for new_val, (block_idx, r, c) in zip(restored_vals, restored_pos):
        dct_blocks[block_idx][r, c] = new_val
    
    # ─── ÉTAPE 9 : Sauvegarder l'image stégo ──────────────────
    print("\n[F5] Étape 9 : Sauvegarde de l'image stégo...")
    save_jpeg(output_path, dct_blocks, block_positions, original_shape)
    
    print("\n" + "="*60)
    print(f"✅ EMBEDDING TERMINÉ")
    print(f"   Image stégo : {output_path}")
    print(f"   Payload caché : {total_bits} bits")
    print("="*60)
    return True


def extract(stego_image_path, output_path, password):
    """
    Fonction principale d'extraction F5.
    Récupère le fichier caché depuis une image stégo.
    
    - stego_image_path : chemin de l'image stégo (.jpg)
    - output_path      : chemin où sauvegarder le fichier extrait
    - password         : mot de passe utilisé lors de l'embedding
    """
    
    print("\n" + "="*60)
    print("EXTRACTION F5 — DÉBUT")
    print("="*60)
    
    # ─── ÉTAPE 1 : Charger l'image stégo ──────────────────────
    print("\n[F5] Étape 1 : Chargement de l'image stégo...")
    image_array, dct_blocks, block_positions, original_shape = load_jpeg(stego_image_path)
    ac_values, ac_positions, ac_nonzero = get_ac_coefficients(dct_blocks)
    
    # ─── ÉTAPE 2 : Shuffling avec le même mot de passe ────────
    print("\n[F5] Étape 2 : Shuffling avec le mot de passe...")
    shuffled_vals, shuffled_pos, mapping = shuffle_coefficients(
        ac_values, ac_positions, password
    )
    
    # ─── ÉTAPE 3 : Lire le header ─────────────────────────────
    print("\n[F5] Étape 3 : Lecture du header...")
    
    header_bits = []
    coeff_idx = 0
    
    while len(header_bits) < 40 and coeff_idx < len(shuffled_vals):
        val = shuffled_vals[coeff_idx]
        if val != 0:
            header_bits.append(abs(val) % 2)
        coeff_idx += 1
    
    header_end_idx = coeff_idx
    
    # Décoder w (8 premiers bits)
    w = 0
    for bit in header_bits[:8]:
        w = (w << 1) | bit
    
    # Décoder la longueur (32 bits suivants)
    total_bits = 0
    for bit in header_bits[8:40]:
        total_bits = (total_bits << 1) | bit
    
    n = (2 ** w) - 1
    print(f"[F5] w lu depuis header       : {w}")
    print(f"[F5] Longueur payload (bits)  : {total_bits}")
    
    # ─── ÉTAPE 4 : Extraction avec Matrix Encoding ────────────
    print(f"\n[F5] Étape 4 : Extraction du message (w={w})...")
    
    extracted_bits = []
    coeff_idx = header_end_idx
    
    while len(extracted_bits) < total_bits and coeff_idx + n <= len(shuffled_vals):
        # Prendre n coefficients
        coeffs_block = shuffled_vals[coeff_idx:coeff_idx + n]
        
        # Décoder avec Matrix Encoding
        msg_bits = decode_block(coeffs_block, w)
        extracted_bits.extend(msg_bits)
        coeff_idx += n
    
    # Tronquer au nombre exact de bits
    extracted_bits = extracted_bits[:total_bits]
    
    print(f"[F5] Bits extraits : {len(extracted_bits)}")
    
    # ─── ÉTAPE 5 : Reconstruire le fichier ────────────────────
    print("\n[F5] Étape 5 : Reconstruction du fichier...")
    bitstream_to_file(extracted_bits, output_path)
    
    print("\n" + "="*60)
    print(f"✅ EXTRACTION TERMINÉE")
    print(f"   Fichier extrait : {output_path}")
    print("="*60)
    return True
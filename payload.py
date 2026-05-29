def file_to_bitstream(filepath):
    """
    Convertit un fichier (image, audio, texte...) en une liste de bits (0 et 1).
    
    Le format du bitstream est :
    [8 bits type] [32 bits longueur en octets] [données binaires du fichier]
    
    Types :
      - "image" → code 00000001
      - "audio" → code 00000010
      - "other" → code 00000011
    
    Retourne : liste de 0 et 1
    """
    
    # Détecter le type de fichier selon son extension
    filepath_lower = filepath.lower()
    if filepath_lower.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
        file_type = 1  # image
        type_name = "image"
    elif filepath_lower.endswith(('.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a')):
        file_type = 2  # audio
        type_name = "audio"
    else:
        file_type = 3  # autre
        type_name = "other"
    
    # Lire le fichier en mode binaire
    with open(filepath, 'rb') as f:
        file_bytes = f.read()
    
    file_length = len(file_bytes)
    
    print(f"[payload] Fichier à cacher : {filepath}")
    print(f"[payload] Type détecté    : {type_name}")
    print(f"[payload] Taille          : {file_length} octets")
    print(f"[payload] Taille en bits  : {file_length * 8 + 40} bits (avec header)")
    
    bits = []
    
    # --- Header partie 1 : type sur 8 bits ---
    for i in range(7, -1, -1):
        bits.append((file_type >> i) & 1)
    
    # --- Header partie 2 : longueur sur 32 bits ---
    for i in range(31, -1, -1):
        bits.append((file_length >> i) & 1)
    
    # --- Données : chaque octet converti en 8 bits ---
    for byte in file_bytes:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    
    print(f"[payload] Bitstream total  : {len(bits)} bits")
    
    return bits


def bitstream_to_file(bits, output_path):
    """
    Reconstitue un fichier à partir d'une liste de bits.
    Lit le header pour connaître le type et la taille, puis reconstruit le fichier.
    
    - bits        : liste de 0 et 1 (avec header)
    - output_path : chemin où sauvegarder le fichier reconstitué
    """
    
    # --- Lire le type (8 premiers bits) ---
    type_bits = bits[0:8]
    file_type = 0
    for b in type_bits:
        file_type = (file_type << 1) | b
    
    # --- Lire la longueur (32 bits suivants) ---
    length_bits = bits[8:40]
    file_length = 0
    for b in length_bits:
        file_length = (file_length << 1) | b
    
    print(f"[payload] Reconstruction du fichier...")
    print(f"[payload] Type code : {file_type}")
    print(f"[payload] Taille    : {file_length} octets")
    
    # --- Lire les données ---
    data_bits = bits[40:40 + file_length * 8]
    
    file_bytes = bytearray()
    for i in range(0, len(data_bits), 8):
        byte_bits = data_bits[i:i+8]
        if len(byte_bits) < 8:
            break
        byte_val = 0
        for b in byte_bits:
            byte_val = (byte_val << 1) | b
        file_bytes.append(byte_val)
    
    # Sauvegarder
    with open(output_path, 'wb') as f:
        f.write(file_bytes)
    
    print(f"[payload] Fichier reconstruit sauvegardé : {output_path}")
    return file_type, file_length
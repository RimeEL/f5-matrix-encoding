import numpy as np

def shuffle_coefficients(ac_coeffs, positions, password):
    """
    Mélange les coefficients AC dans un ordre pseudo-aléatoire
    déterminé par le mot de passe.
    
    - ac_coeffs : liste des valeurs des coefficients AC non-nuls
    - positions : liste des positions (canal, i, j) correspondantes
    - password  : mot de passe (string)
    
    Retourne :
    - shuffled_coeffs   : coefficients dans le nouvel ordre
    - shuffled_positions: positions dans le nouvel ordre
    - mapping           : tableau des indices originaux (pour déshuffle)
    """
    
    n = len(ac_coeffs)
    
    # Créer une seed numérique à partir du mot de passe
    seed = 0
    for char in password:
        seed = (seed * 31 + ord(char)) % (2**32)
    
    # Générer une permutation pseudo-aléatoire des indices
    rng = np.random.default_rng(seed)
    mapping = rng.permutation(n)  # tableau de n indices mélangés
    
    # Appliquer la permutation
    shuffled_coeffs    = [ac_coeffs[i] for i in mapping]
    shuffled_positions = [positions[i] for i in mapping]
    
    print(f"[shuffling] Shuffling appliqué avec password='{password}' → seed={seed}")
    print(f"[shuffling] {n} coefficients mélangés")
    
    return shuffled_coeffs, shuffled_positions, mapping


def deshuffle_coefficients(shuffled_coeffs, shuffled_positions, mapping):
    """
    Remet les coefficients dans leur ordre original (opération inverse du shuffle).
    
    - shuffled_coeffs   : coefficients dans l'ordre mélangé
    - shuffled_positions: positions dans l'ordre mélangé  
    - mapping           : tableau des indices originaux (retourné par shuffle_coefficients)
    
    Retourne :
    - original_coeffs   : coefficients remis dans l'ordre original
    - original_positions: positions remises dans l'ordre original
    """
    
    n = len(shuffled_coeffs)
    
    # Créer les tableaux de sortie
    original_coeffs    = [None] * n
    original_positions = [None] * n
    
    # Inverser la permutation
    for new_idx, orig_idx in enumerate(mapping):
        original_coeffs[orig_idx]    = shuffled_coeffs[new_idx]
        original_positions[orig_idx] = shuffled_positions[new_idx]
    
    print(f"[shuffling] Déshuffling appliqué : {n} coefficients remis en ordre")
    
    return original_coeffs, original_positions
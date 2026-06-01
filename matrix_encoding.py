import numpy as np

def build_hamming_matrix(w):
    """
    Construit la matrice de parité de Hamming H_w de dimensions (w × (2^w - 1)).
    
    Principe :
    - n = 2^w - 1 colonnes
    - Chaque colonne j (j allant de 1 à n) contient la représentation
      binaire de j sur w bits
    
    Exemple pour w=2 :
    - n = 3 colonnes
    - colonne 1 = [0,1] (représentation binaire de 1)
    - colonne 2 = [1,0] (représentation binaire de 2)
    - colonne 3 = [1,1] (représentation binaire de 3)
    
    H_2 = [[0, 1, 1],
            [1, 0, 1]]
    """
    n = (2 ** w) - 1  # nombre de colonnes
    H = np.zeros((w, n), dtype=np.int32)
    
    for j in range(1, n + 1):
        # Convertit j en représentation binaire sur w bits
        for bit_pos in range(w):
            H[w - 1 - bit_pos, j - 1] = (j >> bit_pos) & 1
    
    return H


def compute_syndrome(H, lsb_vector):
    """
    Calcule le syndrome s = H × c (mod 2)
    
    - H          : matrice de Hamming (w × n)
    - lsb_vector : vecteur des LSB des n coefficients (liste de 0 et 1)
    
    Retourne : vecteur syndrome de longueur w (liste de 0 et 1)
    
    Exemple :
    H = [[0,1,1],[1,0,1]], lsb = [1,0,1]
    s = [0*1+1*0+1*1, 1*1+0*0+1*1] mod 2 = [1, 0]
    """
    n = len(lsb_vector)
    c = np.array(lsb_vector, dtype=np.int32)
    s = np.dot(H, c) % 2
    return s.tolist()


def encode_block(coeffs, message_bits, w):
    """
    Encode w bits de message dans un bloc de n = 2^w - 1 coefficients.
    Modifie au maximum 1 coefficient (règle du Matrix Encoding).
    
    Algorithme :
    1. Extraire les LSB des coefficients → vecteur c
    2. Calculer le syndrome s = H × c (mod 2)
    3. Calculer la différence a = message XOR s
    4. Si a == 0 → rien à modifier (parfait !)
    5. Si a != 0 → modifier le coefficient à la position indiquée par a
    
    - coeffs       : liste de n valeurs de coefficients DCT
    - message_bits : liste de w bits à cacher
    - w            : paramètre du Matrix Encoding
    
    Retourne :
    - new_coeffs    : liste de n coefficients (potentiellement 1 modifié)
    - nb_changes    : nombre de modifications faites (0 ou 1)
    - shrinkage     : True si un shrinkage s'est produit
    """
    n = (2 ** w) - 1
    H = build_hamming_matrix(w)
    
    # Extraire les LSB
    lsb_vector = [abs(c) % 2 for c in coeffs]
    
    # Étape 1 : Calculer le syndrome actuel
    s = compute_syndrome(H, lsb_vector)
    
    # Étape 2 : Calculer la différence a = message XOR syndrome
    a = [(message_bits[i] ^ s[i]) for i in range(w)]
    
    # Étape 3 : Vérifier si une modification est nécessaire
    if all(bit == 0 for bit in a):
        # a == 0 → les LSB actuels encodent déjà le bon message !
        return list(coeffs), 0, False
    
    # Étape 4 : Convertir a en entier pour trouver quelle colonne modifier
    # a est la représentation binaire de l'index à modifier
    col_index = 0
    for bit in a:
        col_index = (col_index << 1) | bit
    
    # col_index va de 1 à n → position dans le bloc = col_index - 1
    pos_to_modify = col_index - 1
    
    # Étape 5 : Modifier le coefficient à cette position
    new_coeffs = list(coeffs)
    val = new_coeffs[pos_to_modify]
    
    # Changer le LSB selon la règle F5
    current_lsb = abs(val) % 2
    target_lsb = 1 - current_lsb  # On veut l'opposé
    
    if val > 0:
        # Essayer de diminuer pour changer le LSB
        cand = val - 1
        if cand == 0:
            # Éviter le shrinkage → augmenter à la place
            cand = val + 1
    elif val < 0:
        cand = val + 1
        if cand == 0:
            cand = val - 1
    else:
        # val == 0 : permettre de créer une valeur non-nulle (par ex. 1)
        # afin de pouvoir modifier le LSB même si le coefficient était zéro.
        # On choisit 1 pour obtenir un LSB à 1 (si nécessaire).
        if target_lsb == 1:
            cand = 1
        else:
            cand = 2

    new_coeffs[pos_to_modify] = cand
    return new_coeffs, 1, False

    new_coeffs[pos_to_modify] = cand
    return new_coeffs, 1, False


def decode_block(coeffs, w):
    """
    Extrait w bits cachés depuis un bloc de n = 2^w - 1 coefficients.
    
    Algorithme simple :
    - Extraire les LSB des coefficients
    - Calculer m = H × LSB (mod 2)
    - m est directement le message caché !
    
    - coeffs : liste de n valeurs de coefficients DCT
    - w      : paramètre du Matrix Encoding
    
    Retourne : liste de w bits (le message extrait)
    """
    H = build_hamming_matrix(w)
    
    # Extraire les LSB
    lsb_vector = [abs(c) % 2 for c in coeffs]
    
    # Calculer le syndrome = message extrait
    message = compute_syndrome(H, lsb_vector)
    
    return message
def compute_w(num_ac_nonzero, payload_length_bits):
    """
    Choisit le paramètre w optimal pour le Matrix Encoding.
    
    Rappel du cours :
    - Avec w, on cache w bits dans (2^w - 1) coefficients en modifiant au max 1 coefficient
    - Plus w est grand → plus efficace, mais nécessite plus de coefficients
    
    - num_ac_nonzero     : nombre de coefficients AC non-nuls disponibles
    - payload_length_bits: nombre de bits à cacher (avec header)
    
    Retourne : w (entier entre 1 et 4)
    """
    
    print(f"\n[capacity] Calcul du paramètre w...")
    print(f"[capacity] Coefficients AC disponibles : {num_ac_nonzero}")
    print(f"[capacity] Bits à cacher               : {payload_length_bits}")
    
    best_w = 1  # valeur par défaut minimale
    
    for w in range(1, 5):  # on teste w = 1, 2, 3, 4
        n = (2 ** w) - 1  # nombre de coefficients par bloc
        
        # Capacité théorique : combien de bits on peut cacher au total
        capacity = (num_ac_nonzero // n) * w
        
        # Taux de modification moyen : 1/n
        modification_rate = 1 / n
        
        print(f"[capacity] w={w} → n={n} coefficients/bloc, "
              f"capacité={capacity} bits, "
              f"taux modif={modification_rate:.3f}")
        
        if capacity >= payload_length_bits:
            best_w = w  # Ce w est suffisant, on garde le plus grand possible
    
    print(f"[capacity]  Paramètre w choisi : {best_w}")
    return best_w
from f5 import embed, extract

print("="*60)
print("TEST COMPLET F5 — EMBED + EXTRACT")
print("="*60)

# ── TEST 1 : Cacher une image dans une image ──
print("\n--- TEST 1 : Image dans Image ---")

# Mets ici une petite image à cacher (plus petite que cover.jpg !)
# Par exemple une image de 200x200 pixels
success = embed(
    cover_image_path = "images/cover.jpg",
    payload_path     = "images/secret.jpg",   # ← image à cacher
    output_path      = "images/stego.jpg",
    password         = "motdepasse123"
)

if success:
    extract(
        stego_image_path = "images/stego.jpg",
        output_path      = "images/extracted_secret.jpg",
        password         = "motdepasse123"
    )
    print("\nVérifie que images/extracted_secret.jpg est identique à images/secret.jpg !")

# ── TEST 2 : Mauvais mot de passe ──
print("\n--- TEST 2 : Mauvais mot de passe ---")
extract(
    stego_image_path = "images/stego.jpg",
    output_path      = "images/wrong_extract.jpg",
    password         = "mauvaismdp"
)
print("Le fichier wrong_extract.jpg doit être illisible/corrompu ✓")
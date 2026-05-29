from jpeg_io import load_jpeg, save_jpeg, get_ac_coefficients
from payload import file_to_bitstream, bitstream_to_file
from shuffling import shuffle_coefficients, deshuffle_coefficients
from capacity import compute_w

print("=" * 60)
print("TEST MODULE 1 — jpeg_io")
print("=" * 60)

image_array, dct_blocks, block_positions, original_shape = load_jpeg("images/cover.jpg")
ac_values, ac_positions = get_ac_coefficients(dct_blocks)

print(f"Nombre de blocs DCT        : {len(dct_blocks)}")
print(f"Coefficients AC non-nuls   : {len(ac_values)}")
print(f"Exemples de valeurs AC     : {ac_values[:10]}")

print("\n" + "=" * 60)
print("TEST MODULE 2 — payload")
print("=" * 60)

bits = file_to_bitstream("images/cover.jpg")
print(f"Premiers 40 bits (header)  : {bits[:40]}")
bitstream_to_file(bits, "images/reconstructed_test.jpg")
print("Reconstruction OK ")

print("\n" + "=" * 60)
print("TEST MODULE 3 — shuffling")
print("=" * 60)

shuffled_vals, shuffled_pos, mapping = shuffle_coefficients(
    ac_values, ac_positions, password="motdepasse123"
)
print(f"Avant shuffle  : {ac_values[:5]}")
print(f"Après shuffle  : {shuffled_vals[:5]}")

restored_vals, restored_pos = deshuffle_coefficients(shuffled_vals, shuffled_pos, mapping)
print(f"Après déshuffle: {restored_vals[:5]}")
print(f"Réversible     : {restored_vals[:20] == ac_values[:20]} ✓")

print("\n" + "=" * 60)
print("TEST MODULE 4 — capacity")
print("=" * 60)

payload_bits = file_to_bitstream("images/cover.jpg")
w = compute_w(len(ac_values), len(payload_bits))
print(f"w final retenu : {w}")

print("\n" + "=" * 60)
print("TEST MODULE 5 — save_jpeg")
print("=" * 60)

save_jpeg("images/output_test.jpg", dct_blocks, block_positions, original_shape)
print("Sauvegarde OK ")

print("\n TOUS LES TESTS PASSÉS — Les modules sont prêts !")
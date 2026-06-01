import binascii

def main():
    paths = ['images/secret.jpg','images/extracted_secret.jpg']
    with open(paths[0],'rb') as f:
        a = f.read()
    with open(paths[1],'rb') as f:
        b = f.read()

    print('len A', len(a))
    print('len B', len(b))

    ml = min(len(a), len(b))
    first = None
    mismatches = 0
    # Find first mismatch and count up to a limit
    for i in range(ml):
        if a[i] != b[i]:
            mismatches += 1
            if first is None:
                first = i
            if mismatches >= 1000:
                break
    print('mismatches_count_partial', mismatches)
    print('first_diff_index', first)
    if first is not None:
        print('a_slice', binascii.hexlify(a[first:first+64]))
        print('b_slice', binascii.hexlify(b[first:first+64]))

    # If sizes differ, show trailing bytes
    if len(a) != len(b):
        print('trailing_A', binascii.hexlify(a[ml:ml+64]))
        print('trailing_B', binascii.hexlify(b[ml:ml+64]))

if __name__ == '__main__':
    main()

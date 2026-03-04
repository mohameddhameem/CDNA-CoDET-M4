def make_array_good(a):
    n = len(a)
    b = [1 if ai % 2 == 0 else 2 for ai in a]
    m = min(b)
    for i in range(n):
        if b[i] == m:
            return i
    return -1
def cyclic_shifts_sorting(a):
    n = len(a)
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            if a[j] < a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                return a
    return -1
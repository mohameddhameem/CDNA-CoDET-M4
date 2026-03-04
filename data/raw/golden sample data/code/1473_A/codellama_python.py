def replacing_elements(n, d, a):
    for i in range(n):
        for j in range(i+1, n):
            if a[i] + a[j] <= d:
                a[i] = a[i] + a[j]
                return "YES"
    return "NO"
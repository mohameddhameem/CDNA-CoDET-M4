def who_tested(n, a, b, c, d):
    a.sort()
    b.sort()
    c.sort()
    d.sort()
    for i in range(n):
        if a[i] == b[i] == c[i] == d[i]:
            return a[i]
    return -1
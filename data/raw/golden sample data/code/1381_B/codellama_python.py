def unmerge(p):
    n = len(p)
    if n == 1:
        return False
    elif n == 2:
        return p[0] < p[1]
    else:
        m = n // 2
        a = p[:m]
        b = p[m:]
        if a[0] < b[0]:
            return a == sorted(a) and b == sorted(b) and unmerge(a + b[1:])
        elif a[0] > b[0]:
            return a == sorted(a) and b == sorted(b) and unmerge(b + a[1:])
        else:
            return False
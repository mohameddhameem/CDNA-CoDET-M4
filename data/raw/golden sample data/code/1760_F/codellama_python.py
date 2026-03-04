def euclid(a, b):
    if a < b:
        swap(a, b)
    if b == 0:
        return a
    r = reminder from dividing a by b
    if r > 0:
        append r to the back of t
    return Euclid(b, r)

def guess(t):
    p = []
    for i in range(len(t)):
        t_i = t[i]
        if t_i > m:
            return -1
        p.append((t_i, 1))
    for i in range(len(p)):
        t_i, b_i = p[i]
        p[i] = (b_i, t_i)
    return t

n, m = map(int, input().split())
t = list(map(int, input().split()))
p = guess(t)
if p is not None:
    print(len(p), *p)
else:
    print(-1)
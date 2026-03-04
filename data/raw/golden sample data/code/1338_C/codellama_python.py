def perfect_triples(n):
    s = [1, 2, 3, 4, 8, 12, 5, 10, 15]
    for i in range(n - len(s)):
        a = s[-1]
        b = a + 1
        c = b + 1
        x = a ^ b ^ c
        while x < 256:
            x = x * 2 + 1
        x = x | (x // 256) * 256
        y = x % 256
        while x > 0:
            s.append(y)
            x >>= 8
            y *= 2
            y >>= 8
    return s[n - 1]

t = int(input())
for _ in range(t):
    n = int(input())
    print(perfect_triples(n))
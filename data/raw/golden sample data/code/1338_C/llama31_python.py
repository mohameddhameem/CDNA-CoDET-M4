def perfect_triple(n):
    if n % 3 == 1:
        return n // 3 + 1
    elif n % 3 == 2:
        return n // 3 + 2
    else:
        return n // 3 * 3

def solve():
    t = int(input())
    for _ in range(t):
        n = int(input())
        if n <= 3:
            print(n)
        else:
            n -= 1
            x = 1
            while n > 0:
                n -= 3 * (3 ** (x - 1))
                x += 1
            print(3 ** (x - 1) + perfect_triple(n))

solve()
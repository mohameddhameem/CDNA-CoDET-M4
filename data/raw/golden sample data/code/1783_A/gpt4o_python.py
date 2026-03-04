def is_beautiful(x, d):
    if x == d:
        return False
    x //= d
    if x % 2 == 0:
        return True
    for i in range(2, int(x**0.5) + 1):
        if x % i == 0:
            if i * i != x and (x // i) % 2 == 0:
                return True
    return False

t = int(input())
results = []
for _ in range(t):
    x, d = map(int, input().split())
    results.append("YES" if is_beautiful(x, d) else "NO")

print("\n".join(results))
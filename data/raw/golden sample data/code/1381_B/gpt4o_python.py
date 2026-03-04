def can_merge(n, p):
    seen = set()
    a, b = [], []
    
    for num in p:
        if num in seen:
            return "NO"
        seen.add(num)
        if len(a) < n and (len(a) == 0 or num > a[-1]):
            a.append(num)
        elif len(b) < n and (len(b) == 0 or num > b[-1]):
            b.append(num)
        else:
            return "NO"
    
    return "YES" if len(a) == n and len(b) == n else "NO"

t = int(input())
results = []
for _ in range(t):
    n = int(input())
    p = list(map(int, input().split()))
    results.append(can_merge(n, p))

print("\n".join(results))
def can_kill_monsters(t, cases):
    results = []
    for a, b, c in cases:
        total_health = a + b + c
        if total_health % 9 == 0:
            if min(a, b, c) >= total_health // 9:
                results.append("YES")
                continue
        results.append("NO")
    return results

t = int(input())
cases = [tuple(map(int, input().split())) for _ in range(t)]
results = can_kill_monsters(t, cases)
print("\n".join(results))
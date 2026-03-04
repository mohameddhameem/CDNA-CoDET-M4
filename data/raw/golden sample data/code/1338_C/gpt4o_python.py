def perfect_triples(t, queries):
    results = []
    for n in queries:
        if n % 3 == 1:
            results.append(n)
        elif n % 3 == 2:
            results.append(n + (n // 3) * 3)
        else:
            results.append(n + (n // 3) * 3 - 1)
    return results

import sys
input = sys.stdin.read
data = input().split()
t = int(data[0])
queries = list(map(int, data[1:t + 1]))
results = perfect_triples(t, queries)
print("\n".join(map(str, results)))
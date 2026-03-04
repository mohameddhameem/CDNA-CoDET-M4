def count_seconds(t, test_cases):
    MOD = 998244353
    results = []
    
    for case in test_cases:
        n, m, a, edges = case
        graph = [[] for _ in range(n)]
        in_degree = [0] * n
        
        for x, y in edges:
            graph[x - 1].append(y - 1)
            in_degree[y - 1] += 1
        
        # Find the sink node (the one with no out edges)
        sink = next(i for i in range(n) if len(graph[i]) == 0)

        time = 0
        while True:
            time += 1
            active = [i for i in range(n) if a[i] > 0]
            if not active:
                break
            
            next_a = [0] * n
            for x in active:
                next_a[x] = max(0, a[x] - 1)
                for y in graph[x]:
                    next_a[y] += 1
            
            a = next_a
        
        results.append(time % MOD)
    
    return results


# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

t = int(data[0])
test_cases = []
index = 1

for _ in range(t):
    n, m = map(int, data[index].split())
    a = list(map(int, data[index + 1].split()))
    edges = [tuple(map(int, data[index + 2 + i].split())) for i in range(m)]
    test_cases.append((n, m, a, edges))
    index += 2 + m

results = count_seconds(t, test_cases)

# Print results
for result in results:
    print(result)
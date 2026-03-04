def special_permutation(t, cases):
    results = []
    for n, a, b in cases:
        if a > b or a <= n // 2 or b > n // 2:
            results.append("-1")
            continue
        
        left_half = [i for i in range(1, n // 2 + 1) if i != a]
        right_half = [i for i in range(n // 2 + 1, n + 1) if i != b]
        
        if len(left_half) < n // 2 - 1 or len(right_half) < n // 2 - 1:
            results.append("-1")
            continue
        
        left_half = [a] + left_half[:n // 2 - 1]
        right_half = right_half[:n // 2 - 1] + [b]
        
        permutation = left_half + right_half
        results.append(" ".join(map(str, permutation)))

    return results

# Reading input
t = int(input())
cases = [tuple(map(int, input().split())) for _ in range(t)]

# Getting results
results = special_permutation(t, cases)

# Printing results
for result in results:
    print(result)
def cut_substrings(q, test_cases):
    MOD = 10**9 + 7
    
    results = []
    for s, t in test_cases:
        n, m = len(s), len(t)
        dp = [0] * (n + 1)
        count = [0] * (n + 1)
        dp[0] = 0
        count[0] = 1
        
        positions = []
        
        for i in range(n - m + 1):
            if s[i:i + m] == t:
                positions.append(i + 1)
        
        k = len(positions)
        
        if k == 0:
            results.append((0, 1))
            continue
        
        for i in range(1, k + 1):
            dp[i] = float('inf')
            for j in range(i):
                if positions[i - 1] >= positions[j] + m:
                    if dp[j] + 1 < dp[i]:
                        dp[i] = dp[j] + 1
                        count[i] = count[j]
                    elif dp[j] + 1 == dp[i]:
                        count[i] = (count[i] + count[j]) % MOD
        
        min_moves = float('inf')
        total_count = 0
        
        for i in range(k + 1):
            if dp[i] < min_moves:
                min_moves = dp[i]
                total_count = count[i]
            elif dp[i] == min_moves:
                total_count = (total_count + count[i]) % MOD
        
        if min_moves == float('inf'):
            results.append((0, 1))
        else:
            results.append((min_moves, total_count))
    
    return results

# Reading input
q = int(input())
test_cases = [tuple(input().strip() for _ in range(2)) for _ in range(q)]
results = cut_substrings(q, test_cases)

# Printing output
for move, cnt in results:
    print(move, cnt)
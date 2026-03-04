def max_zero_remainder_sum(n, m, k, matrix):
    half_m = m // 2
    dp = [[-1] * k for _ in range(half_m + 1)]
    dp[0][0] = 0

    for row in matrix:
        current_dp = [dp[j][:] for j in range(half_m + 1)]
        row.sort(reverse=True)
        for j in range(1, half_m + 1):
            for comb in range(min(j, len(row)) + 1):
                if comb > 0:
                    total = sum(row[:comb])
                    for mod in range(k):
                        if current_dp[j - comb][mod] != -1:
                            new_sum = current_dp[j - comb][mod] + total
                            new_mod = new_sum % k
                            current_dp[j][new_mod] = max(current_dp[j][new_mod], new_sum)
        dp = current_dp

    return max(dp[j][0] for j in range(half_m + 1) if dp[j][0] != -1)

# Input reading
n, m, k = map(int, input().strip().split())
matrix = [list(map(int, input().strip().split())) for _ in range(n)]

# Function call and output
print(max_zero_remainder_sum(n, m, k, matrix))
def zero_remainder_sum():
    n, m, k = map(int, input().split())
    a = []
    for _ in range(n):
        a.append(list(map(int, input().split())))

    dp = [[0] * (k + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            for t in range(k, 0, -1):
                if (t - a[i - 1][j - 1]) % k == 0:
                    dp[i][t] = max(dp[i][t], dp[i - 1][t] + a[i - 1][j - 1])
                dp[i][t] = max(dp[i][t], dp[i - 1][t])

    return max(dp[n]) % k == 0 and dp[n] or max(dp[n]) - k

print(zero_remainder_sum())
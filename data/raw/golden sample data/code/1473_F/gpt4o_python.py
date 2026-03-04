def max_strange_set_cost(n, a, b):
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        dp[i] = max(dp[i], dp[i - 1])  # Not including current element
        current_cost = b[i - 1]
        for j in range(1, i):
            if a[j - 1] != 0 and a[i - 1] % a[j - 1] == 0:
                current_cost += dp[j]
        dp[i] = max(dp[i], current_cost)
    return max(dp)

n = int(input().strip())
a = list(map(int, input().strip().split()))
b = list(map(int, input().strip().split()))
print(max_strange_set_cost(n, a, b))
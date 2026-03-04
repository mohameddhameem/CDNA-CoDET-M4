def strange_set(n, a, b):
    dp = [0] * n
    for i in range(n):
        for j in range(i):
            if a[j]!= -1 and a[j]!= -2 and a[j] % a[i] == 0 and dp[j] > dp[i]:
                dp[i] = dp[j]
        if a[i]!= -1:
            dp[i] += b[i]
        else:
            dp[i] = max(dp[i], b[i])
    return max(dp)

n = int(input())
a = list(map(int, input().split()))
b = list(map(int, input().split()))
a = [-2 if x == 1 else -1 if x % 2 == 0 else x for x in a]
print(strange_set(n, a, b))
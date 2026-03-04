def count_substrings(s):
    n = len(s)
    dp = [0] * (n + 1)
    ans = 0
    for i in range(1, n + 1):
        if s[i - 1] == '?':
            dp[i] = dp[i - 1]
        else:
            dp[i] = dp[i - 1] + 1
        if i >= 2 and s[i - 2]!= '?' and s[i - 1]!= '?':
            if (s[i - 2] == '0' and s[i - 1] == '1') or (s[i - 2] == '1' and s[i - 1] == '0'):
                dp[i] = max(dp[i], dp[i - 3] + 2)
        ans += dp[i]
    return ans

t = int(input())
for _ in range(t):
    s = input()
    print(count_substrings(s))
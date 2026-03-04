def kmp_prefix_function(s):
    n = len(s)
    pi = [0] * n
    j = 0
    for i in range(1, n):
        while j > 0 and s[i]!= s[j]:
            j = pi[j - 1]
        if s[i] == s[j]:
            j += 1
        pi[i] = j
    return pi

def main():
    n, m = map(int, input().split())
    a = input()
    b = input()
    pi = kmp_prefix_function(b)
    inf = 10 ** 9
    dp = [[inf] * (n + 1) for _ in range(m + 1)]
    for i in range(n + 1):
        dp[0][i] = i
    for j in range(1, m + 1):
        for i in range(n + 1):
            if i < pi[j - 1]:
                dp[j][i] = dp[j - 1][i]
            else:
                dp[j][i] = min(dp[j - 1][i], dp[j - 1][i - pi[j - 1]] + (b[j - 1]!= a[i - 1]))
    ans = []
    for k in range(n - m + 2):
        ans.append(dp[m][k + m] - k * (b[0]!= a[k + m]))
    print(*ans)

main()
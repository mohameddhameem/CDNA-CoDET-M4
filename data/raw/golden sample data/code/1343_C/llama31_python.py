def alternating_subsequence():
    t = int(input())
    for _ in range(t):
        n = int(input())
        a = list(map(int, input().split()))
        dp = [[float('-inf'), float('-inf')] for _ in range(n)]
        dp[0] = [a[0] if a[0] > 0 else -a[0], float('-inf')]
        for i in range(1, n):
            dp[i][0] = max(dp[i-1][1] + a[i] if a[i] > 0 else float('-inf'), dp[i-1][0])
            dp[i][1] = max(dp[i-1][0] + a[i] if a[i] < 0 else float('-inf'), dp[i-1][1])
        print(max(dp[n-1][0], dp[n-1][1]))

alternating_subsequence()
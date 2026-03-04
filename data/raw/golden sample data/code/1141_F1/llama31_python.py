def same_sum_blocks(n, a):
    prefix_sum = [0] * (n + 1)
    for i in range(n):
        prefix_sum[i + 1] = prefix_sum[i] + a[i]

    dp = [[] for _ in range(n + 1)]
    dp[0] = [[]]
    for i in range(1, n + 1):
        for prev in range(i):
            if prefix_sum[i] - prefix_sum[prev] == prefix_sum[i - 1] - prefix_sum[prev]:
                dp[i].append(dp[prev] + [[prev + 1, i]])

    max_len = 0
    res = []
    for i in range(n + 1):
        if len(dp[i]) > max_len:
            max_len = len(dp[i])
            res = dp[i]

    k = len(res)
    print(k)
    for block in res:
        l, r = block[0]
        while block and block[-1][0] == r:
            l, r = block.pop()
        print(l, r)

n = int(input())
a = list(map(int, input().split()))
same_sum_blocks(n, a)
import sys



input = lambda: sys.stdin.buffer.readline().decode().strip()

n, m, k = map(int, input().split())

if m == 1:

    exit(print(0))



lst = [-10 ** 9] * k

for i in range(n):

    a = [int(x) for x in input().split()]

    dp = [[[-10 ** 9] * (m // 2 + 1) for _ in range(k)] for _ in range(m + 1)]



    for j in range(m):

        dp[j][a[j] % k][1] = a[j]

        for i1 in range(k):

            for j1 in range(m // 2 + 1):

                if dp[j - 1][i1][j1] > dp[j][i1][j1]:

                    dp[j][i1][j1] = dp[j - 1][i1][j1]



                val = a[j] + dp[j - 1][i1][j1 - 1]

                ix = (a[j] + i1) % k



                if j1 and dp[j][ix][j1] < val:

                    dp[j][ix][j1] = val



    mem = [max(dp[m - 1][j]) for j in range(k)]

    old = mem.copy()

    for j1 in range(k):

        mem[j1] = max(mem[j1], lst[j1])

        for j2 in range(k):

            mem[(j2 + j1) % k] = max(mem[(j2 + j1) % k], lst[j1] + old[j2])



    lst = mem

print(max(lst[0], 0))


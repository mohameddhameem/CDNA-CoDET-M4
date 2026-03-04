import sys, math

#sys.setrecursionlimit(1000000)

INF = 1 << 100

#mod = 1000000007

mod = 998244353

input = lambda: sys.stdin.readline().rstrip()

li = lambda: list(map(int, input().split()))



t = int(input())

out = []

for _ in range(t):

    N = int(input())

    A = li()

    ans = 0

    acc = [[0] * (N+1) for _ in range(27)]

    for i in range(N):

        for j in range(27):

            acc[j][i+1] += acc[j][i]

            if A[i] == j:

                acc[j][i+1] += 1

            ans = max(ans, acc[j][i+1])

    

    LR = [[0] * (N+1) for _ in range(N+1)]

    for i in range(27):

        for l in range(1, N+1):

            for r in range(l+1, N+1):

                x = min(acc[i][l], acc[i][N] - acc[i][r]) * 2

                LR[l][r] = max(LR[l][r], x)



    for i in range(27):

        for l in range(1, N+1):

            for r in range(l+1, N+1):

                x = acc[i][r] - acc[i][l] + LR[l][r]

                ans = max(ans, x)

    out.append(ans)

                





for o in out:

    print(o)

    
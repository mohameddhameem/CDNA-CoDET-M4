import sys
input = lambda :sys.stdin.readline()[:-1]
ni = lambda :int(input())
na = lambda :list(map(int,input().split()))
yes = lambda :print("yes");Yes = lambda :print("Yes");YES = lambda : print("YES")
no = lambda :print("no");No = lambda :print("No");NO = lambda : print("NO")
#######################################################################
n = ni()
a = [na() for i in range(n)]
for i in range(n):
    if a[i][0] == 2:
        print(0)
        exit()

dp = [[0 for j in range(n)]for i in range(n)]

mod = 998244353

dp[0][-1] = 2

for i in range(n-1):
    for j in range(n):
        if dp[i][j] == 0:
            continue
        ok = 1
        for k in range(i+1):
            if a[k][i-k+1] == 0:
                continue
            if a[k][i-k+1] == 1 and j + 2 < i - k + 2:
                ok = 0
                break
            if a[k][i-k+1] == 2 and j + 2 >= i - k + 2:
                ok = 0
                break
        #print(i, j, ok)
        if ok:
            dp[i+1][min(j+1, n-1)] += dp[i][j]
            dp[i+1][min(j+1, n-1)] %= mod
        ok = 1
        for k in range(i+1):
            if a[k][i-k+1] == 0:
                continue
            if a[k][i-k+1] == 1:
                ok = 0
                break
        #print(i, j, ok)
        if ok:
            dp[i+1][0] += dp[i][j]
            dp[i+1][0] %= mod

ans = 0
for i in range(n):
    ans += dp[-1][i]
    ans %= mod
print(ans)
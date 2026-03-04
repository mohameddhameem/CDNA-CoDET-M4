def getSpecialNumber(n, k):
    MOD = 10**9 + 7
    ans = 0
    cur = 1
    for i in range(40):
        if k & (1 << i):
            ans = (ans + cur) % MOD
        cur = (cur * n) % MOD
    return ans

t = int(input())
for _ in range(t):
    n, k = map(int, input().split())
    print(getSpecialNumber(n, k))
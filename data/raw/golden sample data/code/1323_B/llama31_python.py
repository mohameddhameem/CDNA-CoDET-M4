def count_subrectangles(n, m, k, a, b):
    def get_count(x, y):
        if x > n or y > m:
            return 0
        if a[x-1] == 0 or b[y-1] == 0:
            return 0
        return 1 + get_count(x, y-1) + get_count(x-1, y) - get_count(x-1, y-1)

    dp = [[0 for _ in range(m+1)] for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(1, m+1):
            dp[i][j] = get_count(i, j)
    ans = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            if dp[i][j] == k:
                ans += 1
    return ans

n, m, k = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))
print(count_subrectangles(n, m, k, a, b))
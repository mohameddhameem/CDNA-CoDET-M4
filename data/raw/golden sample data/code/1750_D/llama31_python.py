def solve():
    MOD = 998244353
    n, m = map(int, input().split())
    grid = []
    for _ in range(n):
        grid.append(list(input()))

    def count_w():
        return sum(1 for row in grid for cell in row if cell == 'o')

    def f(i, j):
        if i < 0 or i >= n or j < 0 or j >= m:
            return 0
        return grid[i][j] == 'o'

    def g(i, j):
        if i < 0 or i >= n or j < 0 or j >= m:
            return 0
        return grid[i][j] == '*'

    def dp(i, j):
        if i < 0 or i >= n or j < 0 or j >= m:
            return 0
        if i == 0 and j == 0:
            return 1
        if f(i, j) or f(i, j-1) or f(i-1, j) or f(i-1, j-1):
            return 0
        if g(i, j) or g(i, j-1) or g(i-1, j) or g(i-1, j-1):
            return 0
        return (dp(i-1, j) + dp(i, j-1) + dp(i-1, j-1)) % MOD

    def calc(i, j):
        if i < 0 or i >= n or j < 0 or j >= m:
            return 0
        if i == 0 and j == 0:
            return 1
        if f(i, j) or f(i, j-1) or f(i-1, j) or f(i-1, j-1):
            return 0
        if g(i, j) or g(i, j-1) or g(i-1, j) or g(i-1, j-1):
            return 0
        return (calc(i-1, j) + calc(i, j-1) + calc(i-1, j-1)) % MOD

    ans = 0
    for i in range(n):
        for j in range(m):
            if f(i, j) or f(i, j-1) or f(i-1, j) or f(i-1, j-1):
                continue
            if g(i, j) or g(i, j-1) or g(i-1, j) or g(i-1, j-1):
                continue
            ans += pow(2, count_w() - 2) * calc(i, j)
            ans %= MOD

    print(ans)

solve()
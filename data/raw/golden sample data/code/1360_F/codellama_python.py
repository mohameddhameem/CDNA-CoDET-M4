def spy_string(n, m, strings):
    dp = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            if strings[i][j]!= 'a':
                dp[i][j] = 1
            else:
                dp[i][j] = 0

    for i in range(n):
        for j in range(m):
            if dp[i][j] == 0:
                continue
            for k in range(j + 1, m):
                if dp[i][k] == 0:
                    continue
                dp[i][k] = min(dp[i][k-1], dp[i][k])

    for i in range(n):
        if dp[i][m - 1] == 0:
            continue
        for j in range(m - 1):
            if dp[i][j] == 0:
                continue
            if strings[i][j]!= strings[i][j + 1]:
                return strings[i][:j] + strings[i][j + 1]
    return "-1"

# Example
test_cases = [
    [2, 4, ["abac", "zbab"]],
    [2, 4, ["aaaa", "bbbb"]],
    [3, 3, ["baa", "aaa", "aab"]],
    [2, 2, ["ab", "bb"]],
    [3, 1, ["a", "b", "c"]]
]

for case in test_cases:
    n, m = case[0], case[1]
    strings = case[2]
    result = spy_string(n, m, strings)
    print(result)
import sys

def find_longest_palindromes(string):
    n = len(string)
    dp = [[False] * n for _ in range(n)]
    longest_palindrome_length = 0
    for i in range(n):
        dp[i][i] = True
        longest_palindrome_length = max(longest_palindrome_length, 1)
    for l in range(2, n + 1):
        for i in range(n - l + 1):
            j = i + l - 1
            if dp[i + 1][j - 1] and string[i] == string[j]:
                dp[i][j] = True
                longest_palindrome_length = max(longest_palindrome_length, l)
    return longest_palindrome_length


t = int(input())
for _ in range(t):
    n = int(input())
    edges = []
    for _ in range(n - 1):
        edges.append(list(map(int, input().split())))
    string = input()
    palindromes = {}
    for v in range(1, n + 1):
        for u in range(1, n + 1):
            if v < u:
                path = find_longest_palindromes(string[v - 1:u])
                if path:
                    palindromes[(v, u)] = path
    print(max(palindromes.values()))
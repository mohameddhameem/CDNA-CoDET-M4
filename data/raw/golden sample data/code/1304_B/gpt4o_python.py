def longest_palindrome_prefix_suffix(s):
    n = len(s)
    lps = [0] * n
    j = 0
    for i in range(1, n):
        while j > 0 and s[i] != s[j]:
            j = lps[j - 1]
        if s[i] == s[j]:
            j += 1
            lps[i] = j
        else:
            lps[i] = 0

    max_len = 0
    for i in range(n):
        if s[i] == s[n - 1 - max_len]:
            max_len += 1
        if max_len > 0 and lps[max_len - 1] == max_len:
            break

    prefix = s[:max_len]
    suffix = s[n - max_len:]
    return prefix + suffix[::-1]

import sys
input = sys.stdin.read
data = input().splitlines()

t = int(data[0])
results = []
for i in range(1, t + 1):
    s = data[i]
    results.append(longest_palindrome_prefix_suffix(s))

print("\n".join(results))
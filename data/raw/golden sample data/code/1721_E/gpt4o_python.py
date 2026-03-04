def prefix_function(s):
    p = [0] * len(s)
    for i in range(1, len(s)):
        j = p[i - 1]
        while j > 0 and s[i] != s[j]:
            j = p[j - 1]
        if s[i] == s[j]:
            j += 1
        p[i] = j
    return p

import sys
input = sys.stdin.read
data = input().splitlines()

s = data[0]
q = int(data[1])
results = []

for i in range(2, 2 + q):
    t = data[i]
    combined = s + t
    p = prefix_function(combined)
    results.append(" ".join(map(str, p[len(s):len(s) + len(t)])))

print("\n".join(results))
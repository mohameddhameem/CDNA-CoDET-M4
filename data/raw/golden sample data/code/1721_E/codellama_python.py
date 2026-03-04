import sys

def prefix_function(s):
    n = len(s)
    p = [0] * n
    for i in range(n-1):
        if s[i] == s[i+1]:
            p[i+1] = p[i] + 1
        else:
            p[i+1] = 0
    return p

s = input()
q = int(input())

for _ in range(q):
    t = input()
    s_plus_t = s + t
    p = prefix_function(s_plus_t)
    print(p[len(s):len(s) + len(t)])
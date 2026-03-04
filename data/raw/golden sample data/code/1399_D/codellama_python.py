import sys

def binary_string_subsequences(s):
    n = len(s)
    k = n // 2
    a = [0] * n
    for i in range(n):
        if s[i] == '0':
            a[i] = l
            l += 1
        else:
            a[i] = k
    return (k, a)

t = int(sys.stdin.readline())
for _ in range(t):
    n = int(sys.stdin.readline())
    s = sys.stdin.readline().strip()
    k, a = binary_string_subsequences(s)
    print(k, end='')
    print(*a, end='')
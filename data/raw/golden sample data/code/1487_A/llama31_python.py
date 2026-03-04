import sys

def arena(n, a):
    a.sort()
    count = 0
    for i in range(n):
        if a[i]!= a[0]:
            count += 1
    return count

t = int(sys.stdin.readline())
for _ in range(t):
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    print(arena(n, a))
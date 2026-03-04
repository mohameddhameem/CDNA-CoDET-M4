import sys

n, m, k = map(int, sys.stdin.readline().split())
a = list(map(int, sys.stdin.readline().split()))
b = list(map(int, sys.stdin.readline().split()))

def count_subrectangles(a, b):
    c = [[int(ai*bi) for ai, bi in zip(a, b)] for _ in range(n)]
    count = 0
    for i in range(n):
        for j in range(m):
            if c[i][j] == 1:
                count += 1
                for x in range(i, n):
                    for y in range(j, m):
                        if c[x][y] == 0:
                            break
                    else:
                        count += 1
    return count

print(count_subrectangles(a, b))
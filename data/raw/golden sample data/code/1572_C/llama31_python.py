from collections import deque

def paint(n, m, a):
    color = {}
    for i in range(n):
        for j in range(m):
            color[(i, j)] = a[i][j]

    q = deque()
    ans = []
    for i in range(n - 1):
        for j in range(m - 1):
            if color[(i, j)] == color[(i + 1, j)] == color[(i, j + 1)] == color[(i + 1, j + 1)]:
                q.append(((i, j), color[(i, j)]))
                ans.append(((i, j), color[(i, j)]))

    while q:
        x, c = q.popleft()
        for dx, dy in [(1, 0), (0, 1), (1, 1)]:
            nx, ny = x[0] + dx, x[1] + dy
            if 0 <= nx < n - 1 and 0 <= ny < m - 1 and color[(nx, ny)]!= c:
                color[(nx, ny)] = c
                q.append(((nx, ny), c))
                ans.append(((nx, ny), c))

    for i in range(n):
        for j in range(m):
            if color[(i, j)] == 0:
                return -1

    ans.reverse()
    return len(ans), ans

n, m = map(int, input().split())
a = [list(map(int, input().split())) for _ in range(n)]
print(*paint(n, m, a))
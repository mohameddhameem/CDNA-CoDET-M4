from collections import deque

def solve_maze(maze):
    n, m = len(maze), len(maze[0])
    queue = deque()
    for i in range(n):
        for j in range(m):
            if maze[i][j] == 'G':
                queue.append((i, j))
                maze[i][j] = '.'

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < m and maze[nx][ny] == '.':
                maze[nx][ny] = 'B'
                queue.append((nx, ny))

    for i in range(n):
        for j in range(m):
            if maze[i][j] == 'B':
                return "No"

    return "Yes"

t = int(input())
for _ in range(t):
    n, m = map(int, input().split())
    maze = [list(input()) for _ in range(n)]
    print(solve_maze(maze))
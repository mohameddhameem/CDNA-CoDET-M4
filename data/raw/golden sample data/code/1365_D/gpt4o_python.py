def solve_maze(t, test_cases):
    results = []
    
    for case in test_cases:
        n, m, maze = case
        can_escape = True

        # Place walls around bad people
        for i in range(n):
            for j in range(m):
                if maze[i][j] == 'B':
                    if i > 0 and maze[i - 1][j] == '.':
                        maze[i - 1] = maze[i - 1][:j] + '#' + maze[i - 1][j + 1:]
                    if i < n - 1 and maze[i + 1][j] == '.':
                        maze[i + 1] = maze[i + 1][:j] + '#' + maze[i + 1][j + 1:]
                    if j > 0 and maze[i][j - 1] == '.':
                        maze[i] = maze[i][:j - 1] + '#' + maze[i][j:]
                    if j < m - 1 and maze[i][j + 1] == '.':
                        maze[i] = maze[i][:j + 1] + '#' + maze[i][j + 2:]
        
        # Check if all good people can escape
        visited = [[False] * m for _ in range(n)]
        
        def dfs(x, y):
            if x < 0 or x >= n or y < 0 or y >= m or visited[x][y] or maze[x][y] == '#':
                return
            visited[x][y] = True
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                dfs(x + dx, y + dy)
        
        # Start DFS from the exit point
        dfs(n - 1, m - 1)
        
        for i in range(n):
            for j in range(m):
                if maze[i][j] == 'G' and not visited[i][j]:
                    can_escape = False
                if maze[i][j] == 'B' and visited[i][j]:
                    can_escape = False
        
        results.append("Yes" if can_escape else "No")
    
    return results

# Reading input
t = int(input())
test_cases = []
for _ in range(t):
    n, m = map(int, input().split())
    maze = [input
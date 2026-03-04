def solve_maze(n, m):
    grid = [input().split() for _ in range(n)]
    goal_row = n - 1
    goal_col = m - 1

    if grid[goal_row][goal_col] == ".":
        return "NO"

    for row in range(n):
        for col in range(m):
            if grid[row][col] == "G" and row!= goal_row or col!= goal_col:
                return "NO"

    for row in range(n):
        for col in range(m):
            if grid[row][col] == "#":
                continue
            if row!= 0 and grid[row - 1][col]!= "#":
                continue
            if col!= 0 and grid[row][col - 1]!= "#":
                continue
            if row!= goal_row and grid[row + 1][col]!= "#":
                continue
            if col!= goal_col and grid[row][col + 1]!= "#":
                continue
            return "NO"
    return "YES"

 t = int(input())
 for _ in range(t):
    n, m = map(int, input().split())
    grid = [list(input().split()) for _ in range(n)]
    print(solve_maze(n, m))
def min_operations_to_portal(n, m, grid):
    def count_changes(x1, y1, a, b):
        changes = 0
        for x in range(x1, x1 + a):
            for y in range(y1, y1 + b):
                if x == x1 or x == x1 + a - 1 or y == y1 or y == y1 + b - 1:
                    if (x == x1 or x == x1 + a - 1) and (y == y1 or y == y1 + b - 1):
                        continue  # corners can be anything
                    if grid[x][y] == '0':
                        changes += 1  # obsidian needed
                else:
                    if grid[x][y] == '1':
                        changes += 1  # empty needed
        return changes
    
    min_changes = float('inf')
    
    for a in range(5, n + 1):
        for b in range(4, m + 1):
            for x in range(n - a + 1):
                for y in range(m - b + 1):
                    min_changes = min(min_changes, count_changes(x, y, a, b))

    return min_changes

t = int(input())
for _ in range(t):
    n, m = map(int, input().split())
    grid = [input().strip() for _ in range(n)]
    result = min_operations_to_portal(n, m, grid)
    print(result)
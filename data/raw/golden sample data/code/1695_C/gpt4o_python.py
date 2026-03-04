def can_zero_path(n, m, grid):
    total = sum(sum(row) for row in grid)
    if total < 0:
        return "NO"
    
    count_positive = sum(1 for row in grid for value in row if value == 1)
    count_negative = sum(1 for row in grid for value in row if value == -1)
    
    if count_positive % 2 == 0 and count_negative % 2 == 0:
        return "YES"
    
    # Check if there's at least one 1 on the path
    if grid[0][0] == 1 or grid[n-1][m-1] == 1:
        return "YES"
    
    return "NO"


import sys
input = sys.stdin.read
data = input().splitlines()

t = int(data[0])
result = []

index = 1
for _ in range(t):
    n, m = map(int, data[index].split())
    grid = []
    for i in range(n):
        grid.append(list(map(int, data[index + 1 + i].split())))
    index += n + 1
    
    result.append(can_zero_path(n, m, grid))

print("\n".join(result))
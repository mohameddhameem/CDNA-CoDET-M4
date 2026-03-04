import sys

def is_zero_path(grid):
    # Initialize the sum to 0
    sum = 0

    # Iterate through each row and column
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            # If the current cell is 1, add it to the sum
            if grid[i][j] == 1:
                sum += 1
            # If the current cell is -1, subtract it from the sum
            elif grid[i][j] == -1:
                sum -= 1
            # If the sum is 0, return True
            if sum == 0:
                return True

    # If the sum is not 0, return False
    return False

# Read the number of test cases
t = int(input())

# Loop through each test case
for i in range(t):
    # Read the size of the grid
    n, m = map(int, input().split())

    # Read the grid
    grid = []
    for _ in range(n):
        grid.append(list(map(int, input().split())))

    # Check if there is a zero path
    if is_zero_path(grid):
        print("YES")
    else:
        print("NO")
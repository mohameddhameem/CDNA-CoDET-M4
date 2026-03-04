def can_walk(t, test_cases):
    results = []
    for case in test_cases:
        a, b, c, d = case[0]
        x, y, x1, y1, x2, y2 = case[1]

        if (x1 == x2 and (a + b > 0)) or (y1 == y2 and (c + d > 0)):
            results.append("NO")
            continue

        left_moves = a
        right_moves = b
        down_moves = c
        up_moves = d

        min_x = x - left_moves + right_moves
        max_x = x + right_moves - left_moves
        min_y = y - down_moves + up_moves
        max_y = y + up_moves - down_moves

        if (x1 <= max_x and x2 >= min_x and y1 <= max_y and y2 >= min_y):
            results.append("YES")
        else:
            results.append("NO")

    return results

# Reading input
import sys
input = sys.stdin.read
data = input().splitlines()

t = int(data[0])
test_cases = []
index = 1
for _ in range(t):
    a, b, c, d = map(int, data[index].split())
    x, y, x1, y1, x2, y2 = map(int, data[index + 1].split())
    test_cases.append(((a, b, c, d), (x, y, x1, y1, x2, y2)))
    index += 2

# Getting results
results = can_walk(t, test_cases)

# Printing output
for result in results:
    print(result)
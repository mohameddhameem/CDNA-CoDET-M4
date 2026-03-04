def area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)

import sys

n = int(sys.stdin.readline().strip())
vertices = [tuple(map(int, sys.stdin.readline().strip().split())) for _ in range(n)]

# Calculate total area
total_area = 0
for i in range(n):
    x1, y1 = vertices[i]
    x2, y2 = vertices[(i + 1) % n]
    x3, y3 = vertices[(i + 2) % n]
    total_area += area(x1, y1, x2, y2, x3, y3)

# Area of the whole pizza
total_area /= 2.0

# Determine who can win
# If n is even, both can eat optimally and will eat exactly half
# If n is odd, Beatrice can always force Alberto to eat more
if n % 2 == 0:
    print("Alberto")
else:
    print("Beatrice")

# Game interaction
chosen = set()
turn = 0 if n % 2 == 0 else 1  # 0 for Alberto's turn, 1 for Beatrice's turn

while len(chosen) < n - 2:
    if turn == 0:  # Alberto's turn
        # Choose the best vertex to minimize his area
        best_vertex = -1
        best_area = float('inf')
        for i in range(n):
            if i not in chosen:
                x1, y1 = vertices[i]
                x2, y2 = vertices[(i + 1) % n]
                x3, y3 = vertices[(i - 1) % n]
                current_area = area(x1, y1, x2, y2, x3, y3)
                if current_area < best_area:
                    best_area = current_area
                    best_vertex = i + 1  # +1 for 1-based index
        print(best_vertex)
        chosen.add(best_vertex - 1)
        turn = 1
    else:  # Beatrice's turn
        judge_choice = int(sys.stdin.readline().strip())
        chosen.add(judge_choice -
def count_cells(t, test_cases):
    results = []
    for n, s in test_cases:
        down_count = s.count('D')
        right_count = s.count('R')
        max_down = min(down_count, n - 1)
        max_right = min(right_count, n - 1)
        total_cells = (max_down + 1) * (max_right + 1)
        results.append(total_cells)
    return results

import sys
input = sys.stdin.read
data = input().splitlines()
t = int(data[0])
test_cases = [(int(data[i * 2 + 1]), data[i * 2 + 2]) for i in range(t)]
results = count_cells(t, test_cases)
print('\n'.join(map(str, results)))
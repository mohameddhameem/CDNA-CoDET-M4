def max_alternating_subsequence_sum(test_cases):
    results = []
    for n, a in test_cases:
        max_sum = 0
        current = a[0]
        
        for i in range(1, n):
            if (current > 0 and a[i] < 0) or (current < 0 and a[i] > 0):
                max_sum += current
                current = a[i]
            else:
                current = max(current, a[i])
        
        max_sum += current
        results.append(max_sum)
    
    return results

import sys
input = sys.stdin.read
data = input().split()
index = 0
t = int(data[index])
index += 1
test_cases = []

for _ in range(t):
    n = int(data[index])
    index += 1
    a = list(map(int, data[index:index+n]))
    index += n
    test_cases.append((n, a))

results = max_alternating_subsequence_sum(test_cases)
print('\n'.join(map(str, results)))
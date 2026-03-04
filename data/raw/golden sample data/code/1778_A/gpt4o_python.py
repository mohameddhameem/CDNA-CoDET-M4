def max_flip_flop_sum(test_cases):
    results = []
    for case in test_cases:
        n, a = case
        current_sum = sum(a)
        max_sum = current_sum
        for i in range(n - 1):
            # Calculate the new sum if we flip a[i] and a[i + 1]
            new_sum = current_sum - a[i] - a[i + 1] + (-a[i]) + (-a[i + 1])
            max_sum = max(max_sum, new_sum)
        results.append(max_sum)
    return results

# Input reading
import sys
input = sys.stdin.read
data = input().splitlines()

t = int(data[0])
test_cases = []
index = 1
for _ in range(t):
    n = int(data[index])
    a = list(map(int, data[index + 1].split()))
    test_cases.append((n, a))
    index += 2

# Get results
results = max_flip_flop_sum(test_cases)

# Print results
for result in results:
    print(result)
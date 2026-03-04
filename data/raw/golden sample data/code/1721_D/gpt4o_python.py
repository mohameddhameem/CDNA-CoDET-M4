def maximum_d(t, test_cases):
    results = []
    for n, b in test_cases:
        a = []
        for i in range(n):
            if b[i] == '0':
                a.append('1')
            else:
                if i == 0 or b[i] != b[i - 1]:
                    a.append('1')
                else:
                    a.append('0')
        results.append(''.join(a))
    return results

# Read input
t = int(input())
test_cases = []
for _ in range(t):
    n = int(input())
    b = input().strip()
    test_cases.append((n, b))

# Get results and print them
results = maximum_d(t, test_cases)
print('\n'.join(results))
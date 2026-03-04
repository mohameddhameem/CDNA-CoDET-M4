def make_array_good(t, test_cases):
    results = []
    for case in test_cases:
        n, a = case
        max_a = max(a)
        operations = []
        for i in range(n):
            if a[i] < max_a:
                operations.append((i + 1, max_a - a[i]))
                a[i] = max_a
        results.append((len(operations), operations))
    return results


t = int(input())
test_cases = []
for _ in range(t):
    n = int(input())
    a = list(map(int, input().split()))
    test_cases.append((n, a))

results = make_array_good(t, test_cases)
for operations_count, operations in results:
    print(operations_count)
    for op in operations:
        print(op[0], op[1])
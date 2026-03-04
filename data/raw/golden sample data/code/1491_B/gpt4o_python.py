def minimal_contests(t, test_cases):
    results = []
    for i in range(t):
        n, x = test_cases[i][0]
        a = test_cases[i][1]
        
        if all(ai == x for ai in a):
            results.append(0)
            continue
        
        if x in a:
            results.append(1)
            continue
        
        total = sum(a)
        if (total + x) % n == 0:
            results.append(2)
        else:
            results.append(1)

    return results

# Input reading
t = int(input().strip())
test_cases = []
for _ in range(t):
    n, x = map(int, input().strip().split())
    a = list(map(int, input().strip().split()))
    test_cases.append(((n, x), a))

# Get results
results = minimal_contests(t, test_cases)

# Output results
for res in results:
    print(res)
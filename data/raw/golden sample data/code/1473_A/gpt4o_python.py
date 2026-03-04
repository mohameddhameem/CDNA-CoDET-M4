def can_replace_elements(t, test_cases):
    results = []
    for n, d, a in test_cases:
        if all(x <= d for x in a):
            results.append("YES")
            continue
        
        a.sort()
        if a[0] + a[1] <= d:
            results.append("YES")
        else:
            results.append("NO")
    
    return results

t = int(input())
test_cases = []
for _ in range(t):
    n, d = map(int, input().split())
    a = list(map(int, input().split()))
    test_cases.append((n, d, a))

results = can_replace_elements(t, test_cases)
for result in results:
    print(result)
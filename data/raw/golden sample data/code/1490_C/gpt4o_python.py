def sum_of_cubes(t, test_cases):
    results = []
    for x in test_cases:
        found = False
        a = 1
        while a**3 < x:
            b_cubed = x - a**3
            if b_cubed > 0:
                b = int(b_cubed ** (1/3))
                if b**3 == b_cubed:
                    found = True
                    break
            a += 1
        results.append("YES" if found else "NO")
    return results

# Input reading
t = int(input().strip())
test_cases = [int(input().strip()) for _ in range(t)]

# Output results
for result in sum_of_cubes(t, test_cases):
    print(result)
def unique_number(t, cases):
    results = []
    for x in cases:
        if x > 45:  # The maximum sum of distinct digits (1 to 9) is 45
            results.append(-1)
            continue
        
        digits = []
        for i in range(9, 0, -1):
            if x >= i:
                digits.append(i)
                x -= i
            if x == 0:
                break
        
        if x == 0:
            results.append(int(''.join(map(str, sorted(digits)))))
        else:
            results.append(-1)
    
    return results

# Read input
t = int(input())
cases = [int(input()) for _ in range(t)]
results = unique_number(t, cases)

# Print results
for result in results:
    print(result)
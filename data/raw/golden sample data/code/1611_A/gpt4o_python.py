def make_even(t, cases):
    results = []
    for n in cases:
        s = str(n)
        if int(s[-1]) % 2 == 0:
            results.append(0)
            continue
        
        found_even = False
        for i in range(len(s) - 1):
            if int(s[i]) % 2 == 0:
                found_even = True
                if i == len(s) - 2:
                    results.append(1)
                else:
                    results.append(2)
                break
        
        if not found_even:
            results.append(-1)
    
    return results

# Reading input
t = int(input())
cases = [int(input()) for _ in range(t)]

# Processing and printing results
results = make_even(t, cases)
for result in results:
    print(result)
def reconstruct_binary_string(t, cases):
    results = []
    for s, x in cases:
        n = len(s)
        w = ['0'] * n
        
        for i in range(n):
            if s[i] == '1':
                if i - x >= 0:
                    w[i - x] = '1'
                if i + x < n:
                    w[i + x] = '1'
        
        # Verify if the constructed w can generate s
        valid = True
        for i in range(n):
            if s[i] == '1':
                if not ((i - x >= 0 and w[i - x] == '1') or (i + x < n and w[i + x] == '1')):
                    valid = False
                    break
        
        if valid:
            results.append(''.join(w))
        else:
            results.append('-1')
    
    return results

# Reading input
import sys
input = sys.stdin.read
data = input().splitlines()

t = int(data[0])
cases = []
for i in range(1, 2*t, 2):
    s = data[i]
    x = int(data[i + 1])
    cases.append((s, x))

# Getting the results
results = reconstruct_binary_string(t, cases)

# Printing output
for result in results:
    print(result)
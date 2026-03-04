def balance_bits(t, test_cases):
    results = []
    for n, s in test_cases:
        a = [''] * n
        b = [''] * n
        open_needed = n // 2

        for i in range(n):
            if s[i] == '1':
                a[i] = '('
                b[i] = '('
                open_needed -= 1
            else:
                if open_needed > 0:
                    a[i] = '('
                    b[i] = ')'
                else:
                    a[i] = ')'
                    b[i] = '('

        if open_needed != 0:
            results.append("NO")
            continue

        balance_a = 0
        balance_b = 0
        valid = True
        for i in range(n):
            balance_a += 1 if a[i] == '(' else -1
            balance_b += 1 if b[i] == '(' else -1
            if balance_a < 0 or balance_b < 0:
                valid = False
                break

        if valid:
            results.append("YES")
            results.append(''.join(a))
            results.append(''.join(b))
        else:
            results.append("NO")
    
    return results

# Input Handling
import sys
input = sys.stdin.read
data = input().splitlines()
t = int(data[0])
test_cases = []

index = 1
for _ in range(t):
    n = int(data[index])
    s = data[index + 1]
    test_cases.append((n, s))
    index += 2

# Solve the problem
results = balance_bits(t, test_cases)

# Output results
for result in results:
    print(result)
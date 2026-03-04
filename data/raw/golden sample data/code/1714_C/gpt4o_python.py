def minimum_varied_number(s):
    if s > 45:
        return -1  # Not possible since we can only use digits 1-9
    digits = []
    for i in range(9, 0, -1):
        if s >= i:
            digits.append(i)
            s -= i
    if s > 0:
        return -1  # If there's still sum left, it's not possible
    return int(''.join(map(str, sorted(digits))))

t = int(input())
results = []
for _ in range(t):
    s = int(input())
    results.append(minimum_varied_number(s))

print('\n'.join(map(str, results)))
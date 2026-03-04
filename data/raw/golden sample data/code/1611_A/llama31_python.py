def make_even(n):
    n = str(n)
    if int(n) % 2 == 0:
        return 0
    if '2' not in n and '4' not in n and '6' not in n and '8' not in n:
        return -1
    for i in range(len(n)):
        if n[i] in ['2', '4', '6', '8']:
            if i == 0:
                return 1
            if n[-i] in ['2', '4', '6', '8']:
                return 1
            return 2
    return -1

t = int(input())
for _ in range(t):
    n = int(input())
    print(make_even(n))
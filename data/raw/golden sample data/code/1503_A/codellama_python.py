def balance_bits(s):
    n = len(s)
    a = [None] * n
    b = [None] * n
    for i in range(n):
        if s[i] == '1':
            a[i] = '('
            b[i] = ')'
        else:
            a[i] = ')'
            b[i] = '('
    return 'YES\n' + ''.join(a) + '\n' + ''.join(b)

t = int(input())
for _ in range(t):
    n = int(input())
    s = input()
    result = balance_bits(s)
    print(result)
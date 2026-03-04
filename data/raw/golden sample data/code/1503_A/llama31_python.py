def balance_bits():
    for _ in range(int(input())):
        n = int(input())
        s = input()

        if s.count('0') % 2!= 0 or s.count('1') < n // 2:
            print('NO')
            continue

        a = [''] * n
        b = [''] * n
        balance = 0
        count = 0

        for i, c in enumerate(s):
            if c == '1':
                balance += 1
                a[i] = b[i] = '('
            elif balance:
                a[i] = b[i] = ')'
                balance -= 1
            else:
                if count % 2 == 0:
                    a[i] = '('
                    b[i] = ')'
                else:
                    a[i] = ')'
                    b[i] = '('
                count += 1

        print('YES')
        print(''.join(a))
        print(''.join(b))

balance_bits()
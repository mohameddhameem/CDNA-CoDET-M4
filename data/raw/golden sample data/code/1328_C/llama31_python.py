def ternary_xor():
    for _ in range(int(input())):
        n = int(input())
        x = input()
        a = ''
        b = ''
        i = 0
        while i < n:
            if x[i] == '2':
                if i == 0:
                    a += '1'
                    b += '1'
                elif x[i-1] == '2':
                    a += '0'
                    b += '2'
                else:
                    a += '2'
                    b += '0'
            else:
                a += x[i]
                b += '0'
            i += 1
        print(a)
        print(b)

ternary_xor()
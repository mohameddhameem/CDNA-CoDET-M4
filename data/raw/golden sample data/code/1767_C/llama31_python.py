def reconstruct_binary_string():
    t = int(input())

    for _ in range(t):
        s = input()
        x = int(input())
        n = len(s)

        w = ['0'] * n
        for i in range(n):
            if (i - x >= 0 and s[i] == '1' and w[i - x] == '0') or \
               (i + x < n and s[i] == '1' and w[i + x] == '0'):
                print(-1)
                break
            else:
                w[i] = '1' if (i - x >= 0 and s[i] == '1' and w[i - x] == '1') or \
                          (i + x < n and s[i] == '1' and w[i + x] == '1') else '0'

        else:
            print(''.join(w))

reconstruct_binary_string()
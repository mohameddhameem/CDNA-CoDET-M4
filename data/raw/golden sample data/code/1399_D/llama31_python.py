def binary_string_to_subsequences():
    t = int(input())
    for _ in range(t):
        n = int(input())
        s = input()
        ones, zeros = [], []
        for i in range(n):
            if s[i] == '0':
                if zeros:
                    zeros.append(zeros[-1] + 1)
                else:
                    zeros.append(1)
            else:
                if ones:
                    ones.append(ones[-1] + 1)
                else:
                    ones.append(1)
        print(len(ones) + len(zeros))
        print(*ones + list(reversed(zeros)))

binary_string_to_subsequences()
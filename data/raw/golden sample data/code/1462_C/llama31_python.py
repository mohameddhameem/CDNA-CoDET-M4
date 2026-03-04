def unique_number(x):
    for i in range(9, 0, -1):
        for j in range(9, 0, -1):
            for k in range(9, 0, -1):
                for l in range(9, 0, -1):
                    if i + j + k + l == x and len(str(i)) + len(str(j)) + len(str(k)) + len(str(l)) == 4:
                        return int(str(i) + str(j) + str(k) + str(l))
                    elif i + j + k + l == x and len(str(i)) + len(str(j)) + len(str(k)) + len(str(l)) == 3:
                        for m in range(9, 0, -1):
                            if i + j + k + m == x and len(str(i)) + len(str(j)) + len(str(k)) + len(str(m)) == 4:
                                return int(str(i) + str(j) + str(k) + str(m))
                    elif i + j + k + l == x and len(str(i)) + len(str(j)) + len(str(k)) + len(str(l)) == 2:
                        for m in range(9, 0, -1):
                            for n in range(9, 0, -1):
                                if i + j + k + m + n == x and len(str(i)) + len(str(j)) + len(str(k)) + len(str(m)) + len(str(n)) == 5:
                                    return int(str(i) + str(j) + str(k) + str(m) + str(n))
                    elif i + j + k + l == x and len(str(i)) + len(str(j)) + len(str(k)) + len(str(l)) == 1:
                        for m in range(9, 0, -1):
                            for n in range(9, 0, -1):
                                for o in range(9, 0, -1):
                                    if i + j + k + m + n + o == x and len(str(i)) + len(str(j)) + len(str(k)) + len(str(m)) + len(str(n)) + len(str(o)) == 6:
                                        return int(str(i) + str(j) + str(k) + str(m) + str(n) + str(o))
    return -1

t = int(input())
for _ in range(t):
    x = int(input())
    print(unique_number(x))
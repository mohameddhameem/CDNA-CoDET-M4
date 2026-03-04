S = set()
for i in range (1,10000):
    S.add(i**3)
for _ in range (int(input())):
    N = int(input())
    flag = False
    for i in S:
        if N-i in S:
            flag = True
            break
    if flag:
        print('YES')
    else:
        print('NO')
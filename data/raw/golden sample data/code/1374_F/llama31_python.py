def cyclic_shifts_sorting():
    n = int(input())
    a = list(map(int, input().split()))

    if a == sorted(a):
        print(0)
        return

    ans = []
    for i in range(n - 3):
        if a[i] == a[i + 2] and a[i + 1] > a[i] and a[i] > a[i + 3]:
            ans.append(i)
            a[i], a[i + 1], a[i + 2] = a[i + 2], a[i], a[i + 1]

    for i in range(n - 3):
        if a[i] == a[i + 2] and a[i + 1] < a[i] and a[i] < a[i + 3]:
            ans.append(i)
            a[i], a[i + 1], a[i + 2] = a[i + 2], a[i], a[i + 1]

    for i in range(n - 3):
        if a[i] == a[i + 2] and a[i + 1] > a[i] and a[i] > a[i + 3]:
            ans.append(i)
            a[i], a[i + 1], a[i + 2] = a[i + 2], a[i], a[i + 1]

    if a == sorted(a):
        print(len(ans))
        print(*[i + 1 for i in ans])
    else:
        print(-1)

t = int(input())
for _ in range(t):
    cyclic_shifts_sorting()
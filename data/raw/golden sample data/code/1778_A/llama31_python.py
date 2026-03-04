def flip_flop_sum():
    t = int(input())
    for _ in range(t):
        n = int(input())
        a = list(map(int, input().split()))
        s = sum(a)
        max_sum = s
        for i in range(n - 1):
            if a[i] == a[i + 1] == -1:
                max_sum = max(max_sum, s + 4)
            elif a[i] == a[i + 1] == 1:
                max_sum = max(max_sum, s - 4)
        print(max_sum)

flip_flop_sum()
for _ in range(int(input())):

    n = int(input())

    a = [int(x) for x in input().split()]



    print(n - a.count(min(a)))


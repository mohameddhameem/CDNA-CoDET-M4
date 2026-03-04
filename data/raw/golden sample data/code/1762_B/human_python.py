def find_good(c):

    b = 2
    while b < c:
        b *= 2
    return b


for _ in range(int(input())):
    n = int(input())
    a = list(map(int, input().split()))

    print(n)
    for i in range(n):
        b = find_good(a[i])
        print("%d %d" % (i+1, b - a[i]))

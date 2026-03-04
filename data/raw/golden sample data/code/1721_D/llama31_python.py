def maximum_and(n, a, b):
    count = [0] * 30
    for i in range(n):
        for j in range(30):
            if (a[i] >> j) & 1 and (b[i] >> j) & 1:
                count[j] += 1
    ans = 0
    for i in range(30):
        if count[i] == n:
            ans |= (1 << i)
    return ans

for _ in range(int(input())):
    n = int(input())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    print(maximum_and(n, a, b))
def count_good_subarrays(n, a):
    ans = n
    for i in range(1, n - 1):
        if a[i - 1] > a[i] > a[i + 1] or a[i - 1] < a[i] < a[i + 1]:
            ans += 1
    for i in range(2, n):
        if a[i - 2] < a[i - 1] > a[i] or a[i - 2] > a[i - 1] < a[i]:
            ans += i - 1
    return ans

t = int(input())
for _ in range(t):
    n = int(input())
    a = list(map(int, input().split()))
    print(count_good_subarrays(n, a))
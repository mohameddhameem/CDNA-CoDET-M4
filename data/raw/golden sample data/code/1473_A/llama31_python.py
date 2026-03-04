def replacing_elements(n, d, a):
    a.sort()
    if a[0] + a[1] > d:
        return "NO"
    return "YES"

t = int(input())
for _ in range(t):
    n, d = map(int, input().split())
    a = list(map(int, input().split()))
    print(replacing_elements(n, d, a))
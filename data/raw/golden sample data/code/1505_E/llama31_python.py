def can_walk(n, a):
    if n == 1:
        return False
    if a[-1] - a[0] >= 120:
        return True
    max_distance = 0
    for i in range(1, n):
        max_distance = max(max_distance, a[i] - a[i - 1])
        if max_distance >= 120:
            return True
    return False

t = int(input())
for _ in range(t):
    n = int(input())
    a = list(map(int, input().split()))
    if can_walk(n, a):
        print("YES")
    else:
        print("NO")
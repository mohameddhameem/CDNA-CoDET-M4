from collections import deque

def frog_traveler(n, a, b):
    q = deque([(n, 0, [])])
    vis = set([n])
    while q:
        x, step, path = q.popleft()
        if x == 0:
            return step, path + [0]
        for i in range(x - a[x - 1], -1, -1):
            y = i + b[i]
            if y not in vis:
                vis.add(y)
                q.append((y, step + 1, path + [x]))
    return -1, []

n = int(input())
a = list(map(int, input().split()))
b = list(map(int, input().split()))
ans = frog_traveler(n, a, b)
if ans[0]!= -1:
    print(ans[0])
    print(*ans[1])
else:
    print(-1)
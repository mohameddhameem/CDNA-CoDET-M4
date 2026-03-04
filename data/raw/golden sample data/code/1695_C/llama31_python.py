from collections import deque
from functools import lru_cache
import sys

input = sys.stdin.readline

n, m, k = map(int, input().split())
g = [[] for _ in range(n)]
deg = [0] * n
for _ in range(m):
    u, v, w = map(int, input().split())
    u -= 1
    v -= 1
    g[u].append((v, w))
    deg[u] += 1

def get_path(u):
    vis = [0] * n
    vis[u] = 1
    q = deque([(u, 0)])
    while q:
        u, i = q.popleft()
        if i == k:
            return u
        for v, _ in g[u]:
            if not vis[v]:
                vis[v] = 1
                q.append((v, i + 1))
    return -1

@lru_cache(maxsize=None)
def dfs(u, c):
    if u == -1:
        return 1
    vis = [0] * n
    vis[u] = 1
    q = deque([(u, 0)])
    while q:
        u, i = q.popleft()
        if i == k:
            if c[i - 1] == i:
                return 0
            for v, _ in g[u]:
                if not vis[v]:
                    vis[v] = 1
                    q.append((v, i + 1))
        else:
            w = sorted([w for v, w in g[u]])[c[i] - 1]
            for v, ww in g[u]:
                if ww == w and not vis[v]:
                    vis[v] = 1
                    q.append((v, i + 1))
    return 0

ans = 0
for u in range(n):
    if deg[u] == 0:
        continue
    v = get_path(u)
    if v == -1:
        continue
    c = [0] * k
    for i in range(k):
        c[i] = deg[v] - dfs(v, c[:i]) + 1
        v = get_path(v)
        if v == -1:
            break
    if v!= -1:
        ans += 1
print(ans)
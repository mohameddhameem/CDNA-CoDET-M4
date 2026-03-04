import sys
input = sys.stdin.readline

class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        while self.parent[x]!= x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def merge(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x!= y:
            if self.size[x] < self.size[y]:
                x, y = y, x
            self.parent[y] = x
            self.size[x] += self.size[y]

def kruskal(n, edges):
    edges.sort(key=lambda x: x[2])
    dsu = DSU(n)
    mst = []
    for u, v, w in edges:
        if dsu.find(u)!= dsu.find(v):
            mst.append((u, v, w))
            dsu.merge(u, v)
    return mst

def solve(n, edges):
    edges.sort(key=lambda x: x[2])
    ans = 0
    for i in range(1, n):
        ans += i
    mst = kruskal(n, edges)
    mst.sort(key=lambda x: x[2])
    for u, v, w in mst:
        if w % 2 == 1:
            ans += n - 1
    return ans

n, m = map(int, input().split())
edges = []
for _ in range(m):
    u, v, w = map(int, input().split())
    edges.append((u - 1, v - 1, w))
ans = solve(n, edges)
print(ans)
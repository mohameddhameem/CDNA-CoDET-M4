#   Author: yumtam

#   Created at: 2021-03-02 20:19



def main():

    n = int(input())

    A = [int(t) for t in input().split()]

    cost = [int(t) for t in input().split()]



    g = Flow(n+2)

    s, t = n, n+1



    last = [-1] * 101

    for i, x in enumerate(A):

        for d in range(1, 101):

            if x % d == 0 and last[d] >= 0:

                g.add_edge(i, last[d], float('inf'))

        last[x] = i

        if cost[i] >= 0:

            g.add_edge(s, i, cost[i])

        else:

            g.add_edge(i, t, -cost[i])



    min_cut = g.calc(s, t)

    ans = sum(max(c, 0) for c in cost) - min_cut



    print(ans)





class Flow:

    def __init__(self, n):

        self.n = n

        self.g = [dict() for _ in range(n)]



    def add_edge(self, u, v, w):

        self.g[u][v] = w

        self.g[v][u] = 0



    def bfs(self, s, t):

        q = [s]

        vis = [0] * self.n

        vis[s] = 1

        prev = [-1] * self.n

        found = False

        for ver in q:

            for nei, w in self.g[ver].items():

                if not vis[nei] and w > 0:

                    vis[nei] = 1

                    prev[nei] = ver

                    q.append(nei)

                    if nei == t:

                        found = True

                        break

            if found:

                break



        if not vis[t]:

            return 0



        flow = float('inf')

        ver = t

        while ver != s:

            p = prev[ver]

            flow = min(flow, self.g[p][ver])

            ver = p



        ver = t

        while ver != s:

            p = prev[ver]

            self.g[p][ver] -= flow

            self.g[ver][p] += flow

            ver = p



        return flow



    def calc(self, s, t):

        res = 0

        while True:

            flow = self.bfs(s, t)

            res += flow

            if not flow:

                return res



main()


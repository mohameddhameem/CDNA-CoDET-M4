import sys, collections, math, bisect, heapq, random, functools, io, os, copy

from heapq import *

from io import BytesIO, IOBase



sys.setrecursionlimit(100000)

BUFSIZE = 4096





class FastIO(IOBase):

    newlines = 0



    def __init__(self, file):

        self._fd = file.fileno()

        self.buffer = BytesIO()

        self.writable = "x" in file.mode or "r" not in file.mode

        self.write = self.buffer.write if self.writable else None



    def read(self):

        while True:

            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))

            if not b:

                break

            ptr = self.buffer.tell()

            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)

        self.newlines = 0

        return self.buffer.read()



    def readline(self):

        while self.newlines == 0:

            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))

            self.newlines = b.count(b"\n") + (not b)

            ptr = self.buffer.tell()

            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)

        self.newlines -= 1

        return self.buffer.readline()



    def flush(self):

        if self.writable:

            os.write(self._fd, self.buffer.getvalue())

            self.buffer.truncate(0), self.buffer.seek(0)





class IOWrapper(IOBase):

    def __init__(self, file):

        self.buffer = FastIO(file)

        self.flush = self.buffer.flush

        self.writable = self.buffer.writable

        self.write = lambda s: self.buffer.write(s.encode("ascii"))

        self.read = lambda: self.buffer.read().decode("ascii")

        self.readline = lambda: self.buffer.readline().decode("ascii")





sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)

input = lambda: sys.stdin.readline().rstrip("\r\n")





def rep():

    a = list(map(int, input().split()))

    return a





def sep():

    a = list(input().split())

    return a





def qpow(x, y, mod):

    ans = 1

    while y:

        if y & 1:

            ans *= x

            ans %= mod

        x *= x

        x %= mod

        y >>= 1

    return ans





def Comb(n, m, p):

    a = (math.factorial(n)) % p

    b = (qpow(math.factorial(m), (p - 2), p)) % p

    c = (qpow(math.factorial(n - m), (p - 2), p)) % p

    return a * b * c % p





def Lucas(n, m, p):

    if m == 0:

        return 1

    return Comb(n % p, m % p, p) * Lucas(n // p, m // p, p) % p





class Trie:

    def __init__(self):

        self.trie = {}



    def insert(self, word):

        cur = self.trie

        for c in word:

            if c not in cur:

                cur[c] = {}

            cur = cur[c]

        if 'end' not in cur:

            cur['end'] = 0

        cur['end'] += 1



    def isprefix(self, word):

        cur = self.trie

        for c in word:

            if c not in word:

                return 0

            cur = cur[c]

        if 'end' not in cur:

            return 0

        return cur['end']





class UnionFind:

    def __init__(self, x) -> None:

        self.uf = [-1] * x



    def find(self, x):

        r = x

        while self.uf[x] >= 0:

            x = self.uf[x]



        while r != x:

            self.uf[r], r = x, self.uf[r]

        return x



    def union(self, x, y):

        ux, uy = self.find(x), self.find(y)

        if ux == uy:

            return

        if self.uf[ux] >= self.uf[uy]:

            self.uf[uy] += self.uf[ux]

            self.uf[ux] = uy

        else:

            self.uf[ux] += self.uf[uy]

            self.uf[uy] = ux

        return



    def count(self):

        ans = 0

        for c in self.uf:

            if c < 0:

                ans += 1

        return ans



    def valid(self):

        n = len(self.uf)

        for c in range(n):

            if self.uf[c] == -n:

                return True

        return False



    def __print__(self):

        return self.uf





def spfa(x, g, n):

    dis = [float('inf') for i in range(n)]

    dis[x] = 0

    state = [False for i in range(n)]

    state[x] = True

    queue = collections.deque()

    queue.append(x)

    while queue:

        cur = queue.popleft()

        state[cur] = False

        for next_ in g[cur]:

            if dis[next_] > dis[cur] + 1:

                dis[next_] = dis[cur] + 1

                if state[next_] == False:

                    state[next_] = True

                    if queue and dis[queue[0]] > dis[next_]:

                        queue.appendleft(next_)

                    else:

                        queue.append(next_)

    return dis





def gcd(x, y):

    if y == 0:

        return x

    return gcd(y, x % y)





def lcm(x, y):

    return x * y // gcd(x, y)





def Topsort(G, deg, n):

    queue = collections.deque()

    vis = set()

    ans = n

    for i in range(n):

        if deg[i] == 0:

            vis.add(i)

            queue.append(i)

    while queue:

        ans -= 1

        cur = queue.popleft()

        for next_ in G[cur]:

            deg[next_] -= 1

            if deg[next_] == 0 and next_ not in vis:

                vis.add(next_)

                queue.append(next_)



    return ans





def solve():

    n,c,d = map(int,input().split())

    a = rep()

    a.sort(reverse = True)

    pre = [0]

    for cc in a:

        pre.append(pre[-1] + cc)

    

    if pre[min(len(pre) - 1,d)] >= c:

        print('Infinity')

    else:

        if max(a) * d < c:

            print('Impossible')

        else:

            l,r = 0,d

            while l <= r:

                  mid = l + (r - l) // 2

                  tot = 0

                  if mid <= n - 1:

                      cnt = d // (mid + 1)

                      left = d % (mid + 1)

                      tot += cnt * pre[mid + 1] + pre[left]

                  else:

                      cnt = d // (mid + 1)

                      

                      left = d % (mid + 1)

                

                      tot += pre[-1] * cnt + pre[min(left,len(pre) - 1)]

                  if tot >= c:

                      l = mid + 1

                  else:

                      r = mid - 1

                

                  # 2 1 0 0 2 1

            print(l - 1)

                     

if __name__ == '__main__':

   for i in range(int(input())):

       solve()
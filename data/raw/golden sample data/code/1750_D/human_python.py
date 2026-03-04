import sys, array, bisect, collections, heapq, itertools, math, functools

sys.setrecursionlimit(100000)



def _r(): return sys.stdin.buffer.readline()

def rs(): return _r().decode('ascii').strip()

def rn(): return int(_r())

def rnt(): return tuple(map(int, _r().split()))

def rnl(): return list(rnt())

def rna(): return array.array('q', rnt())

def pnl(l): print(' '.join(map(str, l)))



def prime_factors(n):

    i = 2

    factors = {}

    while i * i <= n:

        if n % i:

            i += 1

        else:

            n //= i

            factors[i] = factors.get(i, 0) + 1

    if n > 1:

        factors[n] = factors.get(n, 0) + 1

    return factors

 

mod = 998244353



def _calc(m, x, pf):

    sol = base = m // x

    sign = -1

    for take in range(len(pf)):

        for c in itertools.combinations(pf, take+1):

            mult = 1

            for ce in c:

                mult *= ce

            sol = (sol + sign * (base // mult)) % mod

        sign = -sign

    return sol

 

def solve(n, m, a):

    sol = 1

    for i in range(1, n):

        if a[i-1] % a[i] != 0:

            return 0

        diff = prime_factors(a[i-1] // a[i])

        sol = (sol * _calc(m, a[i], diff.keys())) % mod

    return sol

 

for _ in range(rn()):

    n, m = rnt()

    a = rnl()

    print(solve(n, m, a))
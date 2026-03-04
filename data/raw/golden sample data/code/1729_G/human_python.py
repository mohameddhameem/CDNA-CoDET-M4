"""

groupby components

- defined reachable stamps



dp on each group



count out by multiplication of each group











dp is looking at each subcomponent of the component

-> check if we can place it there



-> number of moves to iterate over all components

"""



from collections import *

from heapq import *

from itertools import *

from functools import *

from math import *

from string import *

import operator

import sys

import bisect



input = sys.stdin.readline





def make_components(locs, s, t):

    ans = []

    i = 0



    while i < len(locs):

        anchor = i



        while i + 1 < len(locs) and locs[i][1] + len(t) - 1 >= locs[i + 1][0]:

            i += 1



        start, end = locs[anchor][0], locs[i][1]

        ans.append(s[start : end + 1])

        i += 1



    return ans





def solve():

    MOD = 10 ** 9 + 7

    s, t = [input().strip() for _ in range(2)]



    locs = filter(lambda i: s.startswith(t, i), range(len(s)))

    locs = [[i, i + len(t) - 1] for i in locs]

    components = make_components(locs, s, t)



    if len(components) == 0:

        print(0, 1)

        return



    def query(s):

        locs = [i for i in range(len(s)) if s.startswith(t, i)]

        starts = set(locs)

        n = len(locs)



        def neighbors(i):

            for j in range(locs[i], locs[i] + len(t)):

                if j in starts:

                    yield bisect.bisect_left(locs, j + len(t))



        dp = [inf] * (n + 1)

        ct = [0] * (n + 1)



        dp[n], ct[n] = 0, 1



        for i in range(n - 1, -1, -1):

            for nei in neighbors(i):

                if 1 + dp[nei] < dp[i]:

                    dp[i] = 1 + dp[nei]

                    ct[i] = ct[nei]

                elif 1 + dp[nei] <= dp[i]:

                    ct[i] += ct[nei]



                dp[i] %= MOD

                ct[i] %= MOD



        return dp[0], ct[0]



    ways = []

    total = []

    for shortest, ct in map(query, components):

        ways.append(shortest)

        total.append(ct)



    print(sum(ways) % MOD, reduce(operator.__mul__, total) % MOD)





def main():

    tests = int(input())

    for _ in range(tests):

        solve()





if __name__ == "__main__":

    main()


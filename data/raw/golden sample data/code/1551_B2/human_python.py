import sys

input = sys.stdin.readline

from collections import Counter



for _ in range(int(input())):

    n, k = map(int, input().split())

    w = input()[:-1].split()

    d = Counter()

    s = []

    for i, j in enumerate(w):

        if d[j] < k:

            s.append((j, i))

        d[j] += 1

    s.sort()

    x = len(s)

    x -= x%k

    color = [0]*n

    c = 1

    for i, j in s[:x]:

        color[j] = c

        c = c+1

        if c > k:

            c -= k

    print(' '.join(map(str, color)))
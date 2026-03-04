from collections import Counter, deque, defaultdict

import math, sys

import sys, io, os

from itertools import permutations, accumulate

from heapq import *

from bisect import bisect_left, bisect_right

from functools import cmp_to_key

from random import shuffle, randint

xor = randint(10 ** 7, 10**8)

# https://github.com/cheran-senthil/PyRival/tree/master/pyrival

# https://docs.python.org/3/library/bisect.html

lcm = lambda x, y: (x * y) // math.gcd(x,y)

rotate = lambda seq, k: seq[k:] + seq[:k]

input = sys.stdin.readline

'''

Check for typos before submit, Check if u can get hacked with Dict (use xor)

Observations/Notes: 



'''

for _ in range(int(input())):

    x1, y1 = map(int, input().split())

    if (x1 + y1) & 1:

        print(-1, -1)

    elif x1 & 1 and y1 & 1:

        print(x1 // 2, (y1 // 2) + 1)

    else:

        print(x1 // 2, y1 // 2)








import bisect

import io

import math

import os

import sys



LO = 'abcdefghijklmnopqrstuvwxyz'

Mod = 1000000007



def gcd(x, y):

    while y:

        x, y = y, x % y

    return x



# _input = lambda: io.BytesIO(os.read(0, os.fstat(0).st_size)).readline().decode()

_input = lambda: sys.stdin.buffer.readline().strip().decode()



for _ in range(int(_input())):

    n = int(_input())

    s = _input()

    if s[0] == '0' or s[-1] == '0' or s.count('0') % 2:

        print('NO')

    else:

        print('YES')

        a = ['('] * n

        b = ['('] * n

        c = [i for i in range(n) if s[i] == '1']

        i = 0

        while len(c) > i:

            i += 1

            x = c.pop()

            a[x] = b[x] = ')'

        c = [i for i in range(n) if s[i] == '0']

        for i, x in enumerate(c):

            a[x] = '(' if i % 2 else ')'

            b[x] = ')' if i % 2 else '('

        print(''.join(a))

        print(''.join(b))


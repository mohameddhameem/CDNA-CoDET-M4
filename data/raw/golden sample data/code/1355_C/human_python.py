import decimal
import gc
import heapq
import math
import os
import random
import sys
from collections import Counter, deque, defaultdict
from io import BytesIO, IOBase
import bisect
from types import GeneratorType

BUFSIZE = 8192


class FastIO(IOBase):
    newlines = 0

    def __init__(self, file):
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = 'x' in file.mode or 'r' not in file.mode
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
            self.newlines = b.count(b'\n') + (not b)
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
        self.write = lambda s: self.buffer.write(s.encode('ascii'))
        self.read = lambda: self.buffer.read().decode('ascii')
        self.readline = lambda: self.buffer.readline().decode('ascii')


sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)
input = lambda: sys.stdin.readline().rstrip('\r\n')


def isPrime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i = i + 6
    return True


def lcm(a, b): return (a * b) // math.gcd(a, b)


def ciel(a, b):
    x = a // b
    if a % b == 0:
        return x
    return x + 1


def ints_get(): return map(int, input().strip().split())


def list_get(): return list(map(int, sys.stdin.readline().strip().split()))


def chars_get(): return list(map(str, sys.stdin.readline().strip().split()))


def ipn(): return int(input())


def bootstrap(f, stack=[]):
    def wrappedfunc(*args, **kwargs):
        if stack:
            return f(*args, **kwargs)
        else:
            to = f(*args, **kwargs)
            while True:
                if type(to) is GeneratorType:
                    stack.append(to)
                    to = next(to)
                else:
                    stack.pop()
                    if not stack:
                        break
                    to = stack[-1].send(to)
            return to

    return wrappedfunc


# ******************************************************#
# **************** code starts here ********************#
# ******************************************************#


def main():
    a, b, c, d = ints_get()
    p = []
    q = []
    for i in range(a, b + 1):
        p.append(i + b)
        q.append(i + c)
    p.sort(reverse=True)
    q.sort(reverse=True)
    v = [0 for _ in range(max(q) + 2)]
    val = 0
    for i in range(len(v)):
        while len(p) > 0:
            if p[-1] <= i:
                p.pop()
                val += 1
            else:
                break
        v[i] = val
        while len(q) > 0:
            if q[-1] <= i:
                q.pop()
                val -= 1
            else:
                break
    # print(v)

    out = 0
    for i in range(c + 1, len(v)):
        # print(i, v[i])
        out += v[i] * min(d - c + 1, i - c)
        # print(out)
    print(out)

    return


if __name__ == "__main__":
    main()



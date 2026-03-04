# Init ---------------------------------------------------------
import os
import sys
from io import BytesIO, IOBase
from random import randint
from collections import Counter

RANDOM = randint(1, 10 ** 9)
BUFSIZE = 8192


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


class IntWrapper(int):
    def __init__(self, x):
        int.__init__(x)

    def __hash__(self):
        return super(IntWrapper, self).__hash__() ^ RANDOM


sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)
input = lambda: sys.stdin.readline().rstrip("\r\n")


def LI():
    return list(map(int, input().split()))
def I():
    return int(input())
def LIW():
    return list(map(IntWrapper, input().split()))
def IW():
    return IntWrapper(input())
# End---------------------------------------------------------

for _ in range(I()):
    n, k = LI()
    counts = Counter(LIW())
    values = list(sorted(counts.keys()))

    best, best_l, best_r = -1, 0, 0
    x = 0
    while x < len(values):
        l, r = values[x], values[x]
        while counts[r] >= k:
            r = IntWrapper(r + 1)
        if r > l:
            r -= 1
            if r - l > best:
                best = r - l
                best_l, best_r = l, r
            x += r - l + 1
        else:
            x += 1

    if best != -1:
        print(best_l, best_r)
    else:
        print(-1)

from collections import defaultdict
from queue import deque
from sys import stdin, stdout
from math import log2


def arrinp():
    return [*map(int, stdin.readline().split(' '))]


def mulinp():
    return map(int, stdin.readline().split(' '))


def intinp():
    return int(stdin.readline())


def solution():
    n = intinp()
    if n <= 3:
        print(n)
        return
    if n % 3 == 1:
        k = n.bit_length()-1
        if k & 1:
            k -= 1
        print(2**k + (n-2**k) // 3)
    elif n % 3 == 2:
        n1 = n-1
        k = n1.bit_length()-1
        if k & 1:
            k -= 1
        ans1 = 2**k + (n1-2**k) // 3
        ans = 0
        cnt = 0
        while ans1:
            a = ans1 % 4
            if a == 1:
                a = 2
            elif a == 2:
                a = 3
            elif a == 3:
                a = 1
            ans += a * (4**cnt)
            cnt += 1
            ans1 >>= 2
        print(ans)
    else:
        n1 = n - 2
        k = n1.bit_length()-1
        if k & 1:
            k -= 1
        ans1 = 2 ** k + (n1 - 2 ** k) // 3
        ans = 0
        cnt = 0
        while ans1:
            a = ans1 % 4
            if a == 1:
                a = 3
            elif a == 2:
                a = 1
            elif a == 3:
                a = 2
            ans += a * (4 ** cnt)
            cnt += 1
            ans1 >>= 2
        print(ans)


testcases = 1
testcases = intinp()
for _ in range(testcases):
    solution()

import math

def gcdSum(x):
    return math.gcd(x, sum(int(digit) for digit in str(x)))

def solve():
    t = int(input())
    for _ in range(t):
        n = int(input())
        while True:
            if gcdSum(n) > 1:
                print(n)
                break
            n += 1

solve()
import sys
import math

def gcd_sum(x):
    sum_digits = sum(int(d) for d in str(x))
    return math.gcd(x, sum_digits)

def find_gcd_sum(n):
    while True:
        if gcd_sum(n) > 1:
            return n
        n += 1

input = sys.stdin.read
data = input().split()
t = int(data[0])
results = []

for i in range(1, t + 1):
    n = int(data[i])
    results.append(find_gcd_sum(n))

print("\n".join(map(str, results)))
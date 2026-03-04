import math

def round_down_price(m):
    k = math.floor(math.log10(m))
    return m - 10**k

t = int(input())
for _ in range(t):
    m = int(input())
    print(round_down_price(m))
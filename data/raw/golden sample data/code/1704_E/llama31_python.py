import math
from bisect import bisect_left

def min_seconds(a):
    n = len(a)
    max_val = max(a)
    ans = 0
    for i in range(n):
        max_val = max(max_val, a[i])
        pow2 = 1 << (math.log2(max_val - a[i]) + 1)
        ans = max(ans, pow2)
    return ans

t = int(input())
for _ in range(t):
    n = int(input())
    a = list(map(int, input().split()))
    print(min_seconds(a))
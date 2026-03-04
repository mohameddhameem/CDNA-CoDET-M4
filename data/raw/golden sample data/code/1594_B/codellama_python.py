import math

def is_special(n):
    return (n - math.floor(math.log(n))) % 2 == 1

def find_special_number(n, k):
    for i in range(1, n + 1):
        if is_special(i):
            k -= 1
            if k == 0:
                return i
    return 0

t = int(input())
for _ in range(t):
    n, k = map(int, input().split())
    print(find_special_number(n, k) % (10**9 + 7))
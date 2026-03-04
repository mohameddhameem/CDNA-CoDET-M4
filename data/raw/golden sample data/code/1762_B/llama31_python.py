import math
import sys

def make_array_good(n, a):
    ans = []
    for i in range(n):
        max_val = max(a)
        min_val = min(a)
        g = math.gcd(max_val, min_val)
        x = max_val - (max_val // g * min_val)
        a[i] += x
        ans.append((i+1, x))
    print(len(ans))
    for item in ans:
        print(*item)

t = int(input())
for _ in range(t):
    n = int(input())
    a = list(map(int, input().split()))
    make_array_good(n, a)

This code reads the number of test cases, then for each test case, it reads the length of the array and the array itself, then it calls the `make_array_good` function to perform the operations and print the result. The `make_array_good` function iterates over the array, finds the greatest common divisor of the maximum and minimum values, and adds the difference to the maximum value to make it divisible by the minimum value. The result is then printed.
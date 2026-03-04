#----------------------------IMPORTING LIBRARIES----------------------------#

import math

import itertools

import heapq

from math import floor, ceil

from collections import deque

import sys

from collections import Counter

#----------------------------LAMBDAS AND DEFAULT----------------------------#

I = lambda:list(map(int,input().split()))

M = lambda:map(int, input().split())

input = sys.stdin.readline

sys.setrecursionlimit(10000) 

MOD = 10**9 + 7

#-----------------------------STANDARD FUNCTIONS----------------------------#

def lcm(a, b):

    return abs(a*b) // math.gcd(a, b)



def gcd(a, b):

    if b == 0:

        return a

    return gcd(b, a%b)



def isSubsequence(s, t):

    for i in range (0, len(s)):    

        try:

            index = t.index(s[i])

        except ValueError: 

            return False

        t = t[index+1:]

    return True



def bisect_right(a, x):

        lo, hi = 0, len(a)

        while lo < hi:

            mid = lo + (hi - lo) // 2

            if a[mid] > x: hi = mid

            else: lo = mid + 1

        return lo

    

def bisect_left(a, x):

    lo, hi = 0, len(a)

    while lo < hi:

        mid = lo + (hi - lo) // 2

        if a[mid] < x: lo = mid + 1

        else: hi = mid

    return lo



def binary_search(nums, target):

    left, right = 0, len(nums) - 1

    while left <= right:

        pivot = left + (right - left) // 2

        if nums[pivot] == target:

            return pivot

        if target < nums[pivot]:

            right = pivot - 1

        else:

            left = pivot + 1

    return -1



#------------------------------HELPER FUNCTION------------------------------#

def prefixSum(arr):

    dp = [arr[0]]

    arrLen = len(arr)

    for i in range(1, arrLen): dp.append(dp[-1] + arr[i])

    return dp



def suffixSum(arr):

    arrLen = len(arr)

    dp = [0] * arrLen

    dp[-1] = arr[-1]

    for i in range(N-2, -1, -1): dp[i] = dp[i+1] + arr[i]

    return dp



def fun():

    return 



# sys.stdout.write(str(X) + "\n") 

def solve():

    R, C = M()

    maxDp = []

    minDp = []

    for i in range(R):

        t1, t2 = [], []

        for x in range(C): 

            t1.append(0)

            t2.append(0)

        maxDp.append(t1)

        minDp.append(t2)

    grid = []

    for i in range(R):

        tmp = I()

        grid.append(tmp)

    maxDp[0][0] = grid[0][0]

    minDp[0][0] = grid[0][0]

    for i in range(1, R):

        maxDp[i][0] = maxDp[i-1][0] + grid[i][0]

        minDp[i][0] = minDp[i-1][0] + grid[i][0]

    for i in range(1, C):

        maxDp[0][i] = maxDp[0][i-1] + grid[0][i]

        minDp[0][i] = minDp[0][i-1] + grid[0][i]

    for i in range(1, R):

        for j in range(1, C):

            minDp[i][j] = min(minDp[i-1][j], minDp[i][j-1]) + grid[i][j]

            maxDp[i][j] = max(maxDp[i-1][j], maxDp[i][j-1]) + grid[i][j]

    if (R + C) % 2 == 0 or (minDp[-1][-1] > 0 or 0 > maxDp[-1][-1]):

        print("NO")

    else:

        print("YES")

#---------------------------------DRIVER CODE--------------------------------#

TC = 1

TC = int(input())

for testcases in range(TC):

    solve()
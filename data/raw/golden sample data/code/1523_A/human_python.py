import math
import sys

input = sys.stdin.readline

t = int(input())
inf = 3*10**9

for cs in range(t):
    n, m = map(int, input().split())
    s = input()
    dist = [inf] * n
    last = -inf
    for i in range(n):
        if s[i] == '1':
            last = i
        dist[i] = min(dist[i], i - last)

    last = inf
    dist1 = [inf] * n 
    for i in range(n - 1, -1, -1):
        if s[i] == '1':
            last = i
        dist1[i] = min(dist1[i], last - i)

    answer = ''
    for i in range(n):
        if dist[i] == dist1[i] and s[i] == '0':
            answer += '0'
        elif dist[i] <= m or dist1[i] <= m:
            answer += '1'
        else:
            answer += '0'
    print(answer)


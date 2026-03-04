import sys

input = sys.stdin.readline

s = input().strip()

l = len(s)

d, p = [], [0]

d.append([1 if s[0] == chr(x+97)  else 0 for x in range(26)])

for idx, i in enumerate(s):

    if idx == 0: continue

    x = ord(i) - 97

    d.append(d[p[idx-1]][:])

    d[idx][x] = idx+1

    p.append(d[p[idx-1]][x])

for i in range(10): p.append(0); d.append([])

for i in range(int(input())):

    for idx, t in enumerate(input().strip()):

        x, pos = ord(t) - 97, idx+l

        d[pos] = d[p[pos-1]][:]

        d[pos][x] = pos+1

        p[pos] = d[p[pos-1]][x]

    print(*p[l:l+idx+1])
from collections import defaultdict

def longest_strike(a, k):
    if len(set(a)) < k:
        return -1

    cnt = defaultdict(int)
    for x in a:
        cnt[x] += 1

    max_len = 0
    l, r = -1, -1
    for i in sorted(cnt.keys()):
        if cnt[i] >= k:
            if max_len < r - i + 1:
                max_len = r - i + 1
                l, r = i, i
            else:
                r = i

    if max_len == 0:
        return -1

    return f"{l} {r}"

t = int(input())
for _ in range(t):
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    print(longest_strike(a, k))
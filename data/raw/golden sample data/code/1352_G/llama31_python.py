def special_permutation(n, a, b):
    if a > b:
        return -1
    if a == 1 and b == n:
        return -1
    if a == b:
        return -1
    if a == 1:
        return''.join(map(str, list(range(1, n+1))[::-1]))
    if b == n:
        return''.join(map(str, list(range(1, n+1))))
    left = [i for i in range(1, n+1) if i not in (a, b)]
    left.sort()
    right = [i for i in range(1, n+1) if i not in (a, b)]
    right.sort(reverse=True)
    res = [a] + left[:n//2-1] + [b] + right[:n//2-1]
    if a < b:
        res += [a, b]
    else:
        res += [b, a]
    return''.join(map(str, res))

t = int(input())
for _ in range(t):
    n, a, b = map(int, input().split())
    print(special_permutation(n, a, b))
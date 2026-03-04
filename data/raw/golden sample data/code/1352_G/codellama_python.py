import itertools

def special_permutation(n):
    for p in itertools.permutations(range(1, n+1)):
        if len(set(p)) == n:
            return p
    return None

t = int(input())
for _ in range(t):
    n = int(input())
    p = special_permutation(n)
    if p is not None:
        print(*p)
    else:
        print('NO SOLUTION')
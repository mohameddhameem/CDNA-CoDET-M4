import sys

n = int(input())

def ask(S):
    print('?', len(S), *S)
    sys.stdout.flush()
    return input() == 'YES'

def guess(x):
    print('!', x)
    sys.stdout.flush()
    return input() == ':)'

l, r = 1, n
for _ in range(53):
    if r - l <= 1:
        for i in range(l, r+1):
            if guess(i):
                exit(0)
        exit(1)
    m = (l + r) // 2
    S = [i for i in range(l, m+1)]
    if ask(S):
        r = m
        S = [i for i in range(m+1, r+1)]
    else:
        l = m
        S = [i for i in range(l, r+1)]
    if r - l <= 1:
        for i in range(l, r+1):
            if guess(i):
                exit(0)
        exit(1)

for _ in range(2):
    m = (l + r) // 2
    S = [i for i in range(l, m+1)]
    if ask(S):
        l = m
    else:
        r = m
    for i in range(l, r+1):
        if guess(i):
            exit(0)
exit(1)
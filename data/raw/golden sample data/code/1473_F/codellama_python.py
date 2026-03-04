import sys

n = int(sys.stdin.readline())
a = list(map(int, sys.stdin.readline().split()))
b = list(map(int, sys.stdin.readline().split()))

def get_strange_sets(n, a, b):
    sets = []
    for i in range(n):
        set_ = set()
        for j in range(i, n):
            if a[i] % a[j] == 0 and b[j] >= b[i]:
                set_.add(j)
        sets.append(set_)
    return sets

strange_sets = get_strange_sets(n, a, b)
max_cost = 0
for set_ in strange_sets:
    cost = sum(b[i] for i in set_)
    if cost > max_cost:
        max_cost = cost

print(max_cost)
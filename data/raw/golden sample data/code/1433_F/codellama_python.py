import itertools

def zero_remainder_sum(a, k):
    rows = []
    for r in a:
        rows.append(list(filter(lambda x: x % k == 0, r)))
    return max([sum(x) for x in itertools.product(*rows)])
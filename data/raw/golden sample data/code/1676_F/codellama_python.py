import sys

def longest_strike(a, k):
    # Create a dictionary to store the counts of each number in the array
    counts = {}
    for num in a:
        if num in counts:
            counts[num] += 1
        else:
            counts[num] = 1

    # Find the maximum value of r-l
    max_val = 0
    for num in counts:
        if counts[num] >= k:
            max_val = max(max_val, counts[num])

    # Find the pair of l and r that maximize the value of r-l
    l, r = 0, 0
    for num in counts:
        if counts[num] >= k:
            l = num
            break

    for num in reversed(counts):
        if counts[num] >= k:
            r = num
            break

    return l, r

# Test cases
def test_longest_strike():
    a = [11, 11, 12, 13, 13, 14, 14]
    assert longest_strike(a, 2) == (13, 14)

    a = [11, 12, 13, 13, 14, 14]
    assert longest_strike(a, 2) == (11, 11)

    a = [11, 12, 13, 13, 14, 14, 15, 15, 16, 16]
    assert longest_strike(a, 3) == (13, 15)

    a = [11, 11, 12, 13, 13, 14, 14, 15, 15, 16, 16]
    assert longest_strike(a, 3) == (11, 12)

    a = [11, 12, 13, 13, 14, 14, 15, 15, 16, 16]
    assert longest_strike(a, 2) == (11, 11)

    a = [11, 11, 12, 13, 13, 14, 14, 15, 15, 16, 16]
    assert longest_strike(a, 2) == (11, 12)

    a = [11, 12, 13, 13, 14, 14, 15, 16, 16]
    assert longest_strike(a, 2) == (12, 14)

    a = [11, 12, 13, 13, 14, 14, 15, 16, 16]
    assert longest_strike(a, 3) == (13, 15)

    a = [11, 12, 13, 13, 14, 14, 15, 16, 16]
    assert longest_strike(a, 4) == (14, 16)

    a = [11, 12, 13, 13, 14, 14, 15, 16, 16]
    assert longest_strike(a, 5) == (15, 16)

    a = [11, 12, 13, 13, 14, 14, 15, 16, 16]
    assert longest_strike(a, 6) == (16, 16)

test_longest_strike()

# Read input
_ = input()
n, k = map(int, input().split())
a = list(map(int, input().split()))

# Find the longest strike
l, r = longest_strike(a, k)

# Output
if l == r:
    print(-1)
else:
    print(l, r)
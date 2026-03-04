def good_subarrays(a):
    n = len(a)
    count = 0
    for i in range(n):
        for j in range(i, n):
            if sum(a[i:j+1]) == j - i + 1:
                count += 1
    return count
def count_good_subarrays(n, a):
    prefix_sum = 0
    last_seen = {0: -1}
    good_count = 0

    for i in range(n):
        prefix_sum += a[i]
        if prefix_sum in last_seen:
            good_count += (i - last_seen[prefix_sum])
        else:
            good_count += (i + 1)
        last_seen[prefix_sum] = i

    return good_count

n = int(input())
a = list(map(int, input().split()))
print(count_good_subarrays(n, a))
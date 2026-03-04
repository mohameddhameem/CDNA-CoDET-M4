def same_sum_blocks(n, a):
    from collections import defaultdict
    
    prefix_sum = [0] * (n + 1)
    for i in range(n):
        prefix_sum[i + 1] = prefix_sum[i] + a[i]

    blocks = defaultdict(list)
    
    for l in range(n):
        for r in range(l, n):
            block_sum = prefix_sum[r + 1] - prefix_sum[l]
            blocks[block_sum].append((l + 1, r + 1))  # store as 1-indexed

    result = []
    
    for block_sum, pairs in blocks.items():
        if len(pairs) > len(result):
            result = pairs

    # Now we need to select disjoint intervals
    result.sort(key=lambda x: x[1])  # sort by end index
    selected = []
    last_end = 0

    for l, r in result:
        if l > last_end:
            selected.append((l, r))
            last_end = r

    print(len(selected))
    for l, r in selected:
        print(l, r)

# Input reading
n = int(input().strip())
a = list(map(int, input().strip().split()))

same_sum_blocks(n, a)
def same_sum_blocks(arr):
    n = len(arr)
    max_k = n

    for i in range(n):
        for j in range(i+1, n):
            if sum(arr[i:j+1]) == sum(arr[j:i+1]):
                max_k = min(max_k, j-i+1)
                break

    blocks = []
    for k in range(max_k):
        for i in range(n-k):
            if sum(arr[i:i+k+1]) == sum(arr[i+k+1:i+1]):
                blocks.append((i, i+k))
                break

    return blocks
def reorder(arr):
    n = len(arr)
    for i in range(n-1):
        arr[i] = (arr[i] | arr[i+1]) - arr[i+1]
    return arr
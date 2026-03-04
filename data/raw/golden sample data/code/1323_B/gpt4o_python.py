def count_subrectangles(n, m, k, a, b):
    from collections import defaultdict

    # Step 1: Calculate lengths of consecutive 1s in a and b
    def consecutive_counts(arr):
        counts = []
        current_count = 0
        for value in arr:
            if value == 1:
                current_count += 1
            else:
                if current_count > 0:
                    counts.append(current_count)
                current_count = 0
        if current_count > 0:
            counts.append(current_count)
        return counts

    row_counts = consecutive_counts(a)
    col_counts = consecutive_counts(b)

    # Step 2: Generate the number of ways to form rectangles of area k
    area_count = 0
    for row_length in row_counts:
        if row_length > 0:
            for height in range(1, row_length + 1):
                if height * (k // height) == k and k // height <= m:
                    area_count += (row_length - height + 1) * count_ways(col_counts, k // height)

    return area_count

def count_ways(counts, width):
    total = 0
    for count in counts:
        if count >= width:
            total += (count - width + 1)
    return total

# Input reading
n, m, k = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

# Output result
print(count_subrectangles(n, m, k, a, b))
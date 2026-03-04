def wonderful_coloring(a, k):
    if len(a) == 0:
        return []
    elif len(a) == 1:
        return [0]
    else:
        # Find the maximum value in the sublist a[1:]
        max_val = max(a[1:])
        # Recursively find a coloring for the sublist a[1:]
        coloring = wonderful_coloring(a[1:], k)
        # If the sublist a[1:] has a coloring, return the first color used
        if len(coloring)!= 0:
            return [coloring[0]]
        # If the sublist a[1:] does not have a coloring, find the next color to use
        next_color = coloring[0] + 1
        if next_color == k + 1:
            next_color = 1
        # Recursively find a coloring for the sublist a[1:] with the next color
        new_coloring = wonderful_coloring(a[1:], next_color)
        # If the sublist a[1:] has a coloring, return the next color used
        if len(new_coloring)!= 0:
            return [next_color]
        # If the sublist a[1:] does not have a coloring, return an empty list
        else:
            return []

def main():
    t = int(input())
    for _ in range(t):
        n, k = map(int, input().split())
        a = list(map(int, input().split()))
        coloring = wonderful_coloring(a, k)
        print(*coloring)

if __name__ == '__main__':
    main()
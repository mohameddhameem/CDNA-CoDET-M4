def wonderful_coloring(t, test_cases):
    results = []
    for n, k, a in test_cases:
        color_count = {}
        for num in a:
            if num in color_count:
                color_count[num] += 1
            else:
                color_count[num] = 1
        
        color_map = {}
        current_color = 1
        for i in range(n):
            if a[i] not in color_map and current_color <= k:
                color_map[a[i]] = current_color
                current_color += 1
            results.append(color_map.get(a[i], 0))
        
        # Fill unused colors until we reach k for each unique number
        painted_counts = {}
        for color in results:
            if color > 0:
                if color in painted_counts:
                    painted_counts[color] += 1
                else:
                    painted_counts[color] = 1

        max_paintable = min(painted_counts.values()) if painted_counts else 0
        
        for i in range(n):
            if results[i] > 0 and painted_counts[results[i]] > max_paintable:
                results[i] = 0
                painted_counts[results[i]] -= 1

        results = results[-n:]  # Ensure results length matches n
        print(' '.join(map(str, results)))

# Example usage
t = 6
test_cases = [
    (10, 3, [3, 1, 1, 1, 1, 10, 3, 10, 10, 2]),
    (4, 4, [1, 1, 1, 1]),
    (1, 1, [1]),
    (13, 1, [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9]),
    (13, 2, [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9]),
    (13, 3, [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9]),
]

wonderful_coloring(t, test_cases)
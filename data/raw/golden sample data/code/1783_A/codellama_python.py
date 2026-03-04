n = 1000000000

def is_beautiful(polygon):
    return len(polygon) == n and all(edge.parallel_to_axis for edge in polygon.edges)

for t in range(100000):
    n_i = int(input())
    polygon = Polygon(n_i)
    if is_beautiful(polygon):
        print('YES')
    else:
        print('NO')
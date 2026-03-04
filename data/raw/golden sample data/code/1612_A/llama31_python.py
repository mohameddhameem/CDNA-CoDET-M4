import math

def get_area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)

def get_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def get_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return None

def get_length(x1, y1, x2, y2, x3, y3):
    intersection = get_intersection(x1, y1, x2, y2, x3, y3, 0, 0)
    if intersection is not None:
        return get_distance(x1, y1, intersection[0], intersection[1]) + get_distance(intersection[0], intersection[1], x2, y2)
    return get_distance(x1, y1, x2, y2)

def solve():
    t = int(input())
    for _ in range(t):
        x1, y1, x2, y2, x3, y3 = map(int, input().split())
        area = get_area(x1, y1, x2, y2, x3, y3)
        if area == 0:
            print(0)
            continue
        if min(y1, y2, y3) == 0:
            print(0)
            continue
        if max(y1, y2, y3) == 0:
            print(get_length(x1, y1, x2, y2, x3, y3))
            continue
        safe = False
        for i in range(3):
            for j in range(i + 1, 3):
                x4, y4 = x3, y3
                x3, y3 = x1, y1
                x1, y1 = x2, y2
                x2, y2 = x4, y4
                x4, y4 = map(int, input().split())
                if get_area(x1, y1, x2, y2, x3, y3) == area:
                    safe = True
                    break
            if safe:
                break
        if safe:
            print(0)
        else:
            print(get_length(x1, y1, x2, y2, x3, y3))

solve()
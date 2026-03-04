def can_walk(x, y, x1, y1, x2, y2, a, b, c, d):
    # Check if the walk is valid
    if x < x1 or x > x2 or y < y1 or y > y2:
        return False
    if a + b + c + d < 0:
        return False
    if a + b + c + d > 0 and x == x1 and y == y1:
        return False
    if a + b + c + d > 0 and x == x2 and y == y2:
        return False
    return True
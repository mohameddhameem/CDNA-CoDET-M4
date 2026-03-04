def is_beautiful(matrix):
    a, b = matrix[0]
    c, d = matrix[1]
    
    return (a < b and c < d and a < c and b < d) or (a > b and c > d and a > c and b > d)

def rotate(matrix):
    a, b = matrix[0]
    c, d = matrix[1]
    return [[d, c], [b, a]]

t = int(input())
for _ in range(t):
    matrix = [list(map(int, input().split())) for _ in range(2)]
    if is_beautiful(matrix):
        print("YES")
    else:
        for _ in range(4):
            matrix = rotate(matrix)
            if is_beautiful(matrix):
                print("YES")
                break
        else:
            print("NO")
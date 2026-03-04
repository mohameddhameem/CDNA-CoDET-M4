import sys

def get_distance(x, y):
    return (x - p) + (y - q)

def place_device(i, j):
    global x, y
    x = i
    y = j
    print(get_distance(i, j), end="")
    sys.stdout.flush()

def find_enemy_base():
    global x, y, p, q
    while True:
        if p!= 1 and p!= 10**9:
            x, y = get_distance(p, q), p - x
            p, q = x, y
        if p == 1 or p == 10**9:
            print("!", x, y, p, q, end="")
            sys.stdout.flush()
            break

p, q = int(input()), int(input())
x, y = 1, 1
find_enemy_base()
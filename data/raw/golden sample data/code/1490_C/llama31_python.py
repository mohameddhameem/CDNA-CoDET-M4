def sum_of_cubes(x):
    for i in range(1, int(x ** (1. / 3)) + 1):
        for j in range(i, int((x - i ** 3) ** (1. / 3)) + 1):
            if i ** 3 + j ** 3 == x:
                return "YES"
    return "NO"

t = int(input())
for _ in range(t):
    x = int(input())
    print(sum_of_cubes(x))
import math

def ternary_xor(a, b):
    return tuple(i % 3 for i in (a + b) % 3)

def solve_ternary_xor(x, n):
    a = '0' * (n // 2) + '1' * (n // 2)
    b = '0' * (n // 2) + '2' * (n // 2)
    return ternary_xor(a, b)

def main():
    t = int(input())
    for _ in range(t):
        n = int(input())
        x = input()
        a, b = solve_ternary_xor(x, n)
        print(a + b)

if __name__ == '__main__':
    main()
import sys

def cut_triangle(x1, y1, x2, y2, x3, y3):
    x = sorted([x1, x2, x3])
    y = sorted([y1, y2, y3])
    return x[0]!= x[2] or y[0]!= y[2]

def main():
    input = sys.stdin.readline
    for _ in range(int(input())):
        input()
        x1, y1 = map(int, input().split())
        x2, y2 = map(int, input().split())
        x3, y3 = map(int, input().split())
        if cut_triangle(x1, y1, x2, y2, x3, y3):
            print("YES")
        else:
            print("NO")

if __name__ == "__main__":
    main()
import sys

def flip_flop_sum(a):
    n = len(a)
    max_sum = 0
    for i in range(n-1):
        if a[i] == 1 and a[i+1] == -1:
            a[i] = -1
            a[i+1] = 1
            max_sum += 2
    return max_sum

if __name__ == '__main__':
    t = int(input())
    for _ in range(t):
        n = int(input())
        a = list(map(int, input().split()))
        print(flip_flop_sum(a))
import sys

read = lambda: sys.stdin.readline()

def alternating_subsequence(arr):
    max_sum = 0
    current_sum = 0
    sign = 1
    for element in arr:
        if element * sign > 0:
            current_sum += element * sign
        else:
            sign = -1
            current_sum = 0
        if current_sum > max_sum:
            max_sum = current_sum
    return max_sum

if __name__ == '__main__':
    t = int(read())
    for _ in range(t):
        n = int(read())
        a = list(map(int, read().split()))
        print(alternating_subsequence(a))
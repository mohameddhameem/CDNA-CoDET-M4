#Write your code here

def is_luxurious(x):
    return x % (math.floor(math.sqrt(x))) == 0

def count_changes(l, r):
    count = 0
    for i in range(l, r+1):
        if is_luxurious(i):
            count += 1

    return count

### Code
if __name__ == '__main__':
    t = int(input())
    for i in range(t):
        l, r = map(int, input().split())
        print(count_changes(l, r))
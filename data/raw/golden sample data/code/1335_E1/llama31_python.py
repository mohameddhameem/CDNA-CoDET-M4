def three_blocks_palindrome(n, a):
    if n == 1:
        return 1
    if n == 2:
        return 2 if a[0] == a[1] else 1
    max_length = 0
    for i in range(n):
        for j in range(i + 1, n):
            if a[i] == a[j]:
                count = 0
                for k in range(i + 1, j):
                    if a[k] == a[i]:
                        count += 1
                max_length = max(max_length, count * 2 + 2)
    return max_length

t = int(input())
for _ in range(t):
    n = int(input())
    a = list(map(int, input().split()))
    print(three_blocks_palindrome(n, a))
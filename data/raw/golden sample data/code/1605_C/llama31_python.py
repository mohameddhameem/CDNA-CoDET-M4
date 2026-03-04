def dominant_character(n, s):
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            substring = s[i:i+length]
            if substring.count('a') > substring.count('b') and substring.count('a') > substring.count('c'):
                return length
    return -1

t = int(input())
for _ in range(t):
    n = int(input())
    s = input()
    print(dominant_character(n, s))
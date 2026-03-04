def can_reorder(s):
    for i in range(len(s)):
        for j in range(i + 2, len(s) + 1):
            if s[i:i+2] == s[j:j+2][::-1]:
                return "NO"
    return "YES"

t = int(input())
for _ in range(t):
    n = int(input())
    s = input()
    print(can_reorder(s))
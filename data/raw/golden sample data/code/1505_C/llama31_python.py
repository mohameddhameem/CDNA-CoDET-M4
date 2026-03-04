def is_fibonacci_word(s):
    a, b = 1, 1
    while a <= len(s):
        a, b = a + b, a
    a -= 1
    b -= 1
    for i in range(len(s)):
        if s[i]!= s[(i + a) % len(s)] or s[i]!= s[(i + b) % len(s)]:
            return "NO"
    return "YES"

t = 1
for _ in range(t):
    s = input()
    print(is_fibonacci_word(s))
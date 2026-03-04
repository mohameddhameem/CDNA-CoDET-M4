def wonderful_coloring(s):
    chars = {}
    for c in s:
        chars[c] = chars.get(c, 0) + 1
    n = len(s)
    max_char = max(chars.values())
    return (max_char + n - max_char) // 2

t = int(input())
for _ in range(t):
    s = input()
    print(wonderful_coloring(s))
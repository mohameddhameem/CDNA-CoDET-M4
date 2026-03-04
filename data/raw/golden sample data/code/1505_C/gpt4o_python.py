def is_fibonacci_word(s):
    freq = {c: s.count(c) for c in set(s)}
    values = sorted(freq.values())
    return values == [1, 1] or all(values[i] == values[i - 1] + values[i - 2] for i in range(2, len(values)))

s = input().strip()
print("YES" if is_fibonacci_word(s) else "NO")
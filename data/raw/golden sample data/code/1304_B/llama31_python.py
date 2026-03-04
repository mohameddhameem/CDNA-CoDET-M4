from collections import defaultdict

def longest_palindrome(n, m, strings):
    freq = defaultdict(int)
    for s in strings:
        freq[s] += 1

    even_count = 0
    odd_count = 0
    mid_char = ''
    for s, f in freq.items():
        if f % 2 == 0:
            even_count += f
        else:
            odd_count += f
            mid_char = s

    if odd_count == 0:
        return 0, ''

    half_len = (even_count + 1) // 2
    half_str = ''
    for s, f in freq.items():
        if f % 2 == 0:
            half_str += s * (f // 2)
        else:
            if f > 1:
                half_str += s * (f // 2)
                odd_count -= 1

    result = half_str + mid_char + half_str[::-1]
    return len(result), result

n, m = map(int, input().split())
strings = [input() for _ in range(n)]
length, palindrome = longest_palindrome(n, m, strings)
print(length)
print(palindrome)
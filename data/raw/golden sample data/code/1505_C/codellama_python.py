def is_fibonacci_word(word):
    total = 0
    for i in range(len(word)):
        total += ord(word[i]) - ord('A') + 1
    if total % 2 == 0:
        return "YES"
    else:
        return "NO"
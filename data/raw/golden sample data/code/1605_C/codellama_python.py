import sys

def dominant_character(s):
    # Initialize variables
    a_count, b_count, c_count = 0, 0, 0
    min_len = sys.maxsize
    start_idx, end_idx = 0, 0

    # Count the number of occurrences of each character
    for i in range(len(s)):
        if s[i] == 'a':
            a_count += 1
        elif s[i] == 'b':
            b_count += 1
        elif s[i] == 'c':
            c_count += 1

    # Find the smallest substring that satisfies the conditions
    for i in range(len(s)):
        if s[i] == 'a':
            a_count -= 1
        elif s[i] == 'b':
            b_count -= 1
        elif s[i] == 'c':
            c_count -= 1

        # Check if the current substring satisfies the conditions
        if a_count > b_count and a_count > c_count and len(s[i:]) >= 2:
            min_len = min(min_len, len(s[i:]))
            start_idx = i
            end_idx = len(s)

        # Check if the current substring satisfies the conditions
        if b_count > a_count and b_count > c_count and len(s[i:]) >= 2:
            min_len = min(min_len, len(s[i:]))
            start_idx = i
            end_idx = len(s)

        # Check if the current substring satisfies the conditions
        if c_count > a_count and c_count > b_count and len(s[i:]) >= 2:
            min_len = min(min_len, len(s[i:]))
            start_idx = i
            end_idx = len(s)

    # Return the length of the smallest substring that satisfies the conditions, or -1 if none exists
    if min_len == sys.maxsize:
        return -1
    else:
        return end_idx - start_idx

# Test cases
cases = int(input())
for case in range(cases):
    n = int(input())
    s = input()
    print(dominant_character(s))
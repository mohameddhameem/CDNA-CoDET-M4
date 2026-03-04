def count_beautiful_substrings(s):
    n = len(s)
    count = 0
    last = -1  # last character index that is not '?'
    length = 0  # length of current beautiful substring

    for i in range(n):
        if s[i] == '?':
            length += 1
        else:
            if last == -1 or s[last] != s[i]:
                length += 1
            else:
                length = 1
            last = i

        # Add the number of beautiful substrings ending at current position
        count += length

    return count

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    t = int(data[0])
    results = []
    
    for i in range(1, t + 1):
        results.append(count_beautiful_substrings(data[i]))
    
    print('\n'.join(map(str, results)))

if __name__ == "__main__":
    main()
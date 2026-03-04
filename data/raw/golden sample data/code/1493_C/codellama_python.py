def k_beautiful_strings(s, k):
    t = s
    while len(t) > 0:
        # Replace each letter with a letter that appears k times before the letter in t
        for i in range(len(t)):
            letter = t[i]
            count = 0
            for j in range(len(s)):
                if s[j] == letter:
                    count += 1
            t = t[:i] + (letter * (count // k)) + t[i+1:]
        # Output t if it is beautiful, otherwise output -1
        if all(count % k == 0 for count in t.count(letter) for letter in s):
            return t
    return -1
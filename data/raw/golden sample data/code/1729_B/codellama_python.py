def decode_string(t):
    result = ''
    for c in t:
        if c.isdigit():
            result += chr(ord('a') + int(c))
        else:
            result += c
    return result
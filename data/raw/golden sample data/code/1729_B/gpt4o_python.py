def decode_string(t):
    result = []
    for code in t:
        s = ''
        i = 0
        while i < len(code):
            if i + 1 < len(code) and code[i + 1] == '0':
                s += chr(int(code[i]) + ord('a') - 1)
                i += 2
            else:
                s += chr(int(code[i]) + ord('a') - 1)
                i += 1
        result.append(s)
    return result

q = int(input())
test_cases = [input().strip() for _ in range(2 * q)]
results = decode_string(test_cases[1::2])
print('\n'.join(results))
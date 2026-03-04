def decode_string(s):
    ans = []
    i = 0
    while i < len(s):
        if s[i] == '0':
            num = int(s[i+1:i+2])
            ans.append(chr(96 + num))
            i += 2
        else:
            num = int(s[i])
            ans.append(chr(96 + num))
            i += 1
    return ''.join(ans)

for _ in range(int(input())):
    n = int(input())
    s = input()
    print(decode_string(s))
def perfect_keyboard(s):
    n = len(s)
    used = [False] * 26
    ans = [''] * n
    stack = []
    for i in range(n):
        c = ord(s[i]) - ord('a')
        if stack and stack[-1] == c:
            stack.pop()
            ans[i] = 'a'
            continue
        if stack and abs(stack[-1] - c) == 1:
            ans[i] = chr(stack[-1] + 1) if stack[-1] < c else chr(stack[-1] - 1)
        elif not used[c]:
            stack.append(c)
            ans[i] = chr(c + 97)
            used[c] = True
        else:
            return "NO"
    for i in range(26):
        if not used[i]:
            stack.append(i)
            ans[n-1] = chr(i + 97)
    if len(stack)!= 26:
        return "NO"
    ans = ''.join(ans)
    for i in range(n):
        if ord(ans[i]) - ord('a') == ord(s[i]) - ord('a'):
            continue
        else:
            return "NO"
    return "YES\n" + ans

T = int(input())
for _ in range(T):
    s = input()
    print(perfect_keyboard(s))
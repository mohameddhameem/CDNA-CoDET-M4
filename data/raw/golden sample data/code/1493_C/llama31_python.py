import sys

def k_beautiful_string(n, k, s):
    remain = [0] * 26
    for i in range(n):
        remain[ord(s[i]) - ord('a')] += 1
    remain = [r % k for r in remain]

    for i in range(n):
        for j in range(ord('a'), ord('z') + 1):
            if remain[ord(s[i]) - ord('a')] == 0 and ord(s[i]) < j and (remain[ord(chr(j)) - ord('a')] < k or (remain[ord(chr(j)) - ord('a')] == 0 and s[i]!= chr(j))):
                remain[ord(s[i]) - ord('a')] += 1
                remain[ord(chr(j)) - ord('a')] -= 1
                if sum(remain) == 0:
                    return s[:i] + chr(j) + k_beautiful_string(n - i - 1, k, s[i + 1:])
                else:
                    remain[ord(s[i]) - ord('a')] -= 1
                    remain[ord(chr(j)) - ord('a')] += 1
    return -1

def main():
    input = sys.stdin.readline
    t = int(input())
    for _ in range(t):
        n, k = map(int, input().split())
        s = input().strip()
        print(k_beautiful_string(n, k, s))

if __name__ == "__main__":
    main()
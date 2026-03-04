import sys

from collections import defaultdict





def input():

    return sys.stdin.readline().rstrip()





def main():

    n = int(input())

    A = list(map(int, input().split()))

    B = list(map(int, input().split()))

    res = 0

    bit = max(max(A), max(B)).bit_length()



    def check(res):

        cnt = defaultdict(int)

        for x in A:

            cnt[x & res] += 1

        for x in B:

            cnt[~x & res] -= 1

        ok = 1

        for v in cnt.values():

            ok &= (v == 0)

        return bool(ok)



    for i in range(bit-1, -1, -1):

        if check(res | (1 << i)):

            res |= 1 << i

    print(res)





if __name__ == "__main__":

    t = int(input())

    for _ in range(t):

        main()


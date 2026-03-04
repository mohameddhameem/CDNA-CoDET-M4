import sys





def solve(rem):

    mem2 = [0] * 26



    for k in range(26):

        rem -= memm[k]

        mem2[k] += memm[k]



    mem2[0] += rem

    print(''.join([chr(x + 97) for x in a[:i]]) + chr(j + 97), end='')



    for i1 in range(26):

        print(chr(i1 + 97) * mem2[i1], end='')



    print()

    return True





input = lambda: sys.stdin.buffer.readline().decode().strip()

for _ in range(int(input())):

    n, m = map(int, input().split())

    a, mem, memm, ans = [ord(i) - 97 for i in input()], [0] * 26, [0] * 26, False



    if n % m:

        print(-1)

        continue



    for i in a:

        mem[i] += 1

    for i in a:

        memm[i] = (m - mem[i] % m) % m



    mods = sum(memm)

    for i in reversed(range(n)):

        mem[a[i]] -= 1

        old, new = memm[a[i]], (memm[a[i]] + 1) % m

        memm[a[i]] = new

        mods -= old - new



        for j in range(a[i] + (i != n - 1), 26):

            mem[j] += 1

            old, new = memm[j], (memm[j] - 1) % m

            rem, mods_ = n - i - 1, mods - (old - new)



            if rem >= mods_:

                memm[j] = new

                ans = solve(rem)

                break



            mem[j] -= 1



        if ans:

            break


from random import randint

it = lambda: list(map(int, input().strip().split()))





def solve():

    N = int(input())

    A = it()



    inversion = 0

    duplicate = None

    for i in range(N):

        for j in range(i):

            if A[j] > A[i]:

                inversion += 1

            elif A[j] == A[i]:

                duplicate = A[j]

            

    if inversion % 2:

        if duplicate is None:

            print(-1)

            return



        l = 2

        for i in range(N - 1, -1, -1):

            if A[i] == duplicate:

                l -= 1

                if l == 0:

                    A[i] += 0.1

                    break



    A = [[a, i] for i, a in enumerate(A)]

    B = sorted(A)



    I = []

    for i in range(N):

        if A[i] == B[i]:

            continue

        

        for idx in range(i + 1, N):

            if A[idx] == B[i]:

                break

    

        while idx > i:

            if idx >= i + 2:

                idx -= 2

                I.append(idx + 1)

                A[idx], A[idx + 1], A[idx + 2] = A[idx + 2], A[idx], A[idx + 1]

            else:

                idx -= 1

                I.extend([idx + 1] * 2)

                A[idx], A[idx + 1], A[idx + 2] = A[idx + 1], A[idx + 2], A[idx]



    print(len(I))

    print(' '.join(map(str, I)))        





if __name__ == '__main__':

    T = int(input())

    for _ in range(T):

        solve()
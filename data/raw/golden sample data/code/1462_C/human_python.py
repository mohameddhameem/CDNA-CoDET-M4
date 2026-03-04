t = int(input())

while t:

    n = int(input())

    string = ''

    if n > 45:

        print(-1)

    elif n < 10:

        print(n)

    else:

        for i in range(9, 0, -1):

            if n - i >= 0:

                n -= i

                string = str(i) + string

        print(string)

    t -= 1


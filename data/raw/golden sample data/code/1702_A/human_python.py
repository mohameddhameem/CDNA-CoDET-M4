import bisect

from types import GeneratorType





def gcd(a, b):

    while a and b:

        if a > b:

            a %= b

        else:

            b %= a

    return a + b





def number():

    return int(input())





def lcm(a, b):

    return a * (b // gcd(a, b))





def massint():

    return list(map(int, input().split()))





def massstr(n):

    return [input() for i in range(n)]





def cort():

    return map(int, input().split())





def matrix(n):

    return [[i for i in input()] for i in range(n)]





def bootstrap(f, stack=[]):

    def wrappedfunc(*args, **kwargs):

        if stack:

            return f(*args, **kwargs)

        else:

            to = f(*args, **kwargs)

            while True:

                if type(to) is GeneratorType:

                    stack.append(to)

                    to = next(to)

                else:

                    stack.pop()

                    if not stack:

                        break

                    to = stack[-1].send(to)

            return to

    return wrappedfunc





def solve():

    s = input()



    print(int(s) - 10 ** (len(s) - 1))





for _ in range(int(input())):

    solve()


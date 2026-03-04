# import collections

# from collections import defaultdict

# from re import A, L

# from re import A, I, L, M

import sys

from sys            import setrecursionlimit, stdin, stdout, stderr

# from bisect         import bisect_left, bisect_right

from collections    import defaultdict, deque, Counter

# from itertools      import accumulate, combinations, permutations, product

# from functools      import lru_cache, cmp_to_key, reduce

from heapq          import heapify, heappush, heappop, heappushpop, heapreplace

# import math

# import copy

import os,io

from io import BytesIO, IOBase

# from tkinter import N

# from unittest import TestResult

# from traceback import print_list



# list(map(int,input().split()))

# s = input().strip()

# n = int(input())

# @lru_cache(maxsize=100000)  



from math import sqrt



 

def isprime(x):

    i = 2

    if x <= 3:

        return 1

    if x % 2 == 0:

        return 0

    i += 1

    while i * i <= x:

        if x % i == 0:

            return 0

        i += 2

    return 1



# from itertools import combinations

 

def countbits(x):

    c = 0

    while x:

        if x & 1: c += 1

        x >>= 1

    return c



##############################################

##############################################

############ Code starts here ################

############################################## 







####  BINARY SEARCH  ###  DP ####  MATH  ##########

 

import copy



    

# n = (2 * (10 ** 5)) + 5

# parent = [i for i in range(n)]

# size = [1 for i in range(n)]

 

 

# def find(root):

#     a = root

#     while parent[a] != a:

#         a = parent[a]

#     while root != a:

#         nextnode = parent[root]

#         parent[root] = a

#         root = nextnode

#     return a

 

 

# def union(a, b):

#     a = find(a)

#     b = find(b)

#     if a != b:

#         if size[a] < size[b]:

#             a, b = b, a

#         parent[b] = a

#         size[a] += size[b]

#         return True

#     return False   



import random  

def bootstrap(f, stack=[]):

    from types import GeneratorType

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

 

 

# @bootstrap

 

# from random import getrandbits



# RANDOM = getrandbits(64)



# class Wrapper(int):

#     def __init__(self, x):

#         int.__init__(x)

#     def __hash__(self):

#         return super(Wrapper, self).__hash__() ^ RANDOM 

# from random



# @bootstrap

# sys.setrecursionlimit(10**6)

def ints():

    return int(input())

def inlist():

    return list(map(int,input().split()))





def main():

    # list(map(int,input().split()))

    # s = input().strip()

    # hash={i:2**i for i in range(32)}

    # N=int(input())

    from math import gcd

    # import math

    

    for case in range(int(input())):

    

        n=ints()

        s=input()

        a=0

        b=0

        c=0

        j=0

        m=100

        n=len(s)

        for i in range(1,n-1):

            

            if s[i-1]=='a' and s[i+1]=='a':

                m=min(m,3)

                

        for i in range(1,n-2):

            

            if s[i-1]=='a' and  s[i]=='b' and s[i+1]=='c' and s[i+2]=='a':

                m=min(m,4)

            if s[i-1]=='a' and  s[i]=='c' and s[i+1]=='b' and s[i+2]=='a':

                m=min(m,4)

        

        for i in range(1,n):

            

            if s[i-1]=='a' and s[i]=='a':

                m=min(m,2)

                

        if m==100:

            # if 'aa' in s:

                # print(2)

            if 'accabba' in s:

                print(7)

            elif 'abbacca' in s:

                print(7)

            else:

                

                print(-1)

        else:

            print(m)        

              

            

            

            

        

        

        

        

        



                





        

        

        

        

        

                

                

                

                

            

            

            

     

                

            

                

                

                

                

                

        

  

    

        

                 

                    

            

                

        

        

                

                

                

                

            

              

    

        

                   

                    

                

                

            

        

                            



                

            

            

            

            

            

            

            

            

            

        

            

                    

            

            

            

                        

                    

                            

                

                

                

                

                    

                

                

                

                            

                    

                

                        

                    

            

                

            

            

            

            

        

                

                

            

               

        

        

        

            

            

            

                        

                    

            

            

                    

                                

            

                    

                

                

            

    

    

     

    

# Fast input

BUFSIZE = 8192

class FastIO(IOBase):

    newlines = 0

    def __init__(self, file):

        self._fd = file.fileno()

        self.buffer = BytesIO()

        self.writable = "x" in file.mode or "r" not in file.mode

        self.write = self.buffer.write if self.writable else None

    def read(self):

        while True:

            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))

            if not b:

                break

            ptr = self.buffer.tell()

            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)

        self.newlines = 0

        return self.buffer.read()

    def readline(self):

        while self.newlines == 0:

            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))

            self.newlines = b.count(b"\n") + (not b)

            ptr = self.buffer.tell()

            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)

        self.newlines -= 1

        return self.buffer.readline()

    def flush(self):

        if self.writable:

            os.write(self._fd, self.buffer.getvalue())

            self.buffer.truncate(0), self.buffer.seek(0)

class IOWrapper(IOBase):

    def __init__(self, file):

        self.buffer = FastIO(file)

        self.flush = self.buffer.flush

        self.writable = self.buffer.writable

        self.write = lambda s: self.buffer.write(s.encode("ascii"))

        self.read = lambda: self.buffer.read().decode("ascii")

        self.readline = lambda: self.buffer.readline().decode("ascii")

sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)

input = lambda: sys.stdin.readline().rstrip("\r\n")

# input = sys.stdin.readline



if __name__ == "__main__":

    main()

    

# def bootstrap(f, stack=[]):

#     from types import GeneratorType

#     def wrappedfunc(*args, **kwargs):

#         if stack:

#             return f(*args, **kwargs)

#         else:

#             to = f(*args, **kwargs)

#             while True:

#                 if type(to) is GeneratorType:

#                     stack.append(to)

#                     to = next(to)

#                 else:

#                     stack.pop()

#                     if not stack:

#                         break

#                     to = stack[-1].send(to)

#             return to

 

#     return wrappedfunc

 

 

# @bootstrap
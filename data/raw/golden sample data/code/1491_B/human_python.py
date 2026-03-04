import sys,math,bisect,heapq

input = sys.stdin.readline

print = sys.stdout.write

t=int(input())



def solve():

    ### input init.

    ## num

    # n=int(input())

    n,u,v=map(int,input().split())

    ## string

    # s=input().rstrip("\n")

    ## list

    l=list(map(int,input().split()))

    ## matrix

    # matrix=[list(map(int,input().split())) for _ in range(n)]



    # solve.



    res=float('inf')

    for i in range(n-1):

        if abs(l[i+1]-l[i])>=2:

            res=min(res,0)

        if abs(l[i+1]-l[i])==1:

            res=min(res,min(u,v))



    # return.

    return min(res,min(v+u,v+v))



# print(str(solve()))

for _ in range(t):

    print(str(solve())+"\n")
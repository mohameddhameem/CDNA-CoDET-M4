t=int(input())

while t>0:

    a,b=[int(x) for x in input().split()]

    l=[int(x) for x in input().split()]

    if sum(l)==b:

        

        print("YES")

    else:

        print("NO")

    t-=1    
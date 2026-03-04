def FlipFlopSum():

    for _ in range(int(input())):

        n=int(input())

        a=list(map(int,input().split()))

        if n==2:

            print(-a[0]-a[1])

        elif -1 in a:

            if a.count(-1)<2 or Exist(a,n)==-1:

                print(sum(a))

            else:

                P=Exist(a,n)

                print(sum(a[:P])+2+sum(a[P+2:]))

        else:

            print(sum(a[:n-2])-2)

def Exist(a,n):

    for i in range(n-1):

        if a[i]==a[i+1]==-1:

            return i

    return -1

    

FlipFlopSum()
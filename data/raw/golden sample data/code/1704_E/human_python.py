import io,os,sys

from collections import deque

# sys.stdin=open('D://softwaredata//vscode_file//CP-template//in.txt','r')

input = io.BytesIO(os.read(0, os.fstat(0).st_size)).readline

 

def main(): 

    n,m = map(int,input().split())

    arr = list(map(int,input().split()))

    E=[tuple(map(int,input().split())) for i in range(m)]

    adj=[[] for i in range(n)]

    come=[[] for i in range(n)]

    indeg=[0]*n

    for i,a in enumerate(arr):

        if a>0:

            come[i].append((0,a))

    for a,b in E:

        adj[a-1].append(b-1)

        indeg[b-1]+=1

    

    ans=0

    Q=deque()

    for i in range(n):

        if indeg[i]==0:

            Q.append(i)

    while Q:

        p=Q.popleft()

        come[p].sort()

        go=[]

        cur,now=0,0

        for start,length in come[p]:

            if now==0:

                cur,now=start,length

            elif start<=cur+now:

                now+=length

            else:

                go.append((cur+1,now))

                cur,now=start,length

        

        if now>0:

            go.append((cur+1,now))

        if cur+now>ans:

            ans=cur+now

        for a in adj[p]:

            come[a]+=go

            indeg[a]-=1

            if indeg[a]==0:

                Q.append(a)

    sys.stdout.write(f"{ans%998244353}\n")



t=int(input())

for i in range(t):

    main()
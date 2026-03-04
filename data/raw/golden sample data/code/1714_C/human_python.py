t=int(input())

for x in range(t):

    s,w=int(input()),""

    for i in range(9,0,-1):

        if(s>=i):

            w=str(i)+w

            s-=i

    print(w)


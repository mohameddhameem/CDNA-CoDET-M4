for _ in range(int(input())):

    n=int(input())

    

    s=input()

    # if _==74:

    #     print(s)

    L=[]

    i=0

    m=1

    ans=0

    L=[1]

    p=1

    # while i<n-1:

    #     ans=max(m,ans)

    #     if s[i]!=s[i+1]:

    #         m=1

    #         L.append(p)



    #     else:

    #         p+=1

    #         L.append(p)

    #         m+=1



    #     i+=1

    #     if i==n-2:

    #         if s[-2]==s[-1]:

    #             m+=1

    #             ans=max(ans,m)



    # if len(s)==1:ans=1



    d={0:[],1:[]}

    i=0

    p=1

    final=[1]

    d[int(s[0])].append(1)

    while i < n-1:

        # print(d,'ii',i)

       

        if s[i]==s[i+1]:

          

              

            if len(d[abs(1-int(s[i]))])==0:

                



            





                p+=1

                final.append(p)

                d[int(s[i])].append(p)



            else:

                # if i==n-2:

                #     print('yayayay',s[i])

                k=d[abs(1-int(s[i]))].pop()

                # if i==3:

                #     print(k,'kaka')

                final.append(k)

                d[int(s[i])].append(k)





        else:

            # final.append(p)

            if len(d[int(s[i])])!=0:

                    

                k=d[int(s[i])].pop()

                final.append(k)

                d[abs(1-int(s[i]))].append(k)



        i+=1

    print(len(set(final)))



    for i in final:

        print(i,end=" ")

    print()

    
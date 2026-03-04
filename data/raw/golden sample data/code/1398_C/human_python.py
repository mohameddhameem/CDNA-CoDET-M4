for i in range(int(input())) :

    n = int(input())

    s= [int(num) for num in input()]

    

    mp={0:1} 

    sm, ans=0, 0 

    for i in s: 

        sm+= i-1 

        if sm in mp: 

            ans+= mp[sm]

            mp[sm]+=1 

        else: 

            mp[sm]=1 

    print(ans )

        

 
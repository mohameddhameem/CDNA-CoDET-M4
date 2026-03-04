for t in range(int(input())):
    n,k=map(int,input().split())
    rj,x=0,1
    for i in range(31):
        if k&(1<<i):
            rj+=x
        x*=n

    print(rj%(10**9+7))

  	 		  	   		    			 		  			 	
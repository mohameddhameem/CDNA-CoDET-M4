tc=int(input())

while(tc>0):

    tc-=1

    n=int(input())

    s=input()

    f=1

    for i in range(len(s)):

        if(s[i]=='1'):

            f=0

            break

    if(f==1):

        s1=""

        for i in range(len(s)):

            if(s[i]=='2'):

                s1+='1'

            else:

                s1+='0'

        print(s1)

        print(s1)

    else:

        f=1

        s1=""

        s2=""

        for i in range(len(s)):

            if(f==1):

                if(s[i]=='2'):

                    s1+='1'

                    s2+='1'

                if(s[i]=='1'):

                    s1+='1'

                    s2+='0'

                if(s[i]=='0'):

                    s1+='0'

                    s2+='0'

            else:

                if(s[i]=='2'):

                    s1+='0'

                    s2+='2'

                if(s[i]=='1'):

                    s1+='0'

                    s2+='1'  

                if(s[i]=='0'):

                    s1+='0'

                    s2+='0'

            if(s[i]=='1'):

                f=0

        print(s1)

        print(s2)


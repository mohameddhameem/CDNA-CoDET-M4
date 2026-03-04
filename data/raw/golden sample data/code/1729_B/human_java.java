

import java.util.*;



/*

9

6

315045

4

1100

7

1213121

6

120120

18

315045615018035190

7

1111110

7

1111100

5

11111

4

2606







*/



public class Main{

    

    

    public static void main(String args[]){

        

        Scanner scan = new Scanner(System.in);

        int t=scan.nextInt();

        while(t>0){

            

            int n=scan.nextInt();

            String str=scan.next();

            int p=n;

            String ret="";

            while(p>0){

                if(p>2){

                    String g=str.substring(p-3,p);

                    int m=Integer.parseInt(g);

                    if(m%10==0){

                        m/=10;

                        ret=((char)(96+m))+ret;

                        p-=3;

                    }

                    else{

                        m%=10;

                        //System.out.println("m="+m);

                        ret=((char)(m+96))+ret;

                        p-=1;

                    }

                    //System.out.println(ret);

                }

                else{

                    String g=str.substring(p-1,p);

                    int m=Integer.parseInt(g);

                    ret=((char)(m+96))+ret;

                    p-=1;

                }

                

            }

            

            //for(int j=n-1;j)

            //System.out.println("ab"+(char)(97));

            System.out.println(ret);

            t--;

        }

        

    }

    

    

    

}
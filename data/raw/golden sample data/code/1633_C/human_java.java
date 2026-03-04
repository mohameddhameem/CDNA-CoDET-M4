import java.util.*;

public class practise {



    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);

        int t = sc.nextInt();

        while (t-- > 0) {

            long hc = sc.nextLong();

long dc=sc.nextLong();

long hm=sc.nextLong();

long dm=sc.nextLong();

int k= sc.nextInt();

int w=sc.nextInt();

long a=sc.nextLong();

boolean falg=true;

for(int i=0;i<=k;i++){

    long nhc=hc+(i*a);

    long ndc=dc+(k-i)*w;

    long monmarega=hm/ndc+(hm%ndc==0?0:1);

    long chmarega=(nhc)/(dm)+(nhc%dm==0?0:1);

if(chmarega>=monmarega){

    falg=false;

    break;

}



}



if(!falg){

    System.out.println("YES");

}else{

    System.out.println("NO");

}







 }

    }

}


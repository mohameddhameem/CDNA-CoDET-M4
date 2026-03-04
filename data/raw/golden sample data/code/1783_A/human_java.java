import java.util.*;



public class Main {

    public static void main(String[] args) {

   

      Scanner sc=new Scanner(System.in);

      int t;

      t=sc.nextInt();

      while(t>0){

        

        int n;

        n=sc.nextInt();

        Integer[] arr=new Integer[n];

        

        for(int i=0;i<n;i++){

          arr[i]=sc.nextInt();

        }

        

        Arrays.sort(arr,Collections.reverseOrder());

        

        if(arr[0]==arr[n-1]){

          System.out.println("NO");

        }

        else{

          

          System.out.println("YES");

          if(arr[0]==arr[1]){

            int x=arr[0];

             arr[0]=arr[n-1];

             arr[n-1]=x;

             

          }

          for(int i=0;i<n;i++){

          System.out.print(arr[i]+" ");

          }

          System.out.println();

        }

        

        

        t--;

      }

  }

}
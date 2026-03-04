import java.util.Scanner;



public class Main{

    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);

        int t = sc.nextInt(), n, min, c;

        int a[] = new int[100];

        while(t -- > 0){

            c = 0;

            min = 101;

            n = sc.nextInt();

            for(int i = 0; i < n; i ++){

                a[i] = sc.nextInt();

                if(a[i] < min) min = a[i];

            }

            for(int i = 0; i < n; i ++){

                if(a[i] == min) c ++;

            }

            System.out.println(n - c);

        }

        sc.close();

    }

}
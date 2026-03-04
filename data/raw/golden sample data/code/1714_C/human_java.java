import java.util.Scanner;



public class minimum_varied_number_c {

  public static void main(String[] args) {

    Scanner scan = new Scanner(System.in);

    int t = scan.nextInt();

    while(t-->0) {

      int n = scan.nextInt();

      String ans = "";

      for(int i=9; i>0; i--) {

        if(n>=i) {

          ans = i + ans;

          n -= i;

        }

      }

      System.out.println(ans);

    }

    scan.close();

  }

}


import java.util.*;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int t = sc.nextInt();
        while (t-- > 0) {
            int n = sc.nextInt();
            int d = sc.nextInt();
            int[] A = new int[n];
            int i;
            for (i = 0; i < n; i++) 
                A[i] = sc.nextInt();
            Arrays.sort(A);
            if (A[n - 1] <= d)
                System.out.println("YES");
            else if (A[0] + A[1] <= d)
                System.out.println("YES");
            else
                System.out.println("NO");
        }
        sc.close();
    }
}

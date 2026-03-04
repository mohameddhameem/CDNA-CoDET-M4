import java.util.*;



public class Main {

    public static void main(String[] arg) {

        Scanner in = new Scanner(System.in);

        long a = in.nextLong();

        long b = in.nextLong();

        long c = in.nextLong();

        long d = in.nextLong();



        long ans = 0;

        for (long z = c; z <= d; ++z) {

            long lo = b, hi = c;

            long poss_y = -1;

            while (lo <= hi) {

                long y = (lo + hi) / 2;

                long x = Math.max(a, z - y + 1);

                if (x >= a && x <= b) {

                    poss_y = y;

                    hi = y - 1;

                } else {

                    lo = y + 1;

                }

            }

            if (poss_y != -1) {

                long x = Math.max(a, z - poss_y + 1);

                long sr = b - x + 1;

                long space = c - poss_y + 1;

                long times = Math.min(space, x - a + 1);

                ans += times * (2 * sr + (times - 1)) / 2;

                ans += (space - times) * (b - a + 1);

            }

            // System.out.println(z + " " + ans);

        }

        System.out.println(ans);

    }

}
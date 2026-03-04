/*

        "Everything in the universe is balanced. Every disappointment

                you face in life will be balanced by something good for you!

                        Keep going, never give up."



                        Just have Patience + 1...



*/



















import java.util.*;

import java.lang.*;

import java.io.*;





public class SolutionC {



    static int MOD = 998244353, MAX = 105;

    static long[][][] dp = new long[MAX][MAX][3];

    static int[][] constraints = new int[MAX][MAX];

    static int n;



    public static void main(String[] args) throws Exception {

        out = new PrintWriter(new BufferedOutputStream(System.out));

        sc = new FastReader();



        int test = 1;

        for (int t = 1; t <= test; t++) {

            solve(t);

        }

        out.close();

    }



    private static void solve(int t) {

        n = sc.nextInt();

        for (int i = 1; i <= n; i++) {

            for (int j = i; j <= n; j++) {

                constraints[i][j] = sc.nextInt();

            }

        }

        for (int i = 0; i < MAX; i++) {

            for (int j = 0; j < MAX; j++) {

                Arrays.fill(dp[i][j], -1);

            }

        }

        long ways = dp(1, 0, 2);

        out.println(ways);

    }



    private static long dp(int index, int sameCount, int prev) {

        if (index == n + 1) {

            return 1;

        }

        if (dp[index][sameCount][prev] != -1) {

            return dp[index][sameCount][prev];

        }

        long ways = 0;

        for (int curr = 0; curr < 2; curr++) {

            int nextCount = curr == prev ? sameCount + 1 : 1;

            if (isGood(index, nextCount)) {

                ways = addMod(ways, dp(index + 1, nextCount, curr));

            }

        }

        return dp[index][sameCount][prev] = ways;

    }



    private static boolean isGood(int end, int nextCount) {

        for (int start = end; start >= 0; start--) {

            if (constraints[start][end] == 1 && start < end - nextCount + 1) {

                return false;

            }

            if (constraints[start][end] == 2 && start >= end - nextCount + 1) {

                return false;

            }

        }

        return true;

    }



    private static long addMod(long a, long b) {

        return (a % MOD + b % MOD) % MOD;

    }





    public static FastReader sc;

    public static PrintWriter out;

    static class FastReader

    {

        BufferedReader br;

        StringTokenizer str;



        public FastReader()

        {

            br = new BufferedReader(new

                    InputStreamReader(System.in));

        }



        String next()

        {

            while (str == null || !str.hasMoreElements())

            {

                try



                {

                    str = new StringTokenizer(br.readLine());

                }

                catch (IOException  lastMonthOfVacation)

                {

                    lastMonthOfVacation.printStackTrace();

                }



            }

            return str.nextToken();

        }



        int nextInt()

        {

            return Integer.parseInt(next());

        }



        long nextLong()

        {

            return Long.parseLong(next());

        }



        double nextDouble()

        {

            return Double.parseDouble(next());

        }



        String nextLine()

        {

            String str = "";

            try

            {

                str = br.readLine();

            }

            catch (IOException lastMonthOfVacation)

            {

                lastMonthOfVacation.printStackTrace();

            }

            return str;

        }

    }

}
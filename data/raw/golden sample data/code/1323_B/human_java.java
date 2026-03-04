/*
        "Everything in the universe is balanced. Every disappointment
                you face in life will be balanced by something good for you!
                        Keep going, never give up."

*/


import java.util.*;
import java.lang.*;
import java.io.*;

public class CodeChef {
    public static void main(String[] args) throws java.lang.Exception {
        out = new PrintWriter(new BufferedOutputStream(System.out));
        sc = new FastReader();

        int test = 1;
        for (int t = 0; t < test; t++) {
            solve();
        }
        out.close();
    }

    private static void solve() {
        int n = sc.nextInt();
        int m = sc.nextInt();
        long k = sc.nextLong();

        int[] A = new int[n];
        int[] B = new int[m];

        for (int i = 0; i < n; i++) {
            A[i] = sc.nextInt();
        }
        for (int i = 0; i < m; i++) {
            B[i] = sc.nextInt();
        }

        long[] widths = consecutiveOnes(A);
        long[] depths = consecutiveOnes(B);

        long subRectangles = 0;
        for (int i = 1; i < widths.length; i++) {
            if (k % i == 0 && k / i <= m) {
                subRectangles += widths[i] * depths[(int) k / i];
            }
        }
        out.println(subRectangles);
    }

    private static long[] consecutiveOnes(int[] arr) {
        int n = arr.length;
        long[] res = new long[n + 1];
        int i = 0;
        while (i < n) {
            if (arr[i] == 0) {
                i++;
                continue;
            }
            int j = i;
            while (j < n && arr[j] == 1) {
                j++;
            }
            for (int k = 0; k <= j - i; k++) {
                res[k] += (j - i - k + 1);
            }
            i = j;
        }
        return res;
    }

    public static FastReader sc;
    public static PrintWriter out;
    static class FastReader
    {
        BufferedReader br;
        StringTokenizer st;

        public FastReader()
        {
            br = new BufferedReader(new
                    InputStreamReader(System.in));
        }

        String next()
        {
            while (st == null || !st.hasMoreElements())
            {
                try
                {
                    st = new StringTokenizer(br.readLine());
                }
                catch (IOException  e)
                {
                    e.printStackTrace();
                }
            }
            return st.nextToken();
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
            catch (IOException e)
            {
                e.printStackTrace();
            }
            return str;
        }
    }
}
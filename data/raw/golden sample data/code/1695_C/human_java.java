import java.io.*;

import java.math.*;

import java.util.*;





public class Main {



    static final int INF = 0x3f3f3f3f;

    static final long LNF = 0x3f3f3f3f3f3f3f3fL;

    public static void main(String[] args) throws IOException {

        initReader();

        int t=nextInt();

        while (t--!=0){

            int n=nextInt();

            int m=nextInt();

          int [][] arr=new int[n+1][m+1];

            int[][]mmax=new int[n+1][m+1];

            int[][]mmin=new int[n+1][m+1];

            for(int i=1;i<=n;i++){

                for(int j=1;j<=m;j++){

                    arr[i][j]=nextInt();

                    mmax[i][j]=-INF;

                    mmin[i][j]=INF;

                }

            }

            if((n+m-1)%2!=0){

                pw.println("NO");

                continue;

            }

            mmax[1][1]=mmin[1][1]=arr[1][1];

            for(int i=1;i<=n;i++){

                for(int j=1;j<=m;j++){

                    if(i>1){

                        mmax[i][j]=Math.max(mmax[i][j],mmax[i-1][j]+arr[i][j]);

                        mmin[i][j]=Math.min(mmin[i][j],mmin[i-1][j]+arr[i][j]);

                    }

                   if(j>1){

                        mmax[i][j]=Math.max(mmax[i][j],mmax[i][j-1]+arr[i][j]);

                        mmin[i][j]=Math.min(mmin[i][j],mmin[i][j-1]+arr[i][j]);

                    }

               }

            }

            if(mmax[n][m]>=0&&mmin[n][m]<=0){

                pw.println("YES");

            }else pw.println("NO");

        }



        pw.close();

    }





    /***************************************************************************************/



    static BufferedReader reader;

    static StringTokenizer tokenizer;

    static PrintWriter pw;



    public static void initReader() throws IOException {

        reader = new BufferedReader(new InputStreamReader(System.in));

        tokenizer = new StringTokenizer("");

        pw = new PrintWriter(new BufferedWriter(new OutputStreamWriter(System.out)));



//        从文件读写

//        reader = new BufferedReader(new FileReader("test.in"));

//        tokenizer = new StringTokenizer("");

//        pw = new PrintWriter(new BufferedWriter(new FileWriter("test1.out")));

    }



    public static boolean hasNext() {

        try {

            while (!tokenizer.hasMoreTokens()) {

                tokenizer = new StringTokenizer(reader.readLine());

            }

        } catch (Exception e) {

            return false;

        }

        return true;

    }



    public static String next() throws IOException {

        while (!tokenizer.hasMoreTokens()) {

            tokenizer = new StringTokenizer(reader.readLine());

        }

        return tokenizer.nextToken();

    }



    public static String nextLine() {

        try {

            return reader.readLine();

        } catch (Exception e) {

            return null;

        }

    }



    public static int nextInt() throws IOException {

        return Integer.parseInt(next());

    }



    public static long nextLong() throws IOException {

        return Long.parseLong(next());

    }



    public static double nextDouble() throws IOException {

        return Double.parseDouble(next());

    }



    public static char nextChar() throws IOException {

        return next().charAt(0);

    }

}






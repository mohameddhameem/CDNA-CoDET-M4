/*****  --->         :)    Vijender  Srivastava      (:       <---       *****/

import java.util.*;

import java.lang.*;

import java.io.*;

public class Main  

{

    static FastReader sc =new FastReader();

    static PrintWriter out=new PrintWriter(System.out);

    static long mod=(long)32768;

    static StringBuilder sb = new StringBuilder();

    /* start */

   

    public static void main(String [] args)

    {

        int testcases = 1;

        testcases = i();

        // calc();

        while(testcases-->0)

        {

            

            solve();

        }

        out.flush();

        out.close();

    }

    

    static void solve()

    { 

        int n = i();

        long a[] = inputL(n);

        

        pl(n);

        for(int i=0;i<n;i++)

        {

            long x = pow(a[i])-a[i];

            pl((i+1)+" "+x);

        }

    }

    

    static long pow(long x)

    {

        long y = 1;

        while(y<x)y*=2;

        

        return y;

    }

    

    /* end */

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

    // print code start

    static void p(Object o)

    {

        out.print(o);

    }

    static void pl(Object o)

    {

        out.println(o);

    }

    static void pl()

    {

        out.println("");

    }

    // print code end //

    

    static int i() {

        return sc.nextInt();

    }



    static String s() {

        return sc.next();

    }



    static long l() {

        return sc.nextLong();

    }



    static char[] inputC()

    {

        String s = sc.nextLine();

        return s.toCharArray();

    }



    static int[] input(int n) {

        int A[]=new int[n];

           for(int i=0;i<n;i++) {

               A[i]=sc.nextInt();

           }

        return A;

    }



    static long[] inputL(int n) {

        long A[]=new long[n];

           for(int i=0;i<n;i++) {

               A[i]=sc.nextLong();

           }

           return A;

    }



    static long[] putL(long a[]) {

        long A[]=new long[a.length];

           for(int i=0;i<a.length;i++) {

               A[i]=a[i];

           }

           return A;

    }



    static String[] inputS(int n) {

        String A[]=new String[n];

           for(int i=0;i<n;i++) {

               A[i]=sc.next();

           }

           return A;

    }

      

    static int gcd(int a, int b) 

     { 

         if (a == 0) 

             return b; 

         return gcd(b % a, a); 

     } 



     static long gcd(long a, long b) 

     { 

         if (a == 0) 

             return b; 

         return gcd(b % a, a); 

     } 



     static int lcm(int a, int b)

    {

        return (a / gcd(a, b)) * b;

    }

    

     static String reverse(String s) {

        StringBuffer p=new StringBuffer(s);

        p.reverse();

        return p.toString();

     }



     static int min(int a,int b) {

        return Math.min(a, b);

    }

    static int min(int a,int b,int c) {

        return Math.min(a, Math.min(b, c));

    }

    static int min(int a,int b,int c,int d) {

        return Math.min(a, Math.min(b, Math.min(c, d)));

    }



    static int max(int a,int b) {

        return Math.max(a, b);

    }

    static int max(int a,int b,int c) {

        return Math.max(a, Math.max(b, c));

    }

    static int max(int a,int b,int c,int d) {

        return Math.max(a, Math.max(b, Math.max(c, d)));

    }

    static long min(long a,long b) {

        return Math.min(a, b);

    }

    static long min(long a,long b,long c) {

        return Math.min(a, Math.min(b, c));

    }

    static long min(long a,long b,long c,long d) {

        return Math.min(a, Math.min(b, Math.min(c, d)));

    }

    static long max(long a,long b) {

        return Math.max(a, b);

    }

    static long max(long a,long b,long c) {

        return Math.max(a, Math.max(b, c));

    }

    static long max(long a,long b,long c,long d) {

        return Math.max(a, Math.max(b, Math.max(c, d)));

    }



    static long sum(int A[]) {

        long sum=0;

        for(int i : A) {

            sum+=i;

        }

        return sum;

    }



    static long sum(long A[]) {

        long sum=0;

        for(long i : A) {

            sum+=i;

        }

        return sum;

    }



    static long mod(long x) {

          return ((x%mod + mod)%mod);

    }



    static long power(long x, long y)

    {



	if(y==0)

		return 1;

	if(x==0)

		return 0;

    long res = 1;

    while (y > 0) {



        if (y % 2 == 1)

            res = (res * x) ;



        y = y >> 1; 

        x = (x * x);

    }



    return res;

    }



    static boolean prime(int n) 

	    { 

	        if (n <= 1) 

	            return false; 

	        if (n <= 3) 

	            return true; 

	        if (n % 2 == 0 || n % 3 == 0) 

	            return false; 

	        double sq=Math.sqrt(n);

	  

	        for (int i = 5; i <= sq; i = i + 6) 

	            if (n % i == 0 || n % (i + 2) == 0) 

	                return false; 

	        return true; 

	    } 



     static boolean prime(long n) 

	    { 

	        if (n <= 1) 

	            return false; 

	        if (n <= 3) 

	            return true; 

	        if (n % 2 == 0 || n % 3 == 0) 

	            return false; 

	        double sq=Math.sqrt(n);

	  

	        for (int i = 5; i <= sq; i = i + 6) 

	            if (n % i == 0 || n % (i + 2) == 0) 

	                return false; 

	        return true; 

	    } 



        static long[] sort(long a[]) {

            ArrayList<Long> arr = new ArrayList<>();

            for(long i : a) {

                arr.add(i);

            }

            Collections.sort(arr);

            for(int i = 0; i < arr.size(); i++) {

                a[i] = arr.get(i);

            }

            return a;

        }

    

        static int[] sort(int a[])

        {

            ArrayList<Integer> arr = new ArrayList<>();

            for(Integer i : a) {

                arr.add(i);

            }

            Collections.sort(arr);

            for(int i = 0; i < arr.size(); i++) {

                a[i] = arr.get(i);

            }

            return a;

        }

        //pair class

        private static class Pair implements Comparable<Pair> {

            long first, second;

            public Pair(long f, long s) {

                first = f;

                second = s;

            }

            @Override

            public int compareTo(Pair p) {

                if (first < p.first)

                    return 1;

                else if (first > p.first)

                    return -1;

                else {

                    if (second > p.second)

                        return 1;

                    else if (second < p.second)

                        return -1;

                    else

                        return 0;

                }

            }

     

        }



        // segment t start

        static long seg[] ;

        static void build(long a[], int v, int tl, int tr) {

            if (tl == tr) {

                seg[v] = a[tl];

            } else {

                int tm = (tl + tr) / 2;

                build(a, v*2, tl, tm);

                build(a, v*2+1, tm+1, tr);

                seg[v] = Math.min(seg[v*2] , seg[v*2+1]);

            }

        }

        

        static long query(int v, int tl, int tr, int l, int r) {

            if (l > r || tr < tl) 

                return Integer.MAX_VALUE;

            if (l == tl && r == tr) {

                return seg[v];

            }

            int tm = (tl + tr) / 2;

            return (query(v*2, tl, tm, l, min(r, tm))+ query(v*2+1, tm+1, tr, max(l, tm+1), r));

        }

        

        static void update(int v, int tl, int tr, int pos, long new_val) {

            if (tl == tr) {

                seg[v] = new_val;

            } else {

                int tm = (tl + tr) / 2;

                if (pos <= tm)

                    update(v*2, tl, tm, pos, new_val);

                else

                    update(v*2+1, tm+1, tr, pos, new_val);

                seg[v] = Math.min(seg[v*2] , seg[v*2+1]);

            }

        }



        // segment t end //

}
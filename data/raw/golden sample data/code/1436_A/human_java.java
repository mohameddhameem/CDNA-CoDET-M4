

//  package com.company;





import java.util.*;

import java.lang.*;

import java.io.*;





public final class Main {



    FastReader s;

//    PrintWriter out ;

    public static void main(String[] args) throws java.lang.Exception {





        new Main().run();



    }



    void run() {



//        out = new PrintWriter(new OutputStreamWriter(System.out));

        s = new FastReader();

        solve();

    }







StringBuffer sb;





    void solve() {



        sb = new StringBuffer();







        for (int T = s.nextInt(); T > 0; T--)

        {

            start();



        }



        System.out.print(sb);



    }





    void start()

    {

        int n = s.nextInt();

        int m = s.nextInt();

        long arr[] = longArr(n);

        long sum  =0;

        for(long h :  arr)

        {

            sum+=h;

        }

        if(sum == m)

        {

            sb.append("YES\n");

        }

        else

        {

            sb.append("NO\n");

        }

    }









     void  swap(int arr[], int i, int j)

     {

         int temp = arr[i];

         arr[i] = arr[j];

         arr[j] = temp;

     }



    class Pair {

        int x;

        int y;



//        String hash;



        public Pair(int x, int y) {

            this.x = x;

            this.y = y;



//            this.hash = x+" "+y;

        }



        @Override

        public String toString (){

           return ("{ x = "+this.x +" , "+"y = "+this.y+" }");

        }



        @Override

        public boolean equals(Object ob)

        {

            if(this == ob)

                return true;



            if(ob == null || this.getClass() != ob.getClass())

                return false;

                Pair p = (Pair)ob;

            return (p.x == this.x) && (p.y == this.y);

        }



//        @Override

//        public int hashCode()

//        {

//            return hash.hashCode();

//        }



    }



    class LongPair {

        long x;

        long y;

//        String hash;

        public LongPair(long x, long  y) {

            this.x = x;

            this.y = y;

//            this.hash = x+" "+y;

        }



        @Override

        public String toString (){

            return ("{ x = "+this.x +" , "+"y = "+this.y+" }");

        }



        @Override

        public boolean equals(Object ob)

        {

            if(this == ob)

                return true;



            if(ob == null || this.getClass() != ob.getClass())

                return false;

            LongPair p = (LongPair)ob;

            return (p.x == this.x) && (p.y == this.y);

        }



//        @Override

//        public int hashCode()

//        {

//            return hash.hashCode();

//        }



    }



    class SegmentTree{

        int[] tree;

        int n;

        public SegmentTree(int[] nums) {

            if (nums.length > 0) {

                n = nums.length;

                tree = new int[n * 2];

                buildTree(nums);

            }

        }



        private void buildTree(int[] nums) {

            for (int i = n, j = 0;  i < 2 * n; i++,  j++)

                tree[i] = nums[j];

            for (int i = n - 1; i > 0; --i)

                tree[i] = tree[i * 2] + tree[i * 2 + 1];

        }



        void update(int pos, int val) {

            pos += n;

            tree[pos] = val;

            while (pos > 0) {

                int left = pos;

                int right = pos;

                if (pos % 2 == 0) {

                    right = pos + 1;

                } else {

                    left = pos - 1;

                }

                // parent is updated after child is updated

                tree[pos / 2] = tree[left] + tree[right];

                pos /= 2;

            }

        }



        public int sumRange(int l, int r) {

            // get leaf with value 'l'

            l += n;

            // get leaf with value 'r'

            r += n;

            int sum = 0;

            while (l <= r) {

                if ((l % 2) == 1) {

                    sum += tree[l];

                    l++;

                }

                if ((r % 2) == 0) {

                    sum += tree[r];

                    r--;

                }

                l /= 2;

                r /= 2;

            }

            return sum;

        }



    }



    int gcd(int a, int b) {

        if (b == 0)

            return a;

        return gcd(b, a % b);

    }



    long gcd(long a, long b) {

        if (b == 0)

            return a;

        return gcd(b, a % b);

    }



    long power(long x, long y, long p) {

        long res = 1;





        x = x % p;



        while (y > 0) {



            if ((y & 1) > 0)

                res = (res * x) % p;





            y = y >> 1;

            x = (x * x) % p;

        }

        return res;

    }



    int find(int dsu[], int i) {

        if (dsu[i] == i)

            return i;

        dsu[i] = find(dsu, dsu[i]);

        return dsu[i];

    }



    void union(int dsu[], int i, int j) {

        int a = find(dsu, i);

        int b = find(dsu, j);

        dsu[a] = b;

    }



    static void sort(int[] a) {

        ArrayList<Integer> l = new ArrayList<>();

        for (int i : a)

            l.add(i);

        Collections.sort(l);

        for (int i = 0; i < a.length; i++)

            a[i] = l.get(i);

    }



    static void sort(long[] a) {

        ArrayList<Long> l = new ArrayList<>();

        for (long i : a)

            l.add(i);

        Collections.sort(l);

        for (int i = 0; i < a.length; i++)

            a[i] = l.get(i);

    }



    static String sort(String s) {

        Character ch[] = new Character[s.length()];

        for (int i = 0; i < s.length(); i++) {

            ch[i] = s.charAt(i);

        }

        Arrays.sort(ch);

        StringBuffer st = new StringBuffer("");

        for (int i = 0; i < s.length(); i++) {

            st.append(ch[i]);

        }

        return st.toString();

    }



    // long array input

    public long[] longArr(int len) {

        // long arr input

        long[] arr = new long[len];

        String[] strs = s.nextLine().trim().split("\\s+");

        for (int i = 0; i < len; i++) {

            arr[i] = Long.parseLong(strs[i]);



        }

        return arr;

    }



    // int arr input

    public int[] intArr(int len) {

        // long arr input

        int[] arr = new int[len];

        String[] strs = s.nextLine().trim().split("\\s+");

        for (int i = 0; i < len; i++) {

            arr[i] = Integer.parseInt(strs[i]);



        }

        return arr;

    }



    public void printArray(int[] A) {

        System.out.println(Arrays.toString(A));

    }



    public void printArray(long[] A) {

        System.out.println(Arrays.toString(A));

    }



    static class FastReader {

        BufferedReader br;

        StringTokenizer st;



        public FastReader() {

            br = new BufferedReader(new InputStreamReader(System.in));

        }



        String next() {

            while (st == null || !st.hasMoreElements()) {

                try {

                    st = new StringTokenizer(br.readLine());

                } catch (IOException e) {

                    e.printStackTrace();

                }

            }

            return st.nextToken();

        }



        int nextInt() {

            return Integer.parseInt(next());

        }



        long nextLong() {

            return Long.parseLong(next());

        }



        double nextDouble() {

            return Double.parseDouble(next());

        }



        String nextLine() {

            String str = "";

            try {

                str = br.readLine();

            } catch (IOException e) {

                e.printStackTrace();

            }

            return str;

        }

    }



}


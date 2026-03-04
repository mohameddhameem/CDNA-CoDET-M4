import java.io.*;

import java.util.*;



public class B_Frog_Traveler{

    static FastReader in = new FastReader();

    static PrintWriter out = new PrintWriter(new OutputStreamWriter(System.out));



    static class Pair {

        int p1;

        int p2;

        public Pair(int p1,int p2) {

            this.p1 = p1;

            this.p2 = p2;

        }

    }



    public static void main(String[] args) {

        int n = in.nextInt();

        int[] a = new int[n+1];

        int[] b = new int[n+1];

        for (int i = 1;i <= n;i++) {

            a[i] = in.nextInt();

        }

        for (int i = 1;i <= n;i++) {

            b[i] = in.nextInt();

        }

        TreeMap<Integer,Integer> f = new TreeMap<>();

        for (int i = 1;i <= n;i++) {

            f.put(i, Math.min(i+b[i],n));

        }

        int[] pre = new int[n+1];

        Queue<Pair> Q = new ArrayDeque<>();

        Q.add(new Pair(n,n)); //before after

        while(!Q.isEmpty()) {

            Pair p = Q.poll();

            int u = p.p1;

            int bU = p.p2;

            if (u - a[u] <= 0) {

                pre[0] = bU;

                break;

            }

            Integer l = u-a[u];

            while((l = f.ceilingKey(l))!= null) {

                if (l > u)break;

                pre[l] = bU;

                Q.add(new Pair(f.get(l),l));

                f.remove(l);

            }

        }



        if (pre[0] == 0) {

            out.println(-1);

        } else {

            ArrayList<Integer> ans = new ArrayList<>();

            int p = 0;

            ans.add(0);

            while((p = pre[p])!=n) {

                ans.add(p);

            }

            out.println(ans.size());

            for (int i = ans.size()-1;i >= 0;i--) {

                out.printf("%d ",ans.get(i));

            }

        }

        

        out.close();

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

                }

                catch (IOException e) {

                    e.printStackTrace();

                }

            }

            return st.nextToken();

        }

        

        int nextInt(){

            return Integer.parseInt(next());

        }



        long nextLong() {

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
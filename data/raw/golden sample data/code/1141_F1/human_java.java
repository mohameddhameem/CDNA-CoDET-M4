/*

        "Everything in the universe is balanced. Every disappointment

                you face in life will be balanced by something good for you!

                        Keep going, never give up."



                        Just have Patience + 1...



*/

















import java.util.*;

import java.lang.*;

import java.io.*;





public class Solution {



    static class Segment {

        int start;

        int end;



        Segment(int start, int end) {

            this.start = start;

            this.end = end;

        }

    }



    public static void main(String[] args) throws java.lang.Exception {

        out = new PrintWriter(new BufferedOutputStream(System.out));

        sc = new FastReader();



        int test = 1;

        for (int t = 1; t <= test; t++) {

            solve();

        }

        out.close();

    }



    private static void solve() {

        int n = sc.nextInt();

        int[] arr = new int[n];

        for (int i = 0; i < n; i++) {

            arr[i] = sc.nextInt();

        }



        // divide into max subarray-blocks, such that each have same sum.



        Map<Long, List<Segment>> segmentsWithParticularSum = new HashMap<>();

        // for each possible subarray-sum

        for (int end = 0; end < n; end++) {

            long blockSum = 0;

            for (int start = end; start >= 0; start--) {

                blockSum += arr[start];

                if (!segmentsWithParticularSum.containsKey(blockSum)) {

                    segmentsWithParticularSum.put(blockSum, new ArrayList<>());

                }

                segmentsWithParticularSum.get(blockSum).add(new Segment(start + 1, end + 1));

            }

        }



        List<Segment> bestSegmentsChosen = new ArrayList<>();

        int maxTotalBlocks = 0;

        for (long blockSum : segmentsWithParticularSum.keySet()) {

            // greedily choose the blocks, since blocks are sorted by their right end

            List<Segment> blocksTaken = new ArrayList<>();

            int lastBlockEnd = -1, totalBlocks = 0;

            for (Segment segment : segmentsWithParticularSum.get(blockSum)) {

                if (segment.start > lastBlockEnd) { // can take this block

                    totalBlocks++;

                    blocksTaken.add(segment);

                    lastBlockEnd = segment.end;

                }

            }



            if (totalBlocks > maxTotalBlocks) { // maximize the total blocks taking

                maxTotalBlocks = totalBlocks;

                bestSegmentsChosen = new ArrayList<>(blocksTaken);

            }

        }



        out.println(maxTotalBlocks);

        for (Segment segment : bestSegmentsChosen) {

            out.println(segment.start + " " + segment.end);

        }

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
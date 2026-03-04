import java.io.*;

import java.util.*;



public class goodsubarrays {

    static class FastIO extends PrintWriter {

        private InputStream stream;

        private byte[] buf = new byte[1 << 16];

        private int curChar;

        private int numChars;



        // standard input

        public FastIO() { this(System.in, System.out); }



        public FastIO(InputStream i, OutputStream o) {

            super(o);

            stream = i;

        }



        // file input

        public FastIO(String i, String o) throws IOException {

            super(new FileWriter(o));

            stream = new FileInputStream(i);

        }



        // throws InputMismatchException() if previously detected end of file

        private int nextByte() {

            if (numChars == -1) {

                throw new InputMismatchException();

            }

            if (curChar >= numChars) {

                curChar = 0;

                try {

                    numChars = stream.read(buf);

                } catch (IOException e) {

                    throw new InputMismatchException();

                }

                if (numChars == -1) {

                    return -1;  // end of file

                }

            }

            return buf[curChar++];

        }



        // to read in entire lines, replace c <= ' '

        // with a function that checks whether c is a line break

        public String next() {

            int c;

            do {

                c = nextByte();

            } while (c <= ' ');



            StringBuilder res = new StringBuilder();

            do {

                res.appendCodePoint(c);

                c = nextByte();

            } while (c > ' ');

            return res.toString();

        }



        public int nextInt() {  // nextLong() would be implemented similarly

            int c;

            do {

                c = nextByte();

            } while (c <= ' ');



            int sgn = 1;

            if (c == '-') {

                sgn = -1;

                c = nextByte();

            }



            int res = 0;

            do {

                if (c < '0' || c > '9') {

                    throw new InputMismatchException();

                }

                res = 10 * res + c - '0';

                c = nextByte();

            } while (c > ' ');

            return res * sgn;

        }



        public double nextDouble() { return Double.parseDouble(next()); }

    }



    public static void main(String[] args) {

        FastIO io = new FastIO();

        int t = io.nextInt();

        for(int i = 0; i < t; i++) {

            int n = io.nextInt();

            String s = io.next();



            int[] pre = new int[n+1];

            Map<Integer, Integer> map = new HashMap<>();

            map.put(0, 1);

            long count = 0;

            for(int j = 1; j <= n; j++) {

                pre[j] = pre[j-1] + Integer.parseInt(s.substring(j-1, j)) - 1;

                if(map.containsKey(pre[j])) {

                    count += map.get(pre[j]);

                }

                map.put(pre[j], map.getOrDefault(pre[j], 0) + 1);

            }



            io.println(count);

        }

        io.close();

    }

}


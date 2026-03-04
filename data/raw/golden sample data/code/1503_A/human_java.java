import java.io.*;

import java.util.*;



import static java.lang.Math.*;



public class Main {

    public static void main(String[] args) {

        new Main().solve(new InputReader(System.in), new PrintWriter(System.out));

    }



    private void solve(InputReader in, PrintWriter pw) {

        int tt = in.nextInt();

        while (tt-- > 0) {

            int n = in.nextInt();

            char[] s = in.next().toCharArray();

            int cnt = 0;

            for (char c : s) {

                cnt += c - '0';

            }

            if (s[0] == '0' || s[n - 1] == '0' || cnt % 2 != 0 || n % 2 != 0) {

                pw.println("NO");

                continue;

            }

            StringBuilder a = new StringBuilder(), b = new StringBuilder();

            int k = 0;

            boolean flip = false;

            for (char c : s) {

                if (c == '1') {

                    char t = k * 2 < cnt ? '(' : ')';

                    a.append(t);

                    b.append(t);

                    k++;

                } else {

                    a.append(flip ? "(" : ")");

                    b.append(flip ? ")" : "(");

                    flip = !flip;

                }

            }

            pw.println("YES\n" + a + "\n" + b);

//            int l = 0;

//            for (char c : s) {

//                char t, w;

//                if (c == '1') {

//                    if (l > 0) {

//                        t = w = ')';

//                        l--;

//                    } else {

//                        t = w = '(';

//                        l++;

//                    }

//                } else {

//                    if (l > 0) {

//                        t = ')';

//                        w = '(';

//                        l--;

//                    } else {

//                        t = '(';

//                        w = ')';

//                        l++;

//                    }

//                }

//                a.append(t);

//                b.append(w);

//            }

//            boolean ok = l == 0 && check("" + a) && check("" + b);

//            if (!ok) {

//                pw.println("NO");

//            } else {

//                pw.println("YES");

//                pw.println(a);

//                pw.println(b);

//            }

        }

        pw.close();

    }



    private boolean check(String a) {

        int t = 0;

        for (char c : a.toCharArray()) {

            if (c == '(') {

                t++;

            } else {

                if (--t < 0) {

                    return false;

                }

            }

        }

        return t == 0;

    }

}



class InputReader {

    private final BufferedReader reader;

    private StringTokenizer tokenizer;



    public InputReader(InputStream stream) {

        reader = new BufferedReader(new InputStreamReader(stream), 32768);

        tokenizer = null;

    }



    public String next() {

        while (tokenizer == null || !tokenizer.hasMoreTokens()) {

            try {

                tokenizer = new StringTokenizer(reader.readLine());

            } catch (IOException e) {

                throw new RuntimeException(e);

            }

        }

        return tokenizer.nextToken();

    }



    public String nextLine() {

        String str;

        try {

            str = reader.readLine();

        } catch (IOException e) {

            throw new RuntimeException(e);

        }

        return str;

    }



    public boolean hasNext() {

        while (tokenizer == null || !tokenizer.hasMoreTokens()) {

            String nextLine = null;

            try {

                nextLine = reader.readLine();

            } catch (IOException e) {

                throw new RuntimeException(e);

            }

            if (nextLine == null)

                return false;

            tokenizer = new StringTokenizer(nextLine);

        }

        return true;

    }



    public int nextInt() {

        return Integer.parseInt(next());

    }



    public long nextLong() {

        return Long.parseLong(next());

    }

}
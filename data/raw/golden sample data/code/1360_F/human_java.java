import java.io.*;

import java.util.*;



import static java.lang.Math.*;



public class Main {

    public static void main(String[] args) {

        new Main().solve(new InputReader(System.in), new PrintWriter(System.out));

    }



    private void solve(InputReader in, PrintWriter pw) {

        int tt = in.nextInt();

        outer:while (tt-- > 0) {

            int n = in.nextInt();

            int m = in.nextInt();



            char[][] s = new char[n][m];

            for (int i = 0; i < n; i++) {

                s[i] = in.next().toCharArray();

            }



            char[] ans = s[0];

            for (int i = 0; i < m; i++) {

                char t = ans[i];

                for (char d = 'a'; d <= 'z'; d++) {

                    ans[i] = d;

                    boolean flag = true;

                    for (int z = 0; z < n; z++) {

                        int cnt = 0;

                        for (int c = 0; c < m; c++) {

                            if (s[z][c] != ans[c]) {

                                cnt++;

                            }

                        }

                        if (cnt > 1) {

                            flag = false;

                            break;

                        }

                    }

                    if (flag) {

                        pw.println(ans);

                        continue outer;

                    }

                }

                ans[i] = t;

            }

            pw.println(-1);

        }

        pw.close();

    }



    private void reverse(int[] a) {

        int n = a.length;

        for (int i = 0; i < n / 2; i++) {

            int tmp = a[i];

            a[i] = a[n - i - 1];

            a[n - i - 1] = tmp;

        }

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



    public int nextInt() {

        return Integer.parseInt(next());

    }



    public int[] nextArray(int n) {

        int[] a = new int[n];

        for (int i = 0; i < n; i++) {

            a[i] = nextInt();

        }

        return a;

    }

}
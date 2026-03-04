import java.io.IOException;

import java.io.BufferedReader;

import java.io.InputStreamReader;

import java.util.StringTokenizer;

import java.io.BufferedWriter;

import java.io.OutputStreamWriter;

import java.util.Stack;

public class Solution {

    static Reader input = new Reader();

    static int n;

    static char[] s;

    static int[] a;

    static int min;

    public static void main(String[] args) throws IOException {

        int t = input.nextInt();

        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(System.out));

        while(t --> 0) {

            n = input.nextInt();

            s = input.next().toCharArray();

            a = new int[n];

            solve();

            writer.write(min + "\n");

            for(int i = 0; i < n; ++i) {

                writer.write(a[i] + " ");

            }

            writer.write("\n");

        }

        writer.flush();

    }

    static void solve() {

        min = 1;

        Stack<Integer> z = new Stack<Integer>(), o = new Stack<Integer>();

        int temp;

        for(int i = 0; i < n; ++i) {

            if(s[i] == '0') {

                if(! o.isEmpty()) {

                    temp = o.pop();

                    a[i] = temp;

                    z.push(temp);

                } else {

                    temp = min++;

                    a[i] = temp;

                    z.push(temp);

                }

            } else {

                if(! z.isEmpty()) {

                    temp = z.pop();

                    a[i] = temp;

                    o.push(temp);

                } else {

                    temp = min++;

                    a[i] = temp;

                    o.push(temp);

                }

            }

        }

        --min;

    }

    static class Reader {

        BufferedReader reader;

        StringTokenizer string;

        public Reader() {

            reader = new BufferedReader(new InputStreamReader(System.in));

        }

        public String next() throws IOException {

            while(string == null || ! string.hasMoreElements()) {

                string = new StringTokenizer(reader.readLine());

            }

            return string.nextToken();

        }

        public int nextInt() throws IOException {

            return Integer.parseInt(next());

        }

        public long nextLong() throws IOException {

            return Long.parseLong(next());

        }

    }

}
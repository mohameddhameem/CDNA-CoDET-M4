import java.io.*;

import java.util.*;



public class CF1303C {

    static String alp = "abcdefghijklmnopqrstuvwxyz";

    public static void main(String[] args) throws IOException{

        BufferedReader in = new BufferedReader(new InputStreamReader(System.in));

        int t = Integer.parseInt(in.readLine());

        Outer:

        for (int x = 0; x < t; x++) {

            String s = in.readLine();

            String result = ""+s.charAt(0);

            for (int i = 1; i < s.length(); i++) {

                char c = s.charAt(i);

                char d = s.charAt(i-1);

                int idx1 = result.indexOf(c);

                int idx2 = result.indexOf(d);

                if (idx1 == idx2) {

                    System.out.println("NO");

                    continue Outer;

                }

                if (idx1 < 0) {

                    if (idx2 == 0) result = c+result;

                    else if (idx2 == result.length()-1) result += c;

                    else {

                        System.out.println("NO");

                        continue Outer;

                    }

                }

                else {

                    if (Math.abs(idx1-idx2) != 1) {

                        System.out.println("NO");

                        continue Outer;

                    }

                }

            }

            for (int i = 0; i < 26; i++) {

                if (!result.contains(""+alp.charAt(i))) result += alp.charAt(i);

            }

            System.out.println("YES");

            System.out.println(result);

        }

    }

}
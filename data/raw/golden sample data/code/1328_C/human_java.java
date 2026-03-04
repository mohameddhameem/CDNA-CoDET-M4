import java.util.*;

import java.io.*;



public class Solution {



    public static Scanner scanner = new Scanner(System.in);



    public static void solve() {

        StringBuilder res = new StringBuilder();

        int cases = scanner.nextInt();



        for (int i = 0; i < cases; i++) {

            int n = scanner.nextInt();

            String num = scanner.next();

            StringBuilder larger = new StringBuilder();

            StringBuilder smaller = new StringBuilder();

            boolean gaveOne = false;



            for (int j = 0; j < n; j++) {

                char c = num.charAt(j);



                if (c == '2') {

                    if (gaveOne) {

                        smaller.append('2');

                        larger.append('0');

                    } else {

                        smaller.append('1');

                        larger.append('1');

                    }

                } else if (c == '0') {

                    smaller.append('0');

                    larger.append('0');

                } else {

                    if (gaveOne) {

                        smaller.append('1');

                        larger.append('0');

                    } else {

                        smaller.append('0');

                        larger.append('1');

                        gaveOne = true;

                    }

                }

            }



            res.append(smaller.toString());

            res.append('\n');

            res.append(larger.toString());

            res.append('\n');

        }



        System.out.println(res.toString());

    }



    public static void main(String[] args) {

        solve();

    }

}
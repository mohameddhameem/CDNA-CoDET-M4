

import java.util.*;



public class Training1 {



    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);

        int t = sc.nextInt();

        for (int k = 0; k < t; k++) {

            String s = sc.next();

            if (Integer.parseInt(s.charAt(s.length() - 1) + "") % 2 == 0) {

                System.out.println("0");

            } else if (Integer.parseInt(s.charAt(0) + "") % 2 == 0) {

                System.out.println("1");

            } else {

                int pos = -1;

                for (int i = 1; i < s.length() - 1; i++) {

                    if (Integer.parseInt(s.charAt(i) + "") % 2 == 0) {

                        pos = i;

                        break;

                    }

                }

                if (pos != -1) {

                    System.out.println("2");

                } else {

                    System.out.println("-1");

                }

            }

        }



    }



}


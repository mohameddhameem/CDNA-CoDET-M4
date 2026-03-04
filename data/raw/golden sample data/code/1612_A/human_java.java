

import java.util.Scanner;



public class Distance {

    public static void points(int x,int y){

        if((x + y) % 2 != 0){

            System.out.println(-1 + " " + -1);

            return;

        }

        else {

            int dis = (x + y) / 2;

            int a;

            int b;

            if(x < y){

                a = x;

                b = dis - a;

            }

            else {

                b = y;

                a = dis - b;

            }

            System.out.println(a + " " + b);

        }

    }



    public static void main(String[] args) {

        Scanner input = new Scanner(System.in);

        int t = input.nextInt();

        for (int i = 0; i < t; i++) {

            int x = input.nextInt();

            int y = input.nextInt();

            points(x,y);

        }

    }

}
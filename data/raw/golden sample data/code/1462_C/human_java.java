// package fareeda.ragab.random;



import java.io.BufferedReader;

import java.io.IOException;

import java.io.InputStreamReader;

import java.util.ArrayList;

import java.util.List;



public class UniqueNumber {

    public static void main(String[] args) throws IOException {

        BufferedReader input = new BufferedReader(new InputStreamReader(System.in));



        int T = Integer.parseInt(input.readLine());

        while ((T--) != 0) {



            int x = Integer.parseInt(input.readLine());



            // base cases

            if (x >= 1 && x <= 9)

                System.out.println(x);

            else if (x > 45) // > 123456789

                System.out.println(-1);

            else {

                List<Integer> arr = new ArrayList<>();

                int digit = 9, sumDigits = 0;

                

                // x = 15

                while (true) {

                    // arr: 9, 6

                    if ((x - sumDigits) > digit) {

                        sumDigits += digit;

                        arr.add(digit--);

                    } else {

                        System.out.print(x - sumDigits);

                        for (int i = arr.size() - 1; i >= 0; i--) {

                            System.out.print(arr.get(i));

                        }

                        System.out.println();

                        break;

                    }

                }

            }

        }

    }

}


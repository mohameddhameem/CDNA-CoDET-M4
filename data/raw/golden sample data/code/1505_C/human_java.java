import java.util.*;

public class fibonacciwords{

    public static void main(String args[]){

        Scanner sc = new Scanner(System.in);

        String s = sc.nextLine();

        int a = 0, b = 0,c = 0;

        int i = 2;

        while(i < s.length()) {

        	a = s.charAt(i-2) - 'A'; 

        	b = s.charAt(i-1) - 'A';

        	c = s.charAt(i) - 'A';

//        	System.out.println(a+" "+b+" = "+c);

        	if(c != (a + b)%26) {

        		System.out.println("NO");

        		return;

        	}

        	i++;

        }

        System.out.println("YES");

        return;

    }

}








import java.io.IOException;

import java.util.*;

import java.util.function.IntUnaryOperator;

import java.util.function.UnaryOperator;

import java.util.stream.Collectors;



public class Main {

    public static Scanner in = new Scanner(System.in);

    Map<String,Integer>m=new HashMap<>();

    public static boolean canConstruct(String target,String[]wordBank)

    {

        boolean []arr=new boolean[target.length()+1];

        arr[0]=true;

        for(int i=0;i<arr.length;i++)

        {

            if(arr[i])

                for(String word:wordBank)

                {

                    int end=word.length()+i;

                    if(end<=target.length())

                    if(target.substring(i,end).equals(word))

                        arr[i+word.length()]=true;

                    if(arr[target.length()])

                        return true;

                }

        }

        return false;

    }

    public static int Max=0;

    public static int MaxRoot(int n,int m,int [][]arr)

    {

        if(n==arr.length||m==arr[0].length)

            return -1000000;

        if(n==arr.length-1&&m==arr[0].length-1)

            return arr[n][m];

        return arr[n][m]+Math.max(MaxRoot(n+1,m,arr),MaxRoot(n,m+1,arr));



    }

    public static String reverse(String s)

    {

        String rev="";

        for(int i=s.length()-1;i>=0;i--)

        {

            rev+=s.charAt(i);

        }

        return rev;

    }

    public static void main(String[] args) {

        int n=in.nextInt(),m=in.nextInt();

        in.nextLine();

        String []str=new String[n];

        String []rev=new String [n];

        for(int i=0;i<n;i++){

            str[i]=in.nextLine();

            rev[i]=reverse(str[i]);

        }

        String ans="";

        for(int i=0;i<n;i++)

            if(str[i].equals(rev[i]))

            {ans+=str[i];break;}

        for(int i=0;i<n;i++)

        {

            for(int j=i+1;j<n;j++)

            {

                if(str[i].equals(rev[j])&&i!=j)

                {ans=str[i]+ans+rev[i];break;}

            }

        }

        System.out.println(ans.length()+"\n"+ans);





    }

}






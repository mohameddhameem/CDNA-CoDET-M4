import java.util.*;

import java.lang.*;

import java.io.*;

public class Main

{

    static int dp[][][][]=new int[75][75][75][75];

	public static void main (String[] args) throws java.lang.Exception

	{

	    Scanner sc=new Scanner(System.in);

	   // BufferedReader br=new BufferedReader(new InputStreamReader(System.in));

	    int t=1;

	   // t=sc.nextInt();

	    //int t=Integer.parseInt(br.readLine());

	    while(--t>=0){

	        int n=sc.nextInt();

	        int m=sc.nextInt();

	        int K=sc.nextInt();

	        int a[][]=new int[n][m];

	        for(int i=0;i<n;i++)for(int j=0;j<m;j++)a[i][j]=sc.nextInt();

	        

	        for(int i=0;i<=n+1;i++)

	        for(int j=0;j<=m+1;j++)

	        for(int l=0;l<=m+1;l++)

	        for(int k=0;k<=K;k++)

	            dp[i][j][l][k]=-1;

	         

	         dp[1][0][0][0]=0;

	        

	        for(int i=1;i<=n;i++){

	            

	            for(int j=1;j<=m;j++){

	                for(int l=0;l<=j;l++){

	                    for(int k=0;k<K;k++){

	                        dp[i][j][l][k]=Math.max(dp[i][j-1][l][k],dp[i][j][l][k]);

	                        if(dp[i][j-1][l][k]!=-1){

	                        dp[i][j][l+1][(k+a[i-1][j-1])%K]=Math.max(dp[i][j][l+1][(k+a[i-1][j-1])%K],dp[i][j-1][l][k]+a[i-1][j-1]);

	                        }

	                    }

	                }

	            }

	          for(int l=0;l<=m/2;l++){  

	            for(int k=0;k<K;k++){

	                 dp[i+1][0][0][k]=Math.max(dp[i][m][l][k],dp[i+1][0][0][k]);

	            }

	          }

	        }

	        System.out.println(Math.max(0,dp[n+1][0][0][0]));

	    }

	    

	}

}
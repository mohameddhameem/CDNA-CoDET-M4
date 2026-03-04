import java.util.*;
import java.io.*;

public class C_AlternatingSubsequence {
    public static void main(String[] args) throws IOException{
        BufferedReader b = new BufferedReader(new InputStreamReader(System.in));
        PrintWriter p = new PrintWriter(System.out);

        int test = Integer.parseInt(b.readLine());
        for(int t = 0; t < test; t++){
            int n = Integer.parseInt(b.readLine());
            StringTokenizer st = new StringTokenizer(b.readLine());
            int[] arr = new int[n];
            for(int i = 0; i < n; i++){
                arr[i] = Integer.parseInt(st.nextToken());
            }

            //loop through the array once
            //for each streak of positive/negative, pick the maximum
            int[] sequence = new int[n];
            for(int i = 0; i < n;){
                if(arr[i] > 0){
                    int counter = i+1;
                    int max = arr[i];
                    while(counter <= n-1 && arr[counter] > 0){
                        max = Math.max(max, arr[counter]);
                        counter++;
                    }
                    sequence[i] = max;
                    i = counter;
                }else{
                    int counter = i+1;
                    int min = arr[i];
                    while(counter <= n-1 && arr[counter] < 0){
                        min = Math.max(min, arr[counter]);
                        counter++;
                    }
                    sequence[i] = min;
                    i = counter;
                }
            }
            long answer = 0;
            for(int i = 0; i < n; i++){
                //p.print(sequence[i] + " ");
                answer+=sequence[i];
            }
            //p.println();
            p.println(answer);
        }

        b.close();
        p.close();
    }
}


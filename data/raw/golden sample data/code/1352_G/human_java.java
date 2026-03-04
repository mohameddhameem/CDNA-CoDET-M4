import java.io.*;

import java.util.*;



public class G {

	public static void main(String[] args) {

		FastScanner fs=new FastScanner();

        PrintWriter out = new PrintWriter(System.out);

        int T = fs.nextInt();

        for (int tt=0; tt<T; tt++) {

			int n = fs.nextInt();

			if(n==2 || n==3) {

				out.println(-1);

				continue;

			}

			if(n==1) {

				out.println(1);

				continue;

			}

			if(n==4) {

				out.println("2 4 1 3");

				continue;

			}

			int[] arr = new int[n];

			int k = 0;

			for(int i=2; i<=n; i+=2) {

				arr[k++] = i;

			}

			int firstOdd = k;

			for(int i=n; i>0; i--) {

				if(i%2==0) continue;

				arr[k++] = i;

			}

			int one = arr[firstOdd];

			int two = arr[firstOdd+1];

			int three = arr[firstOdd+2];

			if(n%2==0) {

				arr[firstOdd] = two;

				arr[firstOdd+1] = one;

			} else {

				arr[firstOdd] = three;

				arr[firstOdd+1] = one;

				arr[firstOdd+2] = two;

			}

			





			

			// fix

			// for(int i=0; i<n; i+=2) {

			// 	arr[i] = k++;

			// }

			for(int i: arr) {

				out.print(i+" ");

			}

			out.println();

        }

        out.close();

	}



	static final Random random=new Random();

 

    static void ruffleSort(int[] a) {

        int n=a.length;//shuffle, then sort

        for (int i=0; i<n; i++) {

            int oi=random.nextInt(n), temp=a[oi];

            a[oi]=a[i]; a[i]=temp;

        }

        Arrays.sort(a);

    }

    

	static class FastScanner {

		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));

		StringTokenizer st = new StringTokenizer("");

		String next() {

			while (!st.hasMoreTokens()) {

				try {

					st = new StringTokenizer(br.readLine());

				} catch (IOException e) {

					e.printStackTrace();

				}

			}

			return st.nextToken();

		}



		int nextInt() {

			return Integer.parseInt(next());

		}



		long nextLong() {

			return Long.parseLong(next());

		}



		int[] readArray(int n) {

            int[] a=new int[n];

            for (int i=0; i<n; i++) a[i]=nextInt();

            return a;

        }



	}

}
import java.io.*;

import java.util.*;



public class Solution {

	public static void main(String[] args) throws IOException{

		file_io();

		

		FastScanner sc = new FastScanner();

		int tt = 1;

		tt = sc.nextInt();

		while(tt-->0) {

			int n = sc.nextInt();

			int k = sc.nextInt();

			

			int[] ar = new int[n];

			HashMap<Integer, List<Integer>> map = new HashMap<>(); 

			HashMap<Integer, Integer> occ = new HashMap<>();



			for(int i=0; i<n; i++){

				ar[i] = sc.nextInt();

				map.putIfAbsent(ar[i], new ArrayList<>());

				map.get(ar[i]).add(i);

			}



			int cv = 1;

			int[] colors = new int[n];

			for(List<Integer> list: map.values()) {

				int sz = list.size();

				if(sz>=k) {

					for(int i=0; i<k; i++) {

						colors[list.get(i)] = i+1;

						occ.put(i+1, occ.getOrDefault(i+1, 0)+1);

					}

				}

				else {

					for(int i=0; i<sz; i++) {

						if(cv>k) cv = 1;

						occ.put(cv, occ.getOrDefault(cv, 0)+1);

						colors[list.get(i)] = cv++;

					}

				}

			}

			

			int min = Integer.MAX_VALUE;

			for(int v: occ.values()) 

				min = Math.min(v, min);



			for(int i=0; i<n; i++) {

				int co = occ.getOrDefault(colors[i], 0);

				if(colors[i]!=0 && co>min) {

					occ.put(colors[i], co-1);

					colors[i] = 0;

				}

			}



			for(int i=0; i<n; i++)

				System.out.print(colors[i]+" ");

			System.out.println();

		}

	}



	public static int solve() {

		//write your code

		return 0;

	}





























    public static int mini(int... A) {

    	return Arrays.stream(A).min().getAsInt();

    }



    public static int maxi(int... A) {

    	return Arrays.stream(A).max().getAsInt();

    }





	public static int[] prefix_function(char[] s) {

	    int n = s.length;

	    int[] pi = new int[n];

	    for (int i = 1; i < n; i++) {

	        int j = pi[i-1];

	        while (j > 0 && s[i] != s[j])

	            j = pi[j-1];

	        if (s[i] == s[j])

	            j++;

	        pi[i] = j;

	    }

	    return pi;

	}





	static class FastScanner {

		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));

		StringTokenizer st = new StringTokenizer("");

		String next() {

			while(!st.hasMoreTokens()) {

				try {

					st = new StringTokenizer(br.readLine());

				}

				catch (Exception e) {

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



		double nextDouble() {

			return Double.parseDouble(next());

		}

	}



	public static void file_io() {

		try {

			System.setIn(new FileInputStream("input.txt"));

			System.setOut(new PrintStream(new FileOutputStream("output.txt")));

		} catch(Exception e) {

			System.err.println("Error");

		}

	}

}
import java.util.*;

import java.io.*;



public class aaaaaaaaaaaaaaaa {

	public void run() throws Exception {

		FastReader file = new FastReader();

		int times = file.nextInt();

		for (int asdf = 0; asdf < times; asdf++) {

			int n = file.nextInt();

			int[] x = new int[n];

			boolean pos = false, neg = false, even = false;

			int sum = 0;

			for (int i = 0; i < n; i++) {

				x[i]= file.nextInt();

				sum += x[i];

			}

			for (int i= 0; i < n - 1; i++) {

				if (x[i] == x[i+1]) {

					if (x[i] == -1) neg = true;

					else pos = true;

				}

				else even = true;

			}

			if (neg) {

				System.out.println(sum + 4);

			}

			else if (even) {

				System.out.println(sum);

			}

			else {

				System.out.println(sum - 4);

			}

		}

	}



	public static void main(String[] args) throws Exception {

		new aaaaaaaaaaaaaaaa().run();

	}



	static class FastReader {

		BufferedReader br;

		StringTokenizer st;



		public FastReader() {

			br = new BufferedReader(new InputStreamReader(System.in));

		}



		String next() {

			while (st == null || !st.hasMoreElements()) {

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



		double nextDouble() {

			return Double.parseDouble(next());

		}



		String nextLine() {

			String str = "";

			try {

				str = br.readLine();

			} catch (IOException e) {

				e.printStackTrace();

			}

			return str;

		}

	}

}


//Utilities

import java.io.*;

import java.util.*;



public class a {

	static int t;

	static int n, m;

	static long[] a, dp;

	static long res;

	static ArrayList<Integer>[] adj, revAdj;

	static final int MOD = 998244353;

	

	public static void main(String[] args) throws IOException {

		t = in.iscan();

		outer : while (t-- > 0) {

			n = in.iscan(); m = in.iscan();

			a = new long[n+1];

			boolean allZero = true;

			for (int i = 1; i <= n; i++) {

				a[i] = in.lscan();

				if (a[i] != 0) {

					allZero = false;

				}

			}

			adj = new ArrayList[n+1]; revAdj = new ArrayList[n+1];

			for (int i = 0; i <= n; i++) {

				adj[i] = new ArrayList<Integer>();

				revAdj[i] = new ArrayList<Integer>();

			}

			for (int i = 0; i < m; i++) {

				int x = in.iscan(), y = in.iscan();

				adj[x].add(y); revAdj[y].add(x);

			}

			if (allZero) {

				out.println(0);

				continue outer;

			}

			res = 0;

			for (int time = 1; time <= n; time++) {

				long[] b = new long[n+1];

				for (int i = 0; i <= n; i++) {

					b[i] = a[i];

				}

				for (int i = 1; i <= n; i++) {

					if (a[i] > 0) {

						b[i]--;

						for (int u : adj[i]) {

							b[u]++;

						}

					}

				}

				allZero = true;

				for (int i = 0; i <= n; i++) {

					a[i] = b[i];

					if (a[i] != 0) {

						allZero = false;

					}

				}

				if (allZero) {

					res = time;

					out.println(res);

					continue outer;

				}

			}

//			for (int i = 1; i <= n; i++) {

//				out.print(a[i] + " ");

//			}

//			out.println();

			dp = new long[n+1]; Arrays.fill(dp, -1);

			for (int i = 1; i <= n; i++) {

				if (adj[i].isEmpty()) {

					dfs(i);

					break;

				}

			}

//			out.print("DP: ");

//			for (int i = 1; i <= n; i++) {

//				out.print(dp[i] + " ");

//			}

//			out.println();

			res = -1;

			for (int i = 1; i <= n; i++) {

				if (adj[i].isEmpty()) {

					res = (n + dp[i]) % MOD;

				}

			}

			out.println(res);

		}

		out.close();

	}

	

	static long dfs(int idx) {

		if (dp[idx] != -1) {

			return dp[idx];

		}

		long ret = a[idx] % MOD;

		for (int u : revAdj[idx]) {

			ret = (ret + dfs(u)) % MOD;

		}

		return dp[idx] = ret;

	}

	

	static INPUT in = new INPUT(System.in);

	static PrintWriter out = new PrintWriter(System.out);

	private static class INPUT {



		private InputStream stream;

		private byte[] buf = new byte[1024];

		private int curChar, numChars;



		public INPUT (InputStream stream) {

			this.stream = stream;

		}



		public INPUT (String file) throws IOException {

			this.stream = new FileInputStream (file);

		}



		public int cscan () throws IOException {

			if (curChar >= numChars) {

				curChar = 0;

				numChars = stream.read (buf);

			}

			

			if (numChars == -1)

				return numChars;



			return buf[curChar++];

		}



		public int iscan () throws IOException {

			int c = cscan (), sgn = 1;

			

			while (space (c))

				c = cscan ();



			if (c == '-') {

				sgn = -1;

				c = cscan ();

			}



			int res = 0;



			do {

				res = (res << 1) + (res << 3);

				res += c - '0';

				c = cscan ();

			}

			while (!space (c));



			return res * sgn;

		}



		public String sscan () throws IOException {

			int c = cscan ();

			

			while (space (c))

				c = cscan ();



			StringBuilder res = new StringBuilder ();



			do {

				res.appendCodePoint (c);

				c = cscan ();

			}

			while (!space (c));



			return res.toString ();

		}



		public double dscan () throws IOException {

			int c = cscan (), sgn = 1;

			

			while (space (c))

				c = cscan ();



			if (c == '-') {

				sgn = -1;

				c = cscan ();

			}



			double res = 0;



			while (!space (c) && c != '.') {

				if (c == 'e' || c == 'E')

					return res * UTILITIES.fast_pow (10, iscan ());

				

				res *= 10;

				res += c - '0';

				c = cscan ();

			}



			if (c == '.') {

				c = cscan ();

				double m = 1;



				while (!space (c)) {

					if (c == 'e' || c == 'E')

						return res * UTILITIES.fast_pow (10, iscan ());



					m /= 10;

					res += (c - '0') * m;

					c = cscan ();

				}

			}



			return res * sgn;

		}



		public long lscan () throws IOException {

			int c = cscan (), sgn = 1;

			

			while (space (c))

				c = cscan ();



			if (c == '-') {

				sgn = -1;

				c = cscan ();

			}



			long res = 0;



			do {

				res = (res << 1) + (res << 3);

				res += c - '0';

				c = cscan ();

			}

			while (!space (c));



			return res * sgn;

		}



		public boolean space (int c) {

			return c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == -1;

		}

	}



	public static class UTILITIES {



		static final double EPS = 10e-6;

		

		public static void sort(int[] a, boolean increasing) {

			ArrayList<Integer> arr = new ArrayList<Integer>();

			int n = a.length;

			for (int i = 0; i < n; i++) {

				arr.add(a[i]);

			}

			Collections.sort(arr);

			for (int i = 0; i < n; i++) {

				if (increasing) {

					a[i] = arr.get(i);

				}

				else {

					a[i] = arr.get(n-1-i);

				}

			}

		}

		

		public static void sort(long[] a, boolean increasing) {

			ArrayList<Long> arr = new ArrayList<Long>();

			int n = a.length;

			for (int i = 0; i < n; i++) {

				arr.add(a[i]);

			}

			Collections.sort(arr);

			for (int i = 0; i < n; i++) {

				if (increasing) {

					a[i] = arr.get(i);

				}

				else {

					a[i] = arr.get(n-1-i);

				}

			}

		}

		

		public static void sort(double[] a, boolean increasing) {

			ArrayList<Double> arr = new ArrayList<Double>();

			int n = a.length;

			for (int i = 0; i < n; i++) {

				arr.add(a[i]);

			}

			Collections.sort(arr);

			for (int i = 0; i < n; i++) {

				if (increasing) {

					a[i] = arr.get(i);

				}

				else {

					a[i] = arr.get(n-1-i);

				}

			}

		}



		public static int lower_bound (int[] arr, int x) {

			int low = 0, high = arr.length, mid = -1;



			while (low < high) {

				mid = (low + high) / 2;



				if (arr[mid] >= x)

					high = mid;

				else

					low = mid + 1;

			}



			return low;

		}



		public static int upper_bound (int[] arr, int x) {

			int low = 0, high = arr.length, mid = -1;



			while (low < high) {

				mid = (low + high) / 2;



				if (arr[mid] > x)

					high = mid;

				else

					low = mid + 1;

			}



			return low;

		}

		

		public static void updateMap(HashMap<Integer, Integer> map, int key, int v) {

			if (!map.containsKey(key)) {

				map.put(key, v);

			}

			else {

				map.put(key, map.get(key) + v);

			}

			if (map.get(key) == 0) {

				map.remove(key);

			}

		}



		public static long gcd (long a, long b) {

			return b == 0 ? a : gcd (b, a % b);

		}



		public static long lcm (long a, long b) {

			return a * b / gcd (a, b);

		}



		public static long fast_pow_mod (long b, long x, int mod) {

			if (x == 0) return 1;

			if (x == 1) return b;

			if (x % 2 == 0) return fast_pow_mod (b * b % mod, x / 2, mod) % mod;



			return b * fast_pow_mod (b * b % mod, x / 2, mod) % mod;

		}



		public static long fast_pow (long b, long x) {

			if (x == 0) return 1;

			if (x == 1) return b;

			if (x % 2 == 0) return fast_pow (b * b, x / 2);



			return b * fast_pow (b * b, x / 2);

		}



		public static long choose (long n, long k) {

			k = Math.min (k, n - k);

			long val = 1;



			for (int i = 0; i < k; ++i)

				val = val * (n - i) / (i + 1);



			return val;

		}



		public static long permute (int n, int k) {

			if (n < k) return 0;

			long val = 1;



			for (int i = 0; i < k; ++i)

				val = (val * (n - i));



			return val;

		}

		

		// start of permutation and lower/upper bound template

		public static void nextPermutation(int[] nums) {

		    //find first decreasing digit

		    int mark = -1;

		    for (int i = nums.length - 1; i > 0; i--) {

		        if (nums[i] > nums[i - 1]) {

		            mark = i - 1;

		            break;

		        }

		    }

		 

		    if (mark == -1) {

		        reverse(nums, 0, nums.length - 1);

		        return;

		    }

		 

		    int idx = nums.length-1;

		    for (int i = nums.length-1; i >= mark+1; i--) {

		        if (nums[i] > nums[mark]) {

		            idx = i;

		            break;

		        }

		    }

		 

		    swap(nums, mark, idx);

		 

		    reverse(nums, mark + 1, nums.length - 1);

		}

		 

		public static void swap(int[] nums, int i, int j) {

		    int t = nums[i];

		    nums[i] = nums[j];

		    nums[j] = t;

		}

		 

		public static void reverse(int[] nums, int i, int j) {

		    while (i < j) {

		        swap(nums, i, j);

		        i++;

		        j--;

		    }

		}

		

		static int lower_bound (int[] arr, int hi, int cmp) {

			int low = 0, high = hi, mid = -1;

			while (low < high) {

				mid = (low + high) / 2;

				if (arr[mid] >= cmp) high = mid;

				else low = mid + 1;

			}

			return low;

		}

	 

		static int upper_bound (int[] arr, int hi, int cmp) {

			int low = 0, high = hi, mid = -1;

			while (low < high) {

				mid = (low + high) / 2;

				if (arr[mid] > cmp) high = mid;

				else low = mid + 1;

			}

			return low;

		} 

		// end of permutation and lower/upper bound template

	}

}
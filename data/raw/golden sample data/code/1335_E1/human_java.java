import java.util.*;

import java.io.*;

public class Practice {

	static boolean multipleTC = true;
	final static int mod = 1000000007;
	final static int mod2 = 998244353;
	final double E = 2.7182818284590452354;
	final double PI = 3.14159265358979323846;
	int MAX = 100005;

	void pre() throws Exception {
	}

	// All the best
	void solve(int TC) throws Exception {
		int n = ni();
		arr = readArr(n);
		prefix = new int[n][27];
		prefix[0][arr[0]]++;
		for (int i = 1; i < n; i++) {
			// copy previous values
			for (int k = 0; k < 27; k++) {
				prefix[i][k] = prefix[i - 1][k];
			}
			prefix[i][arr[i]]++;
		}
		// suffix
		suffix = new int[n][27];
		suffix[n - 1][arr[n - 1]]++;
		for (int i = n - 2; i >= 0; i--) {
			// copy previous values
			for (int k = 0; k < 27; k++) {
				suffix[i][k] = suffix[i + 1][k];
			}
			suffix[i][arr[i]]++;
		}
		// bruteforce
		int max = 1;
		for (int a = 1; a <= 26; a++) {
			for (int b = 1; b <= 26; b++) {
				int cur = get(a, b);
				max = max(max, cur);
			}
		}
		pn(max);

	}

	private int get(int a, int b) {
		int n = arr.length;
		int left = 0, right = n - 1;
		int lefta = 0, righta = 0;
		int mx = 0;
		while (left < right) {
			lefta = prefix[left][a];
			righta = suffix[right][a];
			if (lefta == righta) {
				int cntB = 0;
				if (left < right) {
					cntB = prefix[right - 1][b] - prefix[left][b];
				}
				mx = max(mx, cntB + 2 * lefta);
				left++;
				right--;
			} else if (lefta > righta) {
				right--;
			} else {
				left++;
			}

		}
//		dbg(a, b, mx);
		return mx;
	}

	int prefix[][], suffix[][], arr[];

	int upperBound(ArrayList<Integer> list, int target) {
		int ans = -1;
		int left = 0, right = list.size() - 1;
		while (left <= right) {
			int mid = left + (right - left) / 2;
			if (list.get(mid) >= target) {
				ans = mid;
				right = mid - 1;
			} else {
				left = mid + 1;
			}
		}
		return ans;
	}

	int lowerBound(ArrayList<Integer> list, int target) {
		int ans = -1;
		int left = 0, right = list.size() - 1;
		while (left <= right) {
			int mid = left + (right - left) / 2;
			if (list.get(mid) <= target) {
				ans = mid;
				left = mid + 1;
			} else {
				right = mid - 1;
			}
		}
		return ans;
	}

	int[] readArr(int n) throws Exception {
		int arr[] = new int[n];
		for (int i = 0; i < n; i++) {
			arr[i] = ni();
		}
		return arr;
	}

	void sort(int arr[], int left, int right) {
		ArrayList<Integer> list = new ArrayList<>();
		for (int i = left; i <= right; i++)
			list.add(arr[i]);
		Collections.sort(list);
		for (int i = left; i <= right; i++)
			arr[i] = list.get(i - left);
	}

	void sort(char arr[]) {
		ArrayList<Character> list = new ArrayList<>();
		for (int i = 0; i < arr.length; i++)
			list.add(arr[i]);
		Collections.sort(list);
		for (int i = 0; i < arr.length; i++)
			arr[i] = list.get(i);
	}

	public long max(long... arr) {
		long max = arr[0];
		for (long itr : arr)
			max = Math.max(max, itr);
		return max;
	}

	public int max(int... arr) {
		int max = arr[0];
		for (int itr : arr)
			max = Math.max(max, itr);
		return max;
	}

	public long min(long... arr) {
		long min = arr[0];
		for (long itr : arr)
			min = Math.min(min, itr);
		return min;
	}

	public int min(int... arr) {
		int min = arr[0];
		for (int itr : arr)
			min = Math.min(min, itr);
		return min;
	}

	public long sum(long... arr) {
		long sum = 0;
		for (long itr : arr)
			sum += itr;
		return sum;
	}

	public long sum(int... arr) {
		long sum = 0;
		for (int itr : arr)
			sum += itr;
		return sum;
	}

	String bin(long n) {
		return Long.toBinaryString(n);
	}

	String bin(int n) {
		return Integer.toBinaryString(n);
	}

	static int bitCount(int x) {
		return x == 0 ? 0 : (1 + bitCount(x & (x - 1)));
	}

	static void dbg(Object... o) {
		System.err.println(Arrays.deepToString(o));
	}

	int bit(long n) {
		return (n == 0) ? 0 : (1 + bit(n & (n - 1)));
	}

	int abs(int a) {
		return (a < 0) ? -a : a;
	}

	long abs(long a) {
		return (a < 0) ? -a : a;
	}

	void p(Object o) {
		out.print(o);
	}

	void pn(Object o) {
		out.println(o);
	}

	void pni(Object o) {
		out.println(o);
		out.flush();
	}

	String n() throws Exception {
		return in.next();
	}

	String nln() throws Exception {
		return in.nextLine();
	}

	int ni() throws Exception {
		return Integer.parseInt(in.next());
	}

	long nl() throws Exception {
		return Long.parseLong(in.next());
	}

	double nd() throws Exception {
		return Double.parseDouble(in.next());
	}

	public static void main(String[] args) throws Exception {
		new Practice().run();
	}

	FastReader in;
	PrintWriter out;

	void run() throws Exception {
		in = new FastReader();
		out = new PrintWriter(System.out);
		int T = (multipleTC) ? ni() : 1;
		pre();
		for (int t = 1; t <= T; t++)
			solve(t);
		out.flush();
		out.close();
	}

	class FastReader {
		BufferedReader br;
		StringTokenizer st;

		public FastReader() {
			br = new BufferedReader(new InputStreamReader(System.in));
		}

		public FastReader(String s) throws Exception {
			br = new BufferedReader(new FileReader(s));
		}

		String next() throws Exception {
			while (st == null || !st.hasMoreElements()) {
				try {
					st = new StringTokenizer(br.readLine());
				} catch (IOException e) {
					throw new Exception(e.toString());
				}
			}
			return st.nextToken();
		}

		String nextLine() throws Exception {
			String str = "";
			try {
				str = br.readLine();
			} catch (IOException e) {
				throw new Exception(e.toString());
			}
			return str;
		}
	}
}
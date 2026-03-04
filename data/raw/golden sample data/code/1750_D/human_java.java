import java.io.*;

import java.util.*;



public class D {

	static final int mod = 998244353;

	

	static ArrayList<Integer> getPrimes(int n){

		int d = 2;

		ArrayList<Integer> ans = new ArrayList<>();

		

		while(d*d <= n) {

			boolean found = false;

			while(n%d == 0) {

				n /= d;

				found = true;

			}

			if(found)

				ans.add(d);

			++d;

		}

		if(n != 1){

		      ans.add(n);

		}

		return ans;

	}

	

	public static void solve() {

		int n = fs.nextInt(),m = fs.nextInt();

		int[] a = fs.readArray(n);

		

		int g = a[0];

		for(int i=0;i<n;++i) {

			g = gcd(g , a[i]);

			if(g != a[i]) {

				out.println(0);

				return;

			}

		}

		

		ArrayList<Integer> factors = getPrimes(a[0]);

		TreeMap<Integer, Integer> cal = new TreeMap<>();

		

		for(int i=1;i<n;++i) {

			int factorsLeft = a[i-1] / a[i];

			int till = m / a[i];

			

			ArrayList<Integer> leftPrimes = new ArrayList<>();

			for(int f : factors) {

				if(factorsLeft % f == 0)

					leftPrimes.add(f);

			}

			

			int sz = leftPrimes.size();

			int ans = 0;

			

			for(int mask = 0;mask < (1<<sz) ;++mask) {

				int prod = 1;

				int cnt = 0;

				

				for(int j=0;j<sz;++j) {

					if( (mask & (1<<j)) != 0) {

						prod *= leftPrimes.get(j);

						++cnt;

					}

				}

				

				if(cnt % 2 == 0) {

					ans += till/prod;

				}else {

					ans -= till/prod;

				}

				

			}

			

			cal.put(i, ans);

		}

		

		long ans = 1;

		

		for(int i=1;i<n;++i) {

			ans = ( ans * (long) cal.get(i) ) % mod;

		}

		

		out.println(ans);

	}

	

	public static void main(String[] args) {

		fs=new FastScanner();

		out=new PrintWriter(System.out);

		

		int T=1;

		T = fs.nextInt();

		for (int tt=0; tt<T; tt++) {

			solve();

		}

		out.close();

	}

	

	static FastScanner fs;

	static PrintWriter out;

	static final Random random=new Random();

	

	static class FastScanner {

		BufferedReader br=new BufferedReader(new InputStreamReader(System.in));

		StringTokenizer st=new StringTokenizer("");

		String next() {

			while (!st.hasMoreTokens())

				try {

					st=new StringTokenizer(br.readLine());

				} catch (IOException e) {

					e.printStackTrace();

				}

			return st.nextToken();

		}

		

		int nextInt() {

			return Integer.parseInt(next());

		}

		int[] readArray(int n) {

			int[] a=new int[n];

			for (int i=0; i<n; i++) a[i]=nextInt();

			return a;

		}

		long nextLong() {

			return Long.parseLong(next());

		}

	}

	

	static void swap(int a,int b) {

		int temp = a;

		a = b;

		b = temp;

	}

	

	@SuppressWarnings("unused")

	private static int gcd(int a,int b) {

		if(a < b)

			swap(a,b);

		return b==0 ? a : gcd(b,a%b);

	}

	

	static void ruffleSort(int[] a) {

		int n=a.length;//shuffle, then sort 

		for (int i=0; i<n; i++) {

			int oi=random.nextInt(n), temp=a[oi];

			a[oi]=a[i]; a[i]=temp;

		}

		Arrays.sort(a);

	}



}
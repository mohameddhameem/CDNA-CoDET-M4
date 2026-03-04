// package c1721;

//
// Educational Codeforces Round 134 (Rated for Div. 2) 2022-08-27 07:35
// E. Prefix Function Queries
// https://codeforces.com/contest/1721/problem/E
// time limit per test 2 seconds; memory limit per test 256 megabytes
// public class Pseudo for 'Source should satisfy regex [^{}]*public\s+(final)?\s*class\s+(\w+).*'
//
// You are given a string s, consisting of lowercase Latin letters.
//
// You are asked q queries about it: given another string t, consisting of lowercase Latin letters,
// perform the following steps:
//  1. concatenate s and t;
//  2. calculate the prefix function of the resulting string s+t;
//  3. print the values of the prefix function on positions |s|+1, |s|+2, ..., |s|+|t| (|s| and |t|
//     denote the lengths of strings s and t, respectively);
//  4. revert the string back to s.
//
// The prefix function of a string a is a sequence p_1, p_2, ..., p_{|a|}, where p_i is the maximum
// value of k such that k < i and a[1..k]=a[i-k+1..i] (a[l..r] denotes a contiguous substring of a
// string a from a position l to a position r, inclusive). In other words, it's the longest proper
// prefix of the string a[1..i] that is equal to its suffix of the same length.
//
// Input
//
// The first line contains a non-empty string s (1 <= |s| <= 10^6), consisting of lowercase Latin
// letters.
//
// The second line contains a single integer q (1 <= q <= 10^5)-- the number of queries.
//
// Each of the next q lines contains a query: a non-empty string t (1 <= |t| <= 10), consisting of
// lowercase Latin letters.
//
// Output
//
// For each query, print the values of the prefix function of a string s+t on positions |s|+1,
// |s|+2, ..., |s|+|t|.
//
// Example
/*
input:
aba
6
caba
aba
bababa
aaaa
b
forces
output:
0 1 2 3
1 2 3
2 3 4 5 6 7
1 1 1 1
2
0 0 0 0 0 0

input:
aacba
4
aaca
cbbb
aab
ccaca
output:
2 2 3 1
0 0 0 0
2 2 0
0 0 1 0 1
*/
//

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.lang.invoke.MethodHandles;
import java.util.Arrays;
import java.util.Random;
import java.util.StringTokenizer;

public class C1721E {
  static final int MOD = 998244353;
  static final Random RAND = new Random();

  // Time limit exceeded on test 17
  static int[][] solve(String s, String[] queries) {
    int n = s.length();
    int q = queries.length;
    long t0 = System.currentTimeMillis();

    int[] lps = new int[n + 10];
    // appeared[i] tells whether character i appeared so far
    boolean[] appeared = new boolean[256];
    {
      int len = 0;
      int i = 1;
      lps[0] = 0;
      while (i < n) {
        char c = s.charAt(i);
        if (c == s.charAt(len)) {
          len++;
        } else {
          if (!appeared[c]) {
            len = 0;
          } else {
            while (len > 0 && c != s.charAt(len)) {
              len = lps[len - 1];
            }
            if (c == s.charAt(len)) {
              len++;
            }
          }
        }
        lps[i] = len;
        appeared[c] = true;
        i++;
      }
    }
    // System.out.format("down main lps\n");
    //
    int[][] ans = new int[q][10];
    for (int h = 0; h < q; h++) {
      // Not define t as s + queries[h] to avoid TLE on test 7
      String t = queries[h];
      int m = queries[h].length();
      boolean[] used = new boolean[256];
      System.arraycopy(appeared, 0, used, 0, 256);
      // System.out.format("  h:%d t:%s m:%d\n", h, t, m);
      {
        int i = n;
        int len = lps[n-1];
        while (i < n + m) {
          char c = t.charAt(i - n);
          if (c == (len < n ? s.charAt(len) : t.charAt(len - n))) {
            len++;
          } else {
            if (!used[c]) {
              len = 0;
            } else {
              while (len > 0 && c != (len < n ? s.charAt(len) : t.charAt(len - n))) {
                len = lps[len - 1];
              }
              if (c == s.charAt(len)) {
                len++;
              }
            }
          }
          lps[i] = len;
          used[c] = true;
          i++;
        }
      }
      for (int j = 0; j < m; j++) {
        ans[h][j] = lps[n + j];
      }
    }
    myAssert(System.currentTimeMillis() - t0 < 500);
    return ans;
  }

  static boolean startWith(String s, int k, String p) {
    int m = p.length();
    if (k + p.length() > s.length()) {
      return false;
    }
    for (int i = 0; i < m; i++) {
      if (s.charAt(k + i) != p.charAt(i)) {
        return false;
      }
    }
    System.out.format("s:%s k:%d p:%s\n", s, k, p);
    return true;
  }

  // Longest Prefix also Suffix
  // Knuth-Morris-Pratt(KMP) Pattern Matching (Substring search) O(m+n) time
  // Compute longest prefix suffix values for s.
  // For example:
  //   ABABCABAB
  //   001201234
  //     ^^ ^^^^__ ABAB
  //     || |||___ ABA
  //     || ||____ AB
  //     || |_____ A
  //     ||_______ AB
  //     |________ A
  static int[] computeLongestPrefixSuffixArray(String s) {
    int m = s.length();
    int lps[] = new int[m];

    // length of the previous longest prefix suffix
    int len = 0;
    int i = 1;
    lps[0] = 0;

    // appeared[i] tells whether character i appeared so far
    boolean[] appeared = new boolean[256];

    while (i < m) {
      char c = s.charAt(i);
      if (c == s.charAt(len)) {
        len++;
        lps[i] = len;
        appeared[c] = true;
        i++;
      } else {
        if (!appeared[c]) {
          len = 0;
          lps[i] = 0;
          appeared[c] = true;
          i++;
        } else {
          while (len > 0 && c != s.charAt(len)) {
            len = lps[len - 1];
          }
          if (c == s.charAt(len)) {
            len++;
            lps[i] = len;
            appeared[c] = true;
            i++;
          } else {
            lps[i] = 0;
            appeared[c] = true;
            i++;
          }
        }
      }
    }
    // System.out.format("  lps: %s\n", Utils.trace(lps));
    return lps;
  }

  static boolean test = false;
  static void doTest() {
    if (!test) {
      return;
    }
    long t0 = System.currentTimeMillis();
    int n = 1000000;
    char[] ca = new char[n];
    Arrays.fill(ca, 'a');
    String s = new String(ca);
    int q = 100000;
    String[] queries = new String[q];
    for (int i = 0; i < q; i++) {
      queries[i] = "bbabababb";
    }
    int[][] ans = solve(s, queries);
    output(queries, ans);
    System.out.format("%d msec\n", System.currentTimeMillis() - t0);
    System.exit(0);
  }

  public static void main(String[] args) {
    doTest();
    long t0 = System.currentTimeMillis();
    MyScanner in = new MyScanner();
    String s = in.next();
    int q = in.nextInt();
    String[] queries = new String[q];
    int total = 0;
    for (int i = 0; i < q; i++) {
      queries[i] = in.next();
      total += queries[i].length();
      myAssert(System.currentTimeMillis() - t0 < 500);
    }
    myAssert(System.currentTimeMillis() - t0 < 500);

    if (s.startsWith("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa") && total != 549283) {
      // throw new RuntimeException(s.substring(s.length() - 30) + "_" + q + "_" + queries[0]);
    }

    int[][] ans = solve(s, queries);
    output(queries, ans);
  }

  private static void output(String[] queries, int[][] ans) {
    long t0 = System.currentTimeMillis();
    int q = queries.length;
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < q; i++) {
      int m = queries[i].length();
      for (int j = 0; j < m; j++) {
        sb.append(ans[i][j]);
        sb.append(' ');
        if (sb.length() > 4000) {
          System.out.print(sb.toString());
          sb.setLength(0);
        }
      }
      sb.append('\n');
      myAssert(System.currentTimeMillis() - t0 < 500);
    }
    System.out.print(sb.toString());
  }

  static void myAssert(boolean cond) {
    if (!cond) {
      throw new RuntimeException("Unexpected");
    }
  }

  static class MyScanner {
    BufferedReader br;
    StringTokenizer st;

    public MyScanner() {
      try {
        final String USERDIR = System.getProperty("user.dir");
        String cname = MethodHandles.lookup().lookupClass().getCanonicalName().replace(".MyScanner", "");
        cname = cname.lastIndexOf('.') > 0 ? cname.substring(cname.lastIndexOf('.') + 1) : cname;
        final File fin = new File(USERDIR + "/io/c" + cname.substring(1,5) + "/" + cname + ".in");
        br = new BufferedReader(new InputStreamReader(fin.exists()
            ? new FileInputStream(fin) : System.in));
      } catch (Exception e) {
        br = new BufferedReader(new InputStreamReader(System.in));
      }
    }

    public String next() {
      try {
        while (st == null || !st.hasMoreElements()) {
          st = new StringTokenizer(br.readLine());
        }
        return st.nextToken();
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
    }

    public int nextInt() {
      return Integer.parseInt(next());
    }

    public long nextLong() {
      return Long.parseLong(next());
    }
  }
}

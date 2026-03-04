import java.io.BufferedReader;

import java.io.InputStreamReader;

import java.util.ArrayList;

import java.util.Arrays;

import java.util.Collections;

import java.util.PriorityQueue;

import java.util.StringTokenizer;

public class Main {

	public static void main(String[] args) throws Exception {

		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));

		StringBuilder sb = new StringBuilder();

		int T = Integer.parseInt(br.readLine());

		while(T --> 0) {

			StringTokenizer st = new StringTokenizer(br.readLine(), " ");

			int n = Integer.parseInt(st.nextToken());

			long c = Long.parseLong(st.nextToken());

			int d = Integer.parseInt(st.nextToken());

			Integer[] arr = new Integer[n];

			st = new StringTokenizer(br.readLine(), " ");

			for(int i = 0; i < n; i++) {

				arr[i] = Integer.parseInt(st.nextToken());

			}

			Arrays.sort(arr, Collections.reverseOrder());

			int l = 0;

			int r = d+1;

			while(l < r) {

				int k = (l + r) / 2;

				long tot = 0;

				int m = k+1;

				for(int i = 0; i < d; i++) {

					if(i % m < n) {

						tot += arr[i % m];

					}

				}

				if(tot >= c) {

					l = k+1;

				}

				else {

					r = k;

				}

			}

			if(l == d+1) sb.append("Infinity").append('\n');

			else if(l == 0) sb.append("Impossible").append('\n');

			else sb.append(l-1).append('\n');

		}

		System.out.println(sb);

	}

}
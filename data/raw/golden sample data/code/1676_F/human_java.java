import java.io.BufferedReader;

import java.io.IOException;

import java.io.InputStreamReader;

import java.util.ArrayList;

import java.util.Collections;

import java.util.HashMap;

import java.util.StringTokenizer;



public class Main {

 	public static void main(String arg[]) throws IOException {

 		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));

 		StringBuilder sb = new StringBuilder();

 		int T = Integer.parseInt(br.readLine());

 		while(T --> 0) {

 			StringTokenizer st = new StringTokenizer(br.readLine(), " ");

 			int n = Integer.parseInt(st.nextToken());

 			int k = Integer.parseInt(st.nextToken());

 			HashMap<Integer, Integer> hashmap = new HashMap<>();

 			st = new StringTokenizer(br.readLine(), " ");

 			boolean result = false;

 			ArrayList<Integer> list = new ArrayList<>();

 			for(int i = 0; i < n; i++) {

 				int a = Integer.parseInt(st.nextToken());

 				hashmap.put(a, hashmap.getOrDefault(a, 0)+1);

 				int temp = hashmap.get(a);

 				if(temp == k) {

 					list.add(a);

 					result = true;

 				}

 			}

 			if(!result)

 				sb.append(-1).append('\n');

 			else {

 				Collections.sort(list);

 				int ans = 0;

 				int min = list.get(0);

 				int max = list.get(0);

 				int rmin = min;

 				int rmax = max;

 				for(int i = 0; i < list.size()-1; i++) {

 					if(list.get(i+1)-list.get(i) == 1) {

 						if(min == -1 || max == -1) {

 							min = list.get(i);

 							max = list.get(i+1);

 						}

 						else {

 							max = list.get(i+1);

 						}

 						if(max - min > rmax - rmin) {

 							rmax = max;

 							rmin = min;

 						}

 					}

 					else {

 						min = -1;

 						max = -1;

 					}

 				}

 				sb.append(rmin+" "+rmax).append('\n');

 			}

 		}

 		System.out.println(sb);

 	}

}
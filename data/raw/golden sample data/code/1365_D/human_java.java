import java.util.*;

import java.io.*;

public class HMazeFF {

    public static char[][] map;

    public static boolean[][] visited;

    public static int n;

    public static int m;

    public static boolean ans;

    public static int[] R = {0, -1, 0, 1};

    public static int[] C = {-1, 0, 1, 0};

    static class Node {

        public int x, y;

        public Node(int x, int y) {

            this.x = x;

            this.y = y;

        }

    }

    public static void main(String[] args) throws Exception {

        BufferedReader in = new BufferedReader(new InputStreamReader(System.in));

        int t = Integer.parseInt(in.readLine().trim());

        StringBuilder answer = new StringBuilder();

        for (int i = 0; i < t; i++) {

            ans = true;

            String[] input = in.readLine().trim().split("\\s");

            n = Integer.parseInt(input[0]);

            m = Integer.parseInt(input[1]);

            map = new char[n][m];

            visited = new boolean[n][m];

            int countG = 0;

            for (int j = 0; j < n; j++) {

                String read = in.readLine().trim();

                for (int k = 0; k < m; k++) {

                    map[j][k] = read.charAt(k);

                    if (map[j][k] == 'G') {

                        countG++;

                    }

                }

            }

            for (int j = 0; j < n; j++) {

                for (int k = 0; k < m; k++) {

                    if (!visited[j][k] && map[j][k] == 'B') {

                        checkAdj(j, k);

                    }

                }

            }

            if (ans) {

                int check = checkG(n - 1, m - 1);

                if (check != countG) {

                    ans = false;

                }

            }

            if (ans || countG == 0) {

                answer.append("YES\n");

            } else {

                answer.append("NO\n");

            }

        }

        System.out.print(answer);

    }

    public static int checkG(int x, int y) {

        int track = 0;

        Stack<Node> stack = new Stack<>();

        stack.add(new Node(x, y));

        while (!stack.isEmpty()) {

            Node node = stack.pop();

            int row = node.x;

            int col = node.y;

            //if (!visited[row][col]) {

                for (int i = 0; i < 4; i++) {

                    if ((row + R[i] < n && row + R[i] >= 0 && col + C[i] >= 0 && col + C[i] < m) && !visited[row + R[i]][col + C[i]]) {

                        if (map[row + R[i]][col + C[i]] == 'G' || map[row + R[i]][col + C[i]] == '.') {

                            stack.add(new Node(row + R[i], col + C[i]));

                            visited[row + R[i]][col + C[i]] = true;

                        }

                        if (map[row + R[i]][col + C[i]] == 'G') {

                            track++;

                        }

                    }

                }

            //}

        }

        return track;

    }

    public static void checkAdj(int x, int y) {

        Stack<Node> stack = new Stack<>();

        stack.add(new Node(x, y));

        boolean stop = false;

        while (!stack.isEmpty() && !stop) {

            Node node = stack.pop();

            int row = node.x;

            int col = node.y;

            visited[row][col] = true;

            for (int i = 0; i < 4; i++) {

                if ((row + R[i] < n && row + R[i] >= 0 && col + C[i] >= 0 && col + C[i] < m) && !visited[row + R[i]][col + C[i]]) {

                    if (map[row + R[i]][col + C[i]] == 'G' || (row + R[i] == n - 1 && col + C[i] == m - 1)) {

                        stop = true;

                        ans = false;

                    } else if (map[row + R[i]][col + C[i]] == 'B') {

                        stack.add(new Node(row + R[i], col + C[i]));

                    } else {

                        map[row + R[i]][col + C[i]] = '#';

                    }

                }

            }

        }

    }

}


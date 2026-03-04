

import java.util.*;



public class main {



    public static void main(String[] args) {

        Scanner cin = new Scanner(System.in);

        int n = cin.nextInt();

        int m = cin.nextInt();

        char[][] p = new char[n + 10][m + 10];

        List<String> stringList = new ArrayList<>();

        for (int i = 0; i < n; i++) {

            String row = cin.next();

            stringList.add(row);

        }

        for (int i = 1; i <= n; i++) {

            for (int k = 1; k <= m; k++) p[i][k] = stringList.get(i - 1).charAt(k - 1);

        }

        int ans = 0;

        for (int k = 1, i = 1; i <= n && k <= m; ) {

            if (p[i][k] == '*') ++ans;

            if (k < m) {

                if ((p[i + 1][k] != '*' && p[i][k + 1] != '*')

                        || (p[i + 1][k] == '*' && p[i][k + 1] == '*')

                        || p[i][k + 1] == '*') {

                    ++k;

                } else ++i;

            } else ++i;

            if (i == n && k == m) break;

        }

        if (p[n][m] == '*') ++ans;

        System.out.println(ans);

    }

}


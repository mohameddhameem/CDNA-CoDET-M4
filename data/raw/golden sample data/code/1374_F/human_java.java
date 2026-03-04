import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

public class CyclicShiftsSorting {
    static FastScanner fs = new FastScanner();

    public static void main(String[] args) {
        int t = fs.nextInt();
        while(t-- > 0) {
            int n = fs.nextInt();
            Elem[] elements = new Elem[n], sorted = new Elem[n];
            for (int i = 0; i < n; i++) elements[i] = sorted[i] = new Elem(fs.nextInt());

            Arrays.sort(sorted);
            for (int i = 0; i < n; i++) sorted[i].value = i;

            int invs = 0;
            for (int i = 0; i < n; i++) {
                for (int j = i+1; j < n; j++) {
                    if (elements[i].value > elements[j].value) invs++;
                }
            }

            if ((invs & 1) != 0) {
                // swapping a duplicate's mapping to make invs even
                for (int i = 0; i < n-1; i++) {
                    if (sorted[i].original == sorted[i + 1].original) {
                        // Swap mappings to duplicate originals
                        int temp = sorted[i].value;
                        sorted[i].value = sorted[i+1].value;
                        sorted[i+1].value = temp;
                        invs++;
                        break;
                    }
                }
            }

            if ((invs & 1) != 0) {
                System.out.println("-1");
                continue;
            }

            ArrayList<Integer> result = new ArrayList<>();
            for (int i = 0; i < n-2; i++) {
                for (int j = i+1; j < n; j++) {
                    if (elements[j].value == i) {

                        // i is not at its position yet
                        while (elements[i].value != i) {
                            if (j - i >= 2) {
                                cyclicRightShift(elements, j-2, j, 1);
                                result.add(j-2);
                                j -= 2;
                            } else if (j - i == 1) {
                                cyclicRightShift(elements, j-1, j+1, 2);
                                result.add(j-1);
                                result.add(j-1);
                                j -= 1;
                            }
                        }

                        break;
                    }
                }
            }

            System.out.println(result.size());
            result.forEach(integer -> System.out.print(integer + 1 + " "));
            System.out.print(t > 0 ? "\n" : "");
        }
    }

    static void cyclicRightShift(Elem[] arr, int low, int high, int times) {
        while (times-- > 0) {
            swap(arr, low, high);
            swap(arr, high, high-1);
        }
    }

    static void swap(Elem []arr, int i, int j) {
        Elem temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }

    static class Elem implements Comparable<Elem> {
        int original, value;

        Elem (int original) {
            this.original = original;
        }

        @Override
        public int compareTo(Elem o) {
            return Integer.compare(original, o.original);
        }
    }

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
}

import java.io.PrintWriter;

import java.util.*;



public class G {

    private static final int MAX_LENGTH = 500;



    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);

        PrintWriter pw = new PrintWriter(System.out);



        int tests = sc.nextInt();

        for (int t = 0; t < tests; t++) {

            solve(sc, pw);

        }

        sc.close();

        pw.close();

    }



    static void solve(Scanner sc, PrintWriter pw) {

        String s = sc.next();

        String t = sc.next();

        List<Occurrence> occurrences = new ArrayList<>();

        for (int i = 0; i <= s.length() - t.length(); i++) {

            boolean validStartPosition = true;

            for (int j = 0; j < t.length(); j++) {

                if (s.charAt(i + j) != t.charAt(j)) {

                    validStartPosition = false;

                    break;

                }

            }

            if (validStartPosition) {

                occurrences.add(new Occurrence(i));

            }

        }



        Occurrence finalResult = new Occurrence(MAX_LENGTH);

        occurrences.add(finalResult);





        // we can start from any occurrence that intersect with first occurrence

        for (int i = 0; i < occurrences.size(); i++) {

            Occurrence current = occurrences.get(i);

            if (current.idx < occurrences.get(0).idx + t.length()) {

                current.minMoves = 0;

                current.addWay(1);

            } else break;

        }



        for (int i = 0; i < occurrences.size() - 1; i++) {

            Occurrence current = occurrences.get(i);



            // find first occurrence that does not intersect with current

            int nextNotCrossIdx = i + 1;

            Occurrence nextNotCross = occurrences.get(nextNotCrossIdx);

            while (nextNotCross.idx < current.idx + t.length()) {

                nextNotCrossIdx++;

                nextNotCross = occurrences.get(nextNotCrossIdx);

            }

            // find all occurrences that intersect with nextNotCross (including nextNotCross) and update them

            int nextCrossIdx = nextNotCrossIdx;

            while (nextCrossIdx < occurrences.size()) {

                Occurrence nextCross = occurrences.get(nextCrossIdx);

                if (nextCross.idx < nextNotCross.idx + t.length()) {

                    nextCross.addPredecessor(current);

                    nextCross.minMoves = Math.min(nextCross.minMoves, current.minMoves + 1);

                } else {

                    break;

                }



                nextCrossIdx++;

            }



        }



        // update ways for each occurrence

        occurrences.forEach(current -> current.predecessors.forEach(predecessor-> {

            if (predecessor.minMoves + 1 == current.minMoves) {

                current.addWay(predecessor.getWaysCount());

            }

        }));



        pw.println(finalResult.minMoves + " " + finalResult.waysCount);



    }



    static class Occurrence {

        // start position of occurrences t in string S

        int idx;

        // minimum number of moves (remove occurrences of string t in string s) we should do to achieve current idx

        int minMoves = MAX_LENGTH;

        // list of previous start position from which we can jump to current position

        List<Occurrence> predecessors = new ArrayList<>();

        // number of ways we can achieve current position with minimum moves

        private int waysCount;



        Occurrence(int idx) {

            this.idx = idx;

        }



        void addPredecessor(Occurrence predecessor) {

            predecessors.add(predecessor);

        }



        void addWay(int newWays) {

            waysCount += newWays;

            if (waysCount > 1e9 + 7) {

                waysCount -= 1e9 + 7;

            }

        }



        int getWaysCount() {

            return waysCount;

        }

    }

}


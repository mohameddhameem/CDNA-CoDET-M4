import java.util.Scanner;



public class color {



	public static void main(String[] args) {

		//System.out.println("enter coms:");

		Scanner s = new Scanner(System.in);

		String start = "blue";

		String yo = "blue";

		String l = "lock";

		String[] sc = new String[s.nextInt()];

		s.nextLine();

		for (int i = 0; i < sc.length; i++) {

			sc[i] = s.nextLine();

		}

		outer: for (int k = 0; k < 1; k++) {

			for (int n = 0; n < sc.length; n++) {

				switch (sc[n]) {

				default:

					System.out.println("error");

					break outer;

				case "lock":

					yo = start + l;

					break;

				case "unlock":

					yo = start;

					break;

				case "red":

					if (yo.equals(start + l)) {

						break;

					} else {

						start = "red";

					}

					break;

				case "blue":

					if (yo.equals(start + l)) {



						break;

					} else {

						start = "blue";

					}

					break;

				case "green":

					if (yo.equals(start + l)) {



						break;

					} else {

						start = "green";

					}

					break;

				case "orange":

					if (yo.equals(start + l)) {

						break;

					} else {

						start = "orange";

					}

					break;

				case "yellow":

					if (yo.equals(start + l)) {



						break;

					} else {

						start = "yellow";

					}

					break;

				case "indigo":

					if (yo.equals(start + l)) {



						break;

					} else {

						start = "indigo";

					}

					break;

				case "violet":

					if (yo.equals(start + l)) {

						break;

					} else {

						start = "violet";

					}

					break;

				}

			}

			System.out.println(start);

		}

	}

}
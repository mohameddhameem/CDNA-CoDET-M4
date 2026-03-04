#include<iostream>

using namespace std;

long long a[10];

int main()

{

	int t; cin >> t;

	a[0] = 1;

	for (int i = 1; i <= 9; i++)

	{

		a[i] = a[i - 1] * 10;

	}

	while (t--)

	{

		long long n; cin >> n;

		int i = 9;

		while (i >= 1 && a[i] > n)

		{

			i--;

		}

		cout << n - a[i] << endl;

	}



	return 0;

}
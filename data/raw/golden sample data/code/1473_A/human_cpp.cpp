#include <iostream>

#include<vector>

#include<algorithm>

#define FIO ios_base::sync_with_stdio(false);cin.tie(NULL);cout.tie(NULL);

using namespace std;

void test() {

	int n, d;

	cin >> n >> d;

	vector<int> v(n);

	for (size_t i = 0; i < n; i++)

	{

		cin >> v[i];

	}

	sort(v.begin(), v.end());



	cout << (v[n - 1] <= d || v[0] + v[1] <= d ? "Yes\n" : "No\n");



}



int main() {

	FIO

		int t;

	cin >> t ;

	while (t--) {



		test();



	}

		



}
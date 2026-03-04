#include <bits/stdc++.h>



using namespace std;



void solve()

{

    int x;

    cin >> x;

    if (x > 45)

    {

        cout << "-1\n";

        return;

    }

    int cur = 9, su = 0;

    vector<int> res;

    while (su < x)

    {

        res.push_back(min(cur, x - su));

        su += res.back();

        --cur;

    }

    reverse(res.begin(), res.end());

    for (auto& e : res)

    {

        cout << e;

    }

    cout << '\n';

}



int main()

{

    ios_base::sync_with_stdio(false);

    cin.tie(0);

    int t = 1;

    cin >> t;

    while (t--)

    {

        solve();

    }

}


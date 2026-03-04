#include <bits/stdc++.h>



using namespace std;



int main() {



    int t;



    cin >> t;



    while (t--) {



        string s;



        cin >> s;



        if ((s[s.size() - 1] - '0') % 2 == 0)



            cout << 0 << endl;



        else {



            int i;



            for (i = 0 ; i < s.size() ; i++)



                if ((s[i] - '0') % 2 == 0)



                    break;



            if (i == 0)



                cout << 1 << endl;



            else if (i == s.size())



                cout << -1 << endl;



            else



                cout << 2 << endl;



        }



    }



    return 0;

}


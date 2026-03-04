#include <iostream>

///#include <algorithm>

///#include <vector>

using namespace std;

using namespace std::__cxx11;

int main()

{

    ios_base::sync_with_stdio(0);

    cin.tie(0);

    int n;

    string s, color = "blue", status = "unlock";

    cin >> n;

    for(int i = 1; i <= n; ++i)

    {

        cin >> s;

        if(s == "unlock")

            status = "unlock";

        else if(s == "lock")

            status = "lock";

        else

        {

            if(status == "unlock")

                color = s;

        }

    }

    cout << color;

    return 0;

}
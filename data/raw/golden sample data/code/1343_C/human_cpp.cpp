/* Owner : MBILLAH(!)*/



#include<bits/stdc++.h>

#define int long long

bool sign(int a)

{

        if(a<0) return false;

        return true;

}



using namespace std;

int32_t main()

{

        int t;

        cin >> t;

        while(t--)

        {

        int n;

        cin >> n;

        int ara[n];

        for(int i = 0;i<n;i++)

                cin >> ara[i];



        int cnt = 0;

        vector<int>v;



        v.push_back(ara[0]);

        for(int i = 1;i<n;i++)

        {

                int x = v.back();

                if(x<ara[i] and sign(ara[i]) == sign(x) )

                {

                        v.pop_back();

                        v.push_back(ara[i]);

                }

                else if(sign(ara[i])!=sign(x)) v.push_back(ara[i]);

        }



//        for(auto y : v)

//                cout << y << " ";

//        cout << endl;

        int s = 0;

        s = accumulate(v.begin(),v.end(),s);

        cout << s << endl;

        }



        return 0;

}


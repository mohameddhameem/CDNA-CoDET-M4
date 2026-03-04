#include<bits/stdc++.h>

using namespace std;

typedef long long int lli;

typedef unsigned long long int ulli;

typedef long double ld;

const int INF32 = 1e9 + 6e7;

const lli INF64 = (1LL << 61) + 1e17;

const int MOD = 1e9 + 7;

const lli MH1 = 1022063089, MH2 = 2034417103, MH3 = 1090250123, MH4 = 2024491871;

//const __int128 MHP = 3206430037875328613, MHM = 3681348219576961379;

mt19937 rng32(chrono::steady_clock::now().time_since_epoch().count());

mt19937_64 rng64(chrono::steady_clock::now().time_since_epoch().count());

struct splitmix { // https://codeforces.com/blog/entry/62393

    static uint64_t splitmix64(uint64_t x) {

        x += 0x9e3779b97f4a7c15;

        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;

        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;

        return x ^ (x >> 31);

    }

    size_t operator()(uint64_t x) const {

        static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();

        return splitmix64(x + FIXED_RANDOM);

    }

};

class Fraction {

private:

    lli p;

    lli q;

public:

    Fraction() {

        p = 0;

        q = 0;

    }

    Fraction(lli num) {

        p = num;

        q = 1;

    }

    Fraction(lli num, lli denum) {

        p = num;

        q = denum;

        check();

        signify();

    }

    ~Fraction() {}

    Fraction operator+(Fraction fr2) {

        check();

        fr2.check();

        return Fraction(fr2.q * p + q * fr2.p, q * fr2.q);

    }

    Fraction operator-(Fraction fr2) {

        check();

        fr2.check();

        return Fraction(fr2.q * p - q * fr2.p, q * fr2.q);

    }

    Fraction operator*(Fraction fr2) {

        check();

        fr2.check();

        return Fraction(p*fr2.p, q*fr2.q);

    }

    Fraction operator/(Fraction fr2) {

        check();

        fr2.check();

        return Fraction(p*fr2.q, q*fr2.p);

    }

    Fraction& operator+=(Fraction fr2) {

        (*this) = (*this) + fr2;

        return *this;

    }

    Fraction& operator-=(Fraction fr2) {

        (*this) = (*this) - fr2;

        return *this;

    }

    Fraction& operator*=(Fraction fr2) {

        (*this) = (*this) * fr2;

        return *this;

    }

    Fraction& operator/=(Fraction fr2) {

        (*this) = (*this) / fr2;

        return *this;

    }

    bool operator==(Fraction fr2) {

        check();

        fr2.check();

        return this->p*fr2.q == fr2.p*this->q;

    }

    bool operator!=(Fraction fr2) {

        return !((*this) == fr2);

    }

    friend bool operator<(const Fraction& fr1, const Fraction& fr2);

    friend bool operator<=(const Fraction& fr1, const Fraction& fr2);

    friend bool operator>(const Fraction& fr1, const Fraction& fr2);

    friend bool operator>=(const Fraction& fr1, const Fraction& fr2);

    Fraction operator-() {

        return Fraction(-p, q);

    }

    void check() {

        assert(q != 0);

    }

    void normalize() {

        check();

        lli bcd = __gcd(abs(p), abs(q));

        p /= bcd;

        q /= bcd;

        signify();

    }

    void signify() {

        check();

        if(p == 0) {

            q = 1;

        }

        if(q < 0) {

            p = -p;

            q = -q;

        }

    }

    ld eval() {

        check();

        return (ld)(p)/q;

    }

    lli fpart() {

        check();

        return p/q;

    }

    lli fpartnum() {

        check();

        return p - p/q*q;

    }

    void print() {

        check();

        cout << "[" << p << "/" << q << "] ";

    }

    void println() {

        check();

        cout << "[" << p << "/" << q << "]\n";

    }

};

bool operator<(const Fraction& fr1, const Fraction& fr2) {

    return fr1.p * fr2.q < fr2.p * fr1.q;

}

bool operator<=(const Fraction& fr1, const Fraction& fr2) {

    return fr1.p * fr2.q <= fr2.p * fr1.q;

}

bool operator>(const Fraction& fr1, const Fraction& fr2) {

    return fr1.p * fr2.q > fr2.p * fr1.q;

}

bool operator>=(const Fraction& fr1, const Fraction& fr2) {

    return fr1.p * fr2.q >= fr2.p * fr1.q;

}

class Mint {

private:

    int val;

public:

    Mint() {

        val = 0;

    }

    Mint(lli cval) {

        val = (cval % MOD) + MOD;

        if(val >= MOD) val -= MOD;

    }

    ~Mint() {}

    int binPower(lli a, lli b) {

        if(b < 0) return 0;

        lli res = 1;

        while(b > 0) {

            if(b & 1)

                res = res * a % MOD;

            a = a * a % MOD;

            b >>= 1;

        }

        return (int)res;

    }

    Mint invr(Mint mi2) {

        return Mint(binPower(mi2.val, MOD-2));

    }

    Mint& operator+=(const Mint& mi2) {

		val += mi2.val;

		if (val >= MOD) val -= MOD;

		return *this;

	}

	Mint& operator-=(const Mint& mi2) {

		val -= mi2.val;

		if(val < 0) val += MOD;

		return *this;

	}

	Mint& operator*=(const Mint& mi2) {

		val = (lli)val * mi2.val % MOD;

		return *this;

	}

	Mint& operator/=(const Mint& mi2) {

        val = (lli)val * invr(mi2).val % MOD;

        return *this;

	}

	Mint& operator^=(lli pval) {

        if(pval == -1)

            val = invr(Mint(val)).val;

        else

            val = binPower(val, pval);

        return *this;

	}

    Mint operator+(const Mint& mi2) const {

		return Mint(*this) += mi2;

	}

	Mint operator-(const Mint& mi2) const {

		return Mint(*this) -= mi2;

	}

	Mint operator*(const Mint& mi2) const {

		return Mint(*this) *= mi2;

	}

	Mint operator/(const Mint& mi2) const {

		return Mint(*this) /= mi2;

	}

	Mint operator^(lli pval) const {

		return Mint(*this) ^= pval;

	}

    bool operator==(const Mint& mi2) const {

		return val == mi2.val;

	}

    bool operator!=(const Mint& mi2) const {

		return val != mi2.val;

	}

    Mint& operator++() {

        val += 1;

        if(val >= MOD) val -= MOD;

        return *this;

    }

    Mint& operator--() {

        val -= 1;

        if(val < 0) val += MOD;

        return *this;

    }

    Mint operator++(int) {

        Mint prevval = *this;

        ++(*this);

        return prevval;

    }

    Mint operator--(int) {

        Mint prevval = *this;

        --(*this);

        return prevval;

    }

    friend ostream& operator<<(ostream &out, const Mint &mi);

    friend istream& operator>>(istream &in, Mint &mi);

};

istream& operator>>(istream &in, Mint &mi) {

    long long inp;

    in >> inp;

    mi = Mint(inp);

    return in;

}

ostream& operator<<(ostream &out, const Mint &mi) {

    out << mi.val;

    return out;

}

class DSU {

private:

    vector<int> dsp;

    vector<int> dsz;

    int d_n;

public:

    DSU() {

        d_n = 0;

    }

    ~DSU() {}

    void build_dsu(int n) {

        d_n = n;

        dsp.resize(d_n + 1);

        dsz.resize(d_n + 1);

        for(int i = 0; i <= d_n; i++) {

            dsp[i] = i;

            dsz[i] = 1;

        }

    }

    int find_root(int v) {

        if(v == dsp[v]) return v;

        return dsp[v] = find_root(dsp[v]);

    }

    void unite_sets(int a, int b) {

        a = find_root(a);

        b = find_root(b);

        if(a == b) return;



        if(dsz[a] > dsz[b])

            swap(a, b);



        dsz[b] += dsz[a];

        dsp[a] = b;

    }

};

template<typename T>

class Fenwick {

private:

    vector<T> fenw;

    vector<T> arrfw;

    int f_n;

public:

    Fenwick() {

        f_n = 0;

    }

    Fenwick(int n) {

        f_n = n;

        fenw = vector<T>(f_n + 1, 0);

        arrfw = vector<T>(f_n + 1, 0);

    }

    ~Fenwick() {}

    T sump(int i) {

        T ans = 0;

        while(i >= 0) {

            ans += fenw[i];

            i = (i & (i + 1)) - 1;

        }

        return ans;

    }

    void incv(int i, T d) {

        arrfw[i] += d;

        while(i <= f_n) {

            fenw[i] += d;

            i = (i | (i + 1));

        }

    }

    void setv(int i, T v) {

        incv(i, v - arrfw[i]);

    }

    T getv(int i) {

        return arrfw[i];

    }

    T sumr(int l, int r) {

        return sump(r) - sump(l - 1);

    }

};

signed main() {

    ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0);

    cout << fixed << setprecision(15);

    srand(time(0));

    //freopen("input.txt", "r", stdin);

    //freopen("output.txt", "w", stdout);

    cout << "BucketPotato\n";



    return 0;

}


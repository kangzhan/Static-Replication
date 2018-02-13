//康展    1701210012
//衍生工具模型期末大作业程序代码第二部分：代码实现部分
//该文件对头文件"Option-Function.h"中声明的函数、类以及类的方法进行了定义
//该部分的核心函数是亚式期权半静态复制的定价函数double Static_Replication(double S0, double K, int T, double r, double q, double sigma, double stepsize)
//我们对上述函数进行了详细的注释说明

#include <cmath>  
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <limits>
#include <time.h> 
#include <random>
#include "Option-Function.h"

using namespace std;
const double OneOverRootTwoPi = 0.398942280401433;//定义常量来存储1/sqrt(2pi)的值

//第一种随机数生成函数，运用了C++11的特性，但是生成速度较慢
double GetGaussRandn1()
{
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> normal(0, 1);
    return normal(gen);
}

//第二种随机数生成函数，运用了常规的方法
double GetGaussRandn2()
{
    double result;
    double x;
    double y;
    double sizeSquared;
    do
    {
        x = 2.0*rand() / static_cast<double>(RAND_MAX) - 1;
        y = 2.0*rand() / static_cast<double>(RAND_MAX) - 1;
        sizeSquared = x * x + y * y;
    } while
        (sizeSquared >= 1.0);
    result = x * sqrt(-2 * log(sizeSquared) / sizeSquared);
    return result;
}

//正态分布的密度值计算函数
double NormalDensity(double x)
{
    return OneOverRootTwoPi * exp(-x * x / 2);
}

//参考教材中的正态分布累积函数的实现
double CumulativeNormal(double x)
{
    static double a[5] = { 0.319381530,
        -0.356563782,
        1.781477937,
        -1.821255978,
        1.330274429 };
    double result;
    if (x < -7.0)
        result = NormalDensity(x) / sqrt(1. + x * x);
    else
    {
        if (x > 7.0)
            result = 1.0 - CumulativeNormal(-x);
        else
        {
            double tmp = 1.0 / (1.0 + 0.2316419*fabs(x));
            result = 1 - NormalDensity(x)*
                (tmp*(a[0] + tmp * (a[1] + tmp * (a[2] +
                    tmp * (a[3] + tmp * a[4])))));
            if (x <= 0.0)
                result = 1.0 - result;
        }
    }
    return result;
}

//产生股票路径
double* Priceroad(double s0, int T, double r, double sigma)
{
    double *s = new double[T + 1];
    *s = s0;
    for (int i = 0; i < T; i++) {
        *(s + i + 1) = *(s + i)*exp((r - 0.5*sigma*sigma) + sigma * GetGaussRandn2());
    }
    return s;
}

//定义三维数组模板类，其中三维数组是在二维数组的基础上定义得到的，二维数组实在一维数组的基础上得到的
//该类实现了动态分配数组大小，该模板类的其余功能与普通三维数组double** P的用法是类似的
//该类的定义用到了嵌套定义
template <class T>
class CArray3D {
    template <class T>
    class CArray2D {
        template <class T>
        class CArray1D {
        public:
            CArray1D() :p(NULL) {}//一维数组类CArray1D的构造函数
            void set(T a)
            {
                p = new T[a];//为一维数组类CArray1D分配存储空间
                _a = a;
            }
            inline T& operator[](long elem) const//重载运算符[]使得该数组类可以和用[]进行下标访问
            {
                return p[elem];
            }
            ~CArray1D()//一维数组类CArray1D的析构函数
            {
                delete[] p;
            };
            T* p;//一维数组类指向第一个元素的指针
            T _a;//一维数组类的长度
        };
    public:
        CArray2D() :p(NULL) {}//二维数组类CArray2D的构造函数
        void set(T a, T b) {//为二维数组类CArray2D分配空间
            p = new CArray1D<T>[a];
            for (int i = 0; i < a; i++) {
                p[i].set(b);
            }
            _b = b;
        }
        inline CArray1D<T>& operator[](long elem) const//重载运算符[]使得二维数组类CArray2D可以下标访问
        {
            return p[elem];
        }

        ~CArray2D()//二维数组类CArray2D的析构函数
        {
            delete[] p;
        }

        CArray1D<T>* p;//二维数组类CArray2D的成员变量————指向一个一维数组类
        T _b;
    };

public:
    CArray3D(T a, T b, T c) {//为三维数组分配空间
        p = new CArray2D<T>[a];
        for (int i = 0; i < a; i++) {
            p[i].set(b, c);
        }
        _c = c;
    }

    inline CArray2D<T>& operator[](long elem) const//重载[]使得三维数组类可以下标访问
    {
        return p[elem];
    }
    ~CArray3D()//三维数组类的析构函数
    {
        delete[] p;
    }

    CArray2D<T>* p;//三维数组类的成员变量————二维数组指针
    T _c;
};

//欧式看涨期权的定价公式
double Call_Price(double S, double E, int T, double r, double q, double sigma)
{
    if (S < 0.000001)
        return 0;
    if (E < 0.000001)
        return S*exp(-1 * r*T);
    double d1, d2;
    d1 = (log(S / E) + (r - q + 0.5*sigma*sigma)*(T)) / (sigma*sqrt(T));
    d2 = d1 - sigma * sqrt(T);
    return S * exp(-q * T)*CumulativeNormal(d1) - E * exp(-r * T)*CumulativeNormal(d2);
}

//亚式期权半静态复制的定价函数
double Static_Replication(double S0, double K, int T, double r, double q, double sigma, double stepsize)
{
    if (T == 1)//若到期日为1，则亚式期权等价于一个欧式看涨期权
        return Call_Price(S0, K, T, r, q, sigma);
    int A_Size = 2 * S0 / stepsize;//通过步长，确定（A，S）平面横坐标的格子数量
    int S_Size = 2 * S0 / stepsize;//通过步长，确定（A，S）平面纵坐标的格子数量
    double** Position = new double*[T];//二维数组Position用来记录每个时期的持仓数量
    for (int i = 0; i < T; ++i)
    {
        Position[i] = new double[A_Size + 3];
        for (int j = 0; j < A_Size + 3; j++)
            Position[i][j] = 0;//为Position分配空间并赋初值为0
    }
    //构建三维数组Static_Rep，该数组存储了每个时刻每个格点上复制组合的价值
    //通过该数组，我们可以绘制G(t)三维曲面图
    CArray3D<double> Static_Rep(T, A_Size + 1, S_Size + 1);
    //针对T时刻的支付，我们首先构建T-1时刻的复制组合并计算在T-1时刻不同（A，S）下的价格，存储在二维数组Static_Rep[T-1]中
    for (int i = 0; i <= A_Size; i++)
    {
        for (int j = 0; j <= S_Size; j++)
        {
            if (T*K - (T - 1)*i*stepsize <= 0)//若行权价小于0，则等价于持有股票和债券
                Static_Rep[T - 1][i][j] = (j*stepsize + ((T - 1)*i*stepsize - T*K)*exp(-1 * r)) / T;
            else//否则我们持有一定数量的欧式看涨期权
            {
                Static_Rep[T - 1][i][j] = Call_Price(j*stepsize, T*K - (T - 1)*i*stepsize, 1, r, q, sigma) / T;
            }
        }
    }
    //我们通过time+1时刻的Static_Rep[time+1]来构造Static_Rep[time]
    for (int time = T - 2; time >= 1; time--)
    {
        for (int i = 0; i <= A_Size; i++)
        {
            for (int k = 0; k < A_Size + 3; k++)
            {
                Position[time][k] = 0;//首先将持仓组合清零
            }
            for (int m = 0; m <= A_Size; m++)
            {
                int n = (time + 1)*m - time*i;//投影到time+1时刻是一条直线
                if (n == 0)//边界点的处理
                {
                    Position[time][A_Size + 1] = Static_Rep[time + 1][m][0];
                    Position[time][A_Size + 2] = (Static_Rep[time + 1][m + 1][1] - Static_Rep[time + 1][m][0]) / stepsize;
                    Position[time][0] = stepsize*(Static_Rep[time + 1][m + 2][n + 2] + Static_Rep[time + 1][m][n] - 2 * Static_Rep[time + 1][m + 1][n + 1]) / (stepsize*stepsize);
                }
                //在该直线上进行求导，确定复制组合并存储在Position[time]中
                if (n > time & n < A_Size - time)
                {
                    Position[time][n] = stepsize*(Static_Rep[time + 1][m + 1][n + time + 1] + Static_Rep[time + 1][m - 1][n - (time + 1)] - 2 * Static_Rep[time + 1][m][n]) / ((time + 1)*stepsize*stepsize);

                }
                //边界点的处理
                if (n == A_Size)
                {
                    Position[time][n] = stepsize*(Static_Rep[time + 1][m][n] + Static_Rep[time + 1][m - 2][n - 2] - 2 * Static_Rep[time + 1][m - 1][n - 1]) / (stepsize*stepsize);
                }
            }
            //对得到的time时刻的复制组合，在不同的（A，S）下计算该组合的价值，并存储在二维数组Static_Rep[time]中，用来进行下一步的循环
            for (int j = 0; j <= S_Size; j++)
            {
                Static_Rep[time][i][j] = 0;
                //持有期权组合的价值
                for (int k = 0; k <= A_Size; k++)
                {
                    Static_Rep[time][i][j] = Static_Rep[time][i][j] + Position[time][k] * Call_Price(j*stepsize, k*stepsize, 1, r, q, sigma);
                }
                Static_Rep[time][i][j] += Position[time][A_Size + 1] * exp(-1 * r);//持有债券的价值
                Static_Rep[time][i][j] += Position[time][A_Size + 2] * j*stepsize;//持有标的股票的价值
            }
        }
    }
    //通过以上运算我们得到了1时刻的复制组合价值Static_Rep[1]，下面用这个组合价值来找到在0时刻的复制组合
    for (int m = 0; m <= A_Size; m++)
    {
        int n = m;
        if (n == 0)//边界点的处理
        {
            Position[0][A_Size + 1] = Static_Rep[0 + 1][m][0];//债券持仓
            Position[0][A_Size + 2] = (Static_Rep[0 + 1][1][1] - Static_Rep[0 + 1][0][0]) / stepsize;//标的股票持仓
            Position[0][0] = (Static_Rep[0 + 1][2][2] + Static_Rep[0 + 1][0][0] - 2 * Static_Rep[0 + 1][1][1]) / (stepsize*stepsize);
        }
        //期权持仓
        if (n > 0 && n < A_Size)
        {
            Position[0][n] = stepsize*(Static_Rep[0 + 1][n + 1][n + 1] + Static_Rep[0 + 1][n - 1][n - 1] - 2 * Static_Rep[0 + 1][n][n]) / (stepsize*stepsize);

        }
        if (n == A_Size)//边界点的处理
        {
            Position[0][n] = stepsize*(Static_Rep[0 + 1][n][n] + Static_Rep[0 + 1][n - 2][n - 2] - 2 * Static_Rep[0 + 1][n - 1][n - 1]) / (stepsize*stepsize);
        }

    }
    //得到0时刻的投资组合之后，通过0时刻的标的股票价格可以求出复制组合的价值，从而得到亚式期权的价格
    //需要注意的是，我们通过这个0时刻的投资组合，可以计算这个复制组合的Delta值和Gamma值，从而得到亚式期权的希腊字母
    double price = 0;
    for (int i = 0; i <= A_Size; i++)
    {
        price = price + Position[0][i] * Call_Price(S0, i*stepsize, 1, r, q, sigma);//累加期权价值
    }
    price += Position[0][A_Size + 1] * exp(-1 * r);//累加债券价值
    price += Position[0][A_Size + 2] * S0;//累加标的股票价值
    return price;
}

//Option类的构造函数
Option::Option(double _S, double _E, int _T, double _r, double _q, double _sigma) :
    S(_S), E(_E), T(_T), r(_r), q(_q), sigma(_sigma) {}

//Vanilla_Call类的构造函数
Vanilla_Call::Vanilla_Call(double _S, double _E, int _T, double _r, double _q, double _sigma) :
    Option(_S, _E, _T, _r, _q, _sigma) {}

//Vanilla_Call类在BS框架下的的定价公式
double Vanilla_Call::Price()
{
    return Call_Price(S, E, T, r, q, sigma);
}

//Vanilla_Call类的Monte-Carlo定价函数
double Vanilla_Call::MC_Price()
{
    int N = 1000000;
    int i;
    double p;
    double plus = 0;
    for (i = 0; i < N; i++)
    {
        double*s = Priceroad(S, T, r, sigma);
        p = s[T] > E ? s[T] - E : 0;
        plus += p;
    }
    return plus / N;
}

//Vanilla_Call类的Delta值
double Vanilla_Call::Delta()
{
    return  exp(-q * T)*CumulativeNormal((log(S / E) + (r - q + 0.5*sigma*sigma)*(T)) / (sigma*sqrt(T)));
}

//Vanilla_Call类的Gamma值
double Vanilla_Call::Gamma()
{
    double d1 = (log(S / E) + (r - q + 0.5*sigma*sigma)*(T)) / (sigma*sqrt(T));
    return  exp(-q * T)*NormalDensity(d1) / (sigma*S*sqrt(T));
}

//亚式期权类Asian_Call的构造函数
Asian_Call::Asian_Call(double _S, double _E, int _T, double _r, double _q, double _sigma) :
    Option(_S, _E, _T, _r, _q, _sigma) {}

//亚式期权类Asian_Call的半静态复制定价函数
double Asian_Call::Rep_Price()
{
    double stepsize = 1;//定义步长
    return Static_Replication(S, E, T, r, q, sigma, stepsize);//半静态复制法报价
}

//亚式期权类Asian_Call在BS框架下的定价公式，由于没有解析解且虚函数必须实现，我们返回Monte-Carlo的计算结果
double Asian_Call::Price()
{
    return this->MC_Price();
}

//亚式期权类Asian_Call的Monte-Carlo定价结果
double Asian_Call::MC_Price()
{
    int N = 1000000;
    double p = 0;
    double plus = 0;
    for (int i = 0; i < N; i++)
    {
        double* s = Priceroad(S, T, r, sigma);
        double A = 0;
        for (int j = 1; j < T + 1; j++)
            A += s[j];
        A = A / T;
        p = A > E ? A - E : 0;
        plus += p;
    }
    return plus / N;
}

//亚式期权类Asian_Call的Delta值
double Asian_Call::Delta()
{
    double d = 1;
    double S1 = this->MC_Price();
    this->S = S + d;
    double S2 = this->MC_Price();
    return (S2 - S1) / d;
}

//亚式期权类Asian_Call的Gamma值
double Asian_Call::Gamma()
{
    double d = 1;
    double S1 = this->MC_Price();
    this->S = S - d;
    double S2 = this->MC_Price();
    this->S = S + 2 * d;
    double S3 = this->MC_Price();
    return (S3 + S2 - 2 * S1) / (d*d);
}

//Up_and_out_Call期权的构造函数
Up_and_out_Call::Up_and_out_Call(double _S, double _E, int _T, double _r, double _q, double _sigma, double _H) :
    Option(_S, _E, _T, _r, _q, _sigma), H(_H) {};

//Up_and_out_Call类在BS框架下的定价公式，由于没有解析解且虚函数必须实现，我们返回Monte-Carlo的计算结果
double Up_and_out_Call::Price()
{
    return this->MC_Price();
}

//Up_and_out_Call类的Monte-Carlo定价结果
double Up_and_out_Call::MC_Price()
{
    int N = 1000000;
    double p = 0;
    for (int i = 0; i < N; i++)
    {
        double* s = Priceroad(S, T, r, sigma);
        for (int j = 1; j < T + 1; j++)
        {
            if (s[j] >= H)
            {
                p += (s[j] - E)*exp(-1 * r*j);
                break;
            }
            if (j == T)
            {
                double u = s[j] > E ? s[j] - E : 0;
                p += u*exp(-1 * r*j);
            }
        }
    }
    return p / N;
}

//Up_and_out_Call类的半静态复制法定价函数
double Up_and_out_Call::Rep_Price()
{
    double stepsize = 0.001;
    if (T == 1)
        return Call_Price(S, E, T, r, q, sigma);
    int A_Size = 2 * S / stepsize;
    int S_Size = 2 * S / stepsize;
    double** Position = new double*[T];
    for (int i = 0; i < T; ++i)
    {
        Position[i] = new double[A_Size + 3];
        for (int j = 0; j < A_Size + 3; j++)
            Position[i][j] = 0;
    }
    CArray3D<double> Static_Rep(T, A_Size + 1, S_Size + 1);
    for (int i = 0; i <= A_Size; i++)
    {

        for (int j = 0; j <= S_Size; j++)
        {
            if (T*E - (T - 1)*i*stepsize <= 0)
                Static_Rep[T - 1][i][j] = (j*stepsize + ((T - 1)*i*stepsize - T*E)*exp(-1 * r)) / T;
            else
            {
                Static_Rep[T - 1][i][j] = Call_Price(j*stepsize, T*E - (T - 1)*i*stepsize, 1, r, q, sigma) / T;
            }
        }
    }
    for (int time = T - 2; time >= 1; time--)
    {
        for (int i = 0; i <= A_Size; i++)
        {
            for (int k = 0; k < A_Size + 3; k++)
            {
                Position[time][k] = 0;
            }
            for (int m = 0; m <= A_Size; m++)
            {
                int n = (time + 1)*m - time*i;
                if (n == 0)
                {
                    Position[time][A_Size + 1] = Static_Rep[time + 1][m][0];
                    Position[time][A_Size + 2] = (Static_Rep[time + 1][m + 1][1] - Static_Rep[time + 1][m][0]) / stepsize;
                    Position[time][0] = stepsize*(Static_Rep[time + 1][m + 2][n + 2] + Static_Rep[time + 1][m][n] - 2 * Static_Rep[time + 1][m + 1][n + 1]) / (stepsize*stepsize);
                }
                if (n > time & n < A_Size - time)
                {
                    Position[time][n] = stepsize*(Static_Rep[time + 1][m + 1][n + time + 1] + Static_Rep[time + 1][m - 1][n - (time + 1)] - 2 * Static_Rep[time + 1][m][n]) / ((time + 1)*stepsize*stepsize);
                }
                if (n == A_Size)
                {
                    Position[time][n] = stepsize*(Static_Rep[time + 1][m][n] + Static_Rep[time + 1][m - 2][n - 2] - 2 * Static_Rep[time + 1][m - 1][n - 1]) / (stepsize*stepsize);
                }
            }
            for (int j = 0; j <= S_Size; j++)
            {
                Static_Rep[time][i][j] = 0;
                for (int k = 0; k <= A_Size; k++)
                {
                    Static_Rep[time][i][j] = Static_Rep[time][i][j] + Position[time][k] * Call_Price(j*stepsize, k*stepsize, 1, r, q, sigma);
                }
                Static_Rep[time][i][j] += Position[time][A_Size + 1] * exp(-1 * r);
                Static_Rep[time][i][j] += Position[time][A_Size + 2] * j*stepsize;
            }
        }
    }
    for (int m = 0; m <= A_Size; m++)
    {
        int n = m;
        if (n == 0)
        {
            Position[0][A_Size + 1] = Static_Rep[0 + 1][m][0];
            Position[0][A_Size + 2] = (Static_Rep[0 + 1][1][1] - Static_Rep[0 + 1][0][0]) / stepsize;
            Position[0][0] = (Static_Rep[0 + 1][2][2] + Static_Rep[0 + 1][0][0] - 2 * Static_Rep[0 + 1][1][1]) / (stepsize*stepsize);
        }
        if (n > 0 && n < A_Size)
        {
            Position[0][n] = stepsize*(Static_Rep[0 + 1][n + 1][n + 1] + Static_Rep[0 + 1][n - 1][n - 1] - 2 * Static_Rep[0 + 1][n][n]) / (stepsize*stepsize);
        }
        if (n == A_Size)
        {
            Position[0][n] = stepsize*(Static_Rep[0 + 1][n][n] + Static_Rep[0 + 1][n - 2][n - 2] - 2 * Static_Rep[0 + 1][n - 1][n - 1]) / (stepsize*stepsize);
        }
    }
    double price = 0;
    for (int i = 0; i <= A_Size; i++)
    {
        if (1) {
            price = price + Position[0][i] * Call_Price(S, i*stepsize, 1, r, q, sigma);
        }
    }
    price += Position[0][A_Size + 1] * exp(-1 * r);
    price += Position[0][A_Size + 2] * S;
    return price;
}
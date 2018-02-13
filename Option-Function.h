//该文件声明了所有需要用到的函数，期权大类Option及其派生类Vanilla_Call、Asian_Call、Up_and_out_Call，以及每个类内部的函数

#include <vector>
using namespace std;

double GetGaussRandn1();//定义第一种随机数生成函数
double GetGaussRandn2();//定义第二种随机数生成函数
double NormalDensity(double x);//计算正态分布密度值的函数
double CumulativeNormal(double x);//计算正态分布累积分布的函数
double* Priceroad(double s0, int T, double r, double sigma);//生成股票路径的函数

//定义期权类Option
//欧式看涨期权类Vanilla_Call、亚式看涨期权类Asian_Call、向上敲出看涨期权类Up_and_out_Call均由Option大类派生出
class Option {
public:
    double S;//标的当前价格
    double E;//执行价
    int T;//到期日
    double r;//无风险利率
    double q;//标的股息派发率
    double sigma;//标的波动率
    Option() {};//Option类的默认构造函数
    Option(double _S, double _E, int _T, double _r, double _q, double _sigma);//重载Option类的构造函数
    virtual double Price() = 0;//Option类的虚函数————定价函数
    virtual double MC_Price() = 0;//Option类的虚函数————Monte-Carlo定价函数
    ~Option() {};//Option类的析构函数
};

//欧式看涨期权类Vanilla_Call，继承自Option类
class Vanilla_Call :public Option
{
public:
    Vanilla_Call() {};//欧式看涨期权类Vanilla_Call的默认构造函数
    Vanilla_Call(double _S, double _E, int _T, double _r, double _q, double _sigma);//重载欧式看涨期权类Vanilla_Call的构造函数
    double Price(); //Vanilla_Call类在BS框架下的的定价公式，这是对Option类的虚函数virtual double Price()的实现
    double MC_Price();//Vanilla_Call类的Monte-Carlo定价函数，这是对Option类的虚函数virtual double MC_Price()的实现
    double Delta();//Vanilla_Call类的Delta值
    double Gamma();//Vanilla_Call类的Gamma值
    ~Vanilla_Call() {};//Vanilla_Call类的析构函数
};

//亚式期权类Asian_Call，继承自Option类
class Asian_Call :public Option
{
public:
    Asian_Call() {};//亚式期权类Asian_Call的默认构造函数
    Asian_Call(double _S, double _E, int _T, double _r, double _q, double _sigma);//重载亚式期权类Asian_Call的构造函数
    double Delta();//亚式期权类Asian_Call的Delta值
    double Gamma();//亚式期权类Asian_Call的Gamma值
    double Price();//亚式期权类Asian_Call在BS框架下的定价公式，由于没有解析解且虚函数必须实现，我们返回Monte-Carlo的计算结果
    double Rep_Price();//亚式期权类Asian_Call的半静态复制法定价函数
    double MC_Price();//亚式期权类Asian_Call的Monte-Carlo定价结果
    ~Asian_Call() {};////亚式期权类Asian_Call的析构函数
};

//向上敲出看涨期权类Up_and_out_Call，继承自Option类
class Up_and_out_Call :public Option
{
public:
    double H;//Up_and_out_Call类的敲出价
    Up_and_out_Call() {};//Up_and_out_Call类的默认构造函数
    Up_and_out_Call(double _S, double _E, int _T, double _r, double _q, double _sigma, double _H);//重载Up_and_out_Call类的构造函数
    double Price();//Up_and_out_Call类在BS框架下的定价公式，由于没有解析解且虚函数必须实现，我们返回Monte-Carlo的计算结果
    double Rep_Price();//Up_and_out_Call类的半静态复制法定价函数
    double MC_Price();//Up_and_out_Call类的Monte-Carlo定价结果
    ~Up_and_out_Call() {};//Up_and_out_Call类的析构函数
};

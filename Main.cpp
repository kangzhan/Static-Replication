//康展    1701210012
//衍生工具模型期末大作业程序代码第三部分：主函数部分
//该文件是Main函数，其中提供了一些关于我们自己定义的"Option-Function.h"接口的使用例子
//这个自己定义的头文件"Option-Function.h"中包含了本部分需要用到的函数和类
//由于不是本部分的重点研究内容，关于Vanilla_Call和Up_and_out_Call期权的例子我已经注释掉，若要运行只需要将注释符号去掉即可

#include "Option-Function.h"
#include<iostream>
#include "time.h"
#include <fstream>
using namespace std;

int main()
{
    /*
    Vanilla_Call vanilla(100, 100, 5, 0, 0, 0.1);
    cout << "香草期权公式价格" << ':' << vanilla.Price() << endl;
    cout << "香草期权Monte-Carlo价格" << ':' << vanilla.MC_Price() << endl;
    cout << "香草期权Delta值" << ':' << vanilla.Delta() << endl;
    cout << "香草期权Gamma值" << ':' << vanilla.Gamma() << endl;
    cout << endl;
    */

    //本部分的核心部分————亚式期权的定价
    Asian_Call asian(100, 100, 5, 0, 0, 0.1);
    cout << "亚式期权Monte-Carlo价格" << ':' << asian.MC_Price() << endl;
    cout << "亚式期权半静态复制法价格" << ':' << asian.Rep_Price() << endl;
    cout << endl;

    /*
    Up_and_out_Call uo(100, 100, 5, 0, 0, 0.1, 105);
    cout << "向上敲出期权Monte-Carlo价格" << ':' << uo.MC_Price() << endl;
    cout << "向上敲出期权半静态复制法价格" << ':' << uo.Rep_Price() << endl;
    cout << endl;
    */

    system("pause");
}
[toc]

# 算法笔记

## 1.高效算法设计(算法经典)

### 1.渐进时间复杂度（最大连续和为例）

【例题】最大连续和问题，给出一个长度为n的序列
$$
{A_1,A_2,...,A_n}
$$
求最大连续和

**分治法** 分治法一般有以下3个步骤

*划分问题*：把问题的实例划分成子问题

*递归求解*：递归解决子问题

*合并问题*：合并子问题的解得到原问题的解

```c++
int maxsum(int* A,int x,int y){
    int v,L,R,maxs;
    if(y-x==1)return A[x];
    int m=x+(y-x)/2;//取中值
    int maxs=max(maxsum(A,x,m),maxsum(A,m,y));
    int v,L,R;
    v=0;L=A[m-1];
    for(int i=m;i>=x;i--)L=max(L,v+=A[i]);
    v=0;R=A[m];
    for(int i=m+1;i<y;i++)R=max(R,v+=A[i]);
    return max(maxs,L+R);
}
```

此处有一个细节`int m=x+(y-x)/2`这是一个相当重要的技巧，因为计算机的`/`的取整是朝0方向[^1]的所以用这样的方式来确保分界点总是靠近区间起点，在本题中并不是必要的，但是在后面的二分查找中是一个重要的技巧

---

### 2.排序与检索

#### 1.归并排序

同样按照分治三步法

```c++
void merge_sort(int* A,int x,int y,int* T){
    if(y-x>1){
        int m=x+(y-x)/2;
        int p=x,q=m,i=x;//这里的p，q分别为两个序列的头指针
        merge_sort(A,x,m,T);
        merge_sort(A,m,y,T);
        while(p<m || q<y){//合并并排序两个序列
            if(q>=y||(p<m && A[p]<=A[q]))T[i++]=A[p++];//从左半数组复制到临时空间
            else T[i++]=A[q++];//从右半数组复制到临时空间
        }
        for(int i=x;i<y;i++)A[i]=T[i];
    }
}
```

代码中的两个条件是关键。首先，只要有一个序列非空就继续合并`while(p<m || q<y)`，其次，`if(q>=y||(p<m && A[p]<=A[q]))`中的写法也很有讲究，如果将后一个条件写在前的话，很有可能造成指针错误，所以这里我们先将`q>=y`写在前，即当一个序列为空时，将剩余的数依次写入临时空间，这里利用了`||`操作符判断前一个条件满足时，则不会计算第二个条件，从而避免了指针越界的问题。

分治时，我们利用了左闭右开区间，这样的技巧十分有用，对于求解范围题目我们可以用于其中，应用心体会<img src="E:\Picture\guibing .jpg" alt="guibing " style="zoom: 50%;" />

此题的理解算花费了很多的时间，主要是对代码中的变量没有完全的理解，这里我们应该结合归并算法的本质，即该算法到底是怎么样将元素进行排序的，这样我们才能快速的理解算法究竟是怎么样对变量进行操作的

---

#### 2.二分查找

二分查找与初中所学的二分法别无二致，下面附上代码（二分查找可以用递归实现，但是一般写成非递归的）

```c++
int bsearch(int* A,int x,int y,int v){
    int m;
    while(x<y){
        m=x+(y-x)/2;
        if(A[m]==v)return m;
        else if(A[m]>v)y=m;
        else x=m+1;
    }
    return -1;
}
```

---

### 3.递归与分治



【例题1】棋盘覆盖问题 有一个
$$
{2^k\times2^k}
$$
的方格棋盘，恰有一个方格是黑色的，其他为白色，你的任务是用三个白方格的L型牌覆盖所有白色方格，每个L型牌都可以随意旋转。棋盘是正方形的，很容易想到分治，把棋盘切为4块，每一块都是
$$
{2^{k-1}\times2^{k-1}}
$$
有黑格的可以用递归解决，那没有黑格的呢？可以构造一个黑格子，递归边界也不难得出，k=1时一块牌子就行。

---

### 4.贪心法

贪心法是一种解决问题的策略。如果策略正确，那么贪心法往往是易于描述，易于实现的。

---

### 5.KMP算法（字符串匹配）



---

### 6.中途相遇法（四数之和）

例子：UVa1152，题意如下

```
给出四个有n个元素的集合,A,B,C,D(n<=4000)，要求分别从中选出一个元素a,b,c,d，令他们的和为0，输出有多少种选法
```

【分析】如果使用四重循环枚举a,b,c,d，那算法的复制度是n的四次方，肯定超时，最好的算法是，一是先将A,B,C,D，两两分组，用双重循环枚举每个a+b，c+d，然后将他们放到两个数组SumA，SumB中，此时有两种解法，使用二分查找，在数组SumB中寻找与SumA中的值相加为0的情况，二是使用map容器，利用map容器统计不同的a+b的对数，然后再在c+d中找到与他们相加等于0的解。这两种方法的复杂度都是
$$
O(n^2logn)
$$
同时，二分查找的部分可以使用STL库中自带的`lower_bound()`和`upper_bound()`来实现，同时也可以自己实现，因此有两种方法

以下附上代码

1.使用`lower_bound()`和`upper_bound()`来实现

```c++
int main(){
    int T;
    scanf("%d", &T);
    while(T--) {
        long long cnt = 0;
        scanf("%d", &n);
        for(int i = 0; i < n; i++)
            scanf("%d%d%d%d", &a[i], &b[i], &c[i], &d[i]);
        int dex = 0;
        for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            sum[dex++] = a[i] + b[j];
        sort(sum, sum+dex);
        for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++) {
            cnt += upper_bound(sum, sum + dex, -c[i]-d[j]) - lower_bound(sum, sum + dex, -c[i] - d[j]);
            //这里很好理解，如果存在-(c[i]+d[j])的值，那么upper_bound-lower_bound必定为1，若不存在，则为0。因为若该值的下标为i则upper_bound返回的下标为i+1，而lower_bound返回为i，若该值不存在，则他们的返回值都为相同的值
        }
        printf("%lld\n", cnt);
        if(T)   printf("\n");
    }
    return 0;
}
```



2.使用undered_map实现

```c++
#include<bits/stdc++.h>
using namespace std;
const int maxn=4005;
int A[maxn], B[maxn], C[maxn], D[maxn], sumAB[maxn];
int main() {
    int T, n;
    scanf("%d", &T);
    for (int k=0; k < T; k ++) {
        if (k != 0) puts("");
        scanf("%d", &n);
        for (int i=0; i < n; i ++) scanf("%d %d %d %d", &A[i], &B[i], &C[i], &D[i]);
        unordered_map<int, int> mp;
        for (int i=0; i < n; i ++) // A+B
            for (int j=0; j < n; j ++) mp[A[i]+B[j]] ++; // 计算每个和出现次数
        int ans=0;
        for (int i=0; i < n; i ++) // C+D
            for (int j=0; j < n; j ++) ans += mp[-(C[i]+D[j])];
        printf("%d\n", ans);
    }
    return 0;
}
```

这里使用的是`undered_map`而不是`map`为什么呢，因为前者的查找非常迅速，而后者就相对缓慢，map一般用于对元素有排序要求的情况下，而且map的空间占用较大

---



## 2.算法练习（牛客竞赛）

### 1.语法入门

#### 1.指纹锁（对set重载括号运算符）

链接：https://ac.nowcoder.com/acm/contest/19850/L
来源：牛客网

HA实验有一套非常严密的安全保障体系，在HA实验基地的大门，有一个指纹锁。 

​    该指纹锁的加密算法会把一个指纹转化为一个不超过1e7的数字，两个指纹数值之差越小，就说明两个指纹越相似，当两个指纹的数值差≤k时，这两个指纹的持有者会被系统判定为同一个人。
   现在有3种操作，共m个，
 操作1：add x，表示为指纹锁录入一个指纹，该指纹对应的数字为x，如果系统内有一个与x相差≤k的指纹，则系统会忽略这次添加操作
 操作2：del x，表示删除指纹锁中的指纹x，若指纹锁中多个与x相差≤k的指纹，则全部删除，若指纹锁中没有指纹x，则可以忽略该操作，
 操作3：query x，表示有一个持有指纹x的人试图打开指纹锁，你需要设计一个判断程序，返回该人是否可以打开指纹锁（只要x与存入的任何一个指纹相差≤k即可打开锁）。
   初始状态，指纹锁中没有任何指纹。

输入描述:

```
第一行有2个正整数m，k。
接下来m行，每行描述一种操作：add x，del x或query x。
```

输出描述:

```
对于每个query操作，输出一行，包含一个单词“Yes”或“No”，表示该人是否可以打开指纹锁。
```

AC代码

```c++
#include<bits/stdc++.h>
using namespace std;
int m,k;
typedef struct{
    bool operator()(int a,int b)const{
        if(abs(a-b)<=k)return false;
        else return a<b;
    }
}cmp;

int main(){
    ios::sync_with_stdio(false);cin.tie(0);cout.tie(0);
    cin>>m>>k;
    set<int,cmp> st;
    while(m--){
        string s;
        int fig;
        cin>>s>>fig;
        if(s[0]=='a')st.insert(fig);
        else if(s[0]=='d')st.erase(fig);  
        else{
            if(st.count(fig))cout<<"Yes\n";
            else cout<<"No\n";
        }
    }
    system("pause");
}
```

本题主要考察了对set容器的去重规则进行重写，还要注意的点就是读取的性能

---

#### 2.栈与排序

链接：https://ac.nowcoder.com/acm/contest/19850/G
来源：牛客网

给你一个1->n的排列和一个栈，入栈顺序给定，你要在不打乱入栈顺序的情况下，对数组进行从大到小排序 ，当无法完全排序时，请输出字典序最大的出栈序列 

输入描述:

```
第一行一个数n
第二行n个数，表示入栈的顺序，用空格隔开，结尾无空格
```

输出描述:

```
输出一行n个数表示答案，用空格隔开，结尾无空格
```

AC代码

```c++
#include<bits/stdc++.h>
using namespace std;
const int N=1e6+5;
int a[N],maxn[N];
int main(){
    int n;
    cin>>n;
    stack<int> st;
    for(int i=0;i<n;i++)cin>>a[i];//录入数据
    for(int i=n-1;i>=0;i--)maxn[i]=max(a[i+1],maxn[i+1]);//求当前位置是否大于后面的所有数
    for(int i=0;i<n;i++){
        st.push(a[i]);
        while(!st.empty()&&st.top()>maxn[i]){//先判空防止指针越界
            cout<<st.top()<<" ";
            st.pop();
        }
    }
    system("pause");
}
```

本题主要是理解如何利用栈进行排序，本质是将数依次进栈，当当前栈顶的数大于剩下的所有数时出栈，其中关键步骤就是求当前数与后面所有数的最大数的关系，这里的`maxn[N]`数组起到了这样的关键作用，即记录当前位置（除本身外）到剩余未进栈序列的最大的数

---

#### 3.四舍五入（模拟）

链接：https://ac.nowcoder.com/acm/contest/19851/1001?&headNav=acm
来源：牛客网



四舍五入是个好东西。比如你只考了45分，四舍五入后你是50分再四舍五入你就是满分啦！qdgg刚考完拓扑。成绩十分不理想。但老师觉得他每天都很认真的听课很不容易。于是决定给他一个提高成绩的机会：让他的成绩可以在小数点后的任意位置四舍五入（也可以四舍五入为最接近的整数）。
 但是这是有限制的。qdgg只能四舍五入t次。请帮助qdgg找到他在不超过t次四舍五入可获得的最高成绩。请注意，他可以选择不使用全部t次机会。此外，他甚至可以选择完全不对成绩进行四舍五入。
 在这个问题中，使用经典的舍入规则：将数字四舍五入到第n个数字时，必须先看一下数字n + 1，如果小于5，则第n个数字将保持不变，而所有后续数字替换为0。否则，如果n + 1位数大于或等于5，则位置n处的位数将增加1（如果此位数等于9，这也可能会更改其他一些位数），并且随后的所有位数数字将替换为0。最后，所有尾随的零将被丢弃。
 例如，如果将数字1.14舍入到小数点后第一位，则结果为1.1，而如果将1.5舍入到最接近的整数，则结果为2。四舍五入到小数点后第五位的数字1.299996121将得出数字1.3。

输入描述:

```
输入的第一行包含两个整数n和t（1≤b≤2000001 \leq b \leq 2000001≤b≤200000，1≤t≤1091\leq t \leq 10^91≤t≤109）表示小数（含小数点）的长度以及四舍五入的次数。

第二行为一个字符串表示qdgg的初始分数。
```

输出描述:

```
一行表示qdgg能得到的最高分数（请勿输出尾零）
```

AC代码：

```c++
#include<bits/stdc++.h>
using namespace std;
int cnt=0;//进位
string num;
int fun(){
    int ok=1;
    int pos=num.find('.');
    int len=num.size();
    for(int i=pos+1;i<len;i++){//小数部分的进位
        if(num[i]=='x'){break;}
        if(num[i]>='5'&&num[i]!='.'){
            int j=i;
            num[i]='x';
            if(num[j-1]<'9'&&num[j-1]!='.'){num[j-1]++;return ok=0;}
            else{//整数部分
                for(int k=j-2;k>=0;k--)
                    if(num[k]<'9'&&num[k]!='.'){
                        num[k]++;
                        for(int m=k+1;m<pos;m++)num[m]='0';
                        return ok=0;
                    }
                cnt=1;
                for(int m=0;m<pos;m++)num[m]='0';
                return ok=0;
            }
        }
    }
    return ok;
}

int main(){
    int n,t;
    cin>>n>>t;
    cin>>num;
    int pos=0;
    if(num.find('.')!=num.npos)pos=num.find('.');//定位小数点
    else{cout<<num;return 0;}
    
    while(t--){
        if(fun())break;
    }
    if(cnt)cout<<1;
    for(int i=0;i<num.size();i++)
        if(num[i]=='.'&&num[i+1]=='x')break;
        else if(num[i]=='x')break;
        else cout<<num[i];
    system("pause");
    return 0;
}


```

本题不算难题，只是考虑的情况较多，且因为一开始审题不够清楚，存在考虑太多不可能的情景的情况，正确的应该是，认真审题，认真考虑题目中可能存在的多种情况，以本题为例，题目首先说了**在小数点后任意位置四舍五入**所以根本不用考虑整数位的四舍五入，然后是小数位的进位，只有两种可能，在小数位置进为，在整数位置进位，再然后是，在最顶端进位。考虑完这些情况后，本题就可以迎刃而解。模拟类题目一般相对注重细节。

---

#### 4.情诗（字符串子序列问题）

链接：https://ac.nowcoder.com/acm/contest/19851/1010
来源：牛客网

输入描述:

```
共一行：一封若干个字符的情书（大小写不敏感）。
情书不会超过684594个字符（大写、小写字母）。
```

输出描述:

```
共一行：包含一个整数，即iloveyou在情书中作为子序列出现的次数。
由于答案可能很大，请输出对20010905取模后的值。
```

AC代码：

```c++
#include <bits/stdc++.h>
using namespace std;
const int N=20010905;
int main(){
	char c;
	char str[]= "iloveyou";
	int f[9]={0};
	f[0]=1;//初始化
	while(cin>>c){
		for(int i=0;i<8;i++){
			if(c==str[i]||c==str[i]-32)f[i+1]=(f[i+1]+f[i])%N;
		}
	}
	cout<<f[8];
	system("pause");
	return 0;
}
```

用f[i]表示iloveyou匹配了前i个的字符的子序列数，当遇到第i个字符，则之前匹配了前i-1个的字符的子序列都可以变为匹配了前i个的字符的子序列，即
$$
{f[i+1]=f[i+1]+f[i]}
$$
这里其实是用`f[1]`表示`i`的子序列个数，用`f[2]`表示`il`的子序列个数，依次类推。比如当匹配到`l`时，之前的每一个`i`都可以作为组成`il`的子序列，所以用上述公式就可以不断递推，最后得到所求的答案，这种递推的思想很有用处，切记。



---

#### 5.汉诺双塔问题（递推关系和爆long long的解决）

链接：https://ac.nowcoder.com/acm/contest/19851/1011
来源：牛客网

给定A、B、C三根足够长的细柱，在A柱上放有2n个中间有孔的圆盘，共有n个不同的尺寸，每个尺寸都有两个相同的圆盘，注意这两个圆盘是不加区分的（下图为n=3的情形）。现要将这些圆盘移到C柱上，在移动过程中可放在B柱上暂存。要求：
（1）每次只能移动一个圆盘；
（2）A、B、C三根细柱上的圆盘都要保持上小下大的顺序；

  任务：设An为2n个圆盘完成上述任务所需的最少移动次数，对于输入的n，输出An。 

  ![img](https://uploadfiles.nowcoder.com/images/20180615/305281_1528993453083_7673A9888110E5E39181300DE876C2A4)


输入描述:

```
输入一个正整数n，表示在A柱上放有2n个圆盘。
```

输出描述:

```
输出一个正整数, 为完成上述任务所需的最少移动次数An。
```

爆long long代码：

```c++
#include <bits/stdc++.h>
using namespace std;
const int N=20010905;
typedef long long ll;
ll fun(ll a){
	if(a==1)return 2;
	return fun(a-1)*2+2;
}
int main(){
	int n;cin>>n;
	cout<<fun(n);
	system("pause");
	return 0;
}
```

AC代码：

```c++
#include<bits/stdc++.h>
using namespace std;
int main()
{
    int n;
    cin >> n;
    stringstream s;
    s.precision(0);
    s<<fixed<<pow(2,n+1);
    string a=s.str();
    a[a.size()-1]-=2;
    cout<<a;
    return 0;
}
```

问题：之前的我爱把递归理解为下楼梯在上楼梯的过程，现在而言所谓的递归缩小问题规模，实际上就是不断去寻找递归基的过程，通过递推，找到可行解，在回溯到当前问题并求解。这个问题实际上和中学的数学题差不太多。

若存在函数F(x) = F(x-1) + 1，已知F(0)=1,求解F(5)。从题目中可以看到，若求F(5),必须求解F(4),依次递推，最终可得F(5) =F(4) + 1=F(3)+2(1+1)=...F(0)+5=6。这只是临时想到的一个小问题，不过递归的思想却已经体现的淋漓尽致，当前解无法求出，只得借助底层已知解来辅助高层未知解的计算。

汉诺塔问题的递归表示：假设已经熟悉题设，共计64个盘子，三个银柱A，B，C，我们不妨这样想，将64层盘子分成两部分，把上面的63层是为一个整体，把下面的最后一层视为一个独立整体，此时盘子就仅仅就只剩下“两个”啦，一个大问题F(64)<==>拆分成F(63)一个规模减小问题和仅剩最后一个盘子的F(64号盘)复杂度为O(1)一个平凡问题，我们将F(63)的盘子从A柱移动到C柱，再从C柱移动到B柱再将64号盘从A柱直接移动到C柱。最后再把F(63)从B柱移动到A柱再移动到C柱。

其中处理爆long long的问题在stringstream类中得到补充

---







## 3.STL容器/C++补充

### 1.set容器（自定义去重）

set是关联容器的一种，关联容器和顺序容器有着根本的不同：关联容器中的元素是按关键字来保存和访问的。与之相对，顺序容器中的元素是按它们在容器中的位置来顺序保存和访问的set具备的两个特点：

1.set中的元素都是排序好的 2.set中的元素都是唯一的，没有重复的

set的一些基本操作

```c++
begin();            // 返回指向第一个元素的迭代器
end();              // 返回指向迭代器的最末尾处（即最后一个元素的下一个位置）
clear();            // 清除所有元素
count();            // 返回某个值元素的个数
empty();            // 如果集合为空，返回true
equal_range();      //返回集合中与给定值相等的上下限的两个迭代器
erase()–删除集合中的元素
find()–返回一个指向被查找到元素的迭代器
get_allocator()–返回集合的分配器
insert()–在集合中插入元素
lower_bound()–返回指向大于（或等于）某值的第一个元素的迭代器
key_comp()–返回一个用于元素间值比较的函数
max_size()–返回集合能容纳的元素的最大限值
rbegin()–返回指向集合中最后一个元素的反向迭代器
rend()–返回指向集合中第一个元素的反向迭代器
size()–集合中元素的数目
swap()–交换两个集合变量
upper_bound()–返回大于某个值元素的迭代器
value_comp()–返回一个用于比较元素间的值的函数
```

【扩展】set容器的去重规则重写，目前有两种写法来自定义set的去重规则，记住一个准则：永远让比较函数对相同的函数返回false，那么我们的方法就是合法的，自定义规则一般有两种办法

1.重载<运算符

```c++
typedef struct{
    int num;
    bool operator<(const cmp &a){//重载<运算符
        if(num==a.num)return false;//当num值相等时不插入set
        else{
            //降序或者升序代码内容
        }
    }
}cmp;

set<cmp> st;
```

2.重载()运算符

```c++
typedef struct{
    int num;
    bool operator()(const cmp &a){//重载<运算符
        if(num==a.num)return false;//当num值相等时不插入set
        else{
            //降序或者升序代码内容
        }
    }
}cmp;
set<int,cmp> st;//与第一种的写法有区别
```

注意这两种方法在set定义时的写法有区别

还有重载()运算符后，同样会对set自带的函数有所影响，如（例题：指纹锁），在重载()运算符后，其自带的`erase()和count()`函数都会受其影响，始其的运行受我们自定义的规则影响

---

### 2.sort()排序函数自定义

我们在使用`sort()`函数排序时，可以添加一个`cmp()`函数来自定义排序规则

```c++
bool cmp(int a,int b){
    return a>b;
}
int A[m];
sort(A,A+m,cmp);//降序排列
```

`cmp()`函数的作用是，若`cmp()`函数返回结果为假，`sort()`就会把他们互换位置，如果返回为真，则保持原来位置

----

### 3.cin和cout的加速

在c++中，cin和cout为了兼容性牺牲了读取的性能，所以在读取大量信息时很可能超时，但是如果我们在主函数中提前写入以下关键字就可以提高cin和cout的速度，但是这样我们就**不能和C的输入输出混用**

```c++
ios::sync_with_stdio(false);cin.tie(0);cout.tie(0);
```

---

### 4.stringstream类

头文件`<sstream>`中包含了定义了三个**类**：**istringstream**、**ostringstream** 和 **stringstream**，分别用来进行流的输入、输出和输入输出操作。这里以 stringstream 为主，介绍流的输入和输出操作。

1.类型转换

```c++
int main{
    stringstream ss;
    string s;
    int nVal=100;
    ss<<nVal;
    ss>>s;//int型转为string型
}
```

如果多次用一个stringstream进行类型转化，在每次转化后都要使用`ss.clear()`对内存释放，否则会产生错误。这里需要补充的是`ss.clear()`仅仅只是清除标记位，并没有释放内存，如果使用的是`ss.str("")`则会将流清空且释放内存

2.拼接字符串

```c++
int main{
    stringstream ss;
    string s;
    sstream << "first" << " " << "string,";
    sstream << " second string";
    cout << "strResult is: " << sstream.str() << endl;
}
```

3.对未知输入个数的处理

当stringstream遇到空格时会自动分词

例子：

 ```c++
 输入的第一行有一个数字 N 代表接下來有 N 行数字，每一行数字里有不固定个数的整数，打印每一行的总和
 ```

输入：

```
3
1 2 3
20 17 23 54 77 60
111 222 333 444 555 666 777 888 999
```

```c++
string s;
stringstream ss;
int n, i, sum, a;
cin >> n;
getline(cin, s); // 换行读取
for (i=0; i<n; i++){
    getline(cin, s);
    ss.clear();
    ss.str(s);
    sum=0;
    while (1){
        ss >> a;
        if ( ss.fail() )break;
        sum+=a;
    }
	cout << sum << endl;
}

```

4.对于简易高精度的模拟

如汉诺双塔问题，因为`n<=200`，算式是指数级，所以早已超过`long long`的最大值，这里直接将函数`pow(2,n)`的返回值直接传入流中

```c++
#include<bits/stdc++.h>
using namespace std;
int main()
{
    int n;
    cin >> n;
    stringstream s;
    s.precision(0);//输出小数点后0位
    s<<fixed<<pow(2,n+1);//将pow(2,n+1)作为字符串传入流中，fixed指从小数点开始计数
    string a=s.str();
    a[a.size()-1]-=2;
    cout<<a;
    return 0;
}
```

这里再补充一点，因为数值太大仍然会造成溢出的问题，所以函数会以科学计数法的形式返回参数，这里使用了修改输出参数的方法，将完整的数值传入了流中，具体方法还需参照C++输出小数方法



---

### 5.lower_bound()和upper_bound()的使用

这两个函数都是定义在`<algorithm>`头文件中的函数，本质是利用二分查找实现的查找元素的功能，他们的返回值是一个迭代器。

`lower_bound()`从数组的begin位置到end-1位置二分查找第一个大于或等于num的数字，找到返回该数字的地址，不存在则返回end

```c++
int a[]={1,2,3,4,5,7,8,9};
cout<<lower_bound(a,a+8,6)-a;
//输出5
```

`upper_bound()`从数组的begin位置到end-1位置二分查找第一个大于num的数字，找到返回该数字的地址，不存在则返回end。

```c++
vector<int> A;
A.push_back(2); 
A.push_back(4); 
A.push_back(5); 
cout<<upper_bound(A.begin(),A.end(),3)-A.begin();
//输出1
```

他们都需要通过返回的地址减去起始地址begin,得到找到数字在数组中的下标

---

### 6.cout的补充



---

### 7.map和unordered_map容器

我们可以使用map存储这类一对一的数据：

第一个可以称为关键字(key)，每个关键字只能在map中出现一次；
第二个可能称为该关键字的值(value)；

另外需要注意的是，使用 map 容器存储的各个键-值对，键的值既不能重复也不能被修改。换句话说，map 容器中存储的各个键值对不仅键的值独一无二，键的类型也会用 const 修饰，这意味着只要键值对被存储到 map 容器中，其键的值将不能再做任何修改。
```c++
map<int,int> mp;
mp[1]=2;
mp[2]=3;
//map的初始化
```

下面要介绍的是`unordered_map`，它与`map`不同的地方在于，首先他们的实现方式不一样

`map`：其实现是使用了红黑树，`unordered_map`:其实现使用的是哈希表

他们的特点

`map:`

1.元素有序，并且具有自动排序的功能（因为红黑树具有自动排序的功能）
2.元素按照二叉搜索树存储的，也就是说，其左子树上所有节点的键值都小于根节点的键值，右子树所有节点的键值都大于根节点的键值，使用中序遍历可将键值3.按照从小到大遍历出来
3.空间占用率高，因为map内部实现了红黑树，虽然提高了运行效率，但是因为每一个节点都需要额外保存父节点、孩子节点和红/黑性质，使得每一个节点都占用大量的空间
4.适用情况:对顺序有要求的情况下，如排序等

`unordered_map:`

1.元素无序。
2.查找速度非常的快。
3.哈希表的建立比较耗费时间
4.适用情况:对于查找问题
5.对于unordered_map或者unordered_set容器，其遍历顺序与创建该容器时输入元素的顺序是不一定一致的，遍历是按照哈希表从前往后依次遍历的

他们的使用方法都是一样的

---

### 8.max_element()和min_element()的使用

这两个函数都是用来查找容器中最大值和最小值的函数，`max_element()`和`min_element()`返回的都是迭代器，减去头部才能获得下标，如果需要访问元素则加`*`

```c++
int a[]={1,2,3,7,6,5,4};
cout<<*max_element(a,a+7)<<endl;//输出7
cout<<max_element(a,a+7)-a;//输出3
//输出
```



---







[^1]: 朝0取整，指在数轴上整数向左取整，负数向右
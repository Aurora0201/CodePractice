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

####  **复杂度时间表**

![image-20220429002742775](C:\Users\11763\AppData\Roaming\Typora\typora-user-images\image-20220429002742775.png)

可以很好的作为算法是否超时的判断

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

**进阶用法**利用二分查找寻找答案

当我们需要用遍历查找一个复杂度极大的答案时，也行我们可以使用二分法来降低时间复杂度，如《抄书》，《Gates》，他们的数据量太大，如果使用遍历来查找他们的答案肯定会超时TLE，这时我们用二分法来查找，复杂度就会从n降低为logn，这里对于答案的查找都是**十分抽象**的，因为答案本身并不确定，所以我们要二分过程中的区间压缩来逼近我们所需的答案，因为每次二分区间都会折叠，当二分停止时，上下界会重合，这时便是我们需要的答案，下面附上他们的二分过程

```c++
while(L<sum){//抄书二分过程
            ll mid=(L+sum)/2,tmp=0;
            int cnt=0;//分段数
            for(int i=m-1;i>=0;i--){
                if(A[i]>mid){cnt=m+1;break;}//这一行可以替换为if(mid<*max_element(A,A+m)){cnt=m+1;break;}
                if(A[i]+tmp<=mid)tmp+=A[i];
                else{
                    tmp=0;cnt++;i++;
                }
            }
            if(tmp<mid)cnt++;
            if(cnt<=k)sum=mid;
            else L=mid+1;
        }

 while(L<R){//Gates二分过程
            mid=L+(R-L)/2;
            int ans=f(mid);
            ans==a ? L=mid+1 : R=mid;            
        }
```

这里从代码就可见一斑，他们的原理都是大同小异的，都是通过压缩区间来查找答案

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

#### 1.背包相关问题

##### 1.最优装载问题

**最优装载问题** 给出 n 个物体，第 i 个物体重量为 wi。选择尽量多的物体，使得总重量 不超过 C

由于只关心物体的数量，所以装重的没有装轻的划算。只需把所有物体按重量从小到 大排序，依次选择每个物体，直到装不下为止。这是一种典型的贪心算法，它只顾眼前， 但却能得到最优解

---

##### 2.部分背包问题

**部分背包问题 **有 n 个物体，第 i 个物体的重量为 wi，价值为 vi。在总重量不超过 C 的情况下让总价值尽量高。每一个物体都可以只取走一部分，价值和重量按比例计算

本题在上一题的基础上增加了价值，所以不能简单地像上题那样先拿轻的（轻的可能 价值也小），也不能先拿价值大的（可能它特别重），而应该综合考虑两个因素。一种直 观的贪心策略是：优先拿“价值除以重量的值”最大的，直到重量和正好为 C

---

##### 3.乘船问题

**乘船问题 **有 n 个人，第 i 个人重量为 wi。每艘船的最大载重量均为 C，且最多只能乘 两个人。用最少的船装载所有人

考虑最轻的人 i，他应该和谁一起坐呢？如果每个人都无法和他一起坐船，则唯一的方 案就是每人坐一艘船（想一想，为什么）。否则，他应该选择能和他一起坐船的人中最重 的一个 j。这样的方法是贪心的，因此它只是让“眼前”的浪费最少。幸运的是，这个贪心策略也是对的，可以用反证法说明

假设这样做不是最好的，那么最好方案中 i 是什么样的呢？ 

情况 1：i 不和任何一个人坐同一艘船，那么可以把 j 拉过来和他一起坐，总船数不会 增加（而且可能会减少）

情况 2：i 和另外一人 k 同船。由贪心策略，j 是“可以和 i 一起坐船的人”中最重的， 因此 k 比 j 轻。把 j 和 k 交换后 k 所在的船仍然不会超重（因为 k 比 j 轻），而 i 和 j 所在的 船也不会超重（由贪心法过程），因此所得到的新解不会更差。 由此可见，贪心法不会丢失最优解。最后说一下程序实现。在刚才的分析中，比 j 更重 的人只能每人坐一艘船。这样，只需用两个下标 i 和 j 分别表示当前考虑的最轻的人和最重 的人，每次先将 j 往左移动，直到 i 和 j 可以共坐一艘船，然后将 i 加 1，j 减 1，并重复上 述操作。不难看出，程序的时间复杂度仅为 O(n)，是最优算法（别忘了，读入数据也需要 O(n)时间，因此无法比这个更好了）

---

#### 2.区间选择问题

##### 1.选择不相交区间

**选择不相交区间** 数轴上有 n 个开区间(ai, bi)。选择尽量多个区间，使得这些区间两两没有公共点

首先考虑一个问题，如过区间y完全被区间x包含，那么选x是不划算的，因为x占用了更多的空间，选x不如选y。接下来，按bi的大小从小到大排列所有区间，**贪心策略就是：选择第一个区间**

情况1：当a1>a2，也就说此时，区间2包含区间1，此时按照上面的结论，选择区间1。不仅区间2如此，只要有一个区间i的ai<a1，区间i都不要选

情况2：当a1<=a2<=a3···，如果区间1与区间2完全不相交，那么第一次选谁都可以（但是正因如此我们必须选区间1，因为按照我们的贪心策略只会向后选择，不会想去选择区间，如果选了区间2，我们就不能再选区间1）。当他们都有部分相交时，如果不选区间2，那么区间1的黑色部分是无效的（它不会影响任何其他的区间），此时区间1的有效区间就变成了灰色部分，他被区间2包含，按照上面的结论，区间2是不能选择的，依次类推，选择区间1是明智的

![image-20220502114015178](C:\Users\11763\AppData\Roaming\Typora\typora-user-images\image-20220502114015178.png)

这样，选择了区间1后，需要将与区间1相交的区间排除，记录上一个选择的区间，这样在排序后只要扫描一次即可完成贪心获得正确结果

---

##### 2.区间选点问题

**区间选点问题** 数轴上有 n 个闭区间[ai, bi]。取尽量少的点，使得每个区间内都至少有 一个点（不同区间内含的点可以是同一个）

如果区间 i 内已经有一个点被取到，则称此区间已经被满足。受上一题的启发，下面先 讨论区间包含的情况。由于小区间被满足时大区间一定也被满足，所以在区间包含的情况 下，大区间不需要考虑

把所有区间按b从小到大排序（b相同时按a从大到小排序），则如果出现区间包含的情况小区间一定在前面，则第一个区间该如何选点呢？**贪心策略是：取最后一个点**

根据刚才的讨论，所有需要考虑的区间的a也是递增的，那么我们就可以画成下图所示

![image-20220502115257198](C:\Users\11763\AppData\Roaming\Typora\typora-user-images\image-20220502115257198.png)

如果第一个区间不选最后一个点，而是取中间的，如灰色的点，那么把他后移到最后一个点，被满足的区间增加了，而且原先的区间一定被满足，不难看出，这样的贪心策略是正确的

----

##### 3.区间覆盖问题

**区间覆盖问题** 数轴上有 n 个闭区间[ai, bi]，选择尽量少的区间覆盖一条指定线段 [s, t]

本题的突破口仍然是区间包含和排序扫描，不过先要进行一次预处理。每个区间在[s, t] 外的部分都应该预先被切掉，因为它们的存在是毫无意义的。预处理后，在相互包含的情 况下，小区间显然不应该考虑

把各区间按照 a 从小到大排序。如果区间 1 的起点不是 s，无解（因为其他区间的起点 更大，不可能覆盖到 s 点），否则选择起点在 s 的最长区间。选择此区间[ai, bi] 后，新的起 点应该设置为 bi，并且忽略所有区间在 bi 之前的部分，就像预处理一样。虽然贪心策略比 上题复杂，但是仍然只需要一次扫描，如图 8-9 所示。s 为当前有效起点（此前部分已被覆 盖），则应该选择区间 2

![image-20220502140704467](C:\Users\11763\AppData\Roaming\Typora\typora-user-images\image-20220502140704467.png)



---

### 5.KMP算法（字符串匹配）



---

### 6.中途相遇法（四数之和）

例子：UVa1152，题意如下

```
给出四个有n个元素的集合,A,B,C,D(n<=4000)，要求分别从中选出一个元素a,b,c,d，令他们的和为0，输出有多少种选法
```

【分析】如果使用四重循环枚举a,b,c,d，那算法的复杂度是n的四次方，肯定超时，最好的算法是，一是先将A,B,C,D，两两分组，用双重循环枚举每个a+b，c+d，然后将他们放到两个数组SumA，SumB中，此时有两种解法，使用二分查找，在数组SumB中寻找与SumA中的值相加为0的情况，二是使用map容器，利用map容器统计不同的a+b的对数，然后再在c+d中找到与他们相加等于0的解。这两种方法的复杂度都是
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

### 7.扫描法（运酒问题）

例子：UVa11054，题意如下

```
直线上有n个村庄，每个村庄会卖酒或者买酒，他们的供需平衡，每k个单位的就送到附近的村庄所需的劳动力为k，求所需的最少劳动力
```

AC代码：

```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
int main(){
    int n;
    while(cin>>n&&n){
        ll ans=0,a,next=0;
        for(int i=0;i<n;i++){
            cin>>a;
            ans+=abs(next);
            next+=a;
        }
        cout<<ans<<endl;
    }
    system("pause");
}
```

本题关键是需要合并简化问题，并且这里注意的是劳动力是**不可复用**的，且以**临近村庄**为单位，这里我们假设第一个村庄的需求为a，第二个村庄的需求为b，那么我们将第一第二个村庄看作一个整体，则从第三个村庄送过来所需的劳动力为a+b，这里就成为了2~n个村庄的问题，且第二个村庄所需劳动力为a+b，如此计算下去，我们就能得到n-1~n所需的劳动力，问题的解得出，这里的**等价转化思想十分重要**

本题注意要审题的地方有，劳动力的计算是以临近村庄为单位的，且劳动力不可复用

这里使用的是**扫描法**，它在枚举时，维护一些重要的量，从而简化计算

---

### 8.滑动区间问题

---

### 9.使用数据结构来优化算法

使用数据结构，数据结构往往可以在不改变主算法的前提下提高运行效率，具体做法 可能千差万别，但思路却是有规律可循的。下面先介绍一个经典问题

输入正整数 k 和一个长度为 n 的整数序列 A1, A2, A3,…, An。定义 f(i)表示从元素 i 开始 的连续 k 个元素的最小值，即 f(i)=min{Ai, Ai+1,…, Ai+k-1}。要求计算 f(1), f(2), f(3),…, f(n-k+1)。 例如，对于序列 5, 2, 6, 8, 10, 7, 4，k=4，则 f(1)=2, f(2)=2, f(3)=6, f(4)=4

【分析】 如果使用定义，每个 f(i)都需要 O(k)时间计算，总时间复杂度为((n-k)k)，太大了。那么 换一个思路：计算 f(1)时，需要求 k 个元素的最小值——这是一个“窗口”。计算 f(2)时， 这个窗口向右滑动了一个位置，计算 f(3)和 f(4)时，窗口各滑动了一个位置，如图所示

![image-20220426163111102](C:\Users\11763\AppData\Roaming\Typora\typora-user-images\image-20220426163111102.png)

因此，这个问题称为滑动窗口的最小值问题。窗口在滑动的过程中，窗口中的元素“出 去”了一个，又“进来”了一个。借用数据结构中的术语，窗口往右滑动时需要删除一个 元素，然后插入一个元素，还需要取最小值。这不就是优先队列吗？第 5 章中曾经介绍过 用 STL 集合实现一个支持删除任意元素的优先队列。因为窗口中总是有 k 个元素，插入、 删除、取最小值的时间复杂度均为 O(logk)。这样，每次把窗口滑动时都需要 O(logk)的时间， 一共滑动 n-k 次，因此总时间复杂度为 O((n-k)logk)。

其实还可以做得更好。假设窗口中有两个元素 1 和 2，且 1 在 2 的右边，会怎样？这意 味着 2 在离开窗口之前永远不可能成为最小值。换句话说，这个 2 是无用的，应当及时删 除。当删除无用元素之后，滑动窗口中的有用元素从左到右是递增的。为了叙述方便，习 惯上称其为单调队列。在单调队列中求最小值很容易：队首元素就是最小值

当窗口滑动时，首先要删除滑动前窗口的最左边元素（如果是有用元素），然后把新 元素加入单调队列。注意，比新元素大的元素都变得无用了，应当从右往左删除。如图 8-14 所示是滑动窗口的 4 个位置所对应的单调队列

![image-20220426163303144](C:\Users\11763\AppData\Roaming\Typora\typora-user-images\image-20220426163303144.png)

单调队列和普通队列有些不同，因为右端既可以插入又可以删除，因此在代码中通常 用一个数组和 front、rear 两个指针来实现，而不是用 STL 中的 queue。如果一定要用 STL， 则需要用双端队列（即两端都可以插入和删除），即 deque

尽管插入元素时可能会删除多个元素，但因为每个元素最多被删除一次，所以总的时 间复杂度仍为 O(n)，达到了理论下界（因为至少需要 O(n)的时间来检查每个元素）

下面这道例题（防线 UVa1471）更加复杂，但思路是一样的：先排除一些干扰元素（无用元素），然后 把有用的元素组织成易于操作的数据结构



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
	cout<<fun(n)%N;
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

## 3.算法练习（紫书）

### 1.唯一的雪花 （滑动区间+连续子序列问题）

UVa11572

AC代码_1：

```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=1000000+5;
int A[N];
int main(){
    int t,n;
    cin>>t;
    while(t--){
        cin>>n;
        set<int> st;
        for(int i=0;i<n;i++)cin>>A[i];
        int L=0,R=0,ans=0;
        while(R<n){
            while(R<n&&!st.count(A[R]))st.insert(A[R++]);//先判是不是越界
            ans=max(ans,R-L);
            st.erase(A[L++]);
        }
        cout<<ans<<endl;
    } 
    system("pause");
}
```

AC代码_2：

```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=1000000+5;
int A[N],last[N];
int main(){
    int t,n;
    cin>>t;
    while(t--){
        cin>>n;
        unordered_map<int,int> mp;
        for(int i=0;i<n;i++){
            cin>>A[i];
            if(!mp.count(A[i]))last[i]=-1;//-1说明前面没有
            else last[i]=mp[A[i]];
            mp[A[i]]=i;
        }
        int L=0,R=0,ans=0;     
        while(R<n){
           while(R<n&&last[R]<L)R++;
           ans=max(ans,R-L);
           L=last[R]+1;
        }
        cout<<ans<<endl;
    }
    system("pause");
}
```

第一种使用set的方法很容易想到，这里不多赘述。第二种使用map的方法，其实就是使用map来求出当前指针下，该数组元素上一次出现的下标，并储存在last数组中，之前使用的没有使用last数组的方法出现WA的情况是因为，当时set没有不断重置当前元素上次出现的地方，所以出现了错误，这种事情告诉我们，在编写程序还是以更低的复杂性为主，就像上面的程序，使用last数组储存下标，减少了错误的出现

---

### 2.抄书（二分法的活用）

UVa714

“最大值尽量小”是一种很常见的优化目标。下面考虑一个新的问题：能否把输入序列划 分成 m 个连续的子序列，使得所有 S(i)均不超过 x？将这个问题的答案用谓词 P(x)表示，则让 P(x)为真的最小 x 就是原题的答案。P(x)并不难计算，每次尽量往右划分即可（想一想，为什么）。 接下来又可以猜数字了——随便猜一个 x0，如果 P(x0)为假，那么答案比 x0大；如果 P(x0) 为真，则答案小于或等于 x0。至此，解法已经得出：二分最小值 x，把优化问题转化为判定 问题 P(x)。设所有数之和为 M，则二分次数为 O(logM)，计算 P(x)的时间复杂度为 O(n)（从 左到右扫描一次即可），因此总时间复杂度为 O(nlogM) 

AC代码：

```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=500+5;
ll A[N];
int main(){
    int t;cin>>t;
    while(t--){
        int m,k,vis[N]={0};
        ll sum=0,L=0;
        cin>>m>>k;
        for(int i=0;i<m;i++)cin>>A[i],sum+=A[i];
        while(L<sum){
            ll mid=(L+sum)/2,tmp=0;
            int cnt=0;//分段数
            for(int i=m-1;i>=0;i--){
                if(A[i]>mid){cnt=m+1;break;}//这一行可以替换为if(mid<*max_element(A,A+m)){cnt=m+1;break;}
                //但是会稍慢，这里找出最大元素也是需要时间的，所以这里用代码中的写法是相对聪明的
                if(A[i]+tmp<=mid)tmp+=A[i];
                else{
                    tmp=0;cnt++;i++;
                }
            }
            if(tmp<mid)cnt++;
            if(cnt<=k)sum=mid;
            else L=mid+1;
        }
        int cnt=0;
        ll tmp=0;
        for(int i=m-1;i>=0;i--){
            if(tmp+A[i]<=L)tmp+=A[i];
            else{cnt++;tmp=0;vis[i]=1;i++;}
        }
        for(int i=0;i<m;i++){
            if(vis[i])cout<<A[i]<<" / ";
            else if(cnt<k-1){cout<<A[i]<<" / ";cnt++;}
            else if(i!=m-1)cout<<A[i]<<' ';
            else cout<<A[i];
        }
        cout<<endl;
    }
    system("pause");
}
```

一道思维题，在我们不知道最优解的时候，使用二分法来查找最优解，因为如果暴力搜索需要复杂度为n的平方，而n=500，所以肯定会超时，但是我们用查找代替遍历，这样就大大降低了复杂度。

---

### 3.Add All （Huffman编码）

UVa 10954

这里最小cost的构建很像哈夫曼编码的构建过程，所以这里用priority_queue来模拟

AC代码：

```c++
#include<bits/stdc++.h>
using namespace std;
const int N = 505;
typedef long long ll;
int main() {
	int n;
    while(cin>>n&&n){
        priority_queue<int,vector<int>,greater<int> >q;
        int a,b;
        for(int i=0;i<n;i++)cin>>a,q.push(a);
        ll ans=0;     
        for(int i=0;i<n-1;i++){
            a=q.top();q.pop();
            b=q.top();q.pop();
            ans+=b+a;
            q.push(a+b);
        }
        cout<<ans<<endl;
    }
    system("pause");
	return 0;
}
```

优先队列是优先级高的元素先出队，所以用降序优先队列，让小的先出队，这样就得出解

---

### 4.Erratic Expansion（递归）

UVa 12627

如图 8-20 所示，k 小时的情况由 4 个 k-1 小时的情况拼成，其中右下角全是蓝气球， 不用考虑。剩下的 3 个部分有一个共同点：都是前 k-1 小时后“最下面若干行”或者“最上面若干行”的红气球总数。 具体来说，设 f(k, i)表示 k 小时之后最上面 i 行的红气球总数，g(k,i)表示 k 小时之后最 下面 i 行的红气球总数（规定 i≤0 时 f(k,i)=g(k,i)=0），则所求答案为 f(k,b) - f(k, a-1)。 如何计算 f(k,i)和 g(k,i)呢？以 g(k,i)为例，下面分两种情况进行讨论，如图 8-21 所示

![image-20220428155326604](C:\Users\11763\AppData\Roaming\Typora\typora-user-images\image-20220428155326604.png)

如果 i≥2k-1 ，则 g(k,i)=2g(k-1,i-2k-1 )+c(k)，否则 g(k,i)=g(k-1,i)。其中，c(k)表示 k 小时 后红气球的总数，满足递推式 c(k)=3c(k-1)，而 c(0)=1，因此 c(k)=3k

AC代码：

```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
ll g(ll k,ll i){
    if(i<=0)return 0;
    if(k==0)return 1;
    if(i>pow(2,k-1)){
        return 2*g(k-1,i-pow(2,k-1))+pow(3,k-1);
    }else{
        return g(k-1,i);
    }
}

int main() {
	int t,ct=1;cin>>t;
    while(t--){
        ll K,A,B;
        cin>>K>>A>>B;
        cout<<"Case "<<ct++<<": "<<g(K,pow(2,K)+1-A)-g(K,pow(2,K)-B)<<endl;
    }
    system("pause");
	return 0;
}
```

数据量很大，有2的30次方，不可能用模拟，因为是规则的棋盘，所以我们可以想到用递归，但是怎么样递归是一个考验我们思维的问题，观察每一个图形，我们可以发现，第k个小时的图形是由三个第k-1小时的图型和一个全蓝的方块组成，所以这里我们只用考虑三个有红块的区域，这里用的思想是分割组合，与分治法有一点相似

---

### 5.Just Finish it up（贪心+扫描法）

UVa 11093

考虑 1 号加油站，直接模拟判断它是否为解。如果是，直接输出；如果不是，说明在 模拟的过程中遇到了某个加油站 p，在从它开到加油站 p+1 时油没了。这样，以 2, 3,…, p 为起点也一定不是解（想一想，为什么）。这样，使用简单的枚举法便解决了问题，时间复杂度为 O(n)

AC代码：

```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=100000+5;
int A[N],B[N];
int main() {
	int t,n,ct=1;
    cin>>t;
    ios::sync_with_stdio(false);cin.tie(0);cout.tie(0);
    while(t--){
        cin>>n;
        ll sumA=0,sumB=0;
        memset(A,0,sizeof(A));
        memset(B,0,sizeof(B));
        for(int i=0;i<n;i++)cin>>A[i],sumA+=A[i];
        for(int i=0;i<n;i++)cin>>B[i],sumB+=B[i];
        if(sumB>sumA){
            cout<<"Case "<<ct++<<": "<<"Not possible"<<endl;
            continue;
        }
        sumA=0,sumB=0;
        int ans=-1;
        for(int i=n-1;i>=0;i--){
            sumA+=A[i];
            sumB+=B[i];
            if(sumA>=sumB)ans=i,sumA=0,sumB=0;
        }
        cout<<"Case "<<ct++<<": "<<"Possible from station "<<ans+1<<endl;
    }
    system("pause");
	return 0;
}
```



---

### 6.Gates（模拟+二分）

UVa 1607

因为只有一个输入 x，所以整个电路的功能不外乎 4 种：常数 0、常数 1、x 及非 x。先 把 x 设为 0，再把 x 设为 1，如果二者的输出相同，整个电路肯定是常数，任意输出一种方案即可

如果 x=0 和 x=1 的输出不同，说明电路的功能是 x 或者非 x，解至少等于 1。不妨设 x=0 时输出 0，x=1 时输出 1。现在把第一个输入改成 1，其他仍设为 0（记这样的输入为 1000… 0），如果输出是 1，则得到了一个解 x000…0

如果 1000…0 的输出也是 0，再把输入改成 1100…0，如果输出是 1，则又得到了一个 解 1x00…0。如果输出还是 0，再尝试 1110…0，如此等等。由于输入全 1 时输出为 1，这 个算法一定会成功

问题在于 m 太大，而每次“给定输入计算输出”都需要 O(m)时间，逐个尝试会很慢。 好在已经学习了二分查找：只需二分 1 的个数，即可在 O(logm)次计算之内得到结果，总时 间复杂度为 O(mlogm)

AC代码：

```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=200000+5;
int l[N],r[N],val[N];
int n,m;
int f(int k){
    for(int i=1;i<=m;i++){
        val[i]= !((l[i]<0 ? -l[i]<=k : val[l[i]]) & (r[i]<0 ? -r[i]<=k : val[r[i]]));
    }
    return val[m];
}
int main(){
	int t,a,b;
    cin>>t;
    ios::sync_with_stdio(false);cin.tie(0);cout.tie(0);
    while(t--){
        cin>>n>>m;
        for(int i=1;i<=m;i++){
            cin>>a>>b;
            l[i]=a;r[i]=b;
        }
        a=f(0);//全0
        b=f(n);
        if(a==b){//常数情况
            for(int i=1;i<=n;i++)
                cout<<1;
                cout<<endl;
            continue;
        }
        int L=1,R=n,mid;
        while(L<R){//二分
            mid=L+(R-L)/2;
            int ans=f(mid);
            ans==a ? L=mid+1 : R=mid;            
        }
        for(int i=1;i<=n;i++){
            if(i<L)cout<<1;
            else if(i==L)cout<<"x";
            else cout<<0;
        }
        cout<<endl;
    }
    system("pause");
	return 0;
}
```

又是一题二分查找答案的题目，因为数据量有200000，远超n的平方所能承受的10000，所以这里用二分查找来降低复杂度，需要注意的点有，一是分析情况，二是要注意题目中的最少x是不是可以为0。这里对于三元运算符的操作可谓是点睛之笔，利用三元运算符很好的模拟了输入部分，还有，利用数组储存信息时，最好不要使用二维数组（矩阵除外），因为复杂的调用很容易出错。这里我们还是要更加深入的理解二分法，当我们使用二分法寻找一个数组中的值时，查找到会直接返回答案，而当我们用二分法查找抽象的答案时，不能之前的方法去理解，我们知道，每一次二分，区间都会折叠，区间的上下界会不断的逼近我们所求的答案，当区间上下界重合时，这就是我们需要找的答案

---

### 7.Shuffle（滑动区间）

Uva12174

“连续的 s 个数”让你联想到了什么？没错，滑动窗口！这次的窗口大小是“基本” 固定的（因为还需要考虑不完整的段），因此只需要一个指针；而且所有数都是 1~s 的整 数，也不需要 STL 的 set，只需要一个数组即可保存每个数在窗口中出现的次数。再用一个变量记录在窗口中恰好出现一次的数的个数，则可以在 O(n)时间内判断出每个窗口是否满 足要求（每个整数最多出现一次）

这样，就可以枚举所有可能的答案，判断它对应的所有窗口，当且仅当所有窗口均满 足要求时这个答案是可行的

AC代码：

```c
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=100000+5;
int song[N]={0},A[N]={0},ok[2*N]={0};
int t,s,n;
int main(){
    cin>>t;
    ios::sync_with_stdio(false);cin.tie(0);cout.tie(0);
    while(t--){
        cin>>s>>n;
        for(int i=1;i<=n;i++)cin>>song[i];
		for(int i=1;i<=n;i++)A[i]=0;//每次都要重置标记数组
		for(int i=1;i<=n+s;i++)ok[i]=0;
		int R=0,L=1,cnt=0,ans=0;//有效数字个数
		ok[n+s]=1;
        for(int i=1;i<n+s;i++){//枚举n+s-1个位置
			while(R<i&&R<n)if(++A[song[++R]]==1)cnt++;
			while(L<i-s+1&&L<=n)if(!(--A[song[L++]]))cnt--;
			if(R-L==cnt-1)ok[i]=1;
		}
		for(int i=1;i<=s;i++){//拼合区间判断是否合法
			int flag=1;
			for(int j=i;j<=n+s;j+=s){
				if(!ok[j]){flag=0;break;}
			}
			if(flag)ans++;
		}
		cout<<ans<<endl;
    }
    system("pause");
	return 0;
}
```

写这题我是真吐了，一开始一直为了模拟区间的滑动，一直在修改算法，但是出于思路的不清晰，前期的很多错误都浪费了很多时间，特别是对于不完整区间的处理，后面看了别人的代码才真的理解滑动区间，首先滑动区间指的是只遍历一次序列，（刚开始写代码的时候，用了错误的模拟算法，当时的复杂度应该为n的平方，不是WA就是TLE，心态都要爆炸了），但是看了别人的代码时才豁然开朗，这里节省时间的关键就是要利用之前遍历过的部分，同时用`ok[i]`来将他们区分出来，因为遍历时，我们每次只将区间移动了一格，但是通过每隔s格的拼合，我们就能利用他们来判断整个序列是否是合法的，因为我们只遍历了一次，这样复杂度就是O(n)

---

### 8.Fabled Rooks（区间覆盖问题+贪心+问题分解）

UVa11134

AC代码：

```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=5000+5;
int vx[N],vy[N];
/* 
	这题可以分解为，在x维度中，寻找一个值，同时在y的维度中选取一个值
	因为给出了区间，所以是区间选点问题，选择一个点的同时，覆盖最少的区间，贪心策略

*/
struct point{
	int l,r;
	int id;
	bool operator<(point p)const{
		if(r!=p.r)return r<p.r;
		else return l>p.l;
	}
}x[N],y[N];
int px[N],py[N];
bool cmp(point a,point b){//输出时进行id排序
	return a.id < b.id;
}
int main(){
	int t;
    ios::sync_with_stdio(false);cin.tie(0);cout.tie(0);
    while(cin>>t&&t){
		for(int i=1;i<=t;i++)vx[i]=vy[i]=px[i]=py[i]=0;
        for(int i=1;i<=t;i++)cin>>x[i].l>>y[i].l>>x[i].r>>y[i].r,x[i].id=y[i].id=i;
		sort(x+1,x+t+1);
		sort(y+1,y+t+1);
		int fx=0,fy=0;
		for(int i=1;i<=t;i++){
			fx=fy=0;
			for(int j=x[i].l;j<=x[i].r;j++)if(!vx[j]){vx[j]=1;px[x[i].id]=j;fx=1;break;}
			for(int j=y[i].l;j<=y[i].r;j++)if(!vy[j]){vy[j]=1;py[y[i].id]=j;fy=1;break;}
			if(!fx||!fy)break;
		}
		if(fx&&fy)for(int i=1;i<=t;i++)cout<<px[i]<<" "<<py[i]<<endl;
		else cout<<"IMPOSSIBLE"<<endl;
    }
    system("pause");
	return 0;
}
```

两个车相互攻击的条件是处于同一行或者同一列，因此不相互攻击的条件就是不在同 一行，也不在同一列。可以看出：行和列是无关的，因此可以把原题分解成两个一维问题。 在区间[1~n]内选择 n 个不同的整数，使得第 i 个整数在闭区间[n1i, n2i]内。是不是很像前面 讲过的贪心法题目？这也是一个不错的练习，具体解法留给读者思考。 

**等价转换** 与其说这是一种算法设计方法，还不如说是一种思维方式，可以帮助选手 理清思路，甚至直接得到问题的解决方案

---

### 9.*Defense Lines（利用数据结构加速）

UVa1471

为了方便叙述，下面用 L 序列表示“连续递增子序列”。删除一个子序列之后，得到 的最长 L 序列应该是由两个序列拼起来的，如图 8-15 所示

![image-20220504110231862](C:\Users\11763\AppData\Roaming\Typora\typora-user-images\image-20220504110231862.png)

现在我们用`f(j)`和`g(i)`来表示当前以j为尾的递增子序列长度，以i为首的递增子序列长度，这时最笨的方法肯定是O(n3)的暴力枚举，但是如果我们提前将`g(j)`和`f(i)`计算好，算法就能优化到O(n2)，这也是目前能想到最快的方法，但是现在仍然会超时，那该怎么办呢，一般这种情况下，我们应该要用查找来代替一层枚举，这样就能降低算法的复杂度，这里我们只枚举i然后查找当前`A[j]<A[i]`，且最大的`f[j]`，这里我们用STL中的set集合存入每一个二元组`(a[j],g[j])`，那么现在的问题是，如何得到最好的情况，即维护当前的最优解，即在我们插入一个新元素时，我们要与前面一个元素比较，是否前一个元素的`f[j_1]>=f[j_2]`，如果大于等于，则这个元素不要插入set，**因为前一个元素已经是最优解，插入这个元素会影响最优解的情况**，当这个元素需要插入时，我们还需要遍历后面的元素，如果存在这个这个元素的`f[j]`更大的情况，则后面的元素都应该删除。遍历一次后即可得到最优解，复杂度为O(nlogn)

AC代码：

```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=200000+5;
int a[N],f[N],g[N];

struct point{
	int val,f;
	point(int val,int f):val(val),f(f){};
	bool operator<(const point &p)const{
		return val<p.val;
	}
};

int main(){
	int t;
    ios::sync_with_stdio(false);cin.tie(0);cout.tie(0);
	set<point> st;
	cin>>t;
    while(t--){
		st.clear();
		int n;cin>>n;
		for(int i=1;i<=n;i++)cin>>a[i];
		if(n==1){cout<<1<<endl;continue;}//因为下面的运行至少要从2开始，所以要特殊处理1的情况，先后顺序问题，记得先输入再跳过
		f[1]=1;
		for(int i=2;i<=n;i++)a[i]>a[i-1] ? f[i]=f[i-1]+1 : f[i]=1;
		g[n]=1;
		for(int i=n-1;i>=1;i--)a[i]<a[i+1] ? g[i]=g[i+1]+1 : g[i]=1;
		int ans=1;
		st.insert(point(a[1],f[1]));
		for(int i=2;i<=n;i++){//枚举i
			point b(a[i],f[i]);
			set<point>::iterator it=st.lower_bound(b);
			int keep=1;
			if(it!=st.begin()){
				it--;
				ans=max(ans,it->f+g[i]);
				if(it->f>=b.f)keep=0;//因为当前的it是最优解，考虑这次要插入的b是否会影响当前最优解，这个算法成立是因为，每一次更新set时，都不会允许不符合规则的元素存在，因为每次都在维护这个规则，所以算法是成立的
			}
			if(keep){
				st.erase(b);
				st.insert(b);
				it=st.find(b);
				it++;
				while(it!=st.end()&&it->val>b.val&&it->f<=b.f)st.erase(it++);
			}
		}
		cout<<ans<<endl;
    }
    system("pause");
	return 0;
}
```

这个算法重要的地方告诉我们，当我们定义了一个规则，并且在每次枚举时都维护这个规则，那么最终容器中的所有元素都会符合我们的规则，这是理解本题最关键的地方

---

### 10.*Average（数形结合+子序列操作+数形结合）

 UVa1451

先求**前缀和**Si=A1+A2+…+Ai（规定 S0=0），然后令点 Pi=(i, Si)，则子序列 i~j 的平均值 为(Sj-Si-1)/(j-i+1)，也就是直线 Pi-1Pj 的斜率。这样可得到主算法：从小到大枚举 t，快速找 到 t'≤t-L，使得 Pt'Pt 斜率最大。注意题目中的 Ai都是 0 或 1，因此每个 Pi和上一个 Pi-1相 比，都是 x 加 1，y 不变或者加 1

![image-20220504172746420](C:\Users\11763\AppData\Roaming\Typora\typora-user-images\image-20220504172746420.png)

![image-20220504172810339](C:\Users\11763\AppData\Roaming\Typora\typora-user-images\image-20220504172810339.png)

1.上面说的栈指的并不是数据结构中的链栈，而是这里用数组模拟的栈，因为我们在本题中要访问的不仅是栈顶元素，还有栈顶的下面一个元素，所以这里用一个大数组模拟栈即可

2.求斜率的过程是不断的比较当前t点与当前切点的和当前t点与下一个切点的斜率比较，当下一个切点斜率大或者**相等**时，切点更新为下一个切点，相等时可能的情况有，当前字符串更短，所以不能忽略当他们的平均值相等的情况

AC代码：

```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=100000+5;
int a[N],p[N];//栈
char s[N];
int compare(int x1,int x2,int x3,int x4){
	return (a[x2]-a[x1-1]) * (x4-x3+1) - (a[x4]-a[x3-1]) * (x2-x1+1);//用乘法避免除法
}

int main(){
	// ios::sync_with_stdio(false);cin.tie(0);cout.tie(0);
	int t;scanf("%d",&t);
	while(t--){
		int n,L;
		scanf("%d%d%s",&n,&L,s+1);//这种存法可以让下标从1开始
		a[0]=0;
		for(int i=1;i<=n;i++)a[i]=s[i]-'0'+a[i-1];
		int ansL=1,ansR=L;
		int i=0,j=0;
		for(int t=L;t<=n;t++){//从L开始枚举3
			while(j-i>1&&compare(p[j-2],t-L,p[j-1],t-L)>=0)j--;
			p[j++]=t-L+1;//栈中存入的是下标
			while(j-i>1&&compare(p[i+1], t, p[i], t)>=0)i++;
			int c=compare(p[i],t,ansL,ansR);//与之前的答案比较
			if(c>0||c==0&&t-p[i]<ansR-ansL){
				ansL=p[i];ansR=t;
			}
		}
		printf("%d %d\n",ansL,ansR);
	}
    system("pause");
	return 0;
}
```

代码中还有一个技巧，一般遇到斜率我们会用除法，这里会涉及小数的计算，这回使我们的程序更复杂同时更加耗时，所以这里我们用乘法相加来避免除法的出现，这题主要难点在于下标，如果不进行下标转换的话，这里的计算非常容易出错，切记切记

---

### 11.Non-boring sequences（中途相遇法+子序列）

UVa1608

不难想到整体思路：在整个序列中找一个只出现一次的元素，如果不存在，则这个序 列不是不无聊的；如果找到一个只出现一次的元素 A[p]，则只需检查 A[1…p-1]和 A[p+1… n]是否满足条件，设长度为 n 的序列需要 T(n)时间，则有 T(n) = max{T(k-1)  + T(n-k) + 找到唯一元素 k 的时间}。这里取 max 是因为要看最坏情况如果事先算出每个元素左边和右边最近的相同元素，则可以在 O(1)时间内判断在任意一个连续子序列中，某个元素是否唯一。 如果从左边找，最坏情况下唯一元素是最后一个元素，因此 T(n) = T(n-1) + O(n)≥T(n) = O(n 2 )，在n<=200000的数据量下肯定超时，那么，从两边往中间找会怎样？此时 T(n) = max{T(k) + T(n-k) + min(k,n-k)}，刚才 的最坏情况（即第一个元素或最后一个元素是唯一元素）变成了 T(n)=T(n-1)+O(1)（因为一 下子就找到唯一元素了），即 T(n)=O(n)。而此时的最坏情况是唯一元素在中间的情况，它 满足经典递推式 T(n) = 2T(n/2) + O(n)，即 T(n)=O(nlogn)

AC代码：

```c
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=200000+5;
int a[N],l[N],r[N];
int n;
int solve(int L,int R){
	if(L==R)return 1;
	for(int i=L;i<=R;i++){
		if(l[i]<L&&r[i]>R)
			if(i==L)return solve(L+1,R);
			else if(i==R) return solve(L,R-1);
			else return solve(L,i-1)&&solve(i+1,R);
		if(l[R-i+L]<L&&r[R-i+L]>R)//从两侧同时查找
			if(R-i+L==L)return solve(L+1,R);
			else if(R-i+L==R) return solve(L,R-1);
			else return solve(L,R-i+L-1)&&solve(R-i+L+1,R);
	}
	return 0;
}
int main(){
	ios::sync_with_stdio(false);cin.tie(0);cout.tie(0);
	int t;cin>>t;
	while(t--){
		cin>>n;
		for(int i=0;i<n;i++)cin>>a[i];
		unordered_map<int,int> mp;//unordered_map更快
		for(int i=0;i<n;i++){
			mp.count(a[i]) ? l[i]=mp[a[i]] : l[i]=-1;mp[a[i]]=i;
		}
		mp.clear();
		for(int i=n-1;i>=0;i--){
			mp.count(a[i]) ? r[i]=mp[a[i]] : r[i]=N;mp[a[i]]=i;
		}
		int ok=solve(0,n-1);
		if(!ok)cout<<"boring"<<endl;
		else cout<<"non-boring"<<endl;
		// cout<<solve(0,n-1)<<endl;
	}
	system("pause");
	return 0;
}
```

中途相遇的巧妙运用，为了避免最坏情况，从两侧同时查找，大大降低了时间复杂度，最坏的情况也只是元素在中间而已。还有里面的函数递归过程，非常适合这种两端同时判断的写法

---

### 12.Cav（扫描法）

Uva1442

为了方便起见，下面用“水”来代替题目中的燃料。根据物理定律，每一段有水的连 续区间，水位高度必须相等，且水位必须小于等于区间内的最低天花板高度，因此位置[i,i+1] 处的水位满足 h≤si，**且从(i,h)出发往左右延伸出的两条射线均不会碰到天花板**（即两条射 线将一直延伸到洞穴之外或先碰到地板之间的“墙壁”）的最大 h。如果这样的 h 不存在， 则规定 h=pi（也就是“没水”）。 这样，可以先求出“往左延伸不会碰到天花板”的最大值 h1(i)，再求“往右延伸不会 碰到天花板”的最大值 h2(i)，则 hi=min{h1(i), h2(i)}。根据对称性，只考虑 h1(i)的计算

 从左到右扫描。初始时设水位 level=s0，然后依次判断各个位置[i,i+1]处的高度。 如果 p[i] > level，说明水被“隔断”了，需要把 level 提升到 pi。 如果 s[i] < level，说明水位太高，碰到了天花板，需要把 level 下降到 si。 位置[i,i+1]处的水位就是扫描到位置 i 时的 level。 不难发现，两次扫描的时间复杂度均为 O(n)，总时间复杂度为 O(n)

AC代码：

```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=1000000+5;//1000000+5
int p[N],s[N],l[N],r[N];
int main(){
	ios::sync_with_stdio(false);cin.tie(0);cout.tie(0);
	int t;cin>>t;
	while(t--){
		int n;cin>>n;
		for(int i=0;i<n;i++)cin>>p[i];//地面
		for(int i=0;i<n;i++)cin>>s[i];//天花板
		l[0]=s[0];
		for(int i=1;i<n;i++){//从左向右扫描
			if(l[i-1]>=p[i]&&l[i-1]<=s[i])l[i]=l[i-1];
			else if(l[i-1]>s[i])l[i]=s[i];
			else if(l[i-1]<p[i])l[i]=p[i];
		}
		r[n-1]=s[n-1];
		for(int i=n-2;i>=0;i--){
			if(r[i+1]>=p[i]&&r[i+1]<=s[i])r[i]=r[i+1];
			else if(r[i+1]>s[i])r[i]=s[i];
			else if(r[i+1]<p[i])r[i]=p[i];
		}
		ll ans=0;
		for(int i=0;i<n;i++){
			int lev=min(l[i],r[i]);
			ans+=lev-p[i];
		}
		cout<<ans<<endl;
		system("pause");
	}
	return 0;
}
```

这道题的关键就是如何判断水位，上面从(i,h)发射的射线不会碰到天花板就是关键，这里让我们想到从左到右扫描一次，再从右向左扫描一次，这里就是模拟射线的形成，因为本题的数据量有10的6次方，但是这里扫描两次的复杂度均为O(n)，所以本题完美解决

---

### 13.Party Games（暴力穷举）

Uva1610

本来以为需要用理解来做，但是不管怎么样都过不了，但是我们发现数据量极小，不如使用暴力枚举

AC代码：

```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=1000+5;
string s[N];
int main(){
	ios::sync_with_stdio(false);cin.tie(0);cout.tie(0);
	int n;
	while(cin>>n&&n){
		for(int i=0;i<n;i++)cin>>s[i];
		sort(s,s+n);
		int m=n/2-1;
		int len=s[m].size();
		string a,ans;
		while(len>0){
			a=s[m].substr(0,len);
			a[a.size()-1]='Z';
			while(a[a.size()-1]>=s[m][a.size()-1]){//枚举
				if(a>=s[m]&&a<s[m+1])ans=a;
				a[a.size()-1]--;
			}
			len--;
		}
		cout<<ans<<endl;
	}
	system("pause");
	return 0;
}
```

---

### 14.Bits Equalizer（先后顺序的处理）

Uva12545

不难的一题，**但是tm的认真读题**没认真读题浪费大量的时间我超，这题不难，处理好各种操作的先后顺序即可

AC代码：

```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
int main(){
	ios::sync_with_stdio(false);cin.tie(0);cout.tie(0);
	int t;cin>>t;
	int ct=1;
	while(t--){
		string a,b;
		cin>>a>>b;
		int z=0,o=0,u=0;
		for(int i=0;i<a.size();i++){
			if(a[i]=='0')z--;
			if(a[i]=='1')o--;
			if(a[i]=='?')u++;
			if(b[i]=='0')z++;
			if(b[i]=='1')o++;
		}
		if(o<0||z+o!=u){cout<<"Case "<<ct++<<": "<<-1<<endl;continue;}
		int len=a.size(),ans=0;
		if(z<0){
			for(int i=0;i<len;i++){
				if(z<0&&a[i]=='0'&&b[i]=='1'){z++;ans++;o--;a[i]='1';}
			}
		}
		for(int i=0;i<len;i++){//先处理？对应0和1的情况
			if(o>0&&a[i]=='?'&&b[i]=='1'){o--;ans++;a[i]='1';u--;}
			if(z>0&&a[i]=='?'&&b[i]=='0'){z--;ans++;a[i]='0';u--;}
		}
		if(u>0){
			for(int i=0;i<len;i++){
				if(o>0&&a[i]=='?'){o--;ans++;a[i]='1';}
				if(z>0&&a[i]=='?'){z--;ans++;a[i]='0';}
			}
		}
		int cnt=0;
		for(int i=0;i<len;i++){
			if(a[i]!=b[i])cnt++;
		}
		ans+=cnt/2;
		cout<<"Case "<<ct++<<": "<<ans<<endl;
		
	}
	system("pause");
	return 0;
}
```

---

### 15.Erasing and Winning（维护最优解）

UVa11491

本题说实话不难，其基本思路就是尽量将序列中的最小的数先删除，怎么维护当前的最优解问题，是不是很像之前的（防线UVa1471），这里是当有更大的值出现时我们要将之前的小的值删除，因为更大的值才是更优解，这里用一个循环就能实现，注意要处理当删除位数与剩余位数相等时的情况，因为这时不能再添加了

AC代码：

```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=100000+5;
char s[N];
int main(){
	ios::sync_with_stdio(false);cin.tie(0);cout.tie(0);
	int n,d;
	while(cin>>n>>d&&(n||d)){
		char c;
		int del=0,j=0;
		for(int i=0;i<n;i++){
			cin>>c;
			while(j>0&&s[j-1]<c&&del<d){j--;del++;}
			if(d-del==n-i){del++;continue;}
			s[j++]=c;
		}
		s[j]='\0';//记得要补上字符串结尾，因我们进行了反复读写
		cout<<s<<endl;
	}
	system("pause");
	return 0;
}
```

---

### 16.Crane（数学推理分析）

用选择排序的思维来做，每次都将i位置的归位，第一次提交的时候出现RE是因为代码中仍然存在错误，下次出现这种情况还是要自己重写样例进行测试

```c
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=10000+10;//10000+5
int a[N],pos[N];
struct Point{//答案输出
	int x,y;
	Point(int x,int y):x(x),y(y){};
};
vector<Point> v;
int n,ans=0;
void swa(int p1,int p2){
	if(p2>p1+(n-p1+1)/2){
		int p=!((n-p1+1)%2)?p1:p1+1;
		v.push_back(Point(p,n));
		int m=(n-p+1)/2+p;
		for(int i=0;i<(n-p+1)/2;i++)swap(a[p+i],a[m+i]),swap(pos[a[p+i]],pos[a[m+i]]);
		ans++;
	}
	int p3=pos[p1];
	v.push_back(Point(p1,2*p3-p1-1));
	for(int i=0;i<p3-p1;i++)swap(a[p1+i],a[p3+i]),swap(pos[a[p1+i]],pos[a[p3+i]]);
	ans++;
}
int main(){
	ios::sync_with_stdio(false);cin.tie(0);cout.tie(0);
	int t;
    cin>>t;
	while(t--){
		v.clear();
		cin>>n;
		for(int i=1;i<=n;i++)cin>>a[i],pos[a[i]]=i;
		int i=1;
		ans=0;
		while(i<=n){if(a[i]!=i)swa(i,pos[i]);i++;}
		cout<<ans<<endl;
		vector<Point>::iterator it;
		for(it=v.begin();it!=v.end();it++)cout<<it->x<<" "<<it->y<<endl;	
	}
	system("pause");
	return 0;
}
```

---

### 17.Generating Permutations（逆向思维）

逆向思维，原本题目要求的是用1,2,3,···,n的排列生成所给的排列，本来这样是很难想的，但是如果我们这么想，如何用所给的排列用所给的操作还原成1,2,3,···,n这就是一个很简单的问题了，其中要注意的是，所有的操作都要反过来，比如原本是把第一个元素移到最后一个，现在就是把最后一个元素移到第一个，最后把答案输出就行了

AC代码：

```c++
#include<bits/stdc++.h>

using namespace std;
typedef long long ll;
const int N=300+5;
int a[N],n;
/*第一位移到最后为1，前后互换为2*/

bool judge(int ft){
	for(int i=1;i<=n;i++){
		if(a[ft]!=i)return true;
		ft+1>n?ft=1:ft++;
	}
	return false;
}
vector<int> v;
int main(){
	ios::sync_with_stdio(false);cin.tie(0);cout.tie(0);
	// ifstream in("C://Users//11763//Desktop//test//TestFile//in1.txt",ios::in);
	while(cin>>n&&n){
		v.clear();
		for(int i=1;i<=n;i++)cin>>a[i];
		int ft=1,next=ft+1;
		while(1){
			next=ft+1>n?1:ft+1;
			if(a[ft]!=n&&a[ft]>a[next]){swap(a[ft],a[next]);v.push_back(1);}
			else if(a[ft]==n||a[ft]<a[next]){v.push_back(2);ft=ft-1<1?n:ft-1;}
			if(!judge(ft))break;		
		}
		for(vector<int>::reverse_iterator it=v.rbegin();it!=v.rend();it++)cout<<*it;
		cout<<endl;
	}
	// in.close();
	system("pause");
	return 0;
}
```

下面附上别人的题解，真的是非常的厉害，不难发现这种排序题一般都是类似于冒泡排序或者选择排序的思路，用这种思路也许能写出更短更高效的代码

```c++
#includecbits/stdc++.h>
using namespace std;
int main(){
int n, a[300];string s;
while(scanf("%d", &n),n){
    for(int i=0;i<n;++i)
    scanf( "%d", a+i);
    s= "";
    for(int i=0;i<n-1; ++i)
        for(int j =n-1;j>= 0;--j){//冒泡排序
            s+='2';
            if(j !=n-1 && a[j+1]<a[j]){
            	swap(a[i],a[j+1]);s+= '1';
         	}
    	}
	reverse(s.begin(,s.end());//反转答案
    cout << s << endl;
}
return 0;

```

---

### 18.Guess（浮点数精度问题）

1.读题：本题的题目需要好好理解一下，刚开始给的是ID依次1~ n的得分，然后再给的是rank1~ rank n的ID，题目说的不是很清楚。
2.基本思路就是贪心。为了使后面选手的选择空间更大，rank靠前的选手因尽能力分高，这道题就仅此而已了。另外ID并列的情况也很容易分析，详情可以看代码。
3.本题要考虑浮点数精度的问题。比较明智的做法是，由于给的和输出的都是两位小数，所以输入时乘100，输出时除100

AC代码：

```c
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=17000+5;//17000+5
int s[N][3],sc[N];
int id[N];

bool chose(int v,int _id,int id1){
	//v=1能小于等于前者的成绩相同,v=2只能小于前者成绩
	double ans=-1,sum=0;
	for(int i=0;i<3;i++){
		sum+=s[id1][i];
		if(v==1&&sum<=sc[_id-1])ans=max(ans,sum);
		if(v==2&&sum<sc[_id-1])ans=max(ans,sum);
	}sum=0;
	for(int i=1;i<3;i++){
		sum+=s[id1][i];
		if(v==1&&sum<=sc[_id-1])ans=max(ans,sum);
		if(v==2&&sum<sc[_id-1])ans=max(ans,sum);
	}sum=0;
	int j=2;
	for(int i=0;i<2;i++){
		sum+=s[id1][j];
		if(v==1&&sum<=sc[_id-1])ans=max(ans,sum);
		if(v==2&&sum<sc[_id-1])ans=max(ans,sum);
		j=0;
	}
	if(ans!=-1)sc[_id]=ans;
	else return false;
	return true;
}

int main(){
	ios::sync_with_stdio(false);cin.tie(0);cout.tie(0);
	// ifstream in("C://Users//11763//Desktop//test//TestFile//in1.txt",ios::in);
	// ofstream out("C://Users//11763//Desktop//test//TestFile//out.txt",ios::out);
	int n,ct=1;
	while(cin>>n&&n){
		// fill(sc,sc+n,0);
		int ok=1;
		double f1,f2,f3;
		// cout<<"Case "<<ct++<<": ";
		for(int i=1;i<=n;i++)cin>>f1>>f2>>f3,s[i][0]=round(f1*100),s[i][1]=round(f2*100),s[i][2]=round(f3*100);
		for(int i=1;i<=n;i++)cin>>id[i];
		sc[1]=s[id[1]][0]+s[id[1]][1]+s[id[1]][2];
		for(int i=2;i<=n;i++){
			if(id[i]>id[i-1]){//后面的id大于前者时
				if(!chose(1,i,id[i])){cout<<"Case "<<ct++<<": "<<"No solution"<<endl;ok=0;break;}
			}else{
				if(!chose(2,i,id[i])){cout<<"Case "<<ct++<<": "<<"No solution"<<endl;ok=0;break;}
			}
		}
		if(!ok)continue;
		cout<<"Case "<<ct++<<": "<<fixed<<setprecision(2)<<(sc[n]/100.0)<<endl;
	}
	// cout<<endl;
	// in.close();
	// out.close();
	system("pause");
	return 0;
}
```

1.读入的浮点数往往有误差，比如9.53读成9.5299999999这种，这时候为了避免可以使用round函数，其功能是四舍五入。不过讲道理这种误差这么大的情况少见，所以只要记得遇到了可以用round()就可以了

---

### 19.K-Graph Oddity（简单dfs）

这题不难主要是处理邻接矩阵的问题，因为边太多了，开二维数组会报数组过大，所以这里我们用vector数组来存，注意vector的处理，同时这里因为一定有解，所以dfs不用回溯只要获得一种可能就行了

AC代码：

```c
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=10000+5;
int clr[N],k,n,m;//染色数组
vector<int> v[N];//邻接矩阵

int judge(int x,int y){//x当前点，y当前要染的颜色
	for(int i=0;i<v[x].size();i++){
		if(clr[v[x][i]]==y)return 0;
	}
	return 1;
}

void dfs(int x){//x为当前点
	for(int j=1;j<=k;j++){//先给自己染色
		if(judge(x,j)){clr[x]=j;break;}
	}
	for(int i=0;i<v[x].size();i++){
		if(!clr[v[x][i]])dfs(v[x][i]);
	}	
}

int main(){
	ios::sync_with_stdio(false);cin.tie(0);cout.tie(0);
	while(cin>>n>>m&&(n+m)){
		int a,b;
		k=0;
		fill(clr,clr+N,0);
		for(int i=1;i<=n;i++)v[i].clear();//清除数组
		for(int i=1;i<=m;i++)cin>>a>>b,v[a].push_back(b),v[b].push_back(a);//普通数组开不了这么大
		for(int i=1;i<=n;i++)if(k<v[i].size())k=v[i].size();
		if(k%2==0)k++;
		cout<<k<<endl;
		dfs(1);
		for(int i=1;i<=n;i++)cout<<clr[i]<<endl;
		cout<<endl;
	}
	system("pause");
	return 0;
}
```

### 20.Keep the Customer Satisfied（简单区间维护）

AC代码：

```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=800000+5;//800000+5
struct Od{
	int q,d;
	bool operator<(const Od &p)const{
		return d<p.d;
	}
}a[N];
priority_queue<int> q;
int main(){
	ios::sync_with_stdio(false);cin.tie(0);cout.tie(0);
	int t,n;cin>>t;
	while(t--){
		while(!q.empty())q.pop();
		cin>>n;
		for(int i=1;i<=n;i++)cin>>a[i].q>>a[i].d;
		sort(a+1,a+n+1);
		ll cnt=0,ans=0;
		for(int i=1;i<=n;i++){
			if(cnt+a[i].q<=a[i].d)cnt+=a[i].q,ans++,q.push(a[i].q);
			else{
				if(!q.empty()&&a[i].q<q.top())cnt-=q.top(),cnt+=a[i].q,q.pop(),q.push(a[i].q);
			}
		}
		cout<<ans<<endl;
		if(t)cout<<endl;
	}
	system("pause");
	return 0;
}
```

---

### 21.Meeting with Aliens（暴力+贪心）

这题说实话并不难，主要是环形模拟的部分还是花太多时间了，因为想不出更好的办法，以后对环形进行模拟还是预先求出每个元素应该在的位置（打出位置表，以值为索引），否则算法的可读性会大大下降

```c++
#include<bits/stdc++.h>
#define rep(i,n) for(int i=0;i<n;i++)
using namespace std;
typedef long long ll;
const int N=500+5;
int a[N],b[N],c[N];
int n;
int judge(int x){
    for(int i=0;i<n;i++)if(c[b[i]]!=i)return 0;
    return 1;
}

int solve(int x,int t){//x为起始位置，c为方向
    int cnt=0;
    if(t==1)for(int j=0;j<n;j++)c[j]=(x+j)%n;//以值为索引，获得目标位置(正向)
    else for(int j=0;j<n;j++)c[j]=(x+n-j)%n;//以值为索引，获得目标位置(反向)
    while(1){
        rep(i,n)if(c[b[i]]!=i)swap(b[i],b[c[b[i]]]),cnt++;
        if(judge(x))break;
    }
    return cnt;
}
int main(){
	ios::sync_with_stdio(false);cin.tie(0);cout.tie(0);
	while(cin>>n&&n){
		rep(i,n)cin>>a[i],a[i]--;
		int ans=0xfffffff;
		rep(i,n){//枚举起始位置
        rep(j,n)b[j]=a[j];//复制一份a
        int cnt;
        cnt=solve(i,1);
        ans=min(ans,cnt);
        rep(j,n)b[j]=a[j];
        cnt=solve(i,-1);
        ans=min(ans,cnt);
    	}
    	cout<<ans<<endl;
	}
	system("pause");
	return 0;
}
```



---

## 4.算法练习（蓝书）

### 1.Commando War（代数分析）

情况一：交换之前，任务 Y 比 X 先结束，如图 1-1（a）所示。不难发现，交换之后 X 的结束时间延后，Y 的结束时间提前，最终答案不会变好

情况二：交换之前，X 比 Y 先结束，因此交换后答案变好的充要条件是：交换后 X 的 结束时间比交换前 Y 的结束时间早（交换后 Y 的结束时间肯定变早了），如图 1-1（b）所 示。这个条件可以写成 B[Y]+B[X]+J[X]<B[X]+B[Y]+J[Y]，我们可以化简得J[X]<J[Y]，这就是我们贪心的依据

![image-20220509131043627](C:\Users\11763\AppData\Roaming\Typora\typora-user-images\image-20220509131043627.png)

AC代码：

```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=10000+5;
struct sd{
	int b,j;
	bool operator<(const sd &p)const{
		return j>p.j;
	}
};
sd A[N];
int main(){
	ios::sync_with_stdio(false);cin.tie(0);cout.tie(0);
	int n,ct=1;
	while(cin>>n&&n){
		for(int i=0;i<n;i++)cin>>A[i].b>>A[i].j;
		sort(A,A+n);
		int ans=A[0].b+A[0].j,btime=A[0].b;
		for(int i=1;i<n;i++){
			ans=max(ans,btime+A[i].b+A[i].j);
			btime+=A[i].b;
		}
		cout<<"Case "<<ct++<<": "<<ans<<endl;
	}
	system("pause");
	return 0;
}
```

---

### 2.Spreading the Wealth（代数分析+数形结合）

UVa11300

这道题目看起来很复杂，让我们慢慢分析。首先，最终每个人的金币数量可以计算出 来，它等于金币总数除以人数 n。接下来我们用 M 来表示每人最终拥有的金币数。 

假设有 4 个人，按顺序编号为 1, 2, 3, 4。假设 1 号给 2 号 3 枚金币，然后 2 号又给 1 号 5 枚金币，这实际上等价于 2 号给 1 号 2 枚金币，而 1 号什么也没给 2 号。这样，可以设 x2 表示 2 号给了 1 号多少个金币。如果 x2<0，说明实际上是 1 号给了 2 号-x2枚金币。x1， x3 和 x4 的含义类似。注意，由于是环形，x1 指的是 1 号给 4 号多少金币。 

现在假设编号为 i 的人初始有 Ai 枚金币。对于 1 号来说，他给了 4 号 x1枚金币，还剩 Ai-x1 枚；但因为 2 号给了他 x2 枚金币，所以最后还剩 A1-x1+x2枚金币。根据题设，该金币 数等于 M。换句话说，我们得到了一个方程：A1-x1+x2=M。 

同理，对于第 2 个人，有 A2-x2+x3=M。最终，我们可以得到 n 个方程，一共有 n 个变 量，是不是可以直接解方程组了呢？很可惜，还不行。因为从前 n-1 个方程可以推导出最 后一个方程（想一想，为什么）。所以，实际上只有 n-1 个方程是有用的。 

尽管无法直接解出答案，我们还是可以尝试着用 x1 表示出其他的 xi，则本题就变成了 单变量的极值问题。  对于第 1 个人，A1-x1+x2=M  x2=M-A1+x1=x1-C1（规定 C1=A1-M，下面类似） 对于第 2 个人，A2-x2+x3=M  x3=M-A2+x2=2M-A1-A2+x1=x1-C2 对于第 3 个人，A3-x3+x4=M  x4=M-A3+x3=3M-A1-A2-A3+x1=x1-C3 …  对于第 n 个人，An-xn+x1=M。这是一个多余的等式，并不能给我们更多的信息（想一 想，为什么）**因为用之前的所有A与M可以表示An，同时x1与xn是互为相反数的**

|第x个人| 关于x的方程 | 用x1来表示其他变量 |
| :---------: | :-------------: | ----------- |
| 1 | A1-x1+x2=M | x2=M-A1+x1=x1-C1 |
| 2 | A2-x2+x3=M | x3=2M-A1-A2+x1=x1-C2 |
| ··· | ··· | ··· |
| n | An-xn+x1=M |  |

Ps：C1=A1-M，C程递推关系

我们希望所有 xi 的绝对值之和尽量小，即$$min\left.\begin{cases} |x_1|+|x_2|+···+|x_n|\end{cases}\right\}=min\left.\begin{cases} |x_1|+|x_1-C_1|+···+|x_1-C_n|\end{cases}\right\}$$ 这里我们可以发现，这个式子的$|x_1-C_1|$的几何意义是数轴上点 $x_1$到$C_1$的距离，所以我们现在要找一个点，这个点到数轴上所有的点的距离最小，这里我们给出答案，到所有的点距离最短的点就是这些数的中位数，即最中间的点，下面给出证明

我们先把所有点画在一根数轴上

![image-20220509140611406](C:\Users\11763\AppData\Roaming\Typora\typora-user-images\image-20220509140611406.png)

任意找一个点，比如图 1-2 中的灰点。它左边有 4 个输入点，右边有 2 个输入点。把它 往左移动一点，不要移得太多，以免碰到输入点。假设移动了 d 单位距离，则灰点左边 4 个点到它的距离各减少了 d，右边的两个点到它的距离各增加了 d，但总的来说，距离之和 减少了 2d

如果灰点的左边有 2 个点，右边有 4 个点，道理类似，不过应该向右移动。换句话说， 只要灰点左右的输入点不一样多，就不是最优解。什么情况下左右的输入点一样多呢？如 果输入点一共有奇数个，则灰点必须和中间的那个点重合（中位数）；如果有偶数个，则 灰点可以位于最中间的两个点之间的任意位置（还是中位数）

在数轴上的所有点中，中位数离所有顶点的距离之和最小。凡是能转化为这个模型的题目都可以用中位数求解，并不只适用于本题，这也是一种我们需要掌握的思想

AC代码：

```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=1000000+5;
ll A[N],C[N];
int main(){
	ios::sync_with_stdio(false);cin.tie(0);cout.tie(0);
	int n;
	while(cin>>n){
		ll sum=0,m;
		for(int i=0;i<n;i++)cin>>A[i],sum+=A[i];
		m=sum/n;
		C[0]=A[0]-m;
		for(int i=1;i<n;i++)C[i]=C[i-1]+A[i]-m;
		sort(C,C+n);
		ll x=C[n/2];
		sum=0;
		for(int i=0;i<n;i++){
			sum+=abs(x-C[i]);
		}
		cout<<sum<<endl;
	}
	system("pause");
	return 0;
}
```



---

## 5.算法练习（洛谷）

### 1.涂国旗（贪心）

https://www.luogu.com.cn/problem/P3392

某国法律规定，只要一个由 N  M *N*×*M* 个小方块组成的旗帜符合如下规则，就是合法的国旗。（毛熊：阿嚏——）

- 从最上方若干行（至少一行）的格子全部是白色的；
- 接下来若干行（至少一行）的格子全部是蓝色的；
- 剩下的行（至少一行）全部是红色的；

现有一个棋盘状的布，分成了 N*N* 行 M*M* 列的格子，每个格子是白色蓝色红色之一，小 a 希望把这个布改成该国国旗，方法是在一些格子上涂颜料，盖住之前的颜色。

小a很懒，希望涂最少的格子，使这块布成为一个合法的国旗。

输入格式：第一行是两个整数 N,M*N*,*M*。接下来 N*N* 行是一个矩阵，矩阵的每一个小方块是`W`（白），`B`（蓝），`R`（红）中的一个。

输出格式：一个整数，表示至少需要涂多少块

AC代码：

```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=55;
int main(){
    int W[N]={0},B[N]={0},R[N]={0};//涂满0-i行所需的颜料
    int n,m;
    cin>>n>>m;
    char c;
    for(int i=1;i<=n;i++){
        for(int j=0;j<m;j++){
            cin>>c;
            if(c=='W')W[i]++;
            else if(c=='B')B[i]++;
            else R[i]++;
        }
        W[i+1]+=W[i];
        B[i+1]+=B[i];
        R[i+1]+=R[i];
    }
    int ans=0xfffffff;
    for(int i=1;i<n-1;i++)
        for(int j=i+1;j<n;j++){
            ans=min(B[i]+R[i]+W[j]+R[j]-W[i]-R[i]+W[n]-W[j]+B[n]-B[j],ans);
        }
    cout<<ans;
    system("pause");
    return 0;
}
```

题目的数据量为50，因为暴力枚举的复杂度为n的平方，不会超时，所以直接枚举所有可能取最小值

---

### 2.取数游戏（dfs）

https://www.luogu.com.cn/problem/P1123

题目描述

一个N  M*N*×*M*的由非负整数构成的数字矩阵，你需要在其中取出若干个数字，使得取出的任意两个数字不相邻（若一个数字在另外一个数字相邻88个格子中的一个即认为这两个数字相邻），求取出数字和最大是多少。

输入格式：第1行有一个正整数T*T*，表示了有T*T*组数据。对于每一组数据，第一行有两个正整数N*N*和M*M*，表示了数字矩阵为N*N*行M*M*列。接下来N*N*行，每行M*M*个非负整数，描述了这个数字矩阵。

输出格式：T*T*行，每行一个非负整数，输出所求得的答案。

AC代码：

```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;

int M[10][10],vis[10][10]={0};
int _x[]={-1,0,1},_y[]={-1,0,1};
int n,m,ans=0,tmp=0;
void dfs(int x,int y){
    if(y==m)return dfs(x+1,0);
    if(x==n){
        ans=max(tmp,ans);
        return;
    }
    if(!vis[x][y]){
        int k=0,vx[10],vy[10];
        for(int i=0;i<=2;i++)//先标记不能访问的位置
            for(int j=0;j<=2;j++)
                if(x+_x[i]>=0&&y+_y[j]>=0&&!vis[x+_x[i]][y+_y[j]]){
                    vis[x+_x[i]][y+_y[j]]=1;
                    vx[k]=x+_x[i],vy[k++]=y+_y[j];
                }
        tmp+=M[x][y];
        dfs(x,y+1);
        tmp-=M[x][y];
        for(int i=0;i<k;i++)vis[vx[i]][vy[i]]=0;//回溯
    }
    dfs(x,y+1); //下一个起点
}

int main(){
    ios::sync_with_stdio(false);cin.tie(0);cout.tie(0);
    int t;in>>t;
    while(t--){
        ans=0;
        in>>n>>m;
        fill(vis[0],vis[0]+10*10,0);
        for(int i=0;i<n;i++)
            for(int j=0;j<m;j++)in>>M[i][j];
        dfs(0,0);
        cout<<ans<<endl;
    }
    system("pause");
    return 0;
}
```

TLE代码：

```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;

int M[10][10],vis[10][10]={0};
int _x[]={-1,0,1},_y[]={-1,0,1};
int n,m,ans=0,tmp=0;
void dfs(int x,int y){
    int k=0,vx[10],vy[10];
    for(int i=0;i<=2;i++)//先标记不能访问的位置
        for(int j=0;j<=2;j++)
            if(x+_x[i]>=0&&y+_y[j]>=0&&!vis[x+_x[i]][y+_y[j]]){
                vis[x+_x[i]][y+_y[j]]=1;
                vx[k]=x+_x[i],vy[k++]=y+_y[j];
            }
    tmp+=M[x][y];
    for(int i=x;i<n;i++)//暴力穷举(超时)
        for(int j=0;j<m;j++)if(!vis[i][j])dfs(i,j);//还是有重复部分
    ans=max(ans,tmp);
    tmp-=M[x][y];
    for(int i=0;i<k;i++)vis[vx[i]][vy[i]]=0;//回溯
}

int main(){
    ios::sync_with_stdio(false);cin.tie(0);cout.tie(0);
    ifstream in("C://Users//11763//Desktop//test//TestFile//in.txt",ios::in);
    int t;in>>t;
    while(t--){
        ans=0;
        in>>n>>m;
        fill(vis[0],vis[0]+10*10,0);
        for(int i=0;i<n;i++)
            for(int j=0;j<m;j++)in>>M[i][j];
        for(int i=0;i<n;i++)
            for(int j=0;j<m;j++){
                dfs(i,j);
            }
        cout<<ans<<endl;
    }
    in.close();
    system("pause");
    return 0;
}
```

TLE代码出现的问题主要是进行了重复的搜索，已知我们是从左上角搜索到右下角，TLE代码的dfs部分对于下一个位置的搜索在**优化前**是从头开始搜索（从（0，0）开始寻找没被标记的点），相当于不断的重复搜索已经搜过的路径（类似于全排列搜索），这样必然会超时，因为宗旨是不能往前搜索，所以我们**再次优化**将第一个循环从x即当前行开始，第二个循环从0开始，这样就能过第4个测试点，但是这样还是有重复的部分，因为当前行的前y个我们已经搜过了，这样也是冗余的计算。这样的话我们不如**重新编写**代码，改为步进式，即每次都搜索下一个点，这样就避免了重复搜索的问题，最终AC

---

## 6.STL容器/C++补充

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
find()–返回一个指向被查找到元素的迭代器,查找失败则返回end()的迭代器
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
    bool operator<(const cmp &a)const{//重载<运算符
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
    bool operator()(const cmp &a)const{//重载<运算符
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

还有，cin是不能读取空格和换行符的，读取他们需要使用`getline()`

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

`lower_bound()`从数组的begin位置到end-1位置二分查找第一个**大于或等于**num的数字，找到返回该数字的地址，不存在则返回end

```c++
int a[]={1,2,3,4,5,7,8,9};
cout<<lower_bound(a,a+8,6)-a;
//输出5
```

`upper_bound()`从数组的begin位置到end-1位置二分查找第一个**大于**num的数字，找到返回该数字的地址，不存在则返回end。

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

**以不同进制输出数字**

```c++
	int i = 90;
	cout << i << endl;
	cout << dec << i << endl;
	cout << oct << i << endl;
	cout << hex << i << endl;
	cout << setiosflags(ios::uppercase);
	cout << hex << i << endl;
	cout << setbase(8) << i << endl;
```

![输出结果](https://img-blog.csdn.net/20180920214934343?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM1NDgxMTY3/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

其中，oct 是八进制输出， dec 是十进制（效果和默认一样）， hex 是十六进制输出（字母默认是小写字母）。这两个也包含在 std 中，即其全称分别是 std::oct 、 std::dec 、 std::hex ，这三个控制符包含在库 < iostream > 中。
setbase(n) 表示以 n 进制显示，包含在库 < iomanip > 中，n 只能取 8, 10, 16 三个值。
setiosflags(ios::uppercase) 表示将字母大写输出

**输出数字位数的控制**

```c++
	double i = 3333.1415926;
	cout << i << endl;
	cout << setprecision(3) << i << endl;
	cout << setprecision(9) << i << endl;
	cout << setiosflags(ios::fixed);
	cout << i << endl;
	cout << fixed << setprecision(3) << i << endl;
	cout << setprecision(9) << fixed <<  i << endl;
```

![输出结果](https://img-blog.csdn.net/20180920204637119?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM1NDgxMTY3/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

可以看出，C++默认浮点数输出有效位数是 6 位（若前面整数位数大于 6 位，使用科学计数法输出），而通过以下几种方式可以更改输出精度：
1.使用 setprecision(n) 即可设置浮点数输出的有效位数（若前面整数位数大于 n 位，使用科学计数法输出）
2.使用 setiosflags(ios::fixed) 或 fixed，表示对小数点后面数字的输出精度进行控制
所以，和 setprecision(n) 结合使用即可设置浮点数小数点后面数字的输出精度，位数不足的补零
以上均采用 “四舍五入” 的方法控制精度，三个控制符均包含在 std 命名空间中



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

内存占有率的问题就转化成红黑树 VS hash表 , 还是unorder_map占用的内存要高。
但是unordered_map执行效率要比map高很多
对于unordered_map或unordered_set容器，其遍历顺序与创建该容器时输入的顺序不一定相同，因为遍历是按照哈希表从前往后依次遍历的

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

### 9.queue和priority_queue容器

队列queue是一种先进先出(FIFO)的数据结构，基本操作如下

```c++
//队列
q.empty()               如果队列为空返回true，否则返回false
q.size()                返回队列中元素的个数
q.pop()                 删除队列首元素但不返回其值
q.front()               返回队首元素的值，但不删除该元素
q.push()                在队尾压入新元素
q.back()                返回队列尾元素的值，但不删除该元素
//优先队列
q.top()                 返回堆顶元素（即队列头部元素）
```

priority_queue是一种优先级高的元素先出队的数据结构，这里的数据优先级可以通过自带的定义或者通过复杂的计算定义，他是由堆实现的，优先队列具有队列的所有特性，包括基本操作，只是在队列的基础上加入了一个内部的排序，下面是priority_queue的定义

```c++
//升序队列，小顶堆
priority_queue <int,vector<int>,greater<int> > q;
//降序队列，大顶堆
priority_queue <int,vector<int>,less<int> >q;
//greater和less是std实现的两个仿函数（就是使一个类的使用看上去像一个函数。其实现就是类中实现一个operator()，这个类就有了类似函数的行为，就是一个仿函数类了）
//当需要用自定义的数据类型时才需要传入这三个参数，使用基本数据类型时，只需要传入数据类型，默认是大顶堆
```

---

### 10.三元运算符 

众所周知三元运算符`?`的写法

```c++
条件 ? 表达式1 ：表达式2 ;
```

三元运算符的整条语句的返回值为表达式的返回值，当条件为真，返回表达式1的值，反之则为表达式2的值

这里需要注意的点是，表达式可以为**赋值表达式**，**比较表达式**或者是一个**值**

一般来说我们不能直接将比较运算的值赋给一个变量，如

```c++
int a=1<2;//这是错误的
```

但是借助三元运算符，我们可以将这个表达式的返回值也就是`true`赋值给`a`

```c
int a= 1>0 ? 1>0 : 0;//a的值为true就是1
```

同时如果表达式为赋值表达式，也会将返回值赋值给变量

```c++
int a= 1>0 ? b=1 : 0;//b=1的返回值为1
```

这里需要注意的是，赋值表达式返回的值是等号右边的值

我们也可以用一个三元运算符替换一个if-else语句

```c++
ans==a ? L=mid+1 : R=mid;//二分中的一个语句
```

*用三元运算符模拟电路操作尤为方便*

---

### 11.fill()的使用

当我们想对一个容器填充数据时可以使用`fill()`

```c++
int array[8];//对普通数组填充
fill (array,array+4,5);   // myvector: 5 5 5 5 0 0 0 0
fill (array+3,array+6,8);   // myvector: 5 5 5 8 8 8 0 0
vector<int> myvector (8)//对vector数组填充
fill (myvector.begin(),myvector.begin()+4,5);   // myvector: 5 5 5 5 0 0 0 0
fill (myvector.begin()+3,myvector.end()-2,8);   // myvector: 5 5 5 8 8 8 0 0
//fill填充二维数组
int G[6][6];
fill(G[0],G[0]+6*6,6);
```

一般，我们可以用`fill()`来初始化数组

---

### 12.结构体重载与构造

一般来说，当我们自己定义了一个结构体，`sort()`等函数或者`<`等运算符是无法直接对结构体操作的，这时就需要我们在结构体中进行运算符重载

```c++
typedef struct{
    int num;
    bool operator<(const cmp &a)const{//重载<运算符
        if(num==a.num)return false;//当num值相等时不插入set
        else{
            //降序或者升序代码内容
        }
    }
    bool operator()(const cmp &a)const{//重载()运算符
        if(num==a.num)return false;//当num值相等时不插入set
        else{
            //降序或者升序代码内容
        }
    }
    bool operator+(const cmp &a)const{//重载+运算符
        return num+a.num;
    }
}cmp;
```

这里要注意的是不能少那两个`const`不然会出现错误

同时为了方便结构体的定义，我们也能在结构体中写一个构造函数让我们方便定义一个结构体

```c++
struct cmp{
    int num;
    cmp(int num):num(num){};
}
cmp a(1);//这样我们就能以这种方式定义一个结构体
```

---

### 13.C++读写文件

1.写文件

利用流对象`ofstream`

```c++
ofstream out("FilePath",ios::out);
//或者
ofstream out;
out.open("FliePath",ios::out);

out<<"要写入的内容"<<endl;
out.close();//记得关闭文件
```

2.读文件

利用流对象`ifstream`

```c++
ifstream in("FilePath",ios::int);
//或者
ifstream in;
in.open("FilePath",ios::in);
    //第一种方式
	char buf[1024] = { 0 };
	while (ifs >> buf){
		cout << buf << endl;
	}
	//第二种
	char buf[1024] = { 0 };
	while (ifs.getline(buf,sizeof(buf))){
		cout << buf << endl;
	}
	//第三种
	string buf;
	while (getline(ifs, buf)){
		cout << buf << endl;
	}
in.close();//记得关闭文件
```

文件打开方式补充

|  打开方式   |            解释            |
| :---------: | :------------------------: |
|   ios::in   |     为读文件而打开文件     |
|  ios::out   |     为写文件而打开文件     |
|  ios::ate   |      初始位置：文件尾      |
|  ios::app   |       追加方式写文件       |
| ios::trunc  | 如果文件存在先删除，再创建 |
| ios::binary |         二进制方式         |

**注意：** 文件打开方式可以配合使用，利用|操作符

**例如：**用二进制方式写文件 `ios::binary | ios:: out`

---

### 14.reverse()反转函数

使用`reverse()`可以将数组元素翻转

```c
int a[]={1,2,3};
reverse(a,a+3);
//a={3,2,1}
```

---

### 15.floor()，round()，ceil()数学函数用于浮点数操作

ceil(x)返回不小于x的最小整数值（然后转换为double型）。

floor(x)返回不大于x的最大整数值。

round(x)返回x的四舍五入整数值。

---

## 2.每月集训

### 1.位运算

按位与`&`如果两个数据均为1返回1，否则返回0

按位或`|`只要有一个值为1返回1

按位异或`^`两者相同返回0，否则返回1

左移`<<`左移一位相当于乘以2

右移`>>`右移一位相当于除以2

上面仅有左移和右移使用的情况较多，需要注意的是左移1位相当于该数乘以2，左移2位相当于该数乘以2*2。但**此结论只适用于该数左移时被溢出舍弃的高位中不包含1的情况**。由下表可以看出，64在左移1位后相当于乘2，左移2位后，值就等于0了

![image-20220519101724605](C:\Users\11763\AppData\Roaming\Typora\typora-user-images\image-20220519101724605.png)

练习

#### LC.191 位1的数量 

```c++
int hammingWeight(uint32_t n) {
        int ans=0;
        while(n){
            ans+=(n&1);
            n>>=1;
        }
        return ans;
    }
```

---

#### LC.137 只出现一次的数字Ⅱ

很有技巧的一到题目，通过统计每一位1的个数来计算，最后把结果拼接起来

```c++
int singleNumber(vector<int>& nums) {
        int cnt=0;
        for(int i=0;i<32;i++){
            int ans=0;
            for(int j=0;j<nums.size();j++){
                ans+=(nums[j]>>i)&1;
            }
            ans%=3;
            if(ans){cnt+=(unsigned int)1<<i;}
        }
        return cnt;
    }
```

---

#### LC.1442 形成两个异或相等数组的三元组数目

前缀和+枚举即可

```c++
int countTriplets(vector<int>& arr) {
        vector<int> v;//前缀数组
        v.push_back(0);
        for(int i=0;i<arr.size();i++){
            v.push_back(arr[i]^v[i]);
        }
        //枚举i,j,k
        int ans=0;
        for(int i=0;i<arr.size();i++){
            for(int j=i+1;j<arr.size();j++){
                for(int k=j;k<arr.size();k++){
                    int a=v[j]^v[i];
                    int b=v[k+1]^v[j];
                    if(a==b)ans++;
                }
            }
        }
        return ans;
    }
```

---

### 2.前缀和

#### LC.1094 拼车

这道题的关键是，求得每个时刻上，车上的人数，如果使用$log(n^2)$的枚举，很可能超时，所以这里我们提前算好每个时刻的操作，关键点就是上车和下车的时间节点

```c++
bool carPooling(vector<vector<int>>& trips, int capacity) {
        int num[1005];
        fill(num,num+1005,0);
        for(int i=0;i<trips.size();i++){
            int cnt=trips[i][0];
            int l=trips[i][1];
            int r=trips[i][2];
            num[l]+=cnt;
            num[r]-=cnt;
        }
        int sum=0;
        for(int i=0;i<1000;i++){
            sum+=num[i];
            if(sum>capacity)return false;
        }
        return true;
    }
```

---

### 3.递归

#### LC.565 嵌套数组

本题不难，主要是理解，不用清除之前遍历过的地方，因为每次进行嵌套，一定是一个环形，即每个下标开始递推出的答案是一个闭环，不管从其中哪个作为起点都只会得到同样的结果，所以这里我们不用重复清除vis数组

```c++
class Solution {//非递归
public:
    int arrayNesting(vector<int>& nums) {
        int vis[200005];
        fill(vis,vis+200005,0);
        int ans=0;
        for(int i=0;i<nums.size();i++){
            int j=i,cnt=0;
            while(!vis[j]){
                cnt++;
                vis[j]=1;
                j=nums[j];
            }
            ans=max(ans,cnt);
        }
        return ans;
    }
};
```

```c++
class Solution {//递归
    int Max, Cnt;
    int hash[200010];
    
    void dfs(vector<int>& nums, int u, int color) {
        if(hash[u] != -1) {
            return ;
        }
        hash[u] = color;
        ++Cnt;
        dfs(nums, nums[u], color);
    }
public:
    int arrayNesting(vector<int>& nums) {
        int i;
        int n = nums.size();
        int color = 0;
        Max = 0;
        memset(hash, -1, sizeof(hash));
        for(i = 0; i < n; ++i) {
            if(-1 == hash[i]) {
                Cnt = 0;
                dfs(nums, i, ++color);
                Max = max(Max, Cnt);
            }
        }
        return Max;
    }
};

```

---

#### LC.401 二进制手表

不算难，主要是细节的处理

```c
class Solution {
    string s;
    int hour[4]={8,4,2,1},vis[4]={0},vis1[20]={0};
    int cnt=0;
    int minn[6]={32,16,8,4,2,1},vis2[6]={0},vis3[65]={0};
    int cnt1=0;
    vector<string> h;
    vector<string> m;
    vector<string> ans;
    void dfs1(int n){
        if(cnt1>59||vis3[cnt1])return;
        if(!n){vis3[cnt1]=1;stringstream ss;ss<<cnt1;ss>>s;if(cnt1<10)s+=s,s[0]=0+'0';m.push_back(s);return;}
        for(int i=0;i<6;i++){
            if(!vis2[i]){
                vis2[i]=1;cnt1+=minn[i];
                dfs1(n-1);
                vis2[i]=0;cnt1-=minn[i];
            }
        }
    }
    void dfs(int n){
        if(cnt>11||vis1[cnt])return;
        if(!n){vis1[cnt]=1;stringstream ss;ss<<cnt;ss>>s;h.push_back(s);return;}
        for(int i=0;i<4;i++){
            if(!vis[i]){
                vis[i]=1;cnt+=hour[i];
                dfs(n-1);
                vis[i]=0;cnt-=hour[i];
            }
        }
    }

public:
    vector<string> readBinaryWatch(int turnedOn) {
        
        if(turnedOn>8)return ans;
        //枚举小时亮灯数
        for(int i=0;i<4;i++){
            h.clear();m.clear();
            fill(vis1,vis1+20,0);
            fill(vis3,vis3+60,0);
            if(i==0)h.push_back("0");
            else if(i<=turnedOn)dfs(i);
            if(turnedOn-i==0)m.push_back("00");
            else if(turnedOn-i>0)dfs1(turnedOn-i);
            for(int j=0;j<h.size();j++)
                for(int k=0;k<m.size();k++){
                    ans.push_back(h[j]+":"+m[k]);
                }
	    }
        return ans;
    }
};
```

---

[^1]: 朝0取整，指在数轴上整数向左取整，负数向右


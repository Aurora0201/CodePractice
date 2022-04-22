[toc]

# GitLearning

## 1.git初始化与文件提交

每一台新机器装载Git首先要让机器注册，使用以下命令

```git
$ git config --global user.name "Your Name"
$ git config --global user.email "email@example.com"
```

然后创建空目录

```
$ mkdir learngit
$ cd learngit
$ pwd
```

`pwd`是用于显示当前目录的

上面的`mkdir Learngit`是创建一个叫`Learngit`的文件夹

```git
$ git init
```

用`git init`命令来把这个目录变成仓库，创建完成后会多一个.git的目录

```git
$ git add "file"
```

用`git add`命令添加文件（注意文件一定要放在仓库内），然后再运行下面的命令

```git
$ git commit -m "  "
```

添加完后我们就应当进行提交操作，其中双引号的内容最好写对这次提交操作的简单说明，接着运行下面的命令

```git
$ git status
```

这个命令可以让我们掌握仓库当前的状态，每当我们对文件进行修改后，都应该运行这个指令查看当前仓库的状况，如果修改了却没有进行提交，后果会非常麻烦。

如果我们想查看文件被修改了的什么地方那就需要到下面的指令了

```git
$ git diff "file"
```

运行这个指令就可以让我们看到文件被修改了什么地方

知道文件修改了什么地方后我们就可以放心的把文件提交了同样执行下面的命令

```git
$ git add "file"
$ git commit -m"description"
```



---

## 2.版本回溯

首先我们要用下面的命令来获得从近到远的版本信息

```git
$ git log
```

如果嫌输出信息太多，看得眼花缭乱的，可以试试加上`--pretty=oneline`参数即

```git
$ git log --pretty=oneline
```

需要友情提示的是，你看到的一大串类似`1094adb...`的是`commit id`(版本号)

当我们需要回到上一个版本时就可以用`reset`命令

```git
$ git reset --hard HEAD^
```

这里的HEAD^表示的就是上一个版本，HEAD^^就是上上个版本，上100个版本就是HEAD~100

版本回退后我们来看看是不是成功了

```git
$ cat "file"
```

好家伙，成功了

这时我们其实还可以回退到更之前的版本，但是且慢，我们再用`git log`查看版本信息，我们发现刚刚最新的版本已经不见了

那怎么办，1，在命令窗口没有关闭的时候，我们可以通过上翻翻到之前版本的版本号，通过下面的命令来恢复。

```
$ git reset --hard "commit id"
```

2，运行下面的指令

```git
$ git reflag
```

这个命令会显示我们的操作历史，通过它来找到我们提交的记录进而找到版本号，这样我们就可以故技重施，回到之前的版本了。

---

## 3.撤销修改

当我们对文件误操作时有三种情况

1.我们只是在工作区中修改了文件还没有添加到暂存区，我们可以使用来撤销对工作区的修改

```
$ git restore file
```

2.我们已经添加到了暂存区，就是我们已经使用`add`命令，就可以用以下的命令来撤销添加到工作区的操作

```
$ git restore --staged file
```

3.我们已经`add`并且`commit`，我们就可以使用之前的版本回退功能将版本回退

```
$ git reset --hard HEAD^
```

---

## 4.删除文件

我们可以使用以下命令删除文件

```
$ git rm file
$ git commit -m "des"
```

`git rm test.txt` 相当于是删除工作目录中的`test.txt`文件,并把此次删除操作提交到了暂存区

若我们使用`git rm`删除了文件，当我们想恢复文件时，用以下的命令

```
$ git restore --staged file
$ git restore file
```

这里我们可以发现，与上文**撤销修改**异曲同工的是，我们都使用`git restore --staged file`来撤销暂存区的修改

对工作区的修改我们则使用`git restore file`，因为实际上这个命令是用版本库中的版本替换工作区中的内容 

还有一种情况，就是我们只是在工作区中误删了文件，这时我们并没有提交到暂存区，所以我们可以使用`git restore file`直接恢复

如果我们不使用`git rm`命令删除文件，可以先在暂存区将文件删除，然后使用`git add`命令提交，这与前者的操作效果是相同的

如果我们将暂存区的删除命令`git commit `了话，那么我们将无法恢复文件

---

## 5.远程仓库

首先我们要在`Github`创建一个`Repository`(仓库)，然后在本地运行下面的命令

```git
$ git remote add origin https://github.com/Aurora0201/LearnGit.git  或
$ git remote add origin git@github.com:Aurora0201/LearnGit.git
```

千万注意上面的`Aurora0201`和要改成我们自己的账户名,`LearnGit`要改成我们的仓库名，添加后，远程库的名字就叫`origin`这是默认叫法看到`origin`就会知道这是远程库

下一步就是把本地库的所有内容推送到远程库上，运行下面的命令

```git
$ git push -u origin master
```

把本地库的内容推送到远程，用`git push`命令，实际上是把当前分支`master`推送到远程。

由于远程库是空的，我们第一次推送`master`分支时，加上了`-u`参数，Git不但会把本地的`master`分支内容推送的远程新的`master`分支，还会把本地的`master`分支和远程的`master`分支关联起来，在以后的推送或者拉取时就可以简化命令。

当推送成功后，我们就可以看到`Github`远程库已经和本地一模一样,现在只要本地做了提交，就可以通过下面的命令把本地的master分支的最新修改推送至GitHub，现在我们就拥有了真正的分布式版本库。

```
$ git push origin master
```

---

## 6.克隆远程库

先在GitHub创建一个仓库

下一步我们用`git clone`命令克隆一个本地库

```
$ git clone git@github.com:Aurora0201/GitSkill.git
```

然后`Aurora0201`是我们的用户名，`GitSkill`是我们要克隆的仓库

## 7.分支管理

首先我们创建dev分支然后切换到dev分支，运行下面的命令

```
$ git checkout -b dev
```

加上`-b`参数表示创建并切换，相当于下面两条命令

```
$ git branch dev
$ git checkout dev
```

然后用`git branch`查看当前分支

```
$ git branch
* dev
  master
```

`git branch`命令会列出所有分支，当前分支前面会标一个`*`号。

然后，我们就可以在`dev`分支上正常提交，比如对`readme.txt`做个修改，加上一行：

```
Creating a new branch is quick.
```

然后提交：

```
$ git add readme.txt 
$ git commit -m "branch test"
```

现在，`dev`分支的工作完成，我们就可以切换回`master`分支：

```
$ git checkout master
```

切换回`master`分支后，再查看一个`readme.txt`文件，刚才添加的内容不见了！因为那个提交是在`dev`分支上，而`master`分支此刻的提交点并没有变：

现在，我们把`dev`分支的工作成果合并到`master`分支上：

```
$ git merge dev
```

`git merge`命令用于合并指定分支到当前分支。合并后，再查看`readme.txt`的内容，就可以看到，和`dev`分支的最新提交是完全一样的。注意到上面的`Fast-forward`信息，Git告诉我们，这次合并是“快进模式”，也就是直接把`master`指向`dev`的当前提交，所以合并速度非常快。

当然，也不是每次合并都能`Fast-forward`，我们后面会讲其他方式的合并。

合并完成后，就可以放心地删除`dev`分支了：

```
$ git branch -d dev
```

删除后，查看`branch`，就只剩下`master`分支了：

```
$ git branch
```

因为创建、合并和删除分支非常快，所以Git鼓励你使用分支完成某个任务，合并后再删掉分支，这和直接在`master`分支上工作效果是一样的，但过程更安全。

创建并切换到新的`dev`分支，可以使用：

```
$ git switch -c dev
```

直接切换到已有的`master`分支，可以使用：

```
$ git switch master
```

当我们再还没有合并分支时误删了分支，我们需要用下面的命令恢复分支然后，再合并分支

```
$ git branch "branch_name" "commit_id"
```

### 1.分支管理策略

通常，合并分支时，如果可能，Git会用`Fast forward`模式，但这种模式下，删除分支后，会丢掉分支信息，如果要强制禁用`Fast forward`模式，Git就会在merge时生成一个新的commit，这样，从分支历史上就可以看出分支信息

使用的方法如下

先在分支上`add `和 `commit`文件，然后，现在我们再转到`master`上，进行合并操作，代码如下

```
$ git add file (dev)
$ git commit -m"des" (dev)
$ git merge --no-ff -m"des" dev (master)
```

注意，这里第二次的描述是对应此次合并操作的commit

![git-br-policy](https://www.liaoxuefeng.com/files/attachments/919023260793600/0)

在实际开发中，我们应该按照几个基本原则进行分支管理：

首先，`master`分支应该是非常稳定的，也就是仅用来发布新版本，平时不能在上面干活；

那在哪干活呢？干活都在`dev`分支上，也就是说，`dev`分支是不稳定的，到某个时候，比如1.0版本发布时，再把`dev`分支合并到`master`上，在`master`分支发布1.0版本；

你和你的小伙伴们每个人都在`dev`分支上干活，每个人都有自己的分支，时不时地往`dev`分支上合并就可以了。

合并分支时，加上`--no-ff`参数就可以用普通模式合并，合并后的历史有分支，能看出来曾经做过合并，而`fast forward`合并就看不出来曾经做过合并。

### 2.分支冲突

当我们在dev分支提交了一次操作，同时我们又在master分支提交了一次操作，这时，如果我们合并dev分支，就会出现冲突，因为版本库中出现了两个不同的版本，git不知道如何合并，此时就需要我们手动将错误修改然后再次合并


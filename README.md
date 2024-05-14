# gittest
测试是否push？
具体步骤在goodnotes上有标明！

提示443 链接超时的解决
- https://blog.csdn.net/zpf1813763637/article/details/128340109

- sudo vi /etc/hosts
添加了 ping github.com的IP地址

添加本地代码

## Git:execute git fail
commit中存在大文件，出现的错误
表格制作
                                            |image number|
| sample | clsId- | clsName- | instanceNum | train | test | val | 
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| pin| 0-defect-6011 | 1-rust-2000 |2-uninstal-1832| 6579 | 1880 | 940 | 
| Einsulator | 3-burn-475 | 4-defect-508 | 5-dirty-440 | 951 | 272 | 137 | 

## 神经网络可视化工具汇总
- https://cloud.tencent.com/developer/article/2333299

· 直接导入权重pt文件即可显示网络结构(网页在线版)，https://github.com/lutzroeder/Netron?tab=readme-ov-file

· 自己画图时，可参考的形状：https://docs.google.com/presentation/d/11mR1nkIR9fbHegFkcFq8z9oDQ5sjv8E3JJp1LfLGKuk/edit#slide=id.g78327f1586_217_712

###### command useless
      python train.py --yaml ultralytics/cfg/models/v8/yolov8-dyhead.yaml  --info --project runs/train

# github🔗
    ssh -T git@github.com
    cd .git
    ls
    cat config 
    ###
        [core]
        repositoryformatversion = 0
        filemode = true
        bare = false
        logallrefupdates = true
        [remote "origin"]
            url = https://github.com/LiuJiaji1999/power.git
            fetch = +refs/heads/*:refs/remotes/origin/*
        [branch "main"]
            remote = origin
            merge = refs/heads/main
    ###
    vim config 
        url = git@github.com:LiuJiaji1999/power.git

    cd ~/.ssh
    ls
    cat id_ras.pub # github设置中的remote-ssh
    
- A：Added
        
    表示该文件是新添加的文件，已经被Git跟踪，并且将会包含在下一次的提交中。当使用git add命令将新文件添加到暂存区后，文件的状态会从U（Untracked）变为A（Added）。
- U：Untracked
    
    表示该文件是未被Git跟踪的文件，Git不会自动将其包含在版本控制中。这意味着该文件不会被提交到版本库中，也不会被包含在Git的快照中。如果希望Git开始跟踪该文件，需要使用git add命令将其添加到暂存区，然后文件的状态会从U（Untracked）变为A（Added）。
- M：Modified

    表示该文件已被修改。当对已跟踪的文件进行了修改后，文件的状态会从A（Added）变为M（Modified）。这意味着该文件在上一次提交之后发生了变化，但尚未被添加到暂存区。


# gittest
测试是否push？
具体步骤在goodnotes上有标明！

提示443 链接超时的解决
- https://blog.csdn.net/zpf1813763637/article/details/128340109

- sudo vi /etc/hosts
添加了 ping github.com的IP地址

添加本地代码

# Git:execute git fail
commit中存在大文件，出现的错误
表格制作
                                            |image number|
| sample | clsId- | clsName- | instanceNum | train | test | val | 
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| pin| 0-defect-6011 | 1-rust-2000 |2-uninstal-1832| 6579 | 1880 | 940 | 
| Einsulator | 3-burn-475 | 4-defect-508 | 5-dirty-440 | 951 | 272 | 137 | 



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
    
# conda activate base
import py7zr
with py7zr.SevenZipFile("/Users/rl/Documents/PhD-student/Untitled_Folder/MutilModel-494882733.7z", mode='r', password='XK3KUAtEFLnXPhW8915PROP353b2b4ee') as z:
    z.extractall("/Users/rl/Documents/PhD-student/Untitled_Folder/")
    print('解压成功')
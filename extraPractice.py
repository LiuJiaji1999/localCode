import os
 
def count_files_in_folder(folder_path):
    files = os.listdir(folder_path)
    count = len(files)
    return count

# Annotations JPEGImages
count = count_files_in_folder('/Users/rl/Documents/PhD_student/电力领域-dataset、discuss/绝缘子/均压环移位/xml')
print(count)
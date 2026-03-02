import os
import xml.etree.ElementTree as ET
import io
find_path='./CPLID/Defective_Insulators/labels/defect/' #xml所在的文件
savepath='./CPLID/Defective_Insulators/labels/defect_txt/' # 保存文件

class Voc_Yolo(object):
    def __init__(self,find_path):
        self.find_path=find_path

    def Make_txt(self,outfile):
        out=open(outfile,'w')
        print("创建成功：{}".format(outfile))
        return out
    
    def Work(self,count):
    #找到文件路径
        for root,dirs,files in os.walk(self.find_path):
        #找到文件目录中每一个xml文件
            for file in files:
                print(file[:3])
            #记录处理过的文件
                count += 1
                #输入、输出文件定义
                input_file = find_path+file
                outfile = savepath + file[:-4] + '.txt'
                #新建txt文件，确保文件正常保存
                out = self.Make_txt(outfile)
                #分析xml树，取出w_image、h_image
                tree = ET.parse(input_file)
                root = tree.getroot()
                size = root.find('size')
                w_image = float(size.find('width').text)
                h_image = float(size.find('height').text)
                #继续提取有效信息来计算txt中的四个数据
                for obj in root.iter('object'):
                    #将类型提取出来，不同目标类型不同，本文仅有一个类别->0
                    classname = obj.find('name').text
                    if(classname == "insulator"):
                        cls_id = 0
                    else:
                        cls_id = 1   
                    # cls_id = classname
                    xmlbox = obj.find('bndbox')
                    x_min = float(xmlbox.find('xmin').text)
                    x_max = float(xmlbox.find('xmax').text)
                    y_min = float(xmlbox.find('ymin').text)
                    y_max = float(xmlbox.find('ymax').text)
                    #计算公式
                    x_center = ((x_min+x_max)/2-1)/w_image
                    y_center = ((y_min+y_max)/2-1)/h_image
                    w = (x_max-x_min)/w_image
                    h = (y_max-y_min)/h_image
                    #文件写入
                    # out.write(str(cls_id)+""+str(x_center)+""+str(y_center)+""+str(w)+""+str(h)+'\n')
                    out.write(str(cls_id)+" "+str(x_center)+" "+str(y_center)+" "+str(w)+" "+str(h)+'\n')
       
                out.close()
        return count

if __name__ == "__main__":
    data = Voc_Yolo(find_path)
    number=data.Work(0)
    print(number)


### 合并两个tx
# import os
 
# def merge_txt_files(folder1_path, folder2_path, merged_folder_path):
#     # 确保输出文件夹存在
#     os.makedirs(merged_folder_path, exist_ok=True)
    
#     # 获取两个文件夹中所有txt文件的名称
#     file_names = {f for f in os.listdir(folder1_path) if f.endswith('.txt')}
    
#     # 合并文件内容
#     for file_name in file_names:
#         file1_path = os.path.join(folder1_path, file_name)
#         file2_path = os.path.join(folder2_path, file_name)
#         merged_file_path = os.path.join(merged_folder_path, file_name)
        
#         with open(file1_path, 'r', encoding='utf-8') as f1, open(file2_path, 'r', encoding='utf-8') as f2, open(merged_file_path, 'w', encoding='utf-8') as out_file:
#             # out_file.write(f1.read() + '\n' + f2.read()) # 换行 多一空行
#             out_file.write(f1.read() + f2.read())
 
 
# # 使用示例
# folder1_path = '/Users/rl/Documents/PhD_student/Untitled_Folder/CPLID/Defective_Insulators/labels/defect_txt'
# folder2_path = '/Users/rl/Documents/PhD_student/Untitled_Folder/CPLID/Defective_Insulators/labels/insulator_txt'
# merged_folder_path = '/Users/rl/Documents/PhD_student/Untitled_Folder/CPLID/Defective_Insulators/labels/merge_txt'
# merge_txt_files(folder1_path, folder2_path, merged_folder_path)

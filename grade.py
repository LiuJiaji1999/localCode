import pandas as pd
import re

# 读取A和B两个xlsx表格，并指定引擎
try:
    a_df = pd.read_excel('/Users/rl/Documents/PhD_student/Untitled_Folder/knowledge/final.xlsx', engine='openpyxl')
    b_df = pd.read_excel('/Users/rl/Documents/PhD_student/Untitled_Folder/knowledge/mooc.xlsx', engine='openpyxl')
except Exception as e:
    print(f"Error reading Excel files: {e}")
    raise

# 假设A表格的姓名列为 '姓名'，学号列为 '学号'
# 假设B表格的姓名列为 '姓名学号'，成绩列为 '成绩'
a_name_col = '姓名'
a_id_col = '学号'
b_name_id_col = '姓名'
b_score_col = '成绩'

# 使用正则表达式提取B表格的姓名和学号
def extract_name_id(name_id_str):
    # print(name_id_str)
    match1 = re.match(r'(\d+)(\D+)', name_id_str)
    if match1:
        # print(match1.groups())
        return match1.groups()
    
    # 匹配姓名在前，学号在后的情况
    match2 = re.match(r'(\D+)(\d+)', name_id_str)
    if match2:
        # print(match2.groups())
        return match2.groups()[::-1]  # 反转元组顺序，保持学号在前，姓名在后

    # 匹配只有姓名的情况
    match3 = re.match(r'(\D+)', name_id_str)
    if match3:
        # print(match3.group(1))
        return None, match3.group(1)
    # return None, None


# 将B表格的姓名学号列拆解为姓名和学号
# b_df['学号'], b_df['姓名'] = zip(*b_df[b_name_id_col].apply(extract_name_id))
print(b_df['姓名'])
# print(pd.Series(extract_name_id(b_df['姓名'].apply)))
b_df[['学号', '姓名']] = b_df['姓名'].apply(lambda x: pd.Series(extract_name_id(x)))

# print(b_df)
# # 遍历B表格，检查拆解后的姓名和学号是否在A表格中存在
# result_rows = []
# for _, row in b_df.iterrows():
#     b_name = row['姓名']
#     b_id = row['学号']
#     if not a_df[(a_df[a_name_col] == b_name) & (a_df[a_id_col] == b_id)].empty:
#         result_rows.append(row)

# # 创建包含结果的DataFrame
# result_df = pd.DataFrame(result_rows, columns=[b_name_id_col, b_score_col])

# # 保存结果到新的xlsx文件，并指定引擎
# try:
#     result_df.to_excel('result.xlsx', index=False, engine='openpyxl')
# except Exception as e:
#     print(f"Error saving result to Excel: {e}")
#     raise

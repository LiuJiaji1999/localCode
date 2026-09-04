# import pandas as pd
# import re

# # 读取A和B两个xlsx表格，并指定引擎
# try:
#     a_df = pd.read_excel('/Users/rl/Documents/PhD_student/Untitled_Folder/knowledge/final.xlsx', engine='openpyxl')
#     b_df = pd.read_excel('/Users/rl/Documents/PhD_student/Untitled_Folder/knowledge/mooc.xlsx', engine='openpyxl')
# except Exception as e:
#     print(f"Error reading Excel files: {e}")
#     raise

# # 假设A表格的姓名列为 '姓名'，学号列为 '学号'
# # 假设B表格的姓名列为 '姓名学号'，成绩列为 '成绩'
# a_name_col = '姓名'
# a_id_col = '学号'
# b_name_id_col = '姓名'
# b_score_col = '成绩'

# # 使用正则表达式提取B表格的姓名和学号
# def extract_name_id(name_id_str):
#     # print(name_id_str)
#     match1 = re.match(r'(\d+)(\D+)', name_id_str)
#     if match1:
#         # print(match1.groups())
#         return match1.groups()
    
#     # 匹配姓名在前，学号在后的情况
#     match2 = re.match(r'(\D+)(\d+)', name_id_str)
#     if match2:
#         # print(match2.groups())
#         return match2.groups()[::-1]  # 反转元组顺序，保持学号在前，姓名在后

#     # 匹配只有姓名的情况
#     match3 = re.match(r'(\D+)', name_id_str)
#     if match3:
#         # print(match3.group(1))
#         return None, match3.group(1)
#     # return None, None


# # 将B表格的姓名学号列拆解为姓名和学号
# # b_df['学号'], b_df['姓名'] = zip(*b_df[b_name_id_col].apply(extract_name_id))
# print(b_df['姓名'])
# # print(pd.Series(extract_name_id(b_df['姓名'].apply)))
# b_df[['学号', '姓名']] = b_df['姓名'].apply(lambda x: pd.Series(extract_name_id(x)))

# # print(b_df)
# # # 遍历B表格，检查拆解后的姓名和学号是否在A表格中存在
# # result_rows = []
# # for _, row in b_df.iterrows():
# #     b_name = row['姓名']
# #     b_id = row['学号']
# #     if not a_df[(a_df[a_name_col] == b_name) & (a_df[a_id_col] == b_id)].empty:
# #         result_rows.append(row)

# # # 创建包含结果的DataFrame
# # result_df = pd.DataFrame(result_rows, columns=[b_name_id_col, b_score_col])

# # # 保存结果到新的xlsx文件，并指定引擎
# # try:
# #     result_df.to_excel('result.xlsx', index=False, engine='openpyxl')
# # except Exception as e:
# #     print(f"Error saving result to Excel: {e}")
# #     raise

# conda actiavte torch107
import pandas as pd

# ========= 只需要修改这里 =========
模板文件 = "/Users/rl/Desktop/2026全日制专硕作业要求-大数据技术及应用/成绩导入模板信息.xls"
成绩文件 = "/Users/rl/Desktop/2026全日制专硕作业要求-大数据技术及应用/工作簿1.xlsx"
输出文件 = "/Users/rl/Desktop/2026全日制专硕作业要求-大数据技术及应用/成绩导入完成.xlsx"

模板工作表 = "成绩"
成绩工作表 = 0       # 0 表示第一张工作表

# 模板中的列名
模板学号列 = "xh(学号)"
模板姓名列 = "xm(姓名)"
模板成绩列 = "cj(成绩)"

# 成绩表中的列名
成绩表姓名列 = "姓名"
成绩表学号列 = "学号"
成绩表成绩列 = "成绩"
# ================================


def clean_id(x):
    """统一学号格式，避免 Excel 把学号读成 12345.0"""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def clean_name(x):
    """清理姓名前后空格"""
    if pd.isna(x):
        return ""
    return str(x).strip()


# 读取模板所有工作表，方便最后一起输出
模板所有表 = pd.read_excel(模板文件, sheet_name=None, dtype=object)

if 模板工作表 not in 模板所有表:
    raise ValueError(f"模板中找不到工作表：{模板工作表}")

模板 = 模板所有表[模板工作表].copy()

# 读取成绩表
成绩表 = pd.read_excel(成绩文件, sheet_name=成绩工作表, dtype=object)

# 标准化学号、姓名
模板["_学号"] = 模板[模板学号列].apply(clean_id)
模板["_姓名"] = 模板[模板姓名列].apply(clean_name)

成绩表["_学号"] = 成绩表[成绩表学号列].apply(clean_id)
成绩表["_姓名"] = 成绩表[成绩表姓名列].apply(clean_name)


# ---------- 1. 学号 + 姓名精确匹配 ----------
成绩字典 = {}

for _, row in 成绩表.iterrows():
    学号 = row["_学号"]
    姓名 = row["_姓名"]
    成绩 = row[成绩表成绩列]

    if 学号:
        成绩字典[(学号, 姓名)] = 成绩


def 查成绩(row):
    学号 = row["_学号"]
    姓名 = row["_姓名"]

    # 优先按 学号+姓名 匹配
    if (学号, 姓名) in 成绩字典:
        return 成绩字典[(学号, 姓名)]

    # 如果成绩表中该学生没有学号，则允许按唯一姓名匹配
    候选 = 成绩表[
        (成绩表["_姓名"] == 姓名) &
        (成绩表["_学号"] == "")
    ]

    if len(候选) == 1:
        return 候选.iloc[0][成绩表成绩列]

    # 匹配不到或出现重名，不自动填写
    return None


# 填入成绩
模板[模板成绩列] = 模板.apply(查成绩, axis=1)

# 删除辅助列
模板.drop(columns=["_学号", "_姓名"], inplace=True)

# 放回原工作簿
模板所有表[模板工作表] = 模板


# 输出为新的 xlsx
with pd.ExcelWriter(输出文件, engine="openpyxl") as writer:
    for 工作表名, df in 模板所有表.items():
        df.to_excel(writer, sheet_name=工作表名, index=False)


# 输出统计
已填写 = 模板[模板成绩列].notna().sum()
未填写 = 模板[模板成绩列].isna().sum()

print("处理完成！")
print(f"成功填写成绩：{已填写} 人")
print(f"未找到成绩：{未填写} 人")
print(f"输出文件：{输出文件}")

# 显示未匹配人员
if 未填写:
    print("\n以下学生未能自动匹配：")
    print(
        模板.loc[
            模板[模板成绩列].isna(),
            [模板学号列, 模板姓名列]
        ].to_string(index=False)
    )

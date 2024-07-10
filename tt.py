import codecs
import os

# with open('example.txt', 'a') as file:
#     file.write('Hello, Python!')

with open('example.txt', 'a+') as file:
    file.write('\nAppend this line.')
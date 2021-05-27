'''
Created on May 14, 2021

@author: miim
'''
import os

FOLDER = "repos"
EXT = ".rb"
NEW_LINE_REPLACEMENT = '<N>'
MAX_LENGTH = 1024
MIN_LENGTH = 100

def split(data, data_set):
    split_data = data.split(f"{NEW_LINE_REPLACEMENT}{NEW_LINE_REPLACEMENT}")
    accumilator = ""
    for s in split_data:
        s = accumilator + s + f"{NEW_LINE_REPLACEMENT}{NEW_LINE_REPLACEMENT}"
        length = len(s)
        if MIN_LENGTH < length < MAX_LENGTH :
            data_set.write(s + "\n")
            accumilator = ""
        elif length < MIN_LENGTH :
            accumilator = s;
        else:
            break
    

def prepDataForExt(folder,ext):
    counter = 0
    with open("data_set.txt", mode="a", encoding="utf-8") as data_set:
        for dirpath, _, filenames in os.walk(folder) :
            data_set.flush()
            for filename in filenames:
                counter+=1
                if (counter%20==0):
                    print('.', end =" ")
                if (counter%1000==0):
                    print()
                    
                try:
                    fullpath = os.path.join(dirpath,filename)
                    if filename.endswith(ext):
                        data = open(fullpath, mode="r", encoding="utf-8").read()
                        data = data.replace('\n',NEW_LINE_REPLACEMENT)
                        length = len(data)
                        if MIN_LENGTH < length < MAX_LENGTH :
                            data_set.write(data + "\n")
                        elif length > MAX_LENGTH:
                            split(data,data_set)
                except Exception as e:
                    print(fullpath)
                    print(e)
        data_set.close()            
                

prepDataForExt(FOLDER, EXT)                
    
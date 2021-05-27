'''
Created on May 9, 2021

@author: miim
'''
import os

def deletePath(path):
    try:
        os.remove(path)
    except:
        print("error")
    

def cleanFolderExcpet(folder, ext) :
    for dirpath, _, filenames in os.walk(folder) :
        for filename in filenames:
            fullpath = os.path.join(dirpath,filename)
            if not filename.endswith(ext):
                deletePath(fullpath)
                
        
        

    

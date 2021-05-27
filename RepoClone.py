'''
Created on May 5, 2021

@author: miim
'''
from github import Github
from datetime import datetime
import time
import os
from PostProcess import cleanFolderExcpet

TOKEN = open("token","r").read()
git = Github(TOKEN)
NUMBER_OF_DAYS = 5
LANGUAGE = "ruby"
EXT = ".rb"

def  handle_results(res):
    for repo in res :
        destFolder = f"repos/{repo.owner.login}/{repo.name}"
        clone_str = f"git clone {repo.clone_url} {destFolder}"
        os.system(clone_str)
        print(f"Cleaning =====> {destFolder}")
        cleanFolderExcpet(destFolder, EXT)

end_time = time.time()
start_time = end_time - (24 * 60 * 60 )


for i in range(NUMBER_OF_DAYS):
    start_time_str = datetime.utcfromtimestamp(start_time).strftime("%Y-%m-%d")
    end_time_str = datetime.utcfromtimestamp(end_time).strftime("%Y-%m-%d")
    query_str = f"language:{LANGUAGE} created:{start_time_str}..{end_time_str}"
    end_time -= 2 * (24 * 60 * 60 )
    start_time = end_time - (24 * 60 * 60 )
    print(query_str)
    res = git.search_repositories(query_str)
    handle_results(res)
    
    

        
    



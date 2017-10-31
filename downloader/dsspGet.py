
# coding: utf-8

# In[1]:


if __name__ != '__main__':
    raise Exception("`dataGet.py` cannot be imported as a module. Use it as a commandline tool.")

import urllib 
import sys

# In[2]:

# In[3]:

try:
    
    configure = sys.argv[1]

except:
    raise ValueError("Configure file not found! Please check you input arguments")

if configure == '-h':
        print('''use this script in this way:
    
    python dsspGet.py <configure JSON>
        
        JSON format:

        {"pageUrl":"xxx",
         "toStoragePath":"xxx",
         "toUnZippedPath":"xxx"}''')
        sys.exit(1)

class closure:
    pass

import json
with open(configure, 'wb') as toRead:
    conf = json.load(toRead)
    Env  = closure()
    assert set(conf.keys()) == {'pageUrl', 'toStoragePath', 'toUnZippedPath'}
    for key, value in conf.items():
        setattr(Env, key, value)

    
# class Env:
#     pageUrl = r"ftp://ftp.cbi.pku.edu.cn/pub/database/DSSP/20130820" #数据目录地址
#     toStoragePath = r"H:\BioDatas\DSSP"     #数据压缩包下载路径
#     toUnZippedPath=r"H:\BioDatas\DSSP\UnZipped"  #解压路径



# In[4]:


directory = urllib.request.urlopen(Env.pageUrl)
dirs = directory.read().decode("utf-8") # 包含所有压缩文件名的字符串。


# In[4]:


import re


# In[5]:


# 提取文件名
matcher = re.compile("([\w]+\.dssp\.gz)") 
# Test Regex Expr.
fileNames = matcher.findall(dirs)


# In[6]:


import gzip
def storeOneFile(filename):
    url = f"{Env.pageUrl}/{filename}"
    try: 
        content = urllib.request.urlopen(url).read()
    except:
        print(f"error find: in storing processing. filename : {filename}")
        return
    
    zipFile   = f"{Env.toStoragePath}/{filename}"
    unZipFile = f'{Env.toUnZippedPath}/{filename.replace(".gz","")}'
    
    with open(zipFile,"wb") as f:
        f.write(content)
    outUnZippedFile = gzip.GzipFile(zipFile)
    
    with open(unZipFile,'w+',encoding='utf-8') as f:
        f.write(outUnZippedFile.read().decode("utf-8"))
    outUnZippedFile.close()


# test : storeOneFile("101m.dssp.gz")
for task in map(storeOneFile, fileNames): pass





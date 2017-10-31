
# coding: utf-8

# In[11]:


from typing import List
from tools.Dict import Dict as UDict
from copy import deepcopy
from itertools import repeat

import re
tokenize = re.compile("[\w\.,]+")

class Preprocessing:
    @staticmethod
    def getColumnsAndIndex(row):
        keys = tokenize.findall(row)
        res = []
        status = False
        begin = -1
        for i,ch in enumerate(row):
            if status:
                if ch!=" ":
                    continue
                else:
                    status = False
                    res.append(UDict(inf=begin, sup=(i-1) ) )
            else:
                if ch!=" ":
                    status= True
                    begin = i
                else:
                    continue
        return res
    @staticmethod
    def resolveSpace(x):
        if x.startswith(" "):
            for i,ch in enumerate(x):
                if ch!=' ':
                    return x[i:]
            return None
        return x

class Parser:
    def __init__(self, colNamePart:str, backend:str ):
        self.colNamePart = colNamePart 
        self.backend = backend
    def explain(self,article:str) -> List[List[str]]:
        
        if not hasattr(self,'res'):
            lines   :List[str]   = article.splitlines()
            part    :     str    = self.colNamePart
            colNames:     str    = [line for line in lines if part in line][0]
            self.res:List[UDict] = Preprocessing.getColumnsAndIndex(colNames)
            self.columns_row:str = colNames
                
        def cellunit(line, tups):
            tupLast, tup = tups
            if line[tup.inf]==' ':
                return line[tup.inf:tup.sup+1]
            return Preprocessing.resolveSpace(line[tupLast.sup+1:tup.sup+1])
        columns_row : str    = self.columns_row
        
        source  :List[str]   = lines[lines.index(columns_row):]
        tupsS = list(zip(self.res[:-1],self.res[1:]))
        retIter = map(lambda line: map(cellunit, repeat(line), tupsS),  source)
        return  [list(row) for row in retIter]
        


# In[12]:


import pandas as pd
backend_sign = dict(dssp = "  #  RESIDUE AA STRUCTURE BP1 BP2  ACC     N-H-->O ", 
               # part of the row which defines the attributes of our datas.
              )
               
def bio_parse(filename, backend='dssp'):
    with open(filename) as toRead:
        string = toRead.read()
        parser = Parser(colNamePart=backend_sign[backend], backend=backend)
        results = parser.explain(string)
        return pd.DataFrame(results[1:], columns = tuple(results[0]))
    raise SyntaxError("The backend is not {backend}".format(backend = backend))


# In[13]:


if __name__ == '__main__':
    import pandas as pd
    df = bio_parse('../dssp/1a00.dssp')
    print(df)


# In[14]:


if __name__ == '__main__':
    print(df.columns)


# In[15]:


if __name__ == '__main__':
    print( (len(set(df.AA)), len(set(df.STRUCTURE))) )


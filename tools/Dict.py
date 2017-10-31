# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 19:23:03 2017

@author: thautwarm
"""
class Dict(dict):
    def __init__(self,*args,**kwargs):
        super(Dict,self).__init__(*args,**kwargs)
        for key in self.keys():
            self.__setattr__(key,self[key])
    def __setitem__(self,key,value):
        super(Dict,self).__setitem__(key,value)
        super(Dict,self).__setattr__(key,value)
    def __setattr__(self,key,value):
        super(Dict,self).__setitem__(key,value)
        super(Dict,self).__setattr__(key,value)       
    def __delattr__(self,key):
        super(Dict,self).__delattr__(key)
        super(Dict,self).__delitem__(key)
    def __delitem__(self,key):
        self.__delattr__(key)
    def update(self,dic):
        sets=self.keys()|dic.keys()
        items=[self[i]  if i in self else dic[i] for i in sets]
        self.__init__(zip(sets,items))
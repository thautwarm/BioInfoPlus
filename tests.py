# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 17:25:51 2017

@author: misaka-wa
"""

from preprocess.BioParser import bio_parse
dataframe = bio_parse('./dssp/sources/1a00.dssp')
AA = dataframe.AA # amino acid
Structure = dataframe.STRUCTURE # secondary structure

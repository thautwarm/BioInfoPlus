# BioInfoPlus
[![Build Status](https://travis-ci.org/thautwarm/BioInfoPlus.svg?branch=master)](https://travis-ci.org/thautwarm/BioInfoPlus)
[![data](https://img.shields.io/badge/cbi-pku-green.svg?style=flat)](http://www.cbi.pku.edu.cn/main.php?id=100)
## Requirement

- NumPy
- Pandas
- Matplotlib
- Cython(Optional, for speeding up algorithms)

P.S:  
`Cython` compiler requires a C++ compiler, and if you use `Windows 8/8.1/10`, please download **VC++2017 Build Tools** from **Visual Studio 2017 Installer**.   
Next, find out where **vcvarsall.bat** is in your computer, finally, add its path into system environment **PATH**.


## Get datas from bioinformation databases

use [download/dsspGet.py](https://github.com/thautwarm/BioInfoPlus/blob/master/downloader/dsspGet.py).  
see configurations at [downloader/dsspPKU.json](https://github.com/thautwarm/BioInfoPlus/blob/master/downloader/dsspPKU.json).
```
cd downloader && python dsspGet.py dsspPKU.json
```

## Process datas into dataframes

```python
from preprocess.BioParser import bio_parse
dataframe = bio_parse('./dssp/sources/1a00.dssp')
AA = dataframe.AA # amino acid
Structure = dataframe.STRUCTURE # secondary structure
```


## Check the frequency distribution of specific known pattern

```python
cases = np.array([AA, Structure]).T
res =  [each_gram.T.flatten() for each_gram in cython_make_gram(cases, 5)]
from research.specific_regular import specific_report
frequency = specific_report(res, {1:'A', 2:'A'})
print(frequency.values())
```


## An example about biological sequence detection

- [Codes](./main.py)

- [![Simple](https://github.com/thautwarm/BioInfoPlus/raw/master/figure/simple.png)](https://github.com/thautwarm/BioInfoPlus/raw/master/figure/simple.png)


Explanation:
```
c1 :
 ('V',  # 1-st amino acid.
  'A',  # 2-nd amino acid.
  'D',  # ...
  'A',  # ...
  'L',  # ...
  'H  X S+  ',  # 1-st secondary structure
  'H  X S+  ',  # ...
  'H  X S+  ',  # ...
  'H  X S+  ',  # ...
  'H  X S+  '   # 5-th secondary structure
  ),
c2 :
    ...
```

## Analysis on big datasets composed by multiple data files

```python

sources = ['./dssp/sources/1a00.dssp', 
           './dssp/sources/1a0a.dssp',
           './dssp/sources/1a0b.dssp',
           './dssp/sources/1a0c.dssp',
           './dssp/sources/1a0d.dssp']

from research.datasets_report import DatasetsReport
from research.plot import plot_frequency
whole = DatasetsReport(*sources).analyze(filtf=lambda probability, std, mean: probability>0.4)
number_of_dist = len(whole)

for test_some_case_dist in list(whole.keys())[:20]:
    plot_frequency(whole[test_some_case_dist])
```

[![fig1](https://github.com/thautwarm/BioInfoPlus/raw/master/figure/dist-prob1.png)](https://github.com/thautwarm/BioInfoPlus/raw/master/figure/dist-prob1.png)
[![fig2](https://github.com/thautwarm/BioInfoPlus/raw/master/figure/dist-prob2.png)](https://github.com/thautwarm/BioInfoPlus/raw/master/figure/dist-prob2.png)

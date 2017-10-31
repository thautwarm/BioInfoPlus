# BioInfoPlus

## get datas from bioinformation databases

use `./download/dsspGet.py`.  
see configurations at `./download/dsspPKU.json`.
```
python dsspGet.py dsspPKU.json
```

## process datas into dataframe

```python
from preprocess.BioParser import bio_parse
dataframe = bio_parse('./dssp/sources/1a00.dssp')
AA = dataframe.AA # amino acid
Structure = dataframe.STRUCTURE # secondary structure
```




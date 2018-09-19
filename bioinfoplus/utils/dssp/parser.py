from typing import List
from itertools import repeat
import re
import pandas as pd

tokenize = re.compile("[\w\.,]+")


class UDict(dict):
    def __init__(self, *args, **kwargs):
        super(UDict, self).__init__(*args, **kwargs)
        for key in self.keys():
            self.__setattr__(key, self[key])

    def __setitem__(self, key, value):
        super(UDict, self).__setitem__(key, value)
        super(UDict, self).__setattr__(key, value)

    def __setattr__(self, key, value):
        super(UDict, self).__setitem__(key, value)
        super(UDict, self).__setattr__(key, value)

    def __delattr__(self, key):
        super(UDict, self).__delattr__(key)
        super(UDict, self).__delitem__(key)

    def __delitem__(self, key):
        self.__delattr__(key)


class Preprocess:
    @staticmethod
    def get_column_and_index(row):
        res = []
        status = False
        begin = -1
        for i, ch in enumerate(row):
            if status:
                if ch != " ":
                    continue
                else:
                    status = False
                    res.append(UDict(inf=begin, sup=(i - 1)))
            else:
                if ch != " ":
                    status = True
                    begin = i
                else:
                    continue
        return res

    @staticmethod
    def resolve_space(x):
        if x.startswith(" "):
            for i, ch in enumerate(x):
                if ch != ' ':
                    return x[i:]
            return None
        return x


def parse(filename):
    def _parse(article: str) -> List[List[str]]:
        part = "  #  RESIDUE AA STRUCTURE BP1 BP2  ACC     N-H-->O "
        lines: List[str] = article.splitlines()

        col_names: str = [line for line in lines if part in line][0]
        res: List[UDict] = Preprocess.get_column_and_index(col_names)
        columns_row: str = col_names

        def cell_unit(line, tups):
            tup_last, tup = tups
            if line[tup.inf] == ' ':
                return line[tup.inf:tup.sup + 1]
            return Preprocess.resolve_space(line[tup_last.sup + 1:tup.sup + 1])

        source: List[str] = lines[lines.index(columns_row):]
        tuples = list(zip(res[:-1], res[1:]))
        ret_iter = map(lambda line: map(cell_unit, repeat(line), tuples),
                       source)
        return [list(map(str.strip, row)) for row in ret_iter]

    with open(filename) as fr:
        text = fr.read()
        results = _parse(text)
        return pd.DataFrame(results[1:], columns=tuple(results[0]))

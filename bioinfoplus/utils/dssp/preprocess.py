import pandas as pd


def preprocess(df: pd.DataFrame):
    def map_structure(it: str):
        if it and it[0].isalpha():
            return it[0].upper()
        return ''

    structure = df.STRUCTURE.map(map_structure)
    return pd.DataFrame({'STRUCTURE': structure, 'AA': df.AA.map(str.upper)})

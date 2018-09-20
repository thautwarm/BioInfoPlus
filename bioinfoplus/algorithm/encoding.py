import numpy as np


class Encoder:
    def __init__(self, object_set):
        mapping = self.mapping = {}
        for each in object_set:
            mapping[each] = len(mapping)
        self.inverse_mapping = {v: k for k, v in self.mapping.items()}

    def transform(self, vec, strict=True):
        mapping = self.mapping
        get = mapping.get
        maps = mapping.__getitem__ if strict else lambda key: get(key, -1)
        return tuple(map(maps, vec))

    def transform_batch(self, matrix, strict=True):
        mapping = self.mapping
        get = mapping.get
        maps = mapping.__getitem__ if strict else lambda key: get(key, -1)
        return tuple(tuple(maps(cell) for cell in row) for row in matrix)

    def decode(self, vec):
        inverse_mapping = self.inverse_mapping
        return tuple(map(inverse_mapping.__getitem__, vec))

    def decode_batch(self, matrix):
        inverse_mapping = self.inverse_mapping
        return tuple(
            tuple(inverse_mapping[cell] for cell in row) for row in matrix)

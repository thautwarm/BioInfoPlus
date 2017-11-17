# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 10:32:56 2017

@author: misakawa
"""


def gen_permutations(sets):
    if len(sets) is 1:
        yield from ([e] for e in sets[0])
    else:
        yield from ([x]+y for x in sets[0] for y in gen_permutations(sets[1:]))

        
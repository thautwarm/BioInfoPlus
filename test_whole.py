import numba as nb
import numpy as np
@nb.jit((nb.types.string[:, :],))
def gen_permutations(sets):
    if len(sets) == 1:
        for e in sets[0]:
            yield [e]
    else:
        for y in gen_permutations(sets[1:]):
            for x in sets[0]:
                yield [x] + y

if __name__ == '__main__':
    x = np.array([['a','b','c'], ['x','y','z'], ['q','t','e']])
    print(list(gen_permutations(x)))

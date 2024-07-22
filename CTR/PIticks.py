import numpy as np
from fractions import Fraction

def PIticks(start, end, step):
    v = np.arange(start, end+step, step)
    txt = []
    for ii in v:
        f = Fraction(ii)
        if f==0:
            txt.append(0)
            continue
        #
        if   f.numerator ==  1: num = 'π'
        elif f.numerator == -1: num = '-π'
        else: num = f'{f.numerator}π'
        #
        if f.denominator == 1: txt.append(num)
        else : txt.append(f'{num}/{f.denominator}')
    return v * np.pi, txt

__all__ = ['PIticks']

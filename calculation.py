import numpy as np

def convert_radix(number, base_num, length):
    '''
    Convert Decimal number to base_num ary number.

    Args
    ----------
    number: int
        Decimal number.
    base_num: int
        Radix of converted number.
    length: int
        List length of converted number.

    Returns
    ----------
    multi_n: list
        List of converted number.
        multi_n[0] * base_num**0 + multi_n[1] * base_num**1 + ... = number.
    '''

    multi_n = np.zeros(length).astype(np.int64)
    number_ = number
    for i in range(length):
        multi_n[i] = number_%base_num
        number_ = int(number_/base_num)
    return multi_n
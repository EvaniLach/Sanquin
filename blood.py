import numpy as np


class Blood():
    def __init__(self, bg_int, age=0, days_to_issuing=0):
        self.bg_int = bg_int
        self.bg_vector = "{0:b}".format(bg_int).split()
        self.age = age
        self.days_to_issuing = days_to_issuing

def vector_to_bloodgroup_index(vector):
    return int("".join(str(i) for i in vector),2)

# Returns the integer value for the first 'n_antigens' of a given bloodgroup, also expressed as an integer number.
# Example: n_antigens = 3, bg_int = 31 -> 11101 -> 111 -> 7
def comp_antigens(bg_int, n_antigens):
    return int(bin(bg_int)[2:n_antigens+2],2)


def precompute_compatibility(I, R):

    # C = np.zeros([len(I), len(R)])
    # for i in range(len(I)):
    #     for r in range(len(R)):
    #         C[i,r] = 1 - max(1, not_compatible(comp_antigens(r.bg_int,n_antigens), comp_antigens(i.bg_int,n_antigens)))

    # return np.array([[1-max(1,not_compatible(comp_antigens(r.bg_int,n_antigens), comp_antigens(i.bg_int,n_antigens))) for r in R] for i in I]) 
    # return np.array([[not_compatible(r, i) for r in R] for i in I])
    return np.array([[binarray(not_compatible(r, i)) for r in R] for i in I]) 
    # return C


def binarray(a: int, w: int = 8):
    """Integer to binary array conversion
    
    Converts an integer into a list of bits representing
    the number.
    
    :param int a: the integer number to convert
    :param int w: the number of bits to represent.
        Defaults to 8 but will always be as large
        as the number of bits in `a`.    
    """
    b = [int(i) for i in f'{a:0{w}b}']
    return b


def bnot(a: int, m: int = 0):
    """
    Bitwise NOT of an integer
    
    :param int a: integer to be bitwise NOTed
    :param int m: mask, defaults to 0. Don't change this unless you know what you're doing.
    """
    if m == 0:
        m = 2 ** a.bit_length() - 1
    return ~a & m


def not_compatible(a: int, b: int, m: int = 0):
    """
    Determines if blood types are incompatible using integer representation.
    
    :param int a: integer representation of patient/recipient blood type
    :param int b: integer representation of donor blood type
    :param int m: mask, the largest number that can be represented with the bits in the integer representation.
    :return int: integer representation of incompatible blood types.
      When converted to binary, the bits that are set to 1 indicate incompatible blood antigens.
    """
    return bnot(a, m) & b


def bit_mask(n: int):
    """
    Calculates the mask
    :param int n: the number of antigens to consider - number of bits in the integer representation.
    """
    return (2 ** n) - 1


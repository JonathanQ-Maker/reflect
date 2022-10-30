
def to_tuple(x):
    """
    returns x if x is a tuple, else return (x, )
    """
    return x if isinstance(x, tuple) else (x, )

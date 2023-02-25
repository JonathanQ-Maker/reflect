from reflect.constraints.abstract_constraint import AbstractConstraint
from reflect import np

class Clip(AbstractConstraint):


    _limit = 1  # limit >= 0

    @property
    def limit(self):
        return self._limit

    @limit.setter
    def limit(self, value: float):
        self._limit = abs(value)

    def __init__(self, limit: float):
        self.limit = limit
    

    def constrain(self, weight):
        """
        Applies clip constraint inplace on weight
        """
        np.clip(weight, -self._limit, self._limit, out=weight)

    def compile(self, shape: tuple):
        pass

    def is_compiled(self):
        return True
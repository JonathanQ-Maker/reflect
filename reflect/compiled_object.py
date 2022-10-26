from abc import ABC, abstractmethod

class CompiledObject(ABC):
    """
    Objects that have static internal variables that dont often change
    """

    @abstractmethod
    def compile(self):
        """
        Update inetrnal variables
        """
        pass

    @abstractmethod
    def is_compiled(self):
        """
        Check if inetrnal variables are up-to-date
        """
        pass

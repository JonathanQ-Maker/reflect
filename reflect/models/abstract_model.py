from abc import ABC, abstractmethod

class AbstractModel(ABC):
    """Abstract class of Model"""

    @abstractmethod
    def predict(self, X):
        return

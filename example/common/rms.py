import numpy as np

class RunningMean:
    
    def __init__(self, size: int):
        
        self.size = size
        self.data = np.array([], dtype=np.float32)
        
    def update(self, data: np.ndarray):
        
        if data.ndim == 0:
            data = np.array([data], dtype=np.float32)
        self.data = np.concatenate([self.data, data])
        self.data = self.data[-self.size:]
        
    def mean(self):
        
        return np.mean(self.data)
    
    def std(self):
        
        return np.clip(np.std(self.data), 1e-4)
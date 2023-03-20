import numpy as np
import pandas as pd

data = {"y": [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
        "x1": [0, 1, 1, 0, 1, 0, 0, 0, 1, 0],
        "x2": [1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        "x3": [0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
        "x4": [1, 0, 0, 1, 0, 1, 0, 0, 0, 1]}

df = pd.DataFrame(data)
print(df)
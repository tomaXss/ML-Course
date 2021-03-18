import pandas as pd
from sklearn import preprocessing
import numpy as np

df = pd.read_csv("report.csv")

# выборочное среднее для столбца MIP
mip_mean = df["MIP"].mean()
print("Mean MIP:", mip_mean)

# удаление столбца TARGET
df = df.drop(['TARGET'], axis=1)
scaler = preprocessing.MinMaxScaler()

# линейная нормировка
names = df.columns
d = scaler.fit_transform(df)
scaled_df = pd.DataFrame(d, columns=names)
print("MinMax: ", scaled_df["MIP"].mean())

#минимальное расстояние до STAR
STAR = np.array([0.121, 0.009, 0.333, 0.335, 0.655, 0.745, 0.736, 0.536])
def distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2, axis=1))

distance = distance(STAR, scaled_df.values)
print("Min distance: ",min(distance))

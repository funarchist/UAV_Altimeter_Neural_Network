from tensorflow.keras.models import Model
from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
import pickle
datas = []

with open('images.pickle', 'rb') as validationAttrX:
    arrays = pickle.load(images)

with open('list_altitudes.pickle', 'rb') as list_altitudes:
    list_altitudes = pickle.load(list_altitudes)

arrays = np.asarray(arrays)
list_altitudes = np.asarray(list_altitudes)

for f, b in zip(list_altitudes, arrays):
    f = np.array(f)
    b = np.array(b)
    preds = model.predict(b)
    diff = preds.flatten() - f
    diff = diff/1000
    absDiff = np.abs(diff)
    datas.append(absDiff)

with open("datas.txt", "wb") as fp:
    pickle.dump(datas, fp)

datas = np.load("datas.txt", allow_pickle=True)
plt.boxplot(datas[0], datas[1], datas[2],datas[3],datas[4],datas[5], names=c("2.5-7.5","7.5-12.5","12.5-17.5","17.5-22.5","22.5-27.5","27.5-32.5"))
plt.title('Estimation Error(Meter)')
plt.legend()
plt.ylabel('Meter')
plt.legend()
plt.show()
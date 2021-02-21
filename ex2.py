import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

#only need to run once
REBUILD_DATA = False

class DogsVSCats():
    IMG_SIZE = 50
    CATS = "PetImages/Cat"
    DOGS = "PetImages/Dog"
    LABELS = {CATS: 0 ,DOGS: 1}

    training_data = []
    catcount = 0
    dogcount = 0

    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label),desc="Making trainin data"):
                try:
                    path = os.path.join(label,f)
                    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img,(self.IMG_SIZE,self.IMG_SIZE))
                    self.training_data.append([np.array(img),np.eye(2)[self.LABELS[label]]])

                    if label ==  self.CATS:
                        self.catcount += 1
                    if label ==  self.DOGS:
                        self.dogcount += 1
                except Exception as e:
                    pass
        np.random.shuffle(self.training_data)
        np.save("training_data.npy",self.training_data)
        print("Cats: ", self.catcount)
        print("Dogs: ",self.dogcount)
if REBUILD_DATA:
    dogsvcats = DogsVSCats()
    dogsvcats.make_training_data()

#data build, this is a loading test
training_data =np.load("training_data.npy",allow_pickle = True)
print(len(training_data))
print(training_data[10])
plt.imshow(training_data[10][0],cmap="gray")
print(training_data[10][1])
plt.show()
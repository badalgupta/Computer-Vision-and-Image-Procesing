from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from imutils import paths
import argparse
from glob import glob
import os,cv2
import numpy as np



def preprocess(image,width,height,inter):
	return cv2.resize(image,(width,height),interpolation=inter)

def load(imagePaths, verbose=-1):
	data,labels=[],[]
	# images=[]
	for(i,image_path) in enumerate(imagePaths):
		label = image_path.split(os.path.sep)[-2]

		for f_name in os.listdir(image_path):
			img=cv2.imread(os.path.join(image_path,f_name))
			if img is not None:
				img=preprocess(img,32,32,cv2.INTER_AREA)
				data.append(img)
				labels.append(label)
	return (np.array(data),np.array(labels))
		# img=cv2.imread(imagepath)
		# label=imagepath.split()

imagePaths=glob("/media/badal/New Volume/Computer_vision/cat_dog/*/")
(data,labels)=load(imagePaths)
data=data.reshape((data.shape[0],3072))
print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))


#Endoding Labels
le=LabelEncoder()
labels=le.fit_transform(labels)
(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.25, random_state=42)
#Hyper Parameter Tuning
params = {'n_neighbors':[5,6,7,8,9,10],
          # 'leaf_size':[1,2,3,5],
          # 'weights':['uniform', 'distance'],
          # 'algorithm':['auto', 'ball_tree','kd_tree','brute'],
          # 'n_jobs':[-1]

          }


model=KNeighborsClassifier()

model = GridSearchCV(model,param_grid=params,n_jobs=-1)
model.fit(trainX,trainY)

print("Best Hyper Parameters:\n",model.best_params_)
print(classification_report(testY , model.predict(testX), target_names=le.classes_))






# print(imagePaths)
# image_path=imagePaths[0]
# label = image_path.split(os.path.sep)[-2]
# sp=SimplePreprocessor(32,32)
# sdl = DatasetLoader(preprocessors=[sp])
# (data,label)=sdl.load(imagePaths,verbose=200)
# data = data.reshape((data.shape[0], 3072))
# print(data)
# print(label)


# subfolders = [f.path for f in os.scandir(folder) if f.is_dir() ]
# imagePaths=list(paths.list_images())
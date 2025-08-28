> **Road** **Sign** **Detec.on** **–** **Road** **Ragers** **Deep**
> **Neural** **Networks** **Project**


This project implements and compares several neural network
architectures (SimpleCNN, DeepFFN3, AdamNet, SimpleFFN) for detection and
classifying German traﬃc signs using the [<u>GTSRB
dataset</u>.](https://benchmark.ini.rub.de/gtsrb_news.html) All models
are implemented from scratch in NumPy, with manual training loops,
opAmizers, and evaluaAon.

**Project** **Structure** .

├── Models – Codes

└── Models/

> ├── SimpleFFN.py

> ├── DeepFFN3.py

> ├── AdamNet.py

> ├── SimpleCNN.py

├── Results – Jupyter Notebooks – contains the results – executed code

└── Models/

> ├── SimpleFFN.ipynb
> 
> ├── DeepFFN3.ipynb
> 
> ├── AdamNet.ipynb
> 
> ├──SimpleCNN.ipynb

├── Road Ragers Report.pdf


**Dataset/Train/**: Contains 43 subfolders (0–42), each with images for
that class.

**Each** **\*model_name.py**: Script for a speciﬁc model.

**Each** **.ipynb**: Contains the Neural Network models executed with
results

**Each** **script** **includes:**

• Data loading and preprocessing • Model deﬁniAon

• Training and validation with hyperparameter tuning

• Accuracy/loss plotting

**Usage**

1\. **Prepare** **the** **dataset:** Download the [<u>GTSRB
dataset.</u>](https://benchmark.ini.rub.de/gtsrb_news.html)

> Place the training images in **Dataset/Train/0**, **Dataset/Train/1**,
> ..., **Dataset/Train/42**.

2\. **Install** **requirements**:  bash pip install numpy scikit-learn matplotlib pillow opencv-python 

3\.**Run** **a** **model** **script**:  bash python SimpleCNN.py

4\. **View** **results**: Training and validation accuracy/loss will be
printed and plotted.

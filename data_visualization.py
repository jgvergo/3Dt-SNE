# Use the anaconda py2713 environment
# source activate py2713
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Tkinter import *
from mpl_toolkits.mplot3d import Axes3D

# Global variables
master = Tk()

# Array that holds the  on/off state of the columns for the CheckButtons
checked = []

#Initialize
for idx in range (0,1000):
    checked.append(IntVar())
    checked[idx].set(1)

num_cols = 0

def redisplayViz():
    #close the old window
    plt.close()
    plt.title("Reecalculating")
    plt.show(block=False)

    # Copy the original dataframe into a temporary df and eliminate the unchecked columns
    dfa = dataframe_all
    for i in range (num_cols):
        if checked[i].get() == 0:
            dfa = dfa.drop(str(dataframe_all.columns[i]), 1)

    columns = dfa.columns
    
    # step 3: get features (x) and scale the features
    # get x and convert it to numpy array
    x = dfa.ix[:,:-1].values
    standard_scaler = StandardScaler()
    x_std = standard_scaler.fit_transform(x)
    
    # step 4: get class labels y and then encode it into number 
    # get class label data
    y = dfa.ix[:,-1].values
    
    # encode the class label
    class_labels = np.unique(y)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    # step 5: split the data into training set and test set
    test_percentage = 0.1
    x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size = test_percentage, random_state = 0)
    
    # t-distributed Stochastic Neighbor Embedding (t-SNE) visualization
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=3, random_state=0)
    x_test_3d = tsne.fit_transform(x_test)

    # scatter plot the sample points among 5 classes
    markers=('.', '.', '.', '.', '.')
    color_map = {0:'red', 1:'blue', 2:'lightgreen', 3:'purple', 4:'cyan'}
    plt.close()
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    for idx, cl in enumerate(np.unique(y_test)):
        ax.scatter(x_test_3d[y_test==cl,0], x_test_3d[y_test==cl,1], x_test_3d[y_test==cl,2], c=color_map[idx], marker=markers[idx], label=cl)
    

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.legend(loc='upper left')
    plt.title('t-SNE visualization of test data')
    plt.show(block=False)


# step 1: download the data
dataframe_all = pd.read_csv("./pml-training.csv")
num_rows = dataframe_all.shape[0]

# step 2: remove useless data
# count the number of missing elements (NaN) in each column
counter_nan = dataframe_all.isnull().sum()
counter_without_nan = counter_nan[counter_nan==0]

# remove the columns with missing elements
dataframe_all = dataframe_all[counter_without_nan.keys()]
# remove the first 7 columns which contain no discriminative information

dataframe_all = dataframe_all.ix[:,7:]

num_cols = dataframe_all.shape[1]

plt.show(block=False)

Label(master, text="Select variable:").grid(row=0, sticky=W)

# Lay out checkbuttons in a grid that is num_cols wide
for i in range(num_cols):
    ncols = 5
    col = i % ncols
    row = (i - (i % ncols))/ncols
    Checkbutton(master, text=dataframe_all.columns[i], variable=checked[i]).grid(row=row+1, column=col, sticky=W)

Button(master, text='Quit', command=master.quit).grid(row=0, column=ncols, sticky=W, pady=4)
Button(master, text='Show', command=redisplayViz).grid(row=1, column=ncols, sticky=W, pady=4)

# show the viz by default the first time we start
redisplayViz()

mainloop()
pip install streamlit
streamlit hello
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Model Importing
from tensorflow.keras.models import load_model
new_model = load_model('France_model_for_GUI.h5')
new_model.summary()
new_model.get_weights()
new_model.optimizer

test = pd.read_csv("France_test_dataset_for_GUI.csv")
train_dates = pd.to_datetime(test["DateTime"])
test = test.set_index("DateTime")
#test["DateTime"] = pd.to_datetime(test["DateTime"])

#Importing scalar object
from pickle import load
dv_transformer= load(open('dv_transformer.pkl', 'rb'))
iv_transformer= load(open('iv_transformer.pkl', 'rb'))
labelencoder = load(open('labelencoder.pkl', 'rb'))

#Feature Scaling from original scaled functions
iv_columns = ['Year','outdoor_humidity', 'outdoor_temperature', 'wind_speed',"mnth_sin", 
              'mnth_cos', "hr_sin", 'hr_cos', "dy_sin", 'dy_cos']
test['Global_active_power'] = dv_transformer.transform(test[['Global_active_power']])
test.loc[:, iv_columns] = iv_transformer.transform(test[iv_columns].to_numpy())


#Lookback function
dv_columns = ['Global_active_power']

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 72
X_test,   y_test = create_dataset(test , test.loc[:, dv_columns] , time_steps)

#Predicting the test data
y_pred = new_model.predict(X_test)

#Scaling prediction
y_pred_inv = dv_transformer.inverse_transform(y_pred)
y_test_inv = dv_transformer.inverse_transform(y_test)

#Appending dates and prediction
train_dates = train_dates.to_frame()
train_dates = train_dates.iloc[72:,:]
train_dates = train_dates.reset_index(drop = True)
y_pred_inv = pd.DataFrame(y_pred_inv)
Prediction = pd.concat([train_dates,y_pred_inv], sort=False,axis =1)
columns = ['DateTime', 'Predicted_Units']
Prediction['Predicted_Units'] = Prediction.loc[:,0]
Prediction = Prediction[columns]
#Appending dates and y_test
y_test_inv = pd.DataFrame(y_test_inv)
Real = pd.concat([train_dates,y_test_inv], sort=False,axis =1)
columns1 = ['DateTime', 'Real_Units']
Real['Real_Units'] = Real.loc[:,0]
Real = Real[columns1]


Pred_Real = pd.concat([Real,Prediction],axis =1,sort=False)

Pred_Real = Pred_Real.iloc[:,[0,1,3]]

import datetime
from datetime import timedelta

start_date = '2015-10-09'
start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
next_date = start_date + timedelta(days=1)
filtered_dates = Pred_Real[(Pred_Real['DateTime']>= start_date) & (Pred_Real['DateTime']<next_date)]



#import matplotlib.pyplot as plt
#plt.plot(filtered_dates.Units, color='green', linestyle='dashed', linewidth = 3,
        # marker='o', markerfacecolor='blue', markersize=12)
#plt.show()
#----------------------------------------------GUI--------------------------------------#
import tkinter as tk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkcalendar import *
from tkinter import * 
import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from tkinter import ttk

#tkinter GUI
root = tk.Tk()
root.title('Energy_Project')
root.geometry("1920x1080")


#Making Canvas
canvas1 = tk.Canvas(root)
canvas1.pack()

#Making Calendar Form using calendar widget
cal = Calendar(root,selectmode ='day',year = 2015, month =10 , day = 1,date_pattern = 'dd/mm/y')
canvas1.create_window(-130,280,window = cal)
cal.pack()
def values(): 
    start_date = cal.get_date()
    label_Prediction  = tk.Label(root,text = start_date , bg = 'Red')
    canvas1.create_window(260,280,window = label_Prediction)
    label_Prediction.config(font=("Times",18))


def search_dates():
    global filtered_dates 
    filtered_dates = []
    start_date = cal.get_date()
    start_date = datetime.datetime.strptime(start_date, '%d/%m/%Y')
    next_date = start_date + timedelta(days=1)
    filtered_dates = Pred_Real[(Pred_Real['DateTime']>= start_date) & (Pred_Real['DateTime']<next_date)]
    filtered_dates = filtered_dates.set_index("DateTime")
   
    

def graph():
    figure2 = plt.Figure(figsize=(5,4), dpi=100)
    ax2 = figure2.add_subplot(111)
    line2 = FigureCanvasTkAgg(figure2, root)
    line2.get_tk_widget().pack(side = tk.BOTTOM)
    search_dates()
    Predicted_data = filtered_dates.Predicted_Units
    Predicted_data.plot(kind='line', legend=True, ax=ax2, color='r',marker='o', fontsize=10)
    Real_data = filtered_dates.Real_Units
    Real_data.plot(kind='line', legend=True, ax=ax2, color='b',marker='o', fontsize=10)
    ax2.set_title('Year Vs. Unemployment Rate')
    
'''# Add some style
style = ttk.Style()
#Pick a theme
style.theme_use("default")
# Configure our treeview colors

style.configure("Treeview", 
	background="#D3D3D3",
	foreground="black",
	rowheight=10,
	fieldbackground="#D3D3D3"
	)
# Change selected color
style.map('Treeview', 
	background=[('selected', 'blue')])

# Create Treeview Frame
tree_frame = Frame(root)
tree_frame.pack(side = TOP,pady =20,padx = 20)

# Treeview Scrollbar
tree_scroll = Scrollbar(tree_frame)
tree_scroll.pack(side=RIGHT, fill=Y)

# Create Treeview
my_tree = ttk.Treeview(tree_frame, yscrollcommand=tree_scroll.set, selectmode="extended")
# Pack to the screen
my_tree.pack()

#Configure the scrollbar
tree_scroll.config(command=my_tree.yview)

# Define Our Columns
my_tree['columns'] = ("DateTime", "Real_Units","Predicted_Units")

# Formate Our Columns
my_tree.column("#0", width=0, stretch=NO)
my_tree.column("DateTime", anchor=W, width=200)
my_tree.column("Real_Units", anchor=CENTER, width=200)
my_tree.column("Predicted_Units", anchor=W, width=200)

# Create Headings 
my_tree.heading("#0", text="", anchor=W)
my_tree.heading("DateTime", text="DateTime", anchor=W)
my_tree.heading("Real_Units", text="Real_Units", anchor=W)
my_tree.heading("Predicted_Units", text="Predicted_Units", anchor=W)

global count
count=0
search_dates()
units_list=filtered_dates.to_list()

for record in units_list:
        tree.insert(parent='', index='end', iid=count, text="", values=(record[0], record[1], record[2]))
        count += 1

my_tree.pack(pady=20)'''

#Adding Button to link date with ML model   
my_btn = Button(root,text = 'Prediction' , command = graph)
my_btn.pack(pady=20)

#Label of the Button
my_label = Label(root,text= '')
my_label.pack(pady=20)


root.mainloop()



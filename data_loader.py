import numpy as np
import csv
import os
import pandas as pd

def data_loader(root=r"data"):

    ca_path=os.path.join(root, "CrankAnglePosition.csv")
    df = pd.read_csv(ca_path)  
    print(df.head())

    #crank_angle=csv.reader("")
    #print(data)

def print_data():
    root=r"data"
    ca_path=os.path.join(root, "CrankAnglePosition.csv")
    with open(ca_path , 'r') as csvfile:
        # create the object of csv.reader()
        csv_file_reader = csv.reader(csvfile,delimiter=',')
        for row in csv_file_reader:
            print(row)  

if __name__=="__main__":
    data_loader()
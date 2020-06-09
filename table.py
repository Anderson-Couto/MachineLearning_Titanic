from prettytable import PrettyTable
from prettytable import from_csv
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RepeatedKFold

train = open("data/train.csv", "r")
x = from_csv(train)
train.close()

print(x)
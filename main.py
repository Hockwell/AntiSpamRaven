import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import f1_score, accuracy_score

vis_data = pd.read_csv(r"C:\Users\volnu\OneDrive\Data\Edu\ML\2 Регрессии\train.csv", encoding = 'ISO-8859-1', low_memory = False)
data = vis_data[['fine_amount', 'state_fee', 'late_fee', 'discount_amount', 'balance_due', 'compliance']].dropna()

import numpy as np
import pandas as pd
import datetime as dt

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

import itertools
import warnings


def encode_units(x):
    if x:
        return 1
    else :
        return 0


def market_association_rules(min_support):
    warnings.filterwarnings('ignore')
    warnings.simplefilter(action='ignore', category=FutureWarning)

    data = pd.read_csv('market/Market_Basket_Optimisation.csv', header=None)
    # print(data.head())


    # data preprocessing
    transaction = []
    for i in range(data.shape[0]):
        transaction.append([str(data.values[i, j]) for j in range(data.shape[1])])

    transaction = np.array(transaction)
    # print(transaction)

    te = TransactionEncoder()
    te_ary = te.fit(transaction).transform(transaction)
    dataset = pd.DataFrame(te_ary, columns=te.columns_)
    dataset = dataset.drop(['nan'], axis=1)
    # print(dataset)

    dataset = dataset.applymap(encode_units)
    # print(dataset.head(10))

    frequent_itemsets = apriori(dataset, min_support=min_support, use_colnames=True)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    print(frequent_itemsets)

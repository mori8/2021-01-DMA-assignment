from titanic.titanic import titanic_model
from market.market import market_association_rules


print("[ student ID: 1914266 ]")
print("[ Name: 남수연 ]")

while True:
    print()
    print("1. Titanic Survivor Predictor")
    print("2. Market Basket Analyzer")
    print("3. Quit")
    print(">> ", end='')

    n = int(input())
    if n == 1:
        titanic_model()
    elif n == 2:
        print("Enter the minimum support: ", end='')
        min_support = float(input())
        market_association_rules(min_support)
    elif n == 3:
        break


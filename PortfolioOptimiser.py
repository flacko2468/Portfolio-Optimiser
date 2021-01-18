from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import yfinance as yf
from yahoofinancials import YahooFinancials
import numpy as np
import scipy.optimize as spo
import math
import statistics

# Pass a list of portfolio allocations with each number representing the percentage allocation
# of the respective stock and receive a dataframe of the portfolios cumulative return at each day
# along the way
def get_daily_value(portAllocs):
    normed = df / df.iloc[0]
    i = 0
    for column in normed:
        normed[column] = normed[column] * portAllocs[i]
        i = i + 1
    port_val = normed.sum(axis=1)
    return port_val
# Pass a list of portfolio allocations with each number representing the percentage allocation
# of the respective stock and receive a dataframe of the daily returns
def get_daily_rets(portAllocs):
    daily_rets = get_daily_value(portAllocs).copy()
    daily_rets[1:] = (daily_rets[1:] / daily_rets[:-1].values) - 1
    daily_rets.iloc[0] = 0
    return daily_rets

# The constraint on the minimize function, ensuring that the sum of the allocations is equal to 1
def con(alloc):
    allocation_sum = 0
    for x in alloc:
        allocation_sum = allocation_sum + x
    return allocation_sum - 1

cons = {'type':'eq', 'fun': con}

# Pass a list of portfolio allocations with each number representing the percentage allocation
# of the respective stock and print the new calculated allocations in an intuitive format
def print_new_allocations():
    i = 0
    for stock in stocks:
        print(stock + ": " + str(100 * round(new_allocations[i], 2)) + "%")
        i = i + 1

# Pass a list of portfolio allocations with each number representing the percentage allocation
# of the respective stock and receive the Sharpe Ratio calculation multiplied by -1
def sharpe_ratio(x):
    return -1 * math.sqrt(252)*np.mean(get_daily_rets(x))/np.std(get_daily_rets(x))

# Pass the name of the function to be minimised and return the value at which this function
# is minimised
def get_minimum(f):
    min_result = spo.minimize(f, allocation, method='SLSQP',
                              bounds=spo.Bounds(0, 1),
                              constraints=cons, options={'disp': True})
    return min_result


# Create a stocks list and an allocation list, where the indexes of each list relate them to each
# other implicitly
stocks = []
allocation = []

# We would like all available data from 01/01/2000 until todays date, unless another date is specified.
start_date = ''
end_date = datetime.today().strftime('%Y-%m-%d')
while True:
    print("Enter the start date YYYY-MM-DD")
    value = input(" or press enter to default to 2010-01-01: ")
    if value == '':
        start_date = '2010-01-01'
        break
    else:
        start_date = value
        break

# Read the stock symbols the user inputs into the stocks list
while True:
    value = input("Enter Stock Symbol or nothing if done: ").upper()
    if value == '' and stocks:
        break
    if value not in stocks and value != '':
        stocks.append(value)

# Set the value for each stock symbol to an equal percentage, representing the inital assumed
# stock allocation
for stock in stocks:
    allocation.append(1.0/len(stocks))

# Download stocks from yahoo finance
df = yf.download(stocks, 
                          start=start_date, 
                          end=end_date, 
                          progress=False)

# Filter so only the 'Close' data remains
df = df[['Adj Close']]

# Ensure the column names are in the same format as the stock symbols in the stocks list
df.columns = stocks
    
    
# Back and forward filling missing data
df = df.ffill()
df = df.bfill()

# Get the new allocations
new_allocations = get_minimum(sharpe_ratio).x

print_new_allocations()

# Plot a graph comparing the daily value of an equal allocation, and the new allocations
df1 = pd.DataFrame(get_daily_value(allocation), columns=['Equal Allocation'])
df2 = pd.DataFrame(get_daily_value(new_allocations), columns=['Program Adjusted Allocation'])
df1 = df1.join(df2, how = 'inner')
df1.plot()
plt.show()
              






    



    


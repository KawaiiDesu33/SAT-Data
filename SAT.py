
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

path = '/Users/terenceau/Desktop/Python/Basic Data Processing and Visualisation//datasets/2010_SAT'
file = pd.read_csv(path, delimiter = '	')


############################## Simple Analysis ##############################

file_desc = file.describe()
"""
Number of tests range from 7 to 1047. Total 386 Schools in Sample Data
Large Deviation in results. Mean at 104 with Std of 145.
Using Percentiles the 50th Percentile is at 54 tests in the school
"""
"""
Highest mean score was in Mathematics 
Also provides the highest max score
"""

marks = file[['Critical Reading Mean', 'Mathematics Mean', 'Writing Mean']]
marks_corr = marks.corr()
"""Strong Correlation between marks, especially between Critical Reading and Writing"""

sns.heatmap(marks_corr, cmap = 'viridis_r')

############################## Descriptive Plots ##############################
# Boxplots of Marks
plt.subplot(1, 3, 1)
plt.boxplot(x = file['Critical Reading Mean'], labels = ['Reading'])
plt.subplot(1, 3, 2)
plt.title('Boxplots of Marks')
plt.boxplot(x = file['Mathematics Mean'], labels = ['Mathematics'] )
plt.subplot(1, 3, 3)
plt.boxplot(x = file['Writing Mean'], labels = ['Writing'] )
plt.show()

# Histogram of Number of Test Takers
plt.hist(x = file['Number of Test Takers'], bins = 10)
plt.xlabel('Number of Test Takers')
plt.title('Histogram of Number of Test Takers')
plt.show()


######################## Predictive Linear Regressions ########################
from sklearn.linear_model import LinearRegression
lm = LinearRegression()

plt.subplot(1, 2, 1)
plt.scatter(file['Critical Reading Mean'], file['Mathematics Mean'])
plt.xlabel('Reading')
plt.ylabel('Maths')
plt.subplot(1, 2, 2)
plt.scatter(file['Critical Reading Mean'], file['Writing Mean'])
plt.xlabel('Reading')
plt.ylabel('Writing')
plt.show()
"""Strong linear relationships between Variables. Especially Writing and Reading"""

# Predictability of Each Mark Based off other Marks
X = file[['Mathematics Mean', 'Writing Mean']]
Y = file['Critical Reading Mean']

lm.fit(X, Y) #Predictability of Critical Reading
yhat_cr = lm.predict(X)
coeff_cr = lm.coef_
int_cr = lm.intercept_
r2_cr = lm.score(X, Y)

X = file[['Critical Reading Mean', 'Writing Mean']]
Y = file['Mathematics Mean']

lm.fit(X, Y) # Preditability of Mathmatics
yhat_math = lm.predict(X)
coeff_math = lm.coef_
int_math = lm.intercept_
r2_math = lm.score(X, Y)

X = file[['Critical Reading Mean', 'Mathematics Mean']]
Y = file['Writing Mean']

lm.fit(X, Y) # Predictability of Writing Results
yhat_wt = lm.predict(X)
coeff_wt = lm.coef_
int_wt = lm.intercept_
r2_wt = lm.score(X, Y)


######################## Predictive Top Ranking School ########################
""" Must Rank High in all Reading, Mathematics and Writing """

read_school = file[['School Name', 'Critical Reading Mean']]

file['Mean Results'] = file[['Critical Reading Mean', 'Mathematics Mean',
                             'Writing Mean']].mean(axis = 1)
"""Stuvesant HS, Bronx HS of Science, Staten Island Technical HS - Top 3 HS
Using Average Mean of All Test Results"""

"""
Brooklyn HS Leadership Community, HS of Wold Cultures, Kingsbirdge Interantional
- Bottom 3 HS using Average Mean of All Test Results
"""

# Tier List of the Schools
bins = np.linspace(min(file['Mean Results']), max(file['Mean Results']), 6)
groups = ['Lowest', 'Low', 'Average', 'High', 'Highest']

file['School Category'] = pd.cut(file['Mean Results'], bins, labels = groups, include_lowest=True)

plt.hist(x = file['School Category'])
plt.show()


############################# Ranking of Schools  ################################

from scipy import stats

file['Reading Ranks'] = stats.rankdata(file['Critical Reading Mean'], method = 'max')
file['Mathematics Ranks'] = stats.rankdata(file['Mathematics Mean'], method = 'max')
file['Writing Ranks'] = stats.rankdata(file['Writing Mean'], method = 'max')

file['Ranking based on Ranks'] = np.mean(file[['Reading Ranks', 'Mathematics Ranks', 
                                               'Writing Ranks']], axis = 1)
file['Ranking based on Ranks'] = stats.rankdata(file['Ranking based on Ranks'])

file['Ranking based on Results'] = stats.rankdata(file['Mean Results'])


"""
Slight Deviations in Middle Schools due to differences in Measurement.
Results Mean - Takes into a larger deviation of Results. 
Rank Mean - Less Punishment as the scale for Rank is lower
"""
"""
Example: Dual Language Asian Studies High School
Based on Results = 355/385
Based on Ranks = 326/385
"""

"""Not sure how to Reverse the Ranking so lower is better"""

temp = file[['School Name', 'Ranking based on Ranks', 'Ranking based on Results']]

temp_rank = []
temp_result = [] # Making a List of the Values

for i in temp['Ranking based on Ranks']:
    temp_rank.append(i)
    
for i in temp['Ranking based on Results']:
    temp_result.append(i) # Adding Values to List
    
temp_list =[]

for x in range(0, 386, 1): # Function to test each component of both lists
    if temp_rank[x] == temp_result[x]:
        temp_list.append('Same')
    elif temp_rank[x] > temp_result[x]:
        temp_list.append('Rank Higher')
    else:
        temp_list.append('Result Higher')

temp_1 = pd.Series(temp_list) # Convert List of Series
temp_1 = temp_1.to_frame() # Convert Series to Frame
temp_1.rename(columns={0:'Outcome'}, inplace = True) # Renaming the Column
temp = temp.join(temp_1) # Adding to File

plt.subplot(1, 2, 1)
plt.hist(x = temp['Outcome'])
plt.xlabel('Rank Differences')
plt.ylabel('Count out of Data')
plt.title('Bar Graph of Rank Differences')

print(temp['Outcome'].value_counts())
"""Rank = 203, Result = 151, Same = 32"""

x = [203, 151, 32]
y = ['Rank Higher', 'Result Higher', 'Same']

plt.subplot(1, 2, 2)
plt.pie(x, labels = y, autopct = '%.2f%%')
plt.title('Pie Chart of Rank Differences')
plt.show()
        






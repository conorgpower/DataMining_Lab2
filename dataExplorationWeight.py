# Q. Import data from "BSCY4.csv".
# A.
import pandas as pd 
df = pd.read_csv('BSCY4_Lab_2.csv')
print(df.head())

# Q. Assess normality of weight values.
# Kurtosis
skewness = df['weight'].skew(axis = 0, skipna = True)
print("Skewness: ", skewness)
# Skewness
kurtosis = df['weight'].kurtosis()
print("Kurtosis: ", kurtosis)

# (1) Visual Inspection
import seaborn as sns
import matplotlib.pyplot as plt
plt.show(sns.distplot(df['weight']))

# (2) NHST
from scipy import stats
shapiroWeight = stats.shapiro(df['weight'])
print("Shapiro-Wilk Test for 'weight': ",shapiroWeight)

# (3) Q-Q Plot
plt.show(stats.probplot(df['weight'], dist="norm", plot=plt))

# Q. Do the numbers appear to come from a normal distribution? 
# If not, can a transformation be applied so that its result is normal?
# A. The skewness indicates that the distribution is not normal.
# The kurtosis indictes that the data is not normal.
# Upon visual inspection we can see the data appears as though it is not normal.
# The null hypothesis significance testing proves the data is not normal.
# The Q-Q plot confirms non-normality of the data.

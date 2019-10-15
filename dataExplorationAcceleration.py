# Q. Import data from "BSCY4.csv".
# A.
import pandas as pd 
df = pd.read_csv('BSCY4_Lab_2.csv')
print(df.head())

# Q. Assess normality of acceleration values.
# Kurtosis
skewness = df['acceleration'].skew(axis = 0, skipna = True)
print("Skewness: ", skewness)
# Skewness
kurtosis = df['acceleration'].kurtosis()
print("Kurtosis: ", kurtosis)

# (1) Visual Inspection
import seaborn as sns
import matplotlib.pyplot as plt
plt.show(sns.distplot(df['acceleration']))

# (2) NHST
from scipy import stats
shapiroAcceleration = stats.shapiro(df['acceleration'])
print("Shapiro-Wilk Test for 'acceleration': ",shapiroAcceleration)

# (3) Q-Q Plot
plt.show(stats.probplot(df['acceleration'], dist="norm", plot=plt))

# Q. Do the numbers appear to come from a normal distribution? 
# If not, can a transformation be applied so that its result is normal?
# A. The skewness indicates that the distribution may be normal.
# The kurtosis indictes that the data is not normal.
# Upon visual inspection we can see the data appears as though it is normal.
# The null hypothesis significance testing proves the data is normal.
# The Q-Q plot agrees with the NHST.



# Q. Import data from "BSCY4.csv".
# A.
import pandas as pd 
df = pd.read_csv('BSCY4_Lab_2.csv')
print(df.head())

# Q. Assess normality of MPG values.
# Kurtosis
skewness = df['mpg'].skew(axis = 0, skipna = True)
print("Skewness: ", skewness)
# Skewness
kurtosis = df['mpg'].kurtosis()
print("Kurtosis: ", kurtosis)

# (1) Visual Inspection
import seaborn as sns
import matplotlib.pyplot as plt
plt.show(sns.distplot(df['mpg']))

# (2) NHST
from scipy import stats
shapiroMpg = stats.shapiro(df['mpg'])
print("Shapiro-Wilk Test for 'mpg': ",shapiroMpg)

# (3) Q-Q Plot
plt.show(stats.probplot(df['mpg'], dist="norm", plot=plt))

# Q. Do the numbers appear to come from a normal distribution? 
# If not, can a transformation be applied so that its result is normal?
# A. The skewness indicates that the distribution may be normal.
# The kurtosis indictes that the data is not normal.
# Upon visual inspection we can see the data appears as though it may be normal.
# The null hypothesis significance testing proves the data's normality.
# The Q-Q plot matches quintiles of the data and is linear confirming normaility.

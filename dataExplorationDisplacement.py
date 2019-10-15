# Q. Import data from "BSCY4.csv".
# A.
import pandas as pd 
df = pd.read_csv('BSCY4_Lab_2.csv')
print(df.head())

# Q. Assess normality of displacement values.
# Kurtosis
skewness = df['displacement'].skew(axis = 0, skipna = True)
print("Skewness: ", skewness)
# Skewness
kurtosis = df['displacement'].kurtosis()
print("Kurtosis: ", kurtosis)

# (1) Visual Inspection
import seaborn as sns
import matplotlib.pyplot as plt
plt.show(sns.distplot(df['displacement']))

# (2) NHST
from scipy import stats
shapiroDisplacement = stats.shapiro(df['displacement'])
print("Shapiro-Wilk Test for 'displacement': ",shapiroDisplacement)

# (3) Q-Q Plot
plt.show(stats.probplot(df['displacement'], dist="norm", plot=plt))

# Q. Do the numbers appear to come from a normal distribution? 
# If not, can a transformation be applied so that its result is normal?
# A. The skewness indicates that the distribution is not normal.
# The kurtosis indictes that the data is not normal.
# Upon visual inspection we can see the data appears as though it is not normal with left-skew.
# The null hypothesis significance testing proves the data is not normal.
# The Q-Q plot confirms the above.

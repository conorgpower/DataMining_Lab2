import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import statsmodels.api as sm

# Step 1
def importData():
    # Q. Import data from "BSCY4.csv".
    df = pd.read_csv('BSCY4_Lab_2.csv')
    print(df.head())
    return df

# Step 2
def checkNormalityMpg(df):
    # Q. Assess normality of MPG values.
    # Kurtosis
    skewness = df['mpg'].skew(axis = 0, skipna = True)
    print("Skewness: ", skewness)
    # Skewness
    kurtosis = df['mpg'].kurtosis()
    print("Kurtosis: ", kurtosis)

    # (1) Visual Inspection
    plt.title("MPG")
    plt.show(sns.distplot(df['mpg']))

    # (2) NHST
    shapiroMpg = stats.shapiro(df['mpg'])
    print("Shapiro-Wilk Test for 'mpg': ",shapiroMpg)

    # (3) Q-Q Plot
    plt.title("MPG")
    plt.show(stats.probplot(df['mpg'], dist="norm", plot=plt))

    # Q. Do the numbers appear to come from a normal distribution? 
    # If not, can a transformation be applied so that its result is normal?
    # A. The skewness indicates that the distribution may be normal.
    # The kurtosis indictes that the data is not normal.
    # Upon visual inspection we can see the data appears as though it may be normal.
    # The null hypothesis significance testing proves the data's normality.
    # The Q-Q plot matches quintiles of the data and is linear confirming normaility.

# Step 3
def checkNormailityNumeric(df):
    checkNormalityInput(df, 'acceleration')
    # Q. Do the numbers appear to come from a normal distribution? 
    # If not, can a transformation be applied so that its result is normal?
    # A. The skewness indicates that the distribution may be normal.
    # The kurtosis indictes that the data is not normal.
    # Upon visual inspection we can see the data appears as though it is normal.
    # The null hypothesis significance testing proves the data is normal.
    # The Q-Q plot agrees with the NHST.

    checkNormalityInput(df, 'displacement')
    # Q. Do the numbers appear to come from a normal distribution? 
    # If not, can a transformation be applied so that its result is normal?
    # A. The skewness indicates that the distribution is not normal.
    # The kurtosis indictes that the data is not normal.
    # Upon visual inspection we can see the data appears as though it is not normal with left-skew.
    # The null hypothesis significance testing proves the data is not normal.
    # The Q-Q plot confirms the above.

    checkNormalityInput(df, 'horsepower')
    # Q. Do the numbers appear to come from a normal distribution? 
    # If not, can a transformation be applied so that its result is normal?
    # A. The skewness indicates that the distribution is not normal.
    # The kurtosis indictes that the data is not normal.
    # Upon visual inspection we can see the data appears as though it is not normal.
    # The null hypothesis significance testing proves the data is not normal.
    # The Q-Q plot confirms the above.

    checkNormalityInput(df, 'weight')
    # Q. Do the numbers appear to come from a normal distribution? 
    # If not, can a transformation be applied so that its result is normal?
    # A. The skewness indicates that the distribution is not normal.
    # The kurtosis indictes that the data is not normal.
    # Upon visual inspection we can see the data appears as though it is not normal.
    # The null hypothesis significance testing proves the data is not normal.
    # The Q-Q plot confirms non-normality of the data.
    df['weight'] = df["weight"].apply(np.log)
    shapiro = stats.shapiro(df['weight'])
    print("Shapiro-Wilk Test for 'weight log': ", shapiro)
    # A transform can be applied to weight to make it normal
    return df

# Step 4 
def regressionAssumptions(df):
    # Q. Which of numerical fields satisfy the assumptions of regression analysis?
    # The log of weight and acceleration are the only normal predictors and the only fields
    # that can possibly satisfy assumptions.

    print("Correlation of weight and acceleration: ", df['weight'].corr(df['acceleration']))
    # Weight and Acceleration have a correlation of -0.5593202022603644
    # This is below .95 so there is predictor independance. And we can use the two for our regression model.
    # They both satisfy the assumptions of regression analysis

# Step 5
def onePredictorModels(df):
    # Q. Build an initial regression model that incorporates only one numerical 
    # predictor. Ensure the model satisCies all of the regression assumptions

    model = sm.OLS(df['mpg'], df['acceleration'])
    results = model.fit()
    print(results.summary())

    plt.figure()
    plt.scatter(df['mpg'], results.resid)
    plt.title("Acceleration")
    plt.show()

    # The model based on mpg and acceleration, the model does not satisfy the assumptions as
    # the data on the plot scattered wildly and is not normal.

    model = sm.OLS(df['mpg'], df['weight'])
    results = model.fit()
    print(results.summary())

    plt.figure()
    plt.scatter(df['mpg'], results.resid)
    plt.title("Weight")
    plt.show()

    # The model does not pass the assumption of regression as there is no homoscedascity 
    # of residuals.

# Step 6 
def twoPredictorModel(df):
    # Acceleration and log of weight will be used for this regression model 
    # as they are not correlated
    X = np.column_stack((df['weight'], df['acceleration']))
    X = sm.add_constant(X)
    model = sm.OLS(df["mpg"], X)
    results = model.fit()
    print(results.summary())

    plt.figure()
    plt.scatter(df["mpg"], results.resid)
    plt.title("Weight-Acceleration")
    plt.show()

    # From plotting this graph we can see that the data is homoscedastic and is not 
    # as accurate of a representation as acceleration. It does satisfy regression assumptions.

def mediationAnalysis(df):
    # Step 7
    # Q. What can you say about the extended regression model? 
    # Is there mediation effect present?

    # Comparing previous regression results:
    # (1) mpg based on acceleration
    # (2) mpg based on weight
    # (3) mpg based on acceleration and weight

    X = sm.add_constant(df["weight"])
    model = sm.OLS(df["acceleration"], X)
    results = model.fit()
    print(results.summary())

    # Upon examination we see that when weight is added the P-value icreases from 0.000 to 0.278.
    # Also R-Squared value decreases from 0.967 to 0.330.
    # A mediation effect is presented by weight and it should be dropped.

# Step 8
def categoricalModel(df):
    # Q. Introduce a categorical variable into the model.
    # Are all of the categories significant?

    dummies = pd.get_dummies(pd.Series(df["model year"]))
    X = np.column_stack((df["acceleration"], dummies))
    X = sm.add_constant(X)
    model = sm.OLS(df["mpg"], X)
    results = model.fit()
    print(results.summary())

    plt.figure()
    plt.scatter(df["mpg"], results.resid)
    plt.title("Acceleration-Model Year")
    plt.show()

    # Here we can see that three of the categories in 'model year' 
    # are unique and significant.

# Step 9
def mediationAnalysisCategorical(df):
    # Is there a potential mediation effect governed by the categorical variable?
    # If yes, how should the model be updated?

    # Comparing previous regression results:
    # (1) mpg based on acceleration
    # (2) mpg based on model year
    # (3) mpg based on acceleration and model year

    dummies = pd.get_dummies(pd.Series(df["model year"]), drop_first=True)
    X = sm.add_constant(dummies)
    model = sm.OLS(df["mpg"], X)
    results = model.fit()
    print(results.summary())

    plt.figure()
    plt.scatter(df["mpg"], results.resid)
    plt.title("Model Year")
    plt.show()

    # Based on the change in R-Squared we can say that there is a mediation effect.
    # The model should be updated by removing the catagories in Model Year which have a p-value of
    # above .05 as these are not relevant to the regression model.

    # Upon examination it appears that there is a mediation affect.
    # This is evident form the change in R-Squared.
    # The categories of 'model year' with a P-value above 0.05 should be removed from
    # the model as they are not useful.

# General function to check normality of a value in step 3
def checkNormalityInput(df, value):
    # Q. Assess normality of value.
    # Kurtosis
    skewness = df[value].skew(axis = 0, skipna = True)
    print("Skewness " + value + ": ", skewness)
    # Skewness
    kurtosis = df[value].kurtosis()
    print("Kurtosis " + value + ": ", kurtosis)

    # (1) Visual Inspection
    plt.title(value)
    plt.show(sns.distplot(df[value]))

    # (2) NHST
    shapiro = stats.shapiro(df[value])
    print("Shapiro-Wilk Test for '" + value + "': ",shapiro)

    # (3) Q-Q Plot
    plt.title(value)
    plt.show(stats.probplot(df[value], dist="norm", plot=plt))

def main():
    df = importData()
    checkNormalityMpg(df)
    df = checkNormailityNumeric(df)
    regressionAssumptions(df)
    onePredictorModels(df)
    twoPredictorModel(df)
    mediationAnalysis(df)
    categoricalModel(df)
    mediationAnalysisCategorical(df)

main()
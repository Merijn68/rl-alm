import numpy as np
import pandas as pd

# Define the number of time periods and variables
n_periods = 100
n_vars = 3

# Create a random correlation matrix
corr_matrix = np.random.uniform(low=-1, high=1, size=(n_vars, n_vars))
np.fill_diagonal(corr_matrix, 1)
corr_matrix = np.tril(corr_matrix) + np.tril(corr_matrix, k=-1).T

# Generate random normal data
data = np.random.normal(size=(n_periods, n_vars))

# Use Cholesky decomposition to create correlated random data
chol_matrix = np.linalg.cholesky(corr_matrix)
corr_data = np.matmul(data, chol_matrix)

# Convert the data to a Pandas DataFrame
df = pd.DataFrame(corr_data, columns=['Var1', 'Var2', 'Var3'])

# Check the correlation matrix of the simulated data
print(df.corr())

########################################################################


import pandas as pd
import numpy as np

# Load the yield curve data from ECB statistics warehouse
url = "https://sdw.ecb.europa.eu/quickviewexport.do;jsessionid=FE836C6D578C7F8EE22DD93E0C1C9209?SERIES_KEY=143.FM.M.U2.EUR.4F.G_N_A.SV_C_YM_B_A10.A.IRTA_G_N_A"

df = pd.read_csv(url, sep='\t', header=None, skiprows=1, names=['Date', 'Maturity', 'Yield'])

# Pivot the data to create a table with each maturity as a column
pivot_df = df.pivot(index='Date', columns='Maturity', values='Yield')

# Calculate the daily percentage change for each maturity
returns_df = pivot_df.pct_change().dropna()

# Calculate the correlation matrix between the maturities
corr_matrix = returns_df.corr()

# Generate random numbers
n_rows = len(returns_df)
n_cols = len(returns_df.columns)
random_numbers = np.random.normal(size=(n_rows, n_cols))

# Use Cholesky decomposition to convert the correlation matrix into a matrix
# that can be used to generate correlated random numbers
cholesky_matrix = np.linalg.cholesky(corr_matrix)

# Multiply the Cholesky matrix by the random numbers to create correlated
# random numbers
correlated_numbers = np.matmul(cholesky_matrix, random_numbers.T).T

# Convert the correlated numbers into a Pandas DataFrame
simulated_returns_df = pd.DataFrame(data=correlated_numbers, columns=returns_df.columns)

# Scale the simulated returns to match the original data
simulated_returns_df = simulated_returns_df.multiply(returns_df.std()) + returns_df.mean()

# Convert the returns back to yields and add to the original yield curve
simulated_yield_df = pivot_df.shift(1) * (1 + simulated_returns_df)

# Print the first few rows of the data
print(simulated_yield_df.head())

# we load the yield curve data from the ECB statistics warehouse and pivot the data to create a table with each maturity as a column. We then calculate the daily percentage change for each maturity and use this to generate a correlation matrix between the maturities. We generate random numbers using NumPy's random.normal function and use Cholesky decomposition to convert the correlation matrix into a lower triangular matrix that can be used to generate correlated random numbers. We then multiply the lower triangular matrix by the random numbers to create the correlated random numbers and scale them to match the standard deviation and mean of the original data. Finally, we convert the simulated returns back to yields and add them to the original yield curve.
# Defining the portfolio is a crucial step in building a model of a bank. Here are some general guidelines to help you get started:
# Identify the types of assets and liabilities: Start by identifying the types of assets and liabilities that the bank holds. This could include loans, deposits, securities, derivatives, and other financial instruments.
# Collect data: Collect data on the characteristics of each asset and liability in the portfolio, such as the interest rate, maturity, credit rating, and any embedded options or features.
# Group assets and liabilities by risk factors: Group the assets and liabilities into categories based on the risk factors that affect their value. For example, you could group fixed-income assets by the yield curve, or group loans by credit risk.
# Define the risk factors: For each category of assets and liabilities, define the relevant risk factors that affect their value. For example, the risk factors for fixed-income assets might include the level and shape of the yield curve, while the risk factors for loans might include credit risk and interest rates.
# Determine correlations: Determine the correlations between the risk factors. This can be done through statistical analysis of historical data or expert judgment. Correlations are important because they determine how changes in one risk factor affect the value of other assets and liabilities in the portfolio.
# Define assumptions: Define any assumptions or simplifications that you will make in the model. For example, you may assume that the portfolio is static and does not change over time, or you may assume that the bank can hedge its risks perfectly.
# Validate the model: Once you have defined the portfolio and the risk factors, you should validate the model by comparing its output to historical data or expert judgment. This will help you identify any weaknesses or limitations in the model and refine it as needed.
# It's important to note that the specific approach you take to defining the portfolio will depend on the nature of the bank and its business. You may also need to take into account regulatory requirements and accounting standards when defining the portfolio.

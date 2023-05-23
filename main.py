import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing

# read data and drop unnecessary columns
data = pd.read_csv('heart_failure_clinical_records_dataset.csv')
data = data.drop(columns = ['anaemia','diabetes','high_blood_pressure','sex','smoking','time','DEATH_EVENT'])

# define number of bins
n_bins = 20

# plot data before standardization
fig1, axs = plt.subplots(2,3)
columns = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']
for i, column in enumerate(columns):
    axs[i//3, i%3].hist(data[column].values, bins = n_bins)
    axs[i//3, i%3].set_title(column)

# standardize data
scaler = preprocessing.StandardScaler().fit(data)
data_scaled = scaler.transform(data)

# plot data after standardization
fig2, axs = plt.subplots(2,3)
for i, column in enumerate(columns):
    axs[i//3, i%3].hist(data_scaled[:,i], bins = n_bins)
    axs[i//3, i%3].set_title(column)

# calculate mean and standard deviation
print(data.mean())
print(data.std())

# calculate mean and standard deviation by hand
result_b_a=[[0,0,0,0,0,0],[0,0,0,0,0,0]]
for i in range(len(data)):
    for j in range(len(columns)):
        result_b_a[0][j] += data.iloc[i][j]
        result_b_a[1][j] += data_scaled[i][j]
for i in range(len(result_b_a)):
    for j in range(len(result_b_a[i])):
        result_b_a[i][j] /= len(data)
    print(result_b_a[i])

# transform data using MinMaxScaler
min_max_scaler = preprocessing.MinMaxScaler().fit(data)
data_min_max_scaled = min_max_scaler.transform(data)
fig3, axs = plt.subplots(2,3)
for i, column in enumerate(columns):
    axs[i//3, i%3].hist(data_min_max_scaled[:,i], bins = n_bins)
    axs[i//3, i%3].set_title(column)

# transform data using MaxAbsScaler
max_abs_scaler = preprocessing.MaxAbsScaler().fit(data)
data_max_abs_scaler = max_abs_scaler.transform(data)
fig4, axs = plt.subplots(2,3)
for i, column in enumerate(columns):
    axs[i//3, i%3].hist(data_max_abs_scaler[:,i], bins = n_bins)
    axs[i//3, i%3].set_title(column)

# robustscaler
rs = preprocessing.RobustScaler().fit(data)
data_rs = rs.transform(data)
fig5, axs = plt.subplots(2,3)
for i, ax in enumerate(axs.flatten()):
    ax.hist(data_rs[:,i], bins = n_bins)
    ax.set_title(data.columns[i])
# plt.show()

#stand
df_st = data[['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']]
df_norm = (df_st - df_st.min()) / (df_st.max() - df_st.min())
df_norm = df_norm * 15 - 5
print(df_norm.max())
print(df_norm.min())
data[['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']] = df_norm

#quantile transformer
qt = preprocessing.QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal').fit(data)
data_qt = qt.transform(data)
fig6, axs = plt.subplots(2,3)
for i, ax in enumerate(axs.flatten()):
    ax.hist(data_qt[:,i], bins = n_bins)
    ax.set_title(data.columns[i])
# plt.show()

#PowerTransformer
pt = preprocessing.PowerTransformer(method='yeo-johnson', standardize=True, copy=True).fit(data)
data_pt = pt.transform(data)
fig7, axs = plt.subplots(2,3)
for i, ax in enumerate(axs.flatten()):
    ax.hist(data_pt[:,i], bins = n_bins)
    ax.set_title(data.columns[i])
# plt.show()

#discretizer
k_bins = preprocessing.KBinsDiscretizer(n_bins=[3,4,3,10,2,4], encode='ordinal').fit(data)
data_k_bins = k_bins.transform(data)
fig8, axs = plt.subplots(2,3)
for i, ax in enumerate(axs.flatten()):
    ax.hist(data_k_bins[:,i], bins = n_bins)
    ax.set_title(data.columns[i])
plt.show()




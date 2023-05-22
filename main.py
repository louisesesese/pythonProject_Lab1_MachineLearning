import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing

df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
df = df.drop(columns = ['anaemia','diabetes','high_blood_pressure','sex','smoking','time','DEATH_EVENT'])

n_bins = 20

#данные до стандартизации
fig1, axs = plt.subplots(2,3)
axs[0, 0].hist(df['age'].values, bins = n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(df['creatinine_phosphokinase'].values, bins = n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(df['ejection_fraction'].values, bins = n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(df['platelets'].values, bins = n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(df['serum_creatinine'].values, bins = n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(df['serum_sodium'].values, bins = n_bins)
axs[1, 2].set_title('serum_sodium')
# plt.show()

data = df.to_numpy(dtype='float') #стандартизация
scaler = preprocessing.StandardScaler().fit(data)
data_scaled = scaler.transform(data)

#данные после стандартизации
fig2, axs = plt.subplots(2,3)
axs[0, 0].hist(data_scaled[:,0], bins = n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(data_scaled[:,1], bins = n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(data_scaled[:,2], bins = n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(data_scaled[:,3], bins = n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(data_scaled[:,4], bins = n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(data_scaled[:,5], bins = n_bins)
axs[1, 2].set_title('serum_sodium')
# plt.show()

print(df.mean()) #матожидание
print(df.std()) #ско

#мо подсчитанное руками
#массив значений математического ожидания до и после стандартизации
result_b_a=[[0,0,0,0,0,0],[0,0,0,0,0,0]]
#нахождение МО
for i in range(len(data)):
    result_b_a[0][0]+=data[i][0]
    result_b_a[0][1]=data[i][1]
    result_b_a[0][2]=data[i][2]
    result_b_a[0][3]=data[i][3]
    result_b_a[0][4]=data[i][4]
    result_b_a[0][5]=data[i][5]
    result_b_a[1][0] += data_scaled[i][0]
    result_b_a[1][1] = data_scaled[i][1]
    result_b_a[1][2] = data_scaled[i][2]
    result_b_a[1][3] = data_scaled[i][3]
    result_b_a[1][4] = data_scaled[i][4]
    result_b_a[1][5] = data_scaled[i][5]
for i in range(len(result_b_a)):
    for j in range(len(result_b_a[i])):
        result_b_a[i][j]/=len(data)
    print(result_b_a[i])

#приведение к диапазону с помощью minmaxscaler
min_max_scaler = preprocessing.MinMaxScaler().fit(data)
data_min_max_scaled = min_max_scaler.transform(data)
fig3, axs = plt.subplots(2,3)
axs[0, 0].hist(data_min_max_scaled[:,0], bins = n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(data_min_max_scaled[:,1], bins = n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(data_min_max_scaled[:,2], bins = n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(data_min_max_scaled[:,3], bins = n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(data_min_max_scaled[:,4], bins = n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(data_min_max_scaled[:,5], bins = n_bins)
axs[1, 2].set_title('serum_sodium')
# plt.show()

#приведение к диапазону с помощью maxabsscaler
max_abs_scaler = preprocessing.MaxAbsScaler().fit(data)
data_max_abs_scaler = max_abs_scaler.transform(data)
fig4, axs = plt.subplots(2,3)
axs[0, 0].hist(data_max_abs_scaler[:,0], bins = n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(data_max_abs_scaler[:,1], bins = n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(data_max_abs_scaler[:,2], bins = n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(data_max_abs_scaler[:,3], bins = n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(data_max_abs_scaler[:,4], bins = n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(data_max_abs_scaler[:,5], bins = n_bins)
axs[1, 2].set_title('serum_sodium')
# plt.show()

#приведение к диапазону с помощью robustscaler
robust_scaler = preprocessing.RobustScaler().fit(data)
data_robust_scaler = robust_scaler.transform(data)
fig5, axs = plt.subplots(2,3)
axs[0, 0].hist(data_robust_scaler[:,0], bins = n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(data_robust_scaler[:,1], bins = n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(data_robust_scaler[:,2], bins = n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(data_robust_scaler[:,3], bins = n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(data_robust_scaler[:,4], bins = n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(data_robust_scaler[:,5], bins = n_bins)
axs[1, 2].set_title('serum_sodium')
# plt.show()

#стандартизация -5 10
df_st=df[['age', 'creatinine_phosphokinase',
          'ejection_fraction', 'platelets',
          'serum_creatinine', 'serum_sodium']]
df[['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium'] ]= (df_st-df_st.min())/(df_st.max()-df_st.min())*15-5
print(df.max())
print(df.min())

#нелинейные преобразования
#quantile_transformer = preprocessing.QuantileTransformer(n_quantiles = 100, random_state=0).fit(data)
quantile_transformer = preprocessing.QuantileTransformer(n_quantiles = 100, random_state=0, output_distribution='normal').fit(data)
data_quantile_scaled = quantile_transformer.transform(data)
fig6, axs = plt.subplots(2,3)
axs[0, 0].hist(data_quantile_scaled[:,0], bins = n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(data_quantile_scaled[:,1], bins = n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(data_quantile_scaled[:,2], bins = n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(data_quantile_scaled[:,3], bins = n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(data_quantile_scaled[:,4], bins = n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(data_quantile_scaled[:,5], bins = n_bins)
axs[1, 2].set_title('serum_sodium')
# plt.show()


#нелинейные преобразования с помощью PowerTransformer
power_transformer = preprocessing.PowerTransformer(method='yeo-johnson', standardize=True, copy=True).fit(data)
power_transformer_scaled = power_transformer.transform(data)
fig7, axs = plt.subplots(2,3)
axs[0, 0].hist(power_transformer_scaled[:,0], bins = n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(power_transformer_scaled[:,1], bins = n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(power_transformer_scaled[:,2], bins = n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(power_transformer_scaled[:,3], bins = n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(power_transformer_scaled[:,4], bins = n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(power_transformer_scaled[:,5], bins = n_bins)
axs[1, 2].set_title('serum_sodium')
# plt.show()

#дискретизация признаков
k_bins_discretizer = preprocessing.KBinsDiscretizer(n_bins=[3,4,3,10,2,4], encode='ordinal').fit(data)
k_bins_discretizer_scaled = k_bins_discretizer.transform(data)
fig8, axs = plt.subplots(2,3)
axs[0, 0].hist(k_bins_discretizer_scaled[:,0], bins = n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(k_bins_discretizer_scaled[:,1], bins = n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(k_bins_discretizer_scaled[:,2], bins = n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(k_bins_discretizer_scaled[:,3], bins = n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(k_bins_discretizer_scaled[:,4], bins = n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(k_bins_discretizer_scaled[:,5], bins = n_bins)
axs[1, 2].set_title('serum_sodium')
plt.show()


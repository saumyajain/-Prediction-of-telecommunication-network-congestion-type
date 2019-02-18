import numpy as np
import pandas as pd
import seaborn as sns
#import matplotlib.pyplot as plt
import lightgbm as lgbm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc
sns.set_style("whitegrid")
import xgboost as xgb
from sklearn.metrics import matthews_corrcoef



df=pd.read_csv("train.csv")
df_test_results = pd.read_csv("y_test.csv")
df_test = pd.read_csv("test.csv")
y_tests=np.array(df_test_results[df_test_results.columns[-1]]).reshape((df_test_results.shape[0]),1)
df = df[['subscriber_count', 'web_browsing_total_bytes',
       'video_total_bytes', 'social_ntwrking_bytes',
       'cloud_computing_total_bytes', 'web_security_total_bytes',
       'gaming_total_bytes', 'health_total_bytes', 'communication_total_bytes',
       'file_sharing_total_bytes', 'remote_access_total_bytes',
       'photo_sharing_total_bytes', 'software_dwnld_total_bytes',
       'marketplace_total_bytes', 'storage_services_total_bytes',
       'audio_total_bytes', 'location_services_total_bytes',
       'presence_total_bytes', 'advertisement_total_bytes',
       'system_total_bytes', 'voip_total_bytes', 'speedtest_total_bytes',
       'email_total_bytes', 'weather_total_bytes', 'media_total_bytes',
       'mms_total_bytes', 'others_total_bytes','tilt','beam_direction','cell_range','Congestion_Type']]

df_tests = df_test[['subscriber_count', 'web_browsing_total_bytes',
       'video_total_bytes', 'social_ntwrking_bytes',
       'cloud_computing_total_bytes', 'web_security_total_bytes',
       'gaming_total_bytes', 'health_total_bytes', 'communication_total_bytes',
       'file_sharing_total_bytes', 'remote_access_total_bytes',
       'photo_sharing_total_bytes', 'software_dwnld_total_bytes',
       'marketplace_total_bytes', 'storage_services_total_bytes',
       'audio_total_bytes', 'location_services_total_bytes',
       'presence_total_bytes', 'advertisement_total_bytes',
       'system_total_bytes', 'voip_total_bytes', 'speedtest_total_bytes',
       'email_total_bytes', 'weather_total_bytes', 'media_total_bytes',
       'mms_total_bytes', 'others_total_bytes','tilt','beam_direction','cell_range']]

x=np.array(df.drop(df.columns[[-1]],axis=1))
y=np.array(df[df.columns[-1]]).reshape((df.shape[0]),1)

x_tests = np.array(df_tests)
#
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x)
x_tests = sc.transform(x_tests)
##from sklearn.decomposition import PCA
##pca = PCA(n_components=1)
##principalcomponents = pca.fit_transform(x)
#x_tests = pd.DataFrame(pca.transform(x_tests))
#principalDf = pd.DataFrame(data = principalcomponents)
#x_train, x_test, train_y, test_y = train_test_split(x, y, test_size=0.0, random_state = 0)
#x_train = x
train_y = y

print(type(train_y))
train_y = train_y.reshape(-1)
y_tests = y_tests.reshape(-1)
#test_y = test_y.reshape(-1)
train_y = pd.Series(train_y)
#test_y = pd.Series(test_y)
val1 = {"4G_BACKHAUL_CONGESTION":1,"NC":2,"3G_BACKHAUL_CONGESTION":3,"4G_RAN_CONGESTION":4}
train_y = [val1[item] for item in train_y]
#y_tests = [val1[item] for item in y_tests]
y_tests = [val1[item] for item in y_tests] 
train_y = pd.Series(train_y)        
print(train_y)
train_data=lgbm.Dataset(x_train, label=train_y)

#Select Hyper-Parameters
params = {'boosting_type': 'gbdt',
          'max_depth' : -1,
          'objective': 'multiclass',
          'num_class':5,
          'nthread': 5,
          'num_leaves': 64,
          'learning_rate': 0.07,
          'max_bin': 512,
          'subsample_for_bin': 200,
          'subsample': 1,
          'subsample_freq': 1,
          'colsample_bytree': 0.8,
          'reg_alpha': 1.2,
          'reg_lambda': 1.2,
          'min_split_gain': 0.5,
          'min_child_weight': 1,
          'min_child_samples': 5,
          'scale_pos_weight': 1,
          'metric' : 'multi_logloss'
          }

#lgb_cv = lgbm.cv(params, train_x, num_boost_round=10000, nfold=3, shuffle=True, stratified=True, verbose_eval=20, early_stopping_rounds=100)
#nround = lgb_cv['multi_logloss-mean'].index(np.min(lgb_cv['multi_logloss-mean']))
#clf = lgbm.train(params, d_train, num_boost_round=nround)

clf = lgbm.train(params, train_data, 100)
#y_pred=clf.predict(test_x)
y_preds = clf.predict(x_tests)

predictions = []
preds = []

for x in y_preds:
    preds.append(np.argmax(x))

print(preds)
accuracy=accuracy_score(preds,y_tests)
#accuracys = accuracy_score(prediction)
print(clf.score)
print(accuracy)
m_corref = matthews_corrcoef(preds,y_tests)

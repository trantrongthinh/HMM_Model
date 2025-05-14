import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# Đọc và parse file WISDM raw.txt
path = r"D:\TH_Time_Series\WISDM_ar_v1.1\WISDM_ar_v1.1_raw.txt"

records = []
with open(path, 'r') as f:
    for line in f:
        for rec in line.strip().split(';'):
            if not rec:
                continue
            fields = rec.split(',')
            if len(fields) != 6 or any(fld.strip() == '' for fld in fields):
                continue
            records.append(fields)

# Chuyển thành DataFrame
data = pd.DataFrame(records, columns=['user', 'activity', 'timestamp', 'x', 'y', 'z'])
data = data.astype({
    'user': int,
    'timestamp': int,
    'x': float, 'y': float, 'z': float
})

# In phân bố hoạt động
print("Phân bố hoạt động trong dữ liệu thô:")
print(data['activity'].value_counts())

# Lọc các hoạt động cần thiết
activities = ['Walking', 'Jogging', 'Downstairs', 'Upstairs']
data = data[data['activity'].isin(activities)]

# Chuẩn hóa dữ liệu gia tốc
scaler = StandardScaler()
data[['x', 'y', 'z']] = scaler.fit_transform(data[['x', 'y', 'z']])

# Phân đoạn dữ liệu thành cửa sổ
window_size = 50
def extract_features(window):
    features = []
    for axis in ['x', 'y', 'z']:
        features.extend([
            np.mean(window[axis]),
            np.std(window[axis]),
            np.min(window[axis]),
            np.max(window[axis])
        ])
    return np.array(features)

# Tạo danh sách các cửa sổ và nhãn
windows = []
labels = []
for user in data['user'].unique():
    user_data = data[data['user'] == user]
    for i in range(0, len(user_data) - window_size, window_size):
        window = user_data.iloc[i:i + window_size]
        if len(window) == window_size and len(window['activity'].unique()) == 1:
            windows.append(extract_features(window))
            labels.append(window['activity'].iloc[0])

windows = np.array(windows)
labels = np.array(labels)

# In số lượng cửa sổ
print("Số lượng cửa sổ cho mỗi hoạt động:")
print(pd.Series(labels).value_counts())

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(windows, labels, test_size=0.2, random_state=42)

# Lưu tập test
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

# Huấn luyện một HMM cho mỗi hoạt động
n_states = 3
models = {}
for activity in activities:
    X_activity = X_train[y_train == activity]
    if len(X_activity) == 0:
        print(f"No training data for {activity}. Skipping.")
        continue
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100)
    model.fit(X_activity)
    models[activity] = model

# Tạo thư mục để lưu models
os.makedirs('models', exist_ok=True)

# Lưu models và scaler
for activity, model in models.items():
    joblib.dump(model, f'models/hmm_{activity}.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(activities, 'models/activities.pkl')

print("Training completed. Models and scaler saved to 'models/' directory.")
print("Test data saved: X_test.npy, y_test.npy")
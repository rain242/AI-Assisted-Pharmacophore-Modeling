import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from tqdm import tqdm

# 读取CSV文件
file_path = 'table.csv'
df = pd.read_csv(file_path)

# 定义计算分子指纹的函数
def calculate_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return np.array(fp)

# 计算特征并过滤无效的SMILES
df['features'] = df['smiles'].apply(calculate_fingerprint)
df = df.dropna(subset=['features'])

# 过滤掉 y 为 NaN 的样本
df = df.dropna(subset=['ic50(nm)'])

# 提取特征和标签
X = np.array(df['features'].tolist())
y = df['ic50(nm)'].values

# 归一化目标值
y_min = np.min(y)
y_max = np.max(y)
y_normalized = (y - y_min) / (y_max - y_min)

# 定义训练次数
num_epochs = 100  # 你可以根据需要调整这个值

def train_model(model, train_loader, val_loader, criterion, optimizer, device, model_name, fold, num_epochs=3):
    best_mse = np.inf
    best_r2 = -np.inf
    best_model = None
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels.unsqueeze(1))  # 确保维度匹配
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}')

        # 评估模型
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for inputs, labels in val_loader:  # 使用验证集的 DataLoader
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                y_pred.extend(outputs.cpu().numpy().flatten())
                y_true.extend(labels.cpu().numpy().flatten())

        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        if mse < best_mse:  # 更新最佳模型
            best_mse = mse
            best_r2 = r2
            best_model = model

    return best_mse, best_r2

# 定义网格搜索参数
svm_params = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

xgb_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# 修正 XGBoost 模型的继承问题
class XGBRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.model = xgb.XGBRegressor(**kwargs)
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_params(self, deep=True):
        return self.model.get_params(deep)
    
    def set_params(self, **params):
        self.model.set_params(**params)
        return self

# 10-fold 交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 存储所有结果
all_results = []

for random_state in range(1025):
    mse_scores = {model: [] for model in ['SVM', 'Random Forest', 'XGBoost', 'CNN', 'LSTM', 'RNN']}
    r2_scores = {model: [] for model in ['SVM', 'Random Forest', 'XGBoost', 'CNN', 'LSTM', 'RNN']}

    for fold, (train_val_index, test_index) in enumerate(kf.split(X)):
        X_train_val, X_test = X[train_val_index], X[test_index]
        y_train_val, y_test = y_normalized[train_val_index], y_normalized[test_index]
        
        # 再次划分训练集为训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=random_state)

        # 支持向量机
        svm_grid = GridSearchCV(SVR(), svm_params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        svm_grid.fit(X_train, y_train)
        best_svm_model = svm_grid.best_estimator_
        y_pred_svm = best_svm_model.predict(X_val)
        mse_scores['SVM'].append(mean_squared_error(y_val, y_pred_svm))
        r2_scores['SVM'].append(r2_score(y_val, y_pred_svm))

        # 随机森林
        rf_grid = GridSearchCV(RandomForestRegressor(), rf_params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        rf_grid.fit(X_train, y_train)
        best_rf_model = rf_grid.best_estimator_
        y_pred_rf = best_rf_model.predict(X_val)
        mse_scores['Random Forest'].append(mean_squared_error(y_val, y_pred_rf))
        r2_scores['Random Forest'].append(r2_score(y_val, y_pred_rf))

        # XGBoost
        xgb_grid = GridSearchCV(XGBRegressor(), xgb_params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        xgb_grid.fit(X_train, y_train)
        best_xgb_model = xgb_grid.best_estimator_
        y_pred_xgb = best_xgb_model.predict(X_val)
        mse_scores['XGBoost'].append(mean_squared_error(y_val, y_pred_xgb))
        r2_scores['XGBoost'].append(r2_score(y_val, y_pred_xgb))

        # PyTorch Dataset
        class FeatureDataset(Dataset):
            def __init__(self, x, y):
                super().__init__()
                self.x = torch.tensor(x, dtype=torch.float32)
                self.y = torch.tensor(y, dtype=torch.float32)

            def __len__(self):
                return len(self.x)

            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]

        # 数据加载器
        train_loader = DataLoader(FeatureDataset(X_train, y_train), batch_size=32, shuffle=True)
        val_loader = DataLoader(FeatureDataset(X_val, y_val), batch_size=32, shuffle=False)

        # 设定设备
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # CNN模型训练
        class CNNModel(nn.Module):
            def __init__(self):
                super(CNNModel, self).__init__()
                self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
                self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
                self.dropout = nn.Dropout(0.2)
                self.flatten = nn.Flatten()
                self.fc1 = nn.Linear(32 * 2048, 128)  # 调整这里以匹配输入特征数量
                self.fc2 = nn.Linear(128, 1)

            def forward(self, x):
                x = x.unsqueeze(1)  # 增加一个通道维度
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = self.dropout(x)
                x = self.flatten(x)
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        cnn_model = CNNModel().to(device)
        optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        cnn_best_mse, cnn_best_r2 = train_model(cnn_model, train_loader, val_loader, criterion, optimizer, device, 'cnn', fold, num_epochs=num_epochs)
        mse_scores['CNN'].append(cnn_best_mse)
        r2_scores['CNN'].append(cnn_best_r2)

        # LSTM模型训练
        class LSTMModel(nn.Module):
            def __init__(self, input_size=2048):
                super(LSTMModel, self).__init__()
                self.lstm = nn.LSTM(input_size, 64, batch_first=True)
                self.fc = nn.Linear(64, 1)

            def forward(self, x):
                x = x.unsqueeze(1)  # 增加一个时间步维度
                lstm_out, _ = self.lstm(x)
                output = self.fc(lstm_out[:, -1, :])  # 取最后一个时间步的输出
                return output

        lstm_model = LSTMModel().to(device)
        optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        lstm_best_mse, lstm_best_r2 = train_model(lstm_model, train_loader, val_loader, criterion, optimizer, device, 'lstm', fold, num_epochs=num_epochs)
        mse_scores['LSTM'].append(lstm_best_mse)
        r2_scores['LSTM'].append(lstm_best_r2)

        # RNN模型训练
        class RNNModel(nn.Module):
            def __init__(self, input_size=2048):
                super(RNNModel, self).__init__()
                self.rnn = nn.RNN(input_size, 64, batch_first=True)
                self.fc = nn.Linear(64, 1)

            def forward(self, x):
                x = x.unsqueeze(1)  # 增加一个时间步维度
                rnn_out, _ = self.rnn(x)
                output = self.fc(rnn_out[:, -1, :])  # 取最后一个时间步的输出
                return output

        rnn_model = RNNModel().to(device)
        optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        rnn_best_mse, rnn_best_r2 = train_model(rnn_model, train_loader, val_loader, criterion, optimizer, device, 'rnn', fold, num_epochs=num_epochs)
        mse_scores['RNN'].append(rnn_best_mse)
        r2_scores['RNN'].append(rnn_best_r2)

    # 保存参数
    df_models = pd.DataFrame({
        'Model': ['SVM', 'Random Forest', 'XGBoost', 'CNN', 'LSTM', 'RNN'],
        'MSE Mean': [np.mean(mse_scores[model]) for model in mse_scores],
        'MSE Std': [np.std(mse_scores[model]) for model in mse_scores],
        'R2 Mean': [np.mean(r2_scores[model]) for model in r2_scores],
        'R2 Std': [np.std(r2_scores[model]) for model in r2_scores],
        'Random State': [random_state] * 6
    })
    all_results.append(df_models)

# 合并所有结果
final_df = pd.concat(all_results, ignore_index=True)
print(final_df)
# 保存最终结果
final_df.to_excel('model_performance_10fold_normalized_with_random_state.xlsx', index=False)

print('Script execution completed.')

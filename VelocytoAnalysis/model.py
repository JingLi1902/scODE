import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense

def data_augmentation(x, y, noise_factor=0.1):
    noise = noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x.shape)
    x_augmented = x + noise
    y_augmented = y + noise
    return x_augmented, y_augmented

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def build_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def BUILD_MODEL(counts, velocity, genes=None, tfs=None, method='ensemble', n_estimators=10, max_depth=10, lasso_alpha=1, train_size=0.7, noise_factor=0.1):
    '''
    v = f(x, a). return fitted function f
    method: 'rf'(random forest) / 'lasso' / 'linear' / 'ensemble'.
    '''
    if genes is None:
        genes = np.array([True] * counts.shape[1])
    if tfs is None:
        tfs = genes
    
    x, x_val, y, y_val = train_test_split(counts[:, tfs], velocity[:, genes], test_size=1-train_size, random_state=42)
    
    # Data Augmentation
    x_augmented, y_augmented = data_augmentation(x, y, noise_factor)
    x = np.concatenate((x, x_augmented), axis=0)
    y = np.concatenate((y, y_augmented), axis=0)
    
    # Build model
    if method == 'lasso':
        model = linear_model.Lasso(alpha=lasso_alpha)
    elif method == 'linear':
        model = linear_model.LinearRegression(n_jobs=-1)
    elif method == 'rf':
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=9, n_jobs=-1)
    elif method == 'ensemble':
        # Ensemble of models
        model = VotingRegressor([
            ('rf', RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=9, n_jobs=-1)),
            ('gb', GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=9)),
            ('lasso', linear_model.Lasso(alpha=lasso_alpha)),
            ('svr', SVR(kernel='rbf', C=1e3, gamma=0.1)),
            ('xgb', XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=9))
        ])
    
    model = model.fit(x, y)    
    train_score = model.score(x, y)
    test_score = model.score(x_val, y_val)
    
    print('Fitted model | Training R-Square: %.4f; Test R-Square: %.4f' % (train_score, test_score))
    
    # Time Series Analysis with LSTM and GRU
    if counts.ndim == 3:  # If the data has a time series component
        x_train_seq = np.reshape(x, (x.shape[0], x.shape[1], 1))
        x_val_seq = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
        
        # LSTM Model
        lstm_model = build_lstm_model((x_train_seq.shape[1], x_train_seq.shape[2]))
        lstm_model.fit(x_train_seq, y, epochs=50, batch_size=32, validation_data=(x_val_seq, y_val))
        
        train_score_lstm = lstm_model.evaluate(x_train_seq, y, verbose=0)
        test_score_lstm = lstm_model.evaluate(x_val_seq, y_val, verbose=0)
        
        print('LSTM model | Training Loss: %.4f; Test Loss: %.4f' % (train_score_lstm, test_score_lstm))
        
        # GRU Model
        gru_model = build_gru_model((x_train_seq.shape[1], x_train_seq.shape[2]))
        gru_model.fit(x_train_seq, y, epochs=50, batch_size=32, validation_data=(x_val_seq, y_val))
        
        train_score_gru = gru_model.evaluate(x_train_seq, y, verbose=0)
        test_score_gru = gru_model.evaluate(x_val_seq, y_val, verbose=0)
        
        print('GRU model | Training Loss: %.4f; Test Loss: %.4f' % (train_score_gru, test_score_gru))
    
    return model

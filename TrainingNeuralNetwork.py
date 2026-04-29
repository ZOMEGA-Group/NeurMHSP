import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
import time
import pickle as pkl
import random

seed = 42

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


def log_mae_loss(y_true, y_pred):
    epsilon = 1e-6
    y_true_safe = tf.clip_by_value(y_true, 0.0, 1e6)
    y_pred_safe = tf.clip_by_value(y_pred, 0.0, 1e6)
    
    y_true_log = tf.math.log1p(y_true_safe + epsilon)
    y_pred_log = tf.math.log1p(y_pred_safe + epsilon)
    return tf.reduce_mean(tf.abs(y_true_log - y_pred_log))

selected_architectures = {
    0:{'layers': [16,8,4], 'batch size': 32, 'lr':0.01},
    1:{'layers': [32,16,8], 'batch size': 32, 'lr':0.01},
    2:{'layers': [64,32,16], 'batch size': 64, 'lr':0.01},
    3:{'layers': [128,64,32], 'batch size': 32, 'lr':0.01},
    4:{'layers': [256,128,64], 'batch size': 16, 'lr':0.01}
}


n_datasets = 20

for n in range(n_datasets):
    df = pd.read_csv(f'datasets/SRP_5_scenario_{n}.csv')
    df['co2_price'] = 0.018
    
    
    features = df[['0','1','2','3','4','5','6','7','8','9','10','co2_budget', 'co2_price', 'demand_scaling']]
    labels = df['ope_cost']

    scaler = preprocessing.StandardScaler()
    features_scaled = scaler.fit_transform(features)

    df_scaled = pd.DataFrame(features_scaled, columns=features.columns)
    df_scaled['ope_cost'] = labels.reset_index(drop=True)
    
    with open(f'Scalers/scaler_5_scenario_{n}.pkl', 'wb') as f:
        pkl.dump(scaler, f)
    
    X = df_scaled[['0','1','2','3','4','5','6','7','8','9','10','co2_budget', 'co2_price', 'demand_scaling']]
    Y = df_scaled['ope_cost']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    for key, arch in selected_architectures.items():
        
        layers = arch['layers']
        batch_size = arch['batch size']
        lr = arch['lr']
        input_shape = X.shape[1:]
        
        model = keras.Sequential()
        model.add(keras.layers.Input(shape = input_shape))
        model.add(keras.layers.Dense(layers[0],activation='relu', kernel_initializer=keras.initializers.HeNormal()))
        for neurons in layers[1:]:
            model.add(keras.layers.Dense(neurons,activation='relu', kernel_initializer=keras.initializers.HeNormal()))
        model.add(keras.layers.Dense(1))  

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=log_mae_loss)

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)
        
        
        
        start_time = time.time()
        history = model.fit(X_train, y_train, 
                    epochs=1000, 
                    batch_size=batch_size, 
                    validation_data=(X_test, y_test), 
                    callbacks=[early_stopping, reduce_lr],
                    verbose=0)
        end_time = time.time()
        training_time = end_time - start_time

        y_pred = model.predict(X_test)
        y_test_exp = y_test

        mae = metrics.mean_absolute_error(y_test_exp, y_pred)
        
        mask = y_test_exp != 0
        y_test_no_zero = y_test_exp[mask]
        y_pred_no_zero = y_pred[mask]
        
        mape = metrics.mean_absolute_percentage_error(y_test_no_zero, y_pred_no_zero)
        r2 = metrics.r2_score(y_test_exp, y_pred)

        #print(f"Config: dataset = {n} Layers={layers}, LR={lr}, Batch={batch_size} Train metrics: Train time={training_time} Train loss={min(history.history['loss'])} Test metrics: MAE={mae:.4f}, MAPE={mape:.4f}, R²={r2:.4f}")
        print(f"NN_model_5_scenarios_{key}_{n} saved...")
        model.save(f"NN_model_5_scenarios_{key}_{n}.keras")
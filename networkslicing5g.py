import time 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

# 1. Load dataset
start_time = time.time()
df = pd.read_csv('train_dataset_final_cleaned.csv')
X = df.drop(columns=['slice Type']).values.astype('float32')
y = LabelEncoder().fit_transform(df['slice Type'])

# 2. Split data
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=9, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=9, stratify=y_temp)

y_train_cat = to_categorical(y_train)
y_val_cat   = to_categorical(y_val)
y_test_cat  = to_categorical(y_test)

# 3. Define a function to build models
def build_model(config):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    for layer in config['layers']:
        model.add(Dense(**layer['dense']))
        if layer.get('batch_norm'):
            model.add(BatchNormalization())
        if layer.get('dropout'):
            model.add(Dropout(layer['dropout']))
    model.add(Dense(y_train_cat.shape[1], activation='softmax'))
    model.compile(
        optimizer=config['optimizer'],
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 4. Model configurations
config_nn1 = {
    'layers': [
        {'dense': {'units': 8, 'activation': 'relu', 'kernel_regularizer': l2(1e-3)}, 'dropout': 0.3},
        {'dense': {'units': 4, 'activation': 'relu', 'kernel_regularizer': l2(1e-3)}, 'dropout': 0.2},
        {'dense': {'units': 3, 'activation': 'tanh'}}
    ],
    'optimizer': 'adam',
    'early_stop_patience': 5,
    'epochs': 50
}

config_nn2 = {
    'layers': [
        {'dense': {'units': 64, 'activation': 'relu', 'kernel_regularizer': l2(1e-4)}, 'batch_norm': True, 'dropout': 0.4},
        {'dense': {'units': 32, 'activation': 'relu', 'kernel_regularizer': l2(1e-4)}, 'batch_norm': True, 'dropout': 0.3},
        {'dense': {'units': 16, 'activation': 'relu', 'kernel_regularizer': l2(1e-4)}, 'dropout': 0.2}
    ],
    'optimizer': 'adam',
    'early_stop_patience': 6,
    'epochs': 70
}

# 5. Train both models and collect histories
results = {}
for name, cfg in [('NN1', config_nn1), ('NN2', config_nn2)]:
    model = build_model(cfg)
    es = EarlyStopping(monitor='val_loss', patience=cfg['early_stop_patience'], restore_best_weights=True)
    start = time.time()
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=cfg['epochs'], batch_size=64,
        callbacks=[es], verbose=0
    )
    duration = time.time() - start
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    train_loss, train_acc = model.evaluate(X_train, y_train_cat, verbose=0)

    results[name] = {
        'model': model,
        'history': history,
        'train_acc': train_acc * 100,
        'test_acc': test_acc * 100,
        'train_loss': train_loss,
        'test_loss': test_loss,
        'train_time': duration
    }
    print(f"{name}: train_acc={train_acc*100:.2f}%, test_acc={test_acc*100:.2f}%, time={duration:.2f}s")

labels = ['NN1', 'NN2']
n_classes = y_test_cat.shape[1]

# 6. Comparative Plots
# 6.1 Train vs Test Accuracy
train_accs = [results['NN1']['train_acc'], results['NN2']['train_acc']]
test_accs  = [results['NN1']['test_acc'],  results['NN2']['test_acc']]
x = np.arange(len(labels))
width = 0.35
plt.figure()
plt.bar(x - width/2, train_accs, width, label='Train Acc')
plt.bar(x + width/2, test_accs,  width, label='Test Acc')
plt.ylabel('Accuracy (%)')
plt.xticks(x, labels)
plt.title('Train vs Test Accuracy Comparison')
plt.legend()
plt.tight_layout()
plt.show()

# 6.2 Accuracy Curves per Epoch
plt.figure()
for name in labels:
    h = results[name]['history']
    plt.plot(h.history['accuracy'], label=f'{name} Train')
    plt.plot(h.history['val_accuracy'], label=f'{name} Val')
plt.title('Accuracy over Epochs: NN1 vs NN2')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# 6.3 Loss Curves per Epoch
plt.figure()
for name in labels:
    h = results[name]['history']
    plt.plot(h.history['loss'], label=f'{name} Train Loss')
    plt.plot(h.history['val_loss'], label=f'{name} Val Loss')
plt.title('Loss over Epochs: NN1 vs NN2')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# 6.4 Confusion Matrices & Classification Reports
for name in labels:
    model = results[name]['model']
    print(f"\n--- {name} Classification Report & Confusion Matrix ---")
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test_cat, axis=1)

    print(classification_report(y_true, y_pred, target_names=['eMBB','URLLC','mMTC']))

    cm = confusion_matrix(y_true, y_pred)
    print(f"{name} Confusion Matrix:\n", cm)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix: {name}')
    plt.colorbar()
    ticks = np.arange(n_classes)
    plt.xticks(ticks, ['eMBB','URLLC','mMTC'], rotation=45)
    plt.yticks(ticks, ['eMBB','URLLC','mMTC'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

# 6.5 ROC Curves per Model
for name in labels:
    model = results[name]['model']
    y_pred_prob = model.predict(X_test)
    fpr = dict(); tpr = dict(); roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_cat[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.title(f'ROC Curves: {name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

# 7. Overfitting Assessment
for name in labels:
    gap = results[name]['train_acc'] - results[name]['test_acc']
    print(f"{name} Overfitting Gap (Train Acc - Test Acc): {gap:.2f}%")

# 8. Total Script Time
print(f"Total script execution time: {(time.time() - start_time):.2f} sec")

# 9. Feature Distribution Comparison (Figure 4.1)

# Reload datasets in case of earlier modification
df_original = pd.read_csv('train_dataset.csv')
df_cleaned = pd.read_csv('train_dataset_final_cleaned.csv')

# Convert 'slice Type' to numerical for plotting
df_original['slice Type'] = pd.factorize(df_original['slice Type'])[0]
df_cleaned['slice Type'] = pd.factorize(df_cleaned['slice Type'])[0]

# Define features
removed_features = ['IoT', 'LTE/5G', 'Smart Transportation', 'Smart City & Home', 'Non-GBR',
                    'AR/VR/Gaming', 'Healthcare', 'Smartphone', 'GBR', 'Public Safety']
kept_features = ['LTE/5g Category', 'Time', 'Packet Loss Rate', 'Packet delay', 'Industry 4.0', 'IoT Devices']

# Create melted DataFrames
removed_df = df_original[removed_features + ['slice Type']]
kept_df = df_cleaned[kept_features + ['slice Type']]
removed_melted = removed_df.melt(id_vars='slice Type', var_name='Feature', value_name='Value')
kept_melted = kept_df.melt(id_vars='slice Type', var_name='Feature', value_name='Value')

# Plot removed features
plt.figure(figsize=(16, 10))
sns.boxplot(data=removed_melted, x='Feature', y='Value')
plt.title('Κατανομή Χαρακτηριστικών που Αφαιρέθηκαν (Πριν τον Καθαρισμό)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot kept features
plt.figure(figsize=(12, 8))
sns.boxplot(data=kept_melted, x='Feature', y='Value')
plt.title('Κατανομή Τελικών Χαρακτηριστικών (Μετά τον Καθαρισμό)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 10. Πίνακας Συσχέτισης (Heatmap)

corr_matrix = df_cleaned.drop(columns=['slice Type']).corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Πίνακας Συσχέτισης Τελικών Χαρακτηριστικών")
plt.tight_layout()
plt.show()

# 11. Αμοιβαία Πληροφορία (Mutual Information)

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

X = df_cleaned.drop(columns=['slice Type'])
y = df_cleaned['slice Type']

X_scaled = MinMaxScaler().fit_transform(X)

mi_scores = mutual_info_classif(X_scaled, y, discrete_features='auto', random_state=42)
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=mi_series.values, y=mi_series.index)
plt.title("Αμοιβαία Πληροφορία (Mutual Information) με την Slice Type")
plt.xlabel("Mutual Information Score")
plt.tight_layout()
plt.show()

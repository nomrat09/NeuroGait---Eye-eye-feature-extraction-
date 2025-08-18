import pandas as pd
import numpy as np
from scipy.stats import kurtosis, iqr
from mrmr import mrmr_classif
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import optuna
import random
random.seed(42)
np.random.seed(42)

# ------------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------------
df = pd.read_csv(r"C:\Users\pc\Downloads\updated\updated\new_datasets\new_eog (1).csv")

meta_cols = ['subject', 'epoch', 'signal_type']
label_col = 'label'
feature_cols = [c for c in df.columns if c not in meta_cols + [label_col]]

# ------------------------------------------------------------------
# 2. Aggregate features with extra stats
# ------------------------------------------------------------------
agg_df = df.groupby('subject')[feature_cols].agg(['mean', 'std', 'median', 'skew', 'min', 'max'])

# Kurtosis
kurtosis_vals = df.groupby('subject')[feature_cols].apply(
    lambda x: pd.Series(kurtosis(x, fisher=True, bias=False), index=feature_cols)
)
kurtosis_vals.columns = [f"{c}_kurtosis" for c in kurtosis_vals.columns]

# IQR
iqr_vals = df.groupby('subject')[feature_cols].apply(
    lambda x: pd.Series(iqr(x, axis=0), index=feature_cols)
)
iqr_vals.columns = [f"{c}_iqr" for c in iqr_vals.columns]

# Range
range_vals = df.groupby('subject')[feature_cols].apply(
    lambda x: pd.Series(np.max(x, axis=0) - np.min(x, axis=0), index=feature_cols)
)
range_vals.columns = [f"{c}_range" for c in range_vals.columns]

# Combine
agg_df.columns = ['_'.join(col) for col in agg_df.columns]
agg_df = pd.concat([agg_df, kurtosis_vals, iqr_vals, range_vals], axis=1)
agg_df.reset_index(inplace=True)

# Add labels
labels = df.groupby('subject')[label_col].first().reset_index()
agg_df = agg_df.merge(labels, on='subject', how='left')

print(f" Features after aggregation: {agg_df.shape}")

# ------------------------------------------------------------------
# 3. Train-Test Split
# ------------------------------------------------------------------
subjects = agg_df['subject'].values
y_subject = agg_df[label_col].values

train_subjects, test_subjects = train_test_split(
    subjects,
    test_size=0.2,
    random_state=42,
    stratify=y_subject
)

train_df = agg_df[agg_df['subject'].isin(train_subjects)].reset_index(drop=True)
test_df = agg_df[agg_df['subject'].isin(test_subjects)].reset_index(drop=True)

# Inner split for validation
tr_subjs, val_subjs = train_test_split(
    train_df['subject'],
    test_size=0.25,
    random_state=42,
    stratify=train_df[label_col]
)

tr_df = train_df[train_df['subject'].isin(tr_subjs)].reset_index(drop=True)
val_df = train_df[train_df['subject'].isin(val_subjs)].reset_index(drop=True)

# ------------------------------------------------------------------
# 4. Separate features and labels
# ------------------------------------------------------------------
X_tr_df = tr_df.drop(columns=['subject', label_col])
y_tr = tr_df[label_col].values

X_val_df = val_df.drop(columns=['subject', label_col])
y_val = val_df[label_col].values

X_test_df = test_df.drop(columns=['subject', label_col])
y_test = test_df[label_col].values

# ------------------------------------------------------------------
# 5. MRMR Feature Selection
# ------------------------------------------------------------------
print(" Selecting features using MRMR...")
selected_features = mrmr_classif(X=X_tr_df, y=y_tr, K=40)  # Top 40 features
print(f" Selected Features: {selected_features}")

X_tr_df = X_tr_df[selected_features]
X_val_df = X_val_df[selected_features]
X_test_df = X_test_df[selected_features]

# ------------------------------------------------------------------
# 6. Preprocessing
# ------------------------------------------------------------------
imputer = SimpleImputer(strategy='median')
X_tr_imp = imputer.fit_transform(X_tr_df)
X_val_imp = imputer.transform(X_val_df)
X_test_imp = imputer.transform(X_test_df)

scaler = MinMaxScaler()
X_tr_scaled = scaler.fit_transform(X_tr_imp)
X_val_scaled = scaler.transform(X_val_imp)
X_test_scaled = scaler.transform(X_test_imp)

train_pool = Pool(X_tr_scaled, y_tr)
valid_pool = Pool(X_val_scaled, y_val)

# ------------------------------------------------------------------
# 7. Optuna Hyperparameter Tuning
# ------------------------------------------------------------------
def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 200, 1000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_strength": trial.suggest_float("random_strength", 1, 20),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        "od_type": "Iter",
        "od_wait": 50,
        "eval_metric": "Accuracy",
        "random_seed": 42,
        "verbose": False
    }

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=50, verbose=False)
    preds = model.predict(X_val_scaled)
    acc = accuracy_score(y_val, preds)
    return acc

print("\n Running Optuna optimization (50 trials)...")
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))

study.optimize(objective, n_trials=50)

print(f" Best parameters: {study.best_trial.params}")
print(f" Best Validation Accuracy: {study.best_value:.4f}")

# ------------------------------------------------------------------
# 8. Train Final Model on Train+Val
# ------------------------------------------------------------------
best_params = study.best_trial.params
clf = CatBoostClassifier(**best_params)
X_trval_scaled = np.vstack([X_tr_scaled, X_val_scaled])
y_trval = np.concatenate([y_tr, y_val])
clf.fit(X_trval_scaled, y_trval)

# ------------------------------------------------------------------
# 9. Enhanced Test Evaluation
# ------------------------------------------------------------------
y_test_pred = clf.predict(X_test_scaled)
y_test_proba = clf.predict_proba(X_test_scaled)

# Calculate all metrics
test_acc = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

# Print enhanced results
print("\n" + "="*60)
print("FINAL TEST RESULTS")
print("="*60)
print(f" Test Accuracy:  {test_acc:.4f}")
print(f" Test Precision: {test_precision:.4f}")
print(f" Test Recall:    {test_recall:.4f}")
print(f"  Test F1-Score:  {test_f1:.4f}")
print("="*60)

# Detailed classification report
print("\n Detailed Classification Report:")
print(classification_report(y_test, y_test_pred, digits=4))

# ------------------------------------------------------------------
# 10. Feature Importance Analysis
# ------------------------------------------------------------------
print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Get feature importance from the trained model
feature_importance = clf.get_feature_importance()
feature_names = selected_features

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

# Display top 20 most important features
print("\n Top 20 Most Important Features:")
print("-" * 50)
for i, (idx, row) in enumerate(importance_df.head(20).iterrows(), 1):
    print(f"{i:2d}. {row['Feature']:40s} {row['Importance']:8.4f}")

# Create feature importance plot
plt.figure(figsize=(12, 10))

# Plot top 20 features
top_features = importance_df.head(20)
plt.barh(range(len(top_features)), top_features['Importance'], 
         color='steelblue', alpha=0.8)
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Feature Importance')
plt.title('Top 20 Feature Importances (CatBoost Model)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()  # Highest importance at top
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# Additional statistics about feature importance
print(f"\n Feature Importance Statistics:")
print(f" Total Features: {len(feature_importance)}")
print(f" Mean Importance: {np.mean(feature_importance):.4f}")
print(f" Std Importance: {np.std(feature_importance):.4f}")
print(f" Max Importance: {np.max(feature_importance):.4f}")
print(f" Min Importance: {np.min(feature_importance):.4f}")

# Show cumulative importance of top features
cumulative_importance = np.cumsum(importance_df['Importance'].values)
total_importance = np.sum(feature_importance)
cumulative_percentage = (cumulative_importance / total_importance) * 100

print(f"\n Cumulative Feature Importance:")
print(f" Top 5 features contribute:  {cumulative_percentage[4]:.2f}% of total importance")
print(f" Top 10 features contribute: {cumulative_percentage[9]:.2f}% of total importance")
print(f" Top 20 features contribute: {cumulative_percentage[19]:.2f}% of total importance")

print("="*60)

# ------------------------------------------------------------------
# 11. ROC Curve Analysis
# ------------------------------------------------------------------
print("\n" + "="*60)
print("ROC CURVE ANALYSIS")
print("="*60)

# Check if binary or multiclass
n_classes = len(np.unique(y_test))
class_names = np.unique(y_test)

plt.figure(figsize=(12, 8))

if n_classes == 2:
    # Binary classification ROC curve
    if y_test_proba.shape[1] == 2:
        y_scores = y_test_proba[:, 1]  # Probabilities for positive class
    else:
        y_scores = y_test_proba[:, 0]
    
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Binary Classification')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    print(f" AUC Score: {roc_auc:.4f}")

else:
    # Multiclass ROC curve (One-vs-Rest)
    from sklearn.preprocessing import label_binarize
    from itertools import cycle
    
    # Binarize the output
    y_test_bin = label_binarize(y_test, classes=class_names)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i, class_name in enumerate(class_names):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_test_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_test_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot ROC curves
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'Class {class_names[i]} (AUC = {roc_auc[i]:.4f})')
    
    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4,
             label=f'Micro-average (AUC = {roc_auc["micro"]:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Multiclass Classification (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Print AUC scores for each class
    for i, class_name in enumerate(class_names):
        print(f" AUC Score for Class {class_name}: {roc_auc[i]:.4f}")
    print(f" Micro-average AUC Score: {roc_auc['micro']:.4f}")
    
    # Calculate macro-average AUC
    macro_auc = np.mean([roc_auc[i] for i in range(n_classes)])
    print(f" Macro-average AUC Score: {macro_auc:.4f}")

plt.tight_layout()
plt.show()

# Calculate overall AUC score using sklearn's roc_auc_score
if n_classes == 2:
    overall_auc = roc_auc_score(y_test, y_test_proba[:, 1])
else:
    overall_auc = roc_auc_score(y_test, y_test_proba, multi_class='ovr', average='weighted')

print(f"\n Overall AUC Score: {overall_auc:.4f}")
print("="*60)
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------------
df = pd.read_csv(r"C:\Users\pc\Downloads\updated\updated\new_datasets\new_blinks30sec.csv")

meta_cols = ['subject', 'epoch', 'signal_type']
label_col = 'label'
feature_cols = [c for c in df.columns if c not in meta_cols + [label_col]]

# ------------------------------------------------------------------
# 2. Aggregate features to subject level
# ------------------------------------------------------------------
agg_df = df.groupby('subject')[feature_cols].agg(['mean', 'std', 'median'])
agg_df.columns = ['_'.join(col) for col in agg_df.columns]
agg_df.reset_index(inplace=True)

# Subject-level labels
labels = df.groupby('subject')[label_col].first().reset_index()
agg_df = agg_df.merge(labels, on='subject', how='left')

# ------------------------------------------------------------------
# 3. Subject-level split (BEFORE preprocessing)
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
test_df  = agg_df[agg_df['subject'].isin(test_subjects)].reset_index(drop=True)

X_train_df = train_df.drop(columns=['subject', label_col])
X_test_df  = test_df.drop(columns=['subject', label_col])
y_train = train_df[label_col].values
y_test  = test_df[label_col].values

# ------------------------------------------------------------------
# 4. Train-Only Preprocessing
# ------------------------------------------------------------------
imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train_df)
X_test_imp  = imputer.transform(X_test_df)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled  = scaler.transform(X_test_imp)

# ------------------------------------------------------------------
# 5. Hyperparameter Tuning Setup
# ------------------------------------------------------------------
# Conservative hyperparameter grids (focused on key parameters)
param_grids = {
    'HistGradientBoosting': {
        'model__max_depth': [4, 6, 8],  # Your current: 6
        'model__learning_rate': [0.03, 0.05, 0.1],  # Your current: 0.05
        'model__max_iter': [150, 200, 250]  # Your current: 200
    },
    
    'CatBoost': {
        'model__depth': [4, 6, 8],  # Your current: 6
        'model__learning_rate': [0.03, 0.05, 0.1],  # Your current: 0.05
        'model__iterations': [150, 200, 250]  # Your current: 200
    },
    
    'RandomForest': {
        'model__n_estimators': [150, 200, 250],  # Similar to your other models
        'model__max_depth': [4, 6, 8, 10],  # Similar range
        'model__min_samples_split': [2, 3, 5]  # For controlling overfitting
    },
    
    'ExtraTrees': {
        'model__n_estimators': [150, 200, 250],  # Your current: 200
        'model__max_depth': [4, 6, 8, 10],  # Your current: 6
        'model__min_samples_split': [2, 3, 5]  # Default: 2
    }
}

# ------------------------------------------------------------------
# 6. Base Models with Tuning
# ------------------------------------------------------------------
def create_base_models():
    return {
        'HistGradientBoosting': Pipeline([('model', HistGradientBoostingClassifier(random_state=42))]),
        'CatBoost': Pipeline([('model', CatBoostClassifier(random_state=42, verbose=False, allow_writing_files=False))]),
        'RandomForest': Pipeline([('model', RandomForestClassifier(random_state=42, n_jobs=-1))]),
        'ExtraTrees': Pipeline([('model', ExtraTreesClassifier(random_state=42, n_jobs=-1))])
    }

# ------------------------------------------------------------------
# 7. Targeted Hyperparameter Tuning
# ------------------------------------------------------------------
print("="*80)
print("TARGETED HYPERPARAMETER TUNING")
print("="*80)

min_class = np.bincount(y_train).min()
n_splits = min(3, min_class) if min_class > 1 else 2  # Reduced CV for tuning
cv_tuning = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

base_models = create_base_models()
tuned_models = {}
tuning_results = {}

for model_name, model_pipe in base_models.items():
    print(f"\n🔧 Tuning {model_name}...")
    
    # Create GridSearchCV with conservative settings
    grid_search = GridSearchCV(
        model_pipe,
        param_grids[model_name],
        cv=cv_tuning,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    # Fit grid search
    grid_search.fit(X_train_scaled, y_train)
    
    # Store results
    tuned_models[model_name] = grid_search.best_estimator_
    tuning_results[model_name] = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'original_score': None  # Will calculate baseline
    }
    
    print(f"✅ Best params: {grid_search.best_params_}")
    print(f"📊 Best CV score: {grid_search.best_score_:.4f}")

# ------------------------------------------------------------------
# 8. Create Tuned Stacking Ensemble
# ------------------------------------------------------------------
print(f"\n🎯 Creating Tuned Stacking Ensemble...")

# Extract tuned base estimators
tuned_base_estimators = [
    ('hgb', tuned_models['HistGradientBoosting'].named_steps['model']),
    ('catboost', tuned_models['CatBoost'].named_steps['model']),
    ('rf', tuned_models['RandomForest'].named_steps['model']),
    ('extra', tuned_models['ExtraTrees'].named_steps['model'])
]

# Create tuned stacking classifier
tuned_stacking_clf = StackingClassifier(
    estimators=tuned_base_estimators,
    final_estimator=LogisticRegression(random_state=42, max_iter=1000),
    cv=3,
    n_jobs=-1
)

tuned_models['Stacking Ensemble'] = Pipeline([('model', tuned_stacking_clf)])

# ------------------------------------------------------------------
# 9. Compare Original vs Tuned Models
# ------------------------------------------------------------------
print("\n" + "="*80)
print("ORIGINAL VS TUNED COMPARISON")
print("="*80)

# Original models (your baseline)
original_models = {
    'HistGradientBoosting': Pipeline([('model', HistGradientBoostingClassifier(
        max_iter=200, max_depth=6, learning_rate=0.05, random_state=42))]),
    'CatBoost': Pipeline([('model', CatBoostClassifier(
        iterations=200, depth=6, learning_rate=0.05, random_state=42, verbose=False, allow_writing_files=False))]),
    'RandomForest': Pipeline([('model', RandomForestClassifier(
        n_estimators=200, max_depth=6, random_state=42, n_jobs=-1))]),
    'ExtraTrees': Pipeline([('model', ExtraTreesClassifier(
        n_estimators=200, max_depth=6, random_state=42, n_jobs=-1))])
}

# Evaluate both original and tuned
cv_eval = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
comparison_results = {}

for model_name in original_models.keys():
    print(f"\n📊 Comparing {model_name}...")
    
    # Original model
    orig_scores = cross_val_score(original_models[model_name], X_train_scaled, y_train, cv=cv_eval, scoring='accuracy')
    orig_mean = orig_scores.mean()
    
    # Tuned model
    tuned_scores = cross_val_score(tuned_models[model_name], X_train_scaled, y_train, cv=cv_eval, scoring='accuracy')
    tuned_mean = tuned_scores.mean()
    
    # Store comparison
    comparison_results[model_name] = {
        'original': orig_mean,
        'tuned': tuned_mean,
        'improvement': tuned_mean - orig_mean
    }
    
    improvement = tuned_mean - orig_mean
    print(f"   Original: {orig_mean:.4f}")
    print(f"   Tuned:    {tuned_mean:.4f}")
    if improvement > 0:
        print(f"   ⬆️ Improvement: +{improvement:.4f} ({improvement/orig_mean*100:.2f}%)")
    else:
        print(f"   ⬇️ Change: {improvement:.4f} ({improvement/orig_mean*100:.2f}%)")

# ------------------------------------------------------------------
# 10. Final Test Evaluation with Comprehensive Metrics
# ------------------------------------------------------------------
print("\n" + "="*80)
print("FINAL TEST EVALUATION")
print("="*80)

# Use best performing models (tuned if better, original if not)
final_models = {}
for model_name in original_models.keys():
    if comparison_results[model_name]['improvement'] > 0:
        final_models[model_name] = tuned_models[model_name]
        print(f"🔧 Using TUNED {model_name}")
    else:
        final_models[model_name] = original_models[model_name]
        print(f"⚡ Using ORIGINAL {model_name}")

# Add stacking ensemble
final_models['Stacking Ensemble'] = tuned_models['Stacking Ensemble']

# Test evaluation with comprehensive metrics
test_results = {}
model_probabilities = {}

print("\n📈 Evaluating all models on test set...")
for model_name, model_pipe in final_models.items():
    print(f"\n🧪 Testing {model_name}...")
    model_pipe.fit(X_train_scaled, y_train)
    y_pred = model_pipe.predict(X_test_scaled)
    y_proba = model_pipe.predict_proba(X_test_scaled)
    
    # Calculate comprehensive metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, average='weighted')
    test_recall = recall_score(y_test, y_pred, average='weighted')
    test_f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Store results
    test_results[model_name] = {
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1
    }
    model_probabilities[model_name] = y_proba
    
    print(f"   Accuracy:  {test_accuracy:.4f}")
    print(f"   Precision: {test_precision:.4f}")
    print(f"   Recall:    {test_recall:.4f}")
    print(f"   F1-Score:  {test_f1:.4f}")

# ------------------------------------------------------------------
# 11. Feature Importance Analysis for Stacking Ensemble
# ------------------------------------------------------------------
print("\n" + "="*80)
print("🎯 FEATURE IMPORTANCE ANALYSIS - STACKING ENSEMBLE")
print("="*80)

# Get the trained stacking ensemble model
stacking_model = final_models['Stacking Ensemble']

# Extract base estimators from the stacking classifier
base_estimators = stacking_model.named_steps['model'].estimators_

# Get feature names (after preprocessing)
feature_names = X_train_df.columns.tolist()

print(f"📊 Analyzing feature importance across {len(base_estimators)} base models...")
print(f"📋 Total features: {len(feature_names)}")

# ------------------------------------------------------------------
# Base Model Feature Importances
# ------------------------------------------------------------------
base_importances = {}
valid_models = []

print("\n🔍 INDIVIDUAL BASE MODEL IMPORTANCES:")
print("-" * 60)

for name, estimator in zip(['HistGradientBoosting', 'CatBoost', 'RandomForest', 'ExtraTrees'], base_estimators):
    try:
        if hasattr(estimator, 'feature_importances_'):
            importances = estimator.feature_importances_
            base_importances[name] = importances
            valid_models.append(name)
            
            # Get top 10 features for this model
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔧 {name} - Top 10 Features:")
            for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
                print(f"   {i:2d}. {row['feature']:<40} {row['importance']:.6f}")
                
        else:
            print(f"⚠️  {name}: No feature_importances_ attribute available")
    except Exception as e:
        print(f"❌ Error getting importances for {name}: {str(e)}")

# ------------------------------------------------------------------
# Ensemble Feature Importance (Average of Base Models)
# ------------------------------------------------------------------
if len(valid_models) > 0:
    print(f"\n🎯 ENSEMBLE FEATURE IMPORTANCE (Average of {len(valid_models)} models):")
    print("=" * 70)
    
    # Calculate average importance across all valid base models
    ensemble_importance = np.mean([base_importances[model] for model in valid_models], axis=0)
    
    # Create comprehensive importance DataFrame
    importance_summary = pd.DataFrame({
        'feature': feature_names,
        'ensemble_importance': ensemble_importance
    })
    
    # Add individual model importances
    for model in valid_models:
        importance_summary[f'{model}_importance'] = base_importances[model]
    
    # Sort by ensemble importance
    importance_summary = importance_summary.sort_values('ensemble_importance', ascending=False)
    
    # Display top 20 features
    print("\n📊 TOP 20 MOST IMPORTANT FEATURES:")
    print("-" * 70)
    print(f"{'Rank':<4} {'Feature':<40} {'Ensemble':<12} {'Std Dev':<10}")
    print("-" * 70)
    
    for i, (_, row) in enumerate(importance_summary.head(20).iterrows(), 1):
        # Calculate standard deviation across base models
        model_importances = [row[f'{model}_importance'] for model in valid_models]
        std_dev = np.std(model_importances)
        
        print(f"{i:2d}.  {row['feature']:<40} {row['ensemble_importance']:<12.6f} {std_dev:<10.6f}")
    
    # ------------------------------------------------------------------
    # Feature Importance Visualization
    # ------------------------------------------------------------------
    print(f"\n📈 Creating Feature Importance Visualizations...")
    
    # Top 15 features for better visualization
    top_features = importance_summary.head(15)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('🎯 Stacking Ensemble Feature Importance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Ensemble Average Importance (Top 15)
    ax1 = axes[0, 0]
    bars1 = ax1.barh(range(len(top_features)), top_features['ensemble_importance'], 
                     color='darkred', alpha=0.7)
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels([f[:30] + '...' if len(f) > 30 else f for f in top_features['feature']], 
                        fontsize=10)
    ax1.set_xlabel('Ensemble Importance Score', fontsize=12)
    ax1.set_title('Top 15 Features - Ensemble Average', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', ha='left', va='center', fontsize=9)
    
    # 2. Individual Model Comparison (Top 10 features)
    ax2 = axes[0, 1]
    top_10_features = importance_summary.head(10)
    
    x_pos = np.arange(len(top_10_features))
    width = 0.2
    colors = ['red', 'blue', 'green', 'orange']
    
    for i, model in enumerate(valid_models):
        offset = (i - len(valid_models)/2 + 0.5) * width
        bars = ax2.bar(x_pos + offset, top_10_features[f'{model}_importance'], 
                      width, label=model, color=colors[i], alpha=0.7)
    
    ax2.set_xlabel('Features', fontsize=12)
    ax2.set_ylabel('Importance Score', fontsize=12)
    ax2.set_title('Top 10 Features - Individual Model Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f[:15] + '...' if len(f) > 15 else f for f in top_10_features['feature']], 
                       rotation=45, ha='right', fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Feature Importance Distribution
    ax3 = axes[1, 0]
    ax3.hist(ensemble_importance, bins=30, color='darkred', alpha=0.7, edgecolor='black')
    ax3.axvline(np.mean(ensemble_importance), color='blue', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(ensemble_importance):.6f}')
    ax3.axvline(np.median(ensemble_importance), color='green', linestyle='--', linewidth=2, 
                label=f'Median: {np.median(ensemble_importance):.6f}')
    ax3.set_xlabel('Importance Score', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Distribution of Feature Importances', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Cumulative Importance
    ax4 = axes[1, 1]
    sorted_importance = np.sort(ensemble_importance)[::-1]
    cumulative_importance = np.cumsum(sorted_importance) / np.sum(sorted_importance)
    
    ax4.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 
             color='darkred', linewidth=2, marker='o', markersize=1)
    ax4.axhline(0.8, color='blue', linestyle='--', alpha=0.7, label='80% Threshold')
    ax4.axhline(0.9, color='green', linestyle='--', alpha=0.7, label='90% Threshold')
    ax4.set_xlabel('Number of Features', fontsize=12)
    ax4.set_ylabel('Cumulative Importance', fontsize=12)
    ax4.set_title('Cumulative Feature Importance', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Find 80% and 90% thresholds
    features_80 = np.argmax(cumulative_importance >= 0.8) + 1
    features_90 = np.argmax(cumulative_importance >= 0.9) + 1
    
    ax4.axvline(features_80, color='blue', linestyle=':', alpha=0.7)
    ax4.axvline(features_90, color='green', linestyle=':', alpha=0.7)
    ax4.text(features_80, 0.85, f'{features_80} features\n(80%)', ha='center', va='bottom', 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    ax4.text(features_90, 0.95, f'{features_90} features\n(90%)', ha='center', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    # ------------------------------------------------------------------
    # Feature Importance Statistics
    # ------------------------------------------------------------------
    print(f"\n📊 FEATURE IMPORTANCE STATISTICS:")
    print("-" * 50)
    print(f"📈 Total features analyzed: {len(feature_names)}")
    print(f"🔝 Highest importance: {np.max(ensemble_importance):.6f}")
    print(f"📉 Lowest importance: {np.min(ensemble_importance):.6f}")
    print(f"📊 Mean importance: {np.mean(ensemble_importance):.6f}")
    print(f"📊 Median importance: {np.median(ensemble_importance):.6f}")
    print(f"📊 Standard deviation: {np.std(ensemble_importance):.6f}")
    print(f"\n🎯 Feature Selection Insights:")
    print(f"   • {features_80} features explain 80% of total importance")
    print(f"   • {features_90} features explain 90% of total importance")
    print(f"   • Top 20 features explain {cumulative_importance[19]*100:.1f}% of total importance")
    
    # Most consistent features across models
    if len(valid_models) > 1:
        print(f"\n🔄 FEATURE CONSISTENCY ACROSS BASE MODELS:")
        print("-" * 50)
        
        # Calculate coefficient of variation for top features
        top_20_consistency = []
        for _, row in importance_summary.head(20).iterrows():
            model_importances = [row[f'{model}_importance'] for model in valid_models]
            mean_imp = np.mean(model_importances)
            std_imp = np.std(model_importances)
            cv = std_imp / mean_imp if mean_imp > 0 else 0  # Coefficient of variation
            top_20_consistency.append((row['feature'], cv, mean_imp))
        
        # Sort by consistency (lower CV = more consistent)
        top_20_consistency.sort(key=lambda x: x[1])
        
        print("📊 Most Consistent Features (Low Coefficient of Variation):")
        for i, (feature, cv, mean_imp) in enumerate(top_20_consistency[:10], 1):
            print(f"   {i:2d}. {feature:<40} CV: {cv:.4f} | Avg: {mean_imp:.6f}")
    
    # Save feature importance results
    print(f"\n💾 Saving feature importance results...")
    importance_summary.to_csv('stacking_ensemble_feature_importance.csv', index=False)
    print(f"✅ Feature importance saved to 'stacking_ensemble_feature_importance.csv'")
    
else:
    print("❌ No valid base models found with feature importance capabilities!")

# ------------------------------------------------------------------
# 12. ROC Curve Analysis - Focus on Stacking Ensemble
# ------------------------------------------------------------------
print("\n" + "="*80)
print("🎯 ROC CURVE ANALYSIS - STACKING ENSEMBLE vs INDIVIDUAL MODELS")
print("="*80)

# Create comprehensive ROC plot
plt.figure(figsize=(15, 10))

# Check if binary or multiclass
n_classes = len(np.unique(y_test))
class_names = np.unique(y_test)

if n_classes == 2:
    # Binary classification - compare all models
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, (model_name, y_proba) in enumerate(model_probabilities.items()):
        if y_proba.shape[1] == 2:
            y_scores = y_proba[:, 1]  # Probabilities for positive class
        else:
            y_scores = y_proba[:, 0]
        
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Highlight Stacking Ensemble
        if model_name == 'Stacking Ensemble':
            plt.plot(fpr, tpr, color='red', lw=4, alpha=0.8,
                     label=f'🎯 {model_name} (AUC = {roc_auc:.4f})')
        else:
            plt.plot(fpr, tpr, color=colors[i], lw=2, alpha=0.7,
                     label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    # Add random classifier baseline
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', alpha=0.5,
             label='Random Classifier (AUC = 0.5000)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('🎯 ROC Curves: Stacking Ensemble vs Individual Models', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

else:
    # Multiclass ROC for Stacking Ensemble only
    from sklearn.preprocessing import label_binarize
    from itertools import cycle
    
    stacking_proba = model_probabilities['Stacking Ensemble']
    
    # Binarize the output
    y_test_bin = label_binarize(y_test, classes=class_names)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i, class_name in enumerate(class_names):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], stacking_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), stacking_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot ROC curves
    colors = cycle(['red', 'blue', 'green', 'orange', 'purple'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'Class {class_names[i]} (AUC = {roc_auc[i]:.4f})')
    
    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4,
             label=f'Micro-average (AUC = {roc_auc["micro"]:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('🎯 ROC Curves: Stacking Ensemble (Multiclass)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 13. Detailed AUC Analysis
# ------------------------------------------------------------------
print("\n📊 DETAILED AUC ANALYSIS:")
print("-" * 50)

auc_scores = {}
for model_name, y_proba in model_probabilities.items():
    if n_classes == 2:
        if y_proba.shape[1] == 2:
            auc_score = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auc_score = roc_auc_score(y_test, y_proba[:, 0])
    else:
        auc_score = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
    
    auc_scores[model_name] = auc_score
    
    if model_name == 'Stacking Ensemble':
        print(f"🎯 {model_name:.<25} AUC = {auc_score:.4f} ⭐")
    else:
        print(f"   {model_name:.<25} AUC = {auc_score:.4f}")

# ------------------------------------------------------------------
# 14. Model Selection Recommendation
# ------------------------------------------------------------------
print("\n" + "="*80)
print("🎯 MODEL SELECTION RECOMMENDATION")
print("="*80)

# Rank models by multiple criteria
model_rankings = {}
for model_name in final_models.keys():
    metrics = test_results[model_name]
    auc = auc_scores[model_name]
    
    # Combined score (you can adjust weights)
    combined_score = (metrics['accuracy'] * 0.3 + 
                     metrics['f1'] * 0.3 + 
                     auc * 0.4)
    
    model_rankings[model_name] = {
        'combined_score': combined_score,
        'accuracy': metrics['accuracy'],
        'f1': metrics['f1'],
        'auc': auc
    }

# Sort by combined score
sorted_rankings = sorted(model_rankings.items(), key=lambda x: x[1]['combined_score'], reverse=True)

print("\n🏆 FINAL MODEL RANKINGS:")
print("   (Combined Score = 30% Accuracy + 30% F1 + 40% AUC)")
print("-" * 70)

for i, (model_name, scores) in enumerate(sorted_rankings, 1):
    if model_name == 'Stacking Ensemble':
        print(f"   {i}. 🎯 {model_name:.<20} Score: {scores['combined_score']:.4f} ⭐")
    else:
        print(f"   {i}. {model_name:.<23} Score: {scores['combined_score']:.4f}")
    print(f"      └─ Acc: {scores['accuracy']:.4f} | F1: {scores['f1']:.4f} | AUC: {scores['auc']:.4f}")

best_model = sorted_rankings[0][0]
best_scores = sorted_rankings[0][1]

print(f"\n RECOMMENDED MODEL: {best_model}")
print("=" * 50)
print(f" Test Accuracy:  {best_scores['accuracy']:.4f}")
print(f" Test F1-Score:  {best_scores['f1']:.4f}")
print(f" AUC Score:      {best_scores['auc']:.4f}")
print(f" Combined Score: {best_scores['combined_score']:.4f}")

if best_model == 'Stacking Ensemble':
    print(f"\n The Stacking Ensemble combines the strengths of all base models!")
    print(f"   It leverages diversity to achieve superior performance.")
else:
    print(f"\n⚡ Surprisingly, {best_model} outperformed the ensemble!")
    print(f"   Sometimes simpler models work better for specific datasets.")

print("\n" + "="*80)
print(" SUMMARY")
print("="*80)
print(f" Hyperparameter tuning improved {sum(1 for r in comparison_results.values() if r['improvement'] > 0)}/4 base models")
print(f" Best overall model: {best_model}")
print(f" Best AUC score: {max(auc_scores.values()):.4f}")
print(f" Stacking Ensemble AUC: {auc_scores['Stacking Ensemble']:.4f}")

if auc_scores['Stacking Ensemble'] == max(auc_scores.values()):
    print(" Stacking Ensemble achieved the highest AUC score!")
else:
    ensemble_rank = sorted(auc_scores.items(), key=lambda x: x[1], reverse=True)
    ensemble_position = next(i for i, (name, _) in enumerate(ensemble_rank, 1) if name == 'Stacking Ensemble')
    print(f" Stacking Ensemble ranked #{ensemble_position} in AUC performance")

print("="*80)
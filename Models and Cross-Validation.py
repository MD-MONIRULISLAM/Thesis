from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline

models = {
'Logistic Regression': LogisticRegression(
C=0.5, max_iter=2000, class_weight='balanced',
random_state=RANDOM_STATE),
'Random Forest': RandomForestClassifier(
n_estimators=300, max_depth=12, min_samples_split=5,
min_samples_leaf=2, max_features='sqrt', class_weight='balanced',
random_state=RANDOM_STATE),
'XGBoost': XGBClassifier(
n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
scale_pos_weight=1.4, eval_metric='logloss',
random_state=RANDOM_STATE),
'Gradient Boosting': GradientBoostingClassifier(
n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8,
random_state=RANDOM_STATE),
}
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
cv_auc = {}
for name, model in models.items():
# SMOTE is refitted inside each training fold
pipe = Pipeline([('smote', SMOTE(random_state=RANDOM_STATE)),
('clf', model)])
cv_auc[name] = cross_val_score(pipe, X_train, y_train, cv=cv,
scoring='roc_auc')
model.fit(X_train_res, y_train_res)

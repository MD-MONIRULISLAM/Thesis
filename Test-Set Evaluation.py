from sklearn.metrics import (accuracy_score, precision_score, recall_score,
f1_score, roc_auc_score, brier_score_loss,
confusion_matrix)

def bootstrap_ci(y_true, values, metric, n=1000):
scores = []
for _ in range(n):
idx = np.random.choice(len(y_true), len(y_true), replace=True)
if len(np.unique(y_true[idx])) == 2:

scores.append(metric(y_true[idx], values[idx]))

return np.percentile(scores, [2.5, 97.5])

results = {}
for name, model in models.items():
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]
results[name] = {
'Accuracy': accuracy_score(y_test, pred),
'Precision': precision_score(y_test, pred),
'Recall': recall_score(y_test, pred),
'F1': f1_score(y_test, pred),
'AUC': roc_auc_score(y_test, prob),
'AUC_CI': bootstrap_ci(y_test.values, prob, roc_auc_score),
'Recall_CI': bootstrap_ci(y_test.values, pred, recall_score),
'Brier': brier_score_loss(y_test, prob),
'Confusion': confusion_matrix(y_test, pred).ravel(),
}

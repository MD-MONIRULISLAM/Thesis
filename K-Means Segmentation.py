from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu

cluster_features = ['Work_Pressure', 'Productivity', 'WorkLife_Balance',
'Daily_Meetings', 'Daily_Sleep', 'Overtime_Frequency']
Xc = StandardScaler().fit_transform(df[cluster_features])

15silhouette = {}
for k in range(2, 8):
km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
labels = km.fit_predict(Xc)
silhouette[k] = silhouette_score(Xc, labels)

21best_k = max(silhouette, key=silhouette.get)
df['Cluster'] = KMeans(n_clusters=best_k, n_init=10,
random_state=RANDOM_STATE).fit_predict(Xc)
df['Burnout_Risk'] = y

outcomes = ['Productivity', 'Task_Completion', 'Absenteeism', 'Work_Hours']
profiles = df.groupby('Cluster')[cluster_features + outcomes
+ ['Burnout_Risk']].mean()
for outcome in outcomes:
mannwhitneyu(df.loc[df['Cluster'] == 0, outcome],
df.loc[df['Cluster'] == 1, outcome])

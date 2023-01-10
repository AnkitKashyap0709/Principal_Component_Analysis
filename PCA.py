import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np


data = pd.read_csv("C:\\Users\\ANKIT's\\Downloads\\D11.csv")
target_variance = data['NFR_COVERAGE'].var()
covariance_matrix = data.cov()
covariance_target = covariance_matrix['NFR_COVERAGE']
# pd.options.display.max_rows = None

print("Variance of target variable: ", target_variance)
print("Covariance of target variable with other variables: \n", covariance_target)

X = data[['PROJECT_ID','TEAM_ID','SPRINT_QTR','COMMITTED_SP','DELIVERED_SP','COMMITTED_ST','AVG_VELOCITY_RELEASE','TEAM_HAPPINESS_INDEX','P_CSAT','CAPACITY_UTILIZATION','VELOCITY_PREDICTABILITY','SAY_DO_RATIO','TEAM_PRODUCTIVITY','SPRINT_STABILITY_INDEX','TEAM_STABILITY','PER_MINOR_OPEN_DEFECTS','SPRINT_STRETCH_FACTOR','IN_SPRINT_NON_FUNCTIONAL_REQUIREMENTS_TESTED','TOTAL_NON_FUNCTIONAL_REQUIREMENTS']]

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


# For showing the PCA plot in heatmap. For that the above imports of numpy and seaborn will be come into use.

# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)
# cov_mat = np.cov(X_pca.T)
# sns.heatmap(cov_mat, annot = True)
# plt.show()
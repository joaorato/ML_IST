import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

x = np.load("data3_x.npy")
y = np.load("data3_y.npy")

ridge = Ridge(max_iter=10000)
lasso = Lasso(max_iter=10000)

alphas = np.zeros(1001)
alphas[0:1000] = [0.001 + i*0.01 for i in range(1000)]
alphas[1000] = 10

betas_ridge = []
betas_lasso = []

for alpha in alphas:
    ridge.set_params(alpha = alpha)
    ridge.fit(x,y)

    #this appends 3 betas each cycle, one for each feature. the final shape should be 1001 x 3 (beta 0 = 0)
    betas_ridge.append(ridge.coef_)

    lasso.set_params(alpha = alpha)
    lasso.fit(x,y)

    #this appends 3 betas each cycle, one for each feature. the final shape should be 1001 x 3 (beta 0 = 0)
    betas_lasso.append(lasso.coef_)

betas_ridge = np.squeeze(np.array(betas_ridge))
betas_lasso = np.squeeze(np.array(betas_lasso))

plt.plot(alphas, betas_ridge[:,0], label='beta_ridge 0')
plt.plot(alphas, betas_ridge[:,1], label='beta_ridge 1')
plt.plot(alphas, betas_ridge[:,2], label='beta_ridge 2')
plt.plot(alphas, betas_lasso[:,0], label='beta_lasso 0')
plt.plot(alphas, betas_lasso[:,1], label='beta_lasso 1')
plt.plot(alphas, betas_lasso[:,2], label='beta_lasso 2')
plt.legend()
plt.xlabel('alpha')
plt.ylabel('beta')
plt.xscale('log')
plt.show()
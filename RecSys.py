import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from collections import defaultdict

data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

for d in data['userId'].unique():
    if d not in test_data['userId'].unique():
         data = data[data['userId'] != d] 

for d in data['movieId'].unique():
    if d not in test_data['movieId'].unique():
        data = data[data['movieId'] != d] 

# Construct Users vs Movies rating matrix
data['userId'] = data['userId'].astype("category")
data['movieId'] = data['movieId'].astype("category")
rating_matrix = coo_matrix((data['rating'].astype(float),
                            (data['movieId'].cat.codes.copy(),
                             data['userId'].cat.codes.copy())))

R = rating_matrix.toarray()

# Index matrix for training data
I = R.copy()
I[I > 0] = 1
I[I == 0] = 0

lmbda = 0.1 # Regularisation weight
k = 20 # Dimensionality of latent feature space
m, n = R.shape # Number of users and items
n_epochs = 5 # Number of epochs

P = np.random.rand(k,m) / k # Latent user feature matrix
Q = np.random.rand(k,n) / k # Latent movie feature matrix

# Construct Users vs Movies rating matrix
test_data['userId'] = test_data['userId'].astype("category")
test_data['movieId'] = test_data['movieId'].astype("category")
test_rating_matrix = coo_matrix((test_data['rating'].astype(float),
                            (test_data['movieId'].cat.codes.copy(),
                             test_data['userId'].cat.codes.copy())))

T = test_rating_matrix.toarray()

# Index matrix for test data
I2 = T.copy()
I2[I2 > 0] = 1
I2[I2 == 0] = 0

Bu = np.zeros(m)
Bi = np.zeros(n)
mu = np.mean(R[np.where(R != 0)])

# creating dictionary {user: array[item, rating]}.
ratings_by_u=defaultdict(list)
# creating dictionary {item: array[user, rating]}.
ratings_by_i=defaultdict(list)

for u in range(m):
    for i in range(n):
        if R[u][i] > 0:
            ratings_by_u[u].append((i,R[u][i]))
            ratings_by_i[i].append((u,R[u][i]))

def full_matrix():
    return mu + Bu[:,np.newaxis] + Bi[np.newaxis:,] + np.dot(P.T,Q)

# Calculate the RMSE
def rmse(I,R,Q,P):
    u, i = R.nonzero()
    predicted = full_matrix()
    error = 0
    for x, y in zip(u, i):
        error += pow(R[x, y] - predicted[x, y], 2)
    rmse = np.sqrt(error/len(u))
    return rmse

train_errors = []
test_errors = []

# Repeat until convergence
for epoch in range(n_epochs):
    # update Bu
    for i, Ii in enumerate(I):
        accum = 0
        for j, r in ratings_by_u[i]:
            accum += (r - P[:,i].dot(Q[:,j]) - Bi[j] - mu)
        Bu[i] = accum / (len(ratings_by_u[i]) + lmbda*len(ratings_by_u[i]))

    # update P
    for i, Ii in enumerate(I):
        nui = np.count_nonzero(Ii) # Number of items user i has rated
        if (nui == 0): nui = 1 # Be aware of zero counts!
    
        matrix = np.zeros((k, k)) + lmbda*nui*np.eye(k)
        vector = np.zeros(k)
        for j, r in ratings_by_u[i]:
            matrix += np.outer(Q[:,j], Q[:,j])
            vector += (r - Bu[i] - Bi[j] - mu)*Q[:,j]
        P[:,i] = np.linalg.solve(matrix, vector)

    # update Bi
    for j, Ij in enumerate(I.T):
        accum = 0
        for i, r in ratings_by_i[j]:
            accum += (r - P[:,i].dot(Q[:,j]) - Bu[i] - mu)
        Bi[j] = accum / (len(ratings_by_i[j]) + lmbda*len(ratings_by_i[j]))

    # update Q
    for j, Ij in enumerate(I.T):
        nmj = np.count_nonzero(Ij) # Number of users that rated item j
        if (nmj == 0): nmj = 1 # Be aware of zero counts!
            
        matrix = np.zeros((k, k)) + lmbda*nmj*np.eye(k)
        vector = np.zeros(k)
        for i, r in ratings_by_i[j]:
            matrix += np.outer(P[:,i], P[:,i])
            vector += (r - Bu[i] - Bi[j] - mu)*P[:,i]
        Q[:,j] = np.linalg.solve(matrix, vector)
            
    train_rmse = rmse(I,R,Q,P)
    test_rmse = rmse(I2,T,Q,P)
    train_errors.append(train_rmse)
    test_errors.append(test_rmse)

    print("[Epoch %d/%d] train error: %f, test error: %f" \
    %(epoch+1, n_epochs, train_rmse, test_rmse))

print("Algorithm converged")

# Calculate prediction matrix R_hat (low-rank approximation for R)
R_hat = pd.DataFrame(Bu[:,np.newaxis] + Bi[np.newaxis,:] + mu + np.dot(P.T,Q))
R = pd.DataFrame(R)
# Compare true ratings of user 17 with predictions
ratings = pd.DataFrame(data=R.loc[16,R.loc[16,:] > 0]).head(n=5)
ratings['Prediction'] = R_hat.loc[16,R.loc[16,:] > 0]
ratings.columns = ['Actual Rating', 'Predicted Rating']
print(ratings)

predictions = R_hat.loc[16,R.loc[16,:] == 0] # Predictions for movies that the user 17 hasn't rated yet
top5 = predictions.sort_values(ascending=False).head(n=5)
recommendations = pd.DataFrame(data=top5)
recommendations.columns = ['Predicted Rating']
print(recommendations)
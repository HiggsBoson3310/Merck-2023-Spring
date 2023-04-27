import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import DeepQTF as dq
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

NUMBER_SAMPLES = 5000
NEP = 30
BUDGET = 100
BATCH_SIZE = 35


seeds = [658, 682, 533, 27, 889, 224, 205, 338, 559, 163]
X_train, y_train = make_regression(n_samples=NUMBER_SAMPLES,n_features=10,n_informative=5,n_targets=1,noise=0.0)
fstar = np.percentile(y_train,90)

# Greedy sampling function on feature space (just distance to nearest neighboor)
def GS_x(x,x_list):
    distances = np.zeros_like(x_list)

    for ii,xx in enumerate(x_list):
        distances[ii] = np.sqrt(np.sum((x-xx)**2))
    
    return np.min(distances)

def GS_y(x,y_init,model):
    Ypreds = model.predict([x])
    distances =  np.zeros_like(y_init)
    for ii, xx in enumerate(y_init):
        distances[ii] = np.abs(Ypreds-xx)
    return np.min(distances)

def state(model,cand,pool_xs,pool_vals):
    gsx = GS_x(cand,pool_xs)
    gsy = GS_y(cand,pool_vals,model)
    return tf.cast(np.array([[*cand,gsx,gsy]]).reshape(1,-1),dtype=tf.float32)


model1 = dq.Qlearner(12,256,2)

for seed in tqdm(seeds,desc='Episode'):
    randomizer = np.random
    indices = list(range(NUMBER_SAMPLES))
    randomizer.shuffle(indices)
    x_init, y_init = tf.cast(X_train[indices[:100]],dtype=tf.float32), tf.cast(y_train[indices[:100]],dtype=tf.float32)
    x_held, y_held = tf.cast(X_train[indices[100:200]],dtype=tf.float32), tf.cast(y_train[indices[100:200]],dtype=tf.float32)
    x_cand, y_cand = tf.cast(X_train[indices[200:]],dtype=tf.float32), tf.cast(y_train[indices[200:]],dtype=tf.float32)
    initial_model = RandomForestRegressor(100, n_jobs=-2,random_state=158)
    initial_model.fit(x_init,y_init)
    usd = 0 
    error_prev = mean_squared_error(y_held,initial_model.predict(x_held))
    for i in tqdm(range(200),desc='State Looping'):
        s_i = state(initial_model,x_cand[i],x_init,y_init)
        a_i = tf.argmax(tf.transpose(model1.compute_qvalue(s_i)))
        if(a_i==1):
            x_init = np.append(x_init,[x_cand[i]],0)
            y_init = np.append(y_init,[y_cand[i]],0)
            usd += 1
            initial_model.fit(x_init,y_init)
        error_current = mean_squared_error(y_held,initial_model.predict(x_held))
        reward = error_prev-error_current
        error_prev = error_current
        if(usd==BUDGET):
            model1.add_to_memory(s_i,a_i,reward,None)
            model1.finish_episode()
            break
        else:
            s_ip1 = state(initial_model,x_cand[i+1],x_init,y_init)
            model1.add_to_memory(s_i,a_i,reward,s_ip1)

        if(i>BATCH_SIZE):
            cur_mem = model1.get_current_memory()
            indices = randomizer.randint(len(cur_mem)-1,size=BATCH_SIZE)
            batch_sf = tf.reshape(tf.cast(list( cur_mem[i][3] for i in indices),dtype=tf.float32),[BATCH_SIZE,12])
            batch_r = tf.reshape(tf.cast(list( cur_mem[i][2] for i in indices),dtype=tf.float32),[BATCH_SIZE,1])
            batch_si = tf.reshape(tf.cast(list( cur_mem[i][0] for i in indices),dtype=tf.float32),[BATCH_SIZE,12])
            batch_ai = tf.reshape(tf.cast(list( cur_mem[i][1] for i in indices),dtype=tf.float32),[BATCH_SIZE,1])
            ys = model1.y_value(batch_sf,batch_r )
            model1.Training_network(ys,batch_ai,batch_si)
        if(i==199):
            model1.finish_episode()
            
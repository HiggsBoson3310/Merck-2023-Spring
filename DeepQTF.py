import numpy as np
import tensorflow as tf
from tensorflow import keras



class Qlearner():
    def __init__(self, dim_i, dim_l, dim_o, alpha=1e-4,gamma=0.2):
        self.alpha=alpha
        self.gamma = gamma
        self.dim_i = dim_i
        self.dim_l = dim_l
        self.dim_o = dim_o
        self.desicion_memory = [[]]
        self.loss_curve = [[]]
        self.current_loss = 90
        self.current_episode = 0
        self.create_network()
    
    def create_network(self):
        self.M0 = self.weight_variable(self.dim_i,self.dim_l)
        self.b0 = self.bias_variable(1,self.dim_l)#self.weight_variable(1,self.dim_l)
        self.M1 = self.weight_variable(self.dim_l,self.dim_o)
        self.b1 = self.bias_variable(1,self.dim_o)#self.weight_variable(1,self.dim_o)
        self.trainable_vars = [self.M0,self.b0,self.M1,self.b1]
        schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        self.alpha,
        decay_steps=100000,
        decay_rate=1.00,
        staircase=False)

        self.optimizer = tf.keras.optimizers.Adam(schedule)

    def weight_variable(self,dim1, dim2):
        var = tf.random.truncated_normal((dim1,dim2),stddev=0.1)
        return tf.Variable(var,trainable=True)
    
    def bias_variable(self,dim1, dim2):
        var = tf.constant(value = 0.01, shape=(dim1,dim2), dtype=tf.float32)
        return tf.Variable(var,trainable=True)
    
    @tf.function
    def qvalues(self,M0,b0,M1,b1,state_input):
        L0 = tf.nn.relu(tf.matmul(state_input,M0)+b0)
        return tf.matmul(L0,M1)+b1
    
    @tf.function
    def qaction(self,q_values,action_input):
        return tf.reduce_sum(tf.multiply(q_values,action_input),axis=-1)
    
    @tf.function
    def y_value(self,state_batch,reward_batch):
        return reward_batch + self.gamma*tf.reduce_max(self.compute_qvalue(state_batch),axis=-1)

    @tf.function
    def loss(self,M0,b0,M1,b1,y_input,action_inputs,states_input):
        qvals = self.qvalues(M0,b0,M1,b1,states_input)
        qaction = self.qaction(qvals,action_inputs)
        return tf.reduce_mean(tf.square(y_input-qaction))

    def Training_network(self,y_batch_input,action_batch,state_batch):
        with tf.GradientTape() as tp:
            loss_int = self.loss(self.M0,self.b0,self.M1,self.b1,y_batch_input,action_batch,state_batch)
        gradients = tp.gradient(loss_int,self.trainable_vars)
        self.optimizer.apply_gradients(zip(gradients,self.trainable_vars))
        self.current_loss = self.loss(self.M0,self.b0,self.M1,self.b1,y_batch_input,action_batch,state_batch)
        self.update_current_loss()

    def finish_episode(self):
        self.current_episode += 1
        self.desicion_memory.append([])
        self.loss_curve.append([])

    def compute_loss(self,y_in, s_in, a_in):
        return self.loss(self.M0,self.b0,self.M1,self.b1,y_in, s_in, a_in)

    def compute_qvalue(self,state):
        return tf.matmul(tf.nn.relu(tf.matmul(state,self.M0)+self.b0),self.M1)+self.b1

    def add_to_memory(self,si,action,reward,sip1):
        self.desicion_memory[self.current_episode].append(([si,reward,action,sip1]))
    
    def get_current_memory(self):
        return self.desicion_memory[self.current_episode]
    
    def update_current_loss(self):
        self.loss_curve[self.current_episode].append(self.current_loss)
    
        


    



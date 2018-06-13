import pickle
import matplotlib.pyplot as plt
import time
import numpy as np

(x_train, y_train), (x_test, y_test) = pickle.load( open( "dataset.pkl", "rb" ) ) # Loading training and testing data from dummy dataset

learning_rate = 0.001

theta_0 = 0 # Bias
theta_1 = 0 # Coefficient

epochs = 20000
print("PROCESSING...")
for i in range(epochs):
    global count

    # NOTE : The loss function used here is Mean Squared Error

    sigma_0 = (np.subtract(((theta_1 * x_train) + theta_0).T, y_train)).sum() # Gradient of loss function w.r.t theta_0
    sigma_1 = (np.multiply(np.subtract(((theta_1 * x_train) + theta_0).T, y_train), x_train.T)).sum() # Gradient of loss function w.r.t theta_1
    
    # Normalizing gradient over number of training examples
    gradient_0 = sigma_0 / len(x_train)
    gradient_1 = sigma_1 / len(x_train)
    
    # Calculating new values of parameters using gradient descent
    temp_0 = theta_0 - (learning_rate * gradient_0)
    temp_1 = theta_1 - (learning_rate * gradient_1)
    
    #Updating parameters simultaneously
    theta_0 = temp_0
    theta_1 = temp_1
    
print("DONE!")
print("THETA ZERO : " + str(theta_0) + " (Bias)")
print("THETA ONE  : " + str(theta_1) + " (Coefficient)")

# Calculating cost of learned line
cost = np.subtract(((theta_1 * x_train) + theta_0).T, y_train)
cost = ((np.square(cost)).sum()) / (len(x_train) * 2)
print("COST : " + str(cost))


# Plotting the data and learned line
plt.scatter(x_train, y_train, marker=".")
x_axis = [min(x_train), max(x_train)]
y_axis = [theta_0 + theta_1 * x_axis[0], theta_0 + theta_1 * x_axis[1]]
plt.plot(x_axis, y_axis, color="red")
plt.show()
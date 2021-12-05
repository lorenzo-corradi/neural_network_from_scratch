import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
from mlxtend.plotting import plot_decision_regions

warnings.filterwarnings("ignore")

# specifications of neural network:
# 1 input layer, dimension: [n_observations, n_features]
# 1 set of weights [w1] between input layer and hidden layer, dimension: [n_features + 1, n_hidden_nodes]
# 1 hidden layer [nodes of hidden layer can be specified: default = 20]
# 1 set of weights [w2] between hidden layer and output layer, dimension: [n_hidden_nodes + 1, n_classes]
# 1 output layer, dimension: [n_observations, n_classes]
# learning based on batch gradient descent

class MultiLayerPerceptron(object):
    
    def __init__(self, n_features, n_classes, n_hidden = 20, epochs = 100, decrease_const = 0.0,
                 eta = 0.001, shuffle = True, random_state = None):
        
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.w1, self.w2 = self._initialize_weights()
        
        self.eta = eta
        self.epochs = epochs
        self.decrease_const = decrease_const
        
        np.random.seed(random_state)


    def _encode_labels(self, y, classes):
        onehot = np.zeros((y.shape[0], classes))
        for index, value in enumerate(y):
            onehot[index, value] = 1.0
        return onehot
    

    def _initialize_weights(self):
        w1 = np.random.uniform(-1, 1, size = self.n_hidden * (self.n_features + 1))
        w1 = w1.reshape(self.n_hidden, self.n_features + 1)
        w2 = np.random.uniform(-1, 1, size = (self.n_hidden + 1) * self.n_classes)
        w2 = w2.reshape((self.n_hidden + 1), self.n_classes)
        return w1, w2
    
    
    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    
            
    def _sigmoid_gradient(self, z):
        sigmoid = self._sigmoid(z)
        return sigmoid * (1 - sigmoid)
    
    
    def _add_bias_unit(self, data, how = 'column'):
        if how == 'column':
            data_new = np.ones((data.shape[0], data.shape[1] + 1))
            data_new[:, 1:] = data
        elif how == 'row':
            data_new = np.ones((data.shape[0] + 1, data.shape[1]))
            data_new[1:, :] = data
        else:
            raise AttributeError('how parameter must be set as `column` or `row`')
        return data_new
    
                    
    def _feedforward(self, X, w1, w2):
        a1 = self._add_bias_unit(X, how = 'column')
        z2 = np.dot(w1, a1.T)
        a2 = self._add_bias_unit(self._sigmoid(z2), how = 'row')
        z3 = np.dot(a2.T, w2)
        a3 = self._sigmoid(z3)
        return a1, z2, a2, z3, a3
    
            
    def _gradient(self, a1, a2, a3, z2, y_enc, w1, w2):
        sigma3 = a3 - y_enc
        z2 = self._add_bias_unit(z2, how = 'row')
        sigma2 = (np.dot(sigma3, w2.T)) * self._sigmoid_gradient(z2.T)
        sigma2 = sigma2[:, 1:]
        
        grad1 = np.dot(a1.T, sigma2)
        grad2 = np.dot(a2, sigma3)
        
        return grad1, grad2
    

    def _cross_entropy(self, y_enc, output):
        cross_entropy = - np.sum(y_enc * np.log(output)) / output.shape[0]
        return cross_entropy
    
    
    def predict(self, X):
        a1, z2, a2, z3, a3 = self._feedforward(X, self.w1, self.w2)
        y_pred = np.argmax(z3, axis = 1)
        
        return y_pred
        
            
    def fit(self, X, y):
        self.loss_ = []
        
        y = y.reshape(-1, 1)
        y_enc = self._encode_labels(y, self.n_classes)
        
        delta_w1_prev = np.zeros(self.w1.shape)
        delta_w2_prev = np.zeros(self.w2.shape)
        
        for i in range(self.epochs - 1):
            # adaptive learning rate
            self.eta /= (1 + self.decrease_const * i)
                
            loss = []
            
            a1, z2, a2, z3, a3 = self._feedforward(X, self.w1, self.w2)
            
            loss = self._cross_entropy(y_enc = y_enc, output = a3)
            self.loss_.append(loss)
            
            if (len(self.loss_) % 10 == 0):
                print("at epoch {} the cross-entropy value is: {}".format(len(self.loss_), loss))
            
            # compute gradient via backpropagation
            grad1, grad2 = self._gradient(a1, a2, a3, z2, 
                                         y_enc, self.w1, self.w2)
            
            # update weights
            delta_w1, delta_w2 = self.eta * grad1, self.eta * grad2
            self.w1 -= (delta_w1.T + delta_w1_prev)
            
            delta_w1_prev = delta_w1.T
            delta_w2_prev = delta_w2
            
        return self
    
    
# train same neural network for visualization
def visualization_helper(X, y):
    
    n_hidden = 60
    epochs = 150
    decrease_const = 0.001
    eta = 0.02
    
    pca = PCA(n_components = 2)
    X = pca.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    neural_network_visualization = MultiLayerPerceptron(X.shape[1], len(y_unique), n_hidden = n_hidden, epochs = epochs, 
                                                        decrease_const = decrease_const, eta = eta)

    neural_network_visualization.fit(X_train, y_train)
    
    return X_test, y_test, neural_network_visualization


def check_retrain():
    viz = input('Perform PCA and retrain neural network to plot decision approximated decision boundaries? y/n: ')

    while (viz != 'y' and viz != 'n'):
        viz = input('Retry. Only y/n accepted: ')
        
    return viz

    
# helper visualization functions
def make_meshgrid(x, y, h = 0.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    x_mesh, y_mesh = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return x_mesh, y_mesh


def plot_contours(ax, clf, x_mesh, y_mesh, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = neural_network.predict(np.c_[x_mesh.ravel(), y_mesh.ravel()])
    Z = Z.reshape(x_mesh.shape)
    output = ax.contourf(x_mesh, y_mesh, Z, **params)
    return output

if __name__ == "__main__":

    plot_dim = 2
    n_hidden = 60
    epochs = 150
    decrease_const = 0.001
    eta = 0.02
    
    X, y = datasets.make_classification(n_samples = 1000, n_features = 5, n_classes = 4, n_informative = 3,
                                        n_redundant = 0, n_clusters_per_class = 1, random_state = 1)
    y_set = set(y)
    y_unique = list(y_set)
    
    if (X.shape[1] == plot_dim):
        plt.figure(figsize=(15, 8))
        plt.scatter(X[y == 0,0], X[y == 0,1]) # plot needs two arguments: since X has two dims, for plt.plot(X argument) we take first column of X, and as second plt.plot argument (y) we take second column of X. Of course we're only taking the values having label "y == 0" here!
        plt.scatter(X[y == 1,0], X[y == 1,1])
        plt.axis('equal')
        plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    neural_network = MultiLayerPerceptron(X.shape[1], len(y_unique), n_hidden = n_hidden, epochs = epochs, 
                                          decrease_const = decrease_const, eta = eta)

    neural_network.fit(X_train, y_train)
    
    y_pred = neural_network.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
    
            
    if (check_retrain() == 'y' and X.shape[1] > plot_dim):
        print('Perform PCA and train neural network.')
        X_test, y_test, neural_network = visualization_helper(X, y) 
        # re-trains neural network for plotting decision boundaries
    elif (check_retrain() == 'n'):
        print('Execution terminated.')
        
        
    x_mesh, y_mesh = make_meshgrid(X_test[:, 0], X_test[:, 1])
    
    fig, ax = plt.subplots(1, 1)
    
    plot_contours(ax, neural_network, x_mesh, y_mesh, cmap = plt.cm.coolwarm, alpha = 0.8)
    ax.scatter(X_test[:, 0], X_test[:, 1], c = y_test, cmap = plt.cm.coolwarm, s = 20, edgecolors = "k")
    
    ax.set_xlim(x_mesh.min(), x_mesh.max())
    ax.set_ylim(y_mesh.min(), y_mesh.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title('Decision boundaries')
    
    plt.show()
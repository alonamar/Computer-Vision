import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
from sklearn.metrics import confusion_matrix

CATEGORIES = ["walking", "jogging", "running", "boxing", "handwaving","handclapping"]


def plot_curve(plot_method, train_scores, test_scores, param_range, title, xlabel, ylim=None):
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.figure(figsize=(14, 6))
    plt.title(title)
    lw = 2

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(xlabel)
    plt.ylabel("Score")
    lw = 2

    plot_method(param_range, train_scores_mean, label="Training score",
                color="darkorange", lw=lw)
    plot_method(param_range, test_scores_mean, label="Cross-validation score",
                color="navy", lw=lw)
    plt.grid(True)
    plt.legend(loc="best")

    plt.savefig(title + ".png", dpi=300)
    plt.show()



def tuneAndPlot(X, y, clf, tuned_parameters, param_range, name):
    gridClf = GridSearchCV(clf, tuned_parameters, n_jobs=20)
    gridClf.fit(X, y)
    print("Best parameters: ")
    print(gridClf.best_params_)
    clf.set_params(**gridClf.best_params_)
    myRange = param_range['range']
    param = param_range['param']
    train_scores, test_scores = validation_curve(clf, X, y, param, myRange, n_jobs=20)
    title = "Validation Curve_" + name
    plot_curve(param_range['plot'], train_scores, test_scores, myRange, title, param)


def split_data(datafile, test_size=0.75):
    data = np.load(datafile)
    features = data.shape[1] - 1
    X_train, X_test, y_train, y_test = np.array([]).reshape(0, features), np.array([]).reshape(0, features), \
                                       np.array([]), np.array([])
    for i, val in enumerate(CATEGORIES):
        X_train = np.concatenate(
            [X_train, data[data[:, -1] == i][:int(test_size*data[data[:, -1] == 0].shape[0])][:, :features]])
        X_test = np.concatenate(
            [X_test, data[data[:, -1] == i][int(test_size * data[data[:, -1] == 0].shape[0]):][:, :features]])
        y_train = np.concatenate(
            [y_train, data[data[:, -1] == i][:int(test_size*data[data[:, -1] == 0].shape[0])][:, features]])
        y_test = np.concatenate(
            [y_test, data[data[:, -1] == i][int(test_size * data[data[:, -1] == 0].shape[0]):][:, features]])

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def run_clf(X_train, X_test, y_train, y_test, name):
    clf = {'svm': svm.SVC(gamma='auto'),
           'knn': neighbors.KNeighborsClassifier(n_neighbors=10),
           'nn': MLPClassifier(random_state=1)}
    tuned_parameters = {'svm': [{'kernel': ['rbf'], 'gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
                                 'C': [100, 1000, 1e4, 1e5, 1e6, 1e7, 1e8]}],
                        'knn': [{'n_neighbors': range(1, 15, 1),
                                 'weights': ['uniform', 'distance'], 'p': [1, 2, 5]}],
                        'nn': {'hidden_layer_sizes': [(50, 2), (50,), (40,), (60,), (50, 50)],
                               'solver': ['lbfgs', 'sgd', 'adam'],
                               'alpha': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]}}
    param_range = {'svm': {'range': [100, 1000, 1e4, 1e5, 1e6, 1e7, 1e8], 'param': 'C', "plot": plt.semilogx},
                   'knn': {'range': range(1, 20, 1), 'param': 'n_neighbors', "plot": plt.plot},
                   'nn': {'range': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2], 'param': 'alpha', "plot": plt.semilogx}}

    for key in clf.keys():
        clf[key].fit(X_train, y_train)
        y_res = clf[key].predict(X_test)
        print("The accuracy of the initial " + key + " classifier: " + str(accuracy_score(y_test, y_res)))
        tuneAndPlot(X_train, y_train, clf[key], tuned_parameters[key], param_range[key], name + "_" + key)
        clf[key].fit(X_train, y_train)
        y_res = clf[key].predict(X_test)
        print("The accuracy of the tuned " + key + " classifier: " + str(accuracy_score(y_test, y_res)))
        cm = confusion_matrix(y_test, y_res)
        print("Confusion matrix:")
        print(cm)
        np.savetxt(name + "_" + key + ".csv", cm.astype(int), delimiter=",")
        # print(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])

    #return clf['nn']

def run(full_frame=False):
    if full_frame:
        datafile = "train_data_10th_full_frame.npy"
        X_train, X_test, y_train, y_test, scaler = split_data(datafile)
        run_clf(X_train, X_test, y_train, y_test, "full_frame")
    else:
        datafile = "train_data_10th_20tau.npy"
        X_train, X_test, y_train, y_test, scaler = split_data(datafile)
        run_clf(X_train, X_test, y_train, y_test, "20tau")


if __name__ == "__main__":
    run()
    run(True)

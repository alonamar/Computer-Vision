import preprocessing
import clf_finder
import analyze
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def run(preproecess = False, full_frame=False):
    if preproecess:
        preprocessing.run(full_frame)
    clf_finder.run(full_frame)
    analyze.run(full_frame)


if __name__ == "__main__":
    run()
    run(full_frame=True) #optional - running the full frame version

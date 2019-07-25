"""Problem Set 6: PCA, Boosting, Haar Features, Viola-Jones."""
import numpy as np
import cv2
import os

from helper_classes import WeakClassifier, VJ_Classifier


# assignment code
def load_images(folder, size=(32, 32)):
    """Load images to workspace.

    Args:
        folder (String): path to folder with images.
        size   (tuple): new image sizes

    Returns:
        tuple: two-element tuple containing:
            X (numpy.array): data matrix of flatten images
                             (row:observations, col:features) (float).
            y (numpy.array): 1D array of labels (int).
    """
    images_files = [f for f in os.listdir(folder) if f.endswith(".png")]
    X = []
    y = []
    for file in images_files:
        img = cv2.imread(folder + "/" + file, 0)
        img = cv2.resize(img, tuple(size), interpolation=cv2.INTER_CUBIC)
        X.append(np.ndarray.flatten(img))
        y.append(file.split(".")[0][-2:])
    return np.array(X, dtype=float), np.array(y, dtype=int)


def split_dataset(X, y, p):
    """Split dataset into training and test sets.

    Let M be the number of images in X, select N random images that will
    compose the training data (see np.random.permutation). The images that
    were not selected (M - N) will be part of the test data. Record the labels
    accordingly.

    Args:
        X (numpy.array): 2D dataset.
        y (numpy.array): 1D array of labels (int).
        p (float): Decimal value that determines the percentage of the data
                   that will be the training data.

    Returns:
        tuple: Four-element tuple containing:
            Xtrain (numpy.array): Training data 2D array.
            ytrain (numpy.array): Training data labels.
            Xtest (numpy.array): Test data test 2D array.
            ytest (numpy.array): Test data labels.
    """
    ind = np.random.permutation(X.shape[0])
    N = int(round(X.shape[0] * p))
    return X[ind[:N]], y[ind[:N]], X[ind[N:]], y[ind[N:]]


def get_mean_face(x):
    """Return the mean face.

    Calculate the mean for each column.

    Args:
        x (numpy.array): array of flattened images.

    Returns:
        numpy.array: Mean face.
    """
    return np.mean(x, axis=0)


def pca(X, k):
    """PCA Reduction method.

    Return the top k eigenvectors and eigenvalues using the covariance array
    obtained from X.


    Args:
        X (numpy.array): 2D data array of flatten images (row:observations,
                         col:features) (float).
        k (int): new dimension space

    Returns:
        tuple: two-element tuple containing
            eigenvectors (numpy.array): 2D array with the top k eigenvectors.
            eigenvalues (numpy.array): array with the top k eigenvalues.
    """
    mean = get_mean_face(X)
    sigma = np.matmul(np.transpose(X - mean), (X - mean))
    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    return eigenvectors[:, -k:][:, ::-1], eigenvalues[-k:][::-1]


class Boosting:
    """Boosting classifier.

    Args:
        X (numpy.array): Data array of flattened images
                         (row:observations, col:features) (float).
        y (numpy.array): Labels array of shape (observations, ).
        num_iterations (int): number of iterations
                              (ie number of weak classifiers).

    Attributes:
        Xtrain (numpy.array): Array of flattened images (float32).
        ytrain (numpy.array): Labels array (float32).
        num_iterations (int): Number of iterations for the boosting loop.
        weakClassifiers (list): List of weak classifiers appended in each
                               iteration.
        alphas (list): List of alpha values, one for each classifier.
        num_obs (int): Number of observations.
        weights (numpy.array): Array of normalized weights, one for each
                               observation.
        eps (float): Error threshold value to indicate whether to update
                     the current weights or stop training.
    """

    def __init__(self, X, y, num_iterations):
        self.Xtrain = np.float32(X)
        self.ytrain = np.float32(y)
        self.num_iterations = num_iterations
        self.weakClassifiers = []
        self.alphas = []
        self.num_obs = X.shape[0]
        self.weights = np.array([1.0 / self.num_obs] * self.num_obs)  # uniform weights
        self.eps = 0.0001

    def train(self):
        """Implement the for loop shown in the problem set instructions."""
        for i in range(self.num_iterations):
            self.weights = self.weights/self.weights.sum()
            clf = WeakClassifier(self.Xtrain, self.ytrain, self.weights)
            clf.train()
            self.weakClassifiers.append(clf)
            res = np.array([clf.predict(x) for x in self.Xtrain])
            ind = np.argwhere(np.not_equal(self.ytrain, res))
            eps = sum(self.weights[ind])
            self.alphas.append(0.5*np.log((1-eps)/eps))
            if eps > self.eps:
                self.weights[ind] = self.weights[ind]*np.exp(-self.ytrain[ind]*self.alphas[i]*res[ind])
            else:
                break

    def evaluate(self):
        """Return the number of correct and incorrect predictions.

        Use the training data (self.Xtrain) to obtain predictions. Compare
        them with the training labels (self.ytrain) and return how many
        where correct and incorrect.

        Returns:
            tuple: two-element tuple containing:
                correct (int): Number of correct predictions.
                incorrect (int): Number of incorrect predictions.
        """
        res = self.predict(self.Xtrain)
        correct = np.sum(np.equal(self.ytrain, res))
        return correct, res.shape[0]-correct

    def predict(self, X):
        """Return predictions for a given array of observations.

        Use the alpha values stored in self.alphas and the weak classifiers
        stored in self.weakClassifiers.

        Args:
            X (numpy.array): Array of flattened images (observations).

        Returns:
            numpy.array: Predictions, one for each row in X.
        """
        #res = np.array([np.array([clf.predict(x) for x in X]) for clf in self.weakClassifiers])
        #res = np.array([clf.predict(X) for clf in self.weakClassifiers])
        res = []
        for clf in self.weakClassifiers:
            res.append(np.array([clf.predict(x) for x in X]))
        res = np.array(res)
        return np.sign(np.sum(np.array(self.alphas).reshape(len(self.alphas), 1) * res, axis=0))


class HaarFeature:
    """Haar-like features.

    Args:
        feat_type (tuple): Feature type {(2, 1), (1, 2), (3, 1), (2, 2)}.
        position (tuple): (row, col) position of the feature's top left corner.
        size (tuple): Feature's (height, width)

    Attributes:
        feat_type (tuple): Feature type.
        position (tuple): Feature's top left corner.
        size (tuple): Feature's width and height.
    """

    def __init__(self, feat_type, position, size):
        self.feat_type = feat_type
        self.position = position
        self.size = size

    def _create_feat(self):
        img = np.zeros(self.size)
        indRow = np.linspace(0, self.size[0], self.feat_type[0]+1).astype(int)
        indCol = np.linspace(0, self.size[1], self.feat_type[1]+1).astype(int)
        color = 0 # 0: white, 1:gray
        for i, row in enumerate(indRow[:-1]):
            img[row: indRow[i+1]] = np.logical_xor(img[row: indRow[i+1]], color)
            color = 1 - color
        color = 1 if self.feat_type[0] == self.feat_type[1] else 0  # 0: white, 1:gray
        for i, col in enumerate(indCol[:-1]):
            img[:, col: indCol[i+1]] = np.logical_xor(img[:, col: indCol[i+1]], color)
            color = 1 - color
        img[img == 0] = 255
        img[img == 1] = 126
        return img

    def _merge_feat(self, img, feat):
        row, col = self.position
        h, w = self.size
        img[row:row + h, col:col + w] = feat

    def _create_two_horizontal_feature(self, shape):
        """Create a feature of type (2, 1).

        Use int division to obtain half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        img = np.zeros(shape)
        feat = self._create_feat()
        self._merge_feat(img, feat)
        return img

    def _create_two_vertical_feature(self, shape):
        """Create a feature of type (1, 2).

        Use int division to obtain half the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        img = np.zeros(shape)
        feat = self._create_feat()
        self._merge_feat(img, feat)
        return img

    def _create_three_horizontal_feature(self, shape):
        """Create a feature of type (3, 1).

        Use int division to obtain a third of the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        img = np.zeros(shape)
        feat = self._create_feat()
        self._merge_feat(img, feat)
        return img

    def _create_three_vertical_feature(self, shape):
        """Create a feature of type (1, 3).

        Use int division to obtain a third of the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        img = np.zeros(shape)
        feat = self._create_feat()
        self._merge_feat(img, feat)
        return img

    def _create_four_square_feature(self, shape):
        """Create a feature of type (2, 2).

        Use int division to obtain half the width and half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        img = np.zeros(shape)
        feat = self._create_feat()
        self._merge_feat(img, feat)
        return img

    def preview(self, shape=(24, 24), filename=None):
        """Return an image with a Haar-like feature of a given type.

        Function that calls feature drawing methods. Each method should
        create an 2D zeros array. Each feature is made of a white area (255)
        and a gray area (126).

        The drawing methods use the class attributes position and size.
        Keep in mind these are in (row, col) and (height, width) format.

        Args:
            shape (tuple): Array numpy-style shape (rows, cols).
                           Defaults to (24, 24).

        Returns:
            numpy.array: Array containing a Haar feature (float or uint8).
        """

        if self.feat_type == (2, 1):  # two_horizontal
            X = self._create_two_horizontal_feature(shape)

        if self.feat_type == (1, 2):  # two_vertical
            X = self._create_two_vertical_feature(shape)

        if self.feat_type == (3, 1):  # three_horizontal
            X = self._create_three_horizontal_feature(shape)

        if self.feat_type == (1, 3):  # three_vertical
            X = self._create_three_vertical_feature(shape)

        if self.feat_type == (2, 2):  # four_square
            X = self._create_four_square_feature(shape)

        if filename is None:
            cv2.imwrite("output/{}_feature.png".format(self.feat_type), X)

        else:
            cv2.imwrite("output/{}.png".format(filename), X)

        return X

    def evaluate(self, ii):
        """Evaluates a feature's score on a given integral image.

        Calculate the score of a feature defined by the self.feat_type.
        Using the integral image and the sum / subtraction of rectangles to
        obtain a feature's value. Add the feature's white area value and
        subtract the gray area.

        For example, on a feature of type (2, 1):
        score = sum of pixels in the white area - sum of pixels in the gray area

        Keep in mind you will need to use the rectangle sum / subtraction
        method and not numpy.sum(). This will make this process faster and
        will be useful in the ViolaJones algorithm.

        Args:
            ii (numpy.array): Integral Image.

        Returns:
            float: Score value.
        """
        '''
        Too slow!
        ii = ii.astype(np.float32)
        row, col = self.position
        indRow = np.linspace(0, self.size[0], self.feat_type[0]+1).astype(int) + row - 1
        indCol = np.linspace(0, self.size[1], self.feat_type[1]+1).astype(int) + col - 1
        w_g = np.zeros(self.feat_type) # 0: white, 1: gray
        color = 0
        for i, row in enumerate(indRow[:-1]):
            w_g[i] = np.logical_xor(w_g[i], color)
            color = 1 - color
        color = 1 if self.feat_type[0] == self.feat_type[1] else 0  # 0: white, 1:gray
        for i, col in enumerate(indCol[:-1]):
            w_g[:, i] = np.logical_xor(w_g[:, i], color)
            color = 1 - color

        gray = 0
        white = 0
        for i, row in enumerate(indRow[:-1]):
            for j, col in enumerate(indCol[:-1]):
                if w_g[i, j] == 1:
                    gray += ii[row, col] - ii[row, indCol[j+1]] - ii[indRow[i+1], col] + ii[indRow[i+1], indCol[j+1]]
                else:
                    white += ii[row, col] - ii[row, indCol[j+1]] - ii[indRow[i+1], col] + ii[indRow[i+1], indCol[j+1]]
        return white - gray
        '''
        ii = ii.astype(np.float32)
        row, col = self.position[0]-1, self.position[1]-1
        h, w = self.size
        white = 0
        gray = 0
        if self.feat_type == (2, 1):
            midRow = row + h//2
            white = ii[row, col] - ii[row, col+w] - ii[midRow, col] + ii[midRow, col+w]
            gray = ii[midRow, col] - ii[midRow, col+w] - ii[row+h, col] + ii[row+h, col+w]
        if self.feat_type == (3, 1):
            midRow1, midRow2 = row + h//3, row + 2*h//3
            white = ii[row, col] - ii[row, col+w] - ii[midRow1, col] + ii[midRow1, col+w]
            gray = ii[midRow1, col] - ii[midRow1, col+w] - ii[midRow2, col] + ii[midRow2, col+w]
            white += ii[midRow2, col] - ii[midRow2, col+w] - ii[row+h, col] + ii[row+h, col+w]
        if self.feat_type == (1, 2):
            midCol = col + w//2
            white = ii[row, col] - ii[row, midCol] - ii[row+h, col] + ii[row+h, midCol]
            gray = ii[row, midCol] - ii[row+h, midCol] - ii[row, col+w] + ii[row+h, col+w]
        if self.feat_type == (1, 3):
            midCol1, midCol2 = col + w // 3, col + 2 * w // 3
            white = ii[row, col] - ii[row, midCol1] - ii[row + h, col] + ii[row + h, midCol1]
            gray = ii[row, midCol1] - ii[row + h, midCol1] - ii[row, midCol2] + ii[row + h, midCol2]
            white += ii[row, midCol2] - ii[row + h, midCol2] - ii[row, col+w] + ii[row + h, col+w]
        if self.feat_type == (2, 2):
            midRow, midCol = row + h//2, col + w//2
            gray = ii[row, col] - ii[row, midCol] - ii[midRow, col] + ii[midRow, midCol]
            white = ii[row, midCol] - ii[midRow, midCol] - ii[row, col+w] + ii[midRow, col+w]
            white += ii[midRow, col] - ii[row+h, col] - ii[midRow, midCol] + ii[row+h, midCol]
            gray += ii[midRow, midCol] - ii[midRow, col + w] - ii[row+h, midCol] + ii[row + h, col + w]

        return white - gray


def convert_images_to_integral_images(images):
    """Convert a list of grayscale images to integral images.

    Args:
        images (list): List of grayscale images (uint8 or float).

    Returns:
        (list): List of integral images.
    """
    tmp = np.array(images)
    return list(tmp.cumsum(axis=1).cumsum(axis=2))


class ViolaJones:
    """Viola Jones face detection method

    Args:
        pos (list): List of positive images.
        neg (list): List of negative images.
        integral_images (list): List of integral images.

    Attributes:
        haarFeatures (list): List of haarFeature objects.
        integralImages (list): List of integral images.
        classifiers (list): List of weak classifiers (VJ_Classifier).
        alphas (list): Alpha values, one for each weak classifier.
        posImages (list): List of positive images.
        negImages (list): List of negative images.
        labels (numpy.array): Positive and negative labels.
    """
    def __init__(self, pos, neg, integral_images):
        self.haarFeatures = []
        self.integralImages = integral_images
        self.classifiers = []
        self.alphas = []
        self.posImages = pos
        self.negImages = neg
        self.labels = np.hstack((np.ones(len(pos)), -1*np.ones(len(neg))))

    def createHaarFeatures(self):
        # Let's take detector resolution of 24x24 like in the paper
        FeatureTypes = {"two_horizontal": (2, 1),
                        "two_vertical": (1, 2),
                        "three_horizontal": (3, 1),
                        "three_vertical": (1, 3),
                        "four_square": (2, 2)}

        haarFeatures = []
        for _, feat_type in FeatureTypes.items():
            for sizei in range(feat_type[0], 24 + 1, feat_type[0]):
                for sizej in range(feat_type[1], 24 + 1, feat_type[1]):
                    for posi in range(0, 24 - sizei + 1, 4):
                        for posj in range(0, 24 - sizej + 1, 4):
                            haarFeatures.append(
                                HaarFeature(feat_type, [posi, posj],
                                            [sizei-1, sizej-1]))
        self.haarFeatures = haarFeatures

    def train(self, num_classifiers):

        # Use this scores array to train a weak classifier using VJ_Classifier
        # in the for loop below.
        scores = np.zeros((len(self.integralImages), len(self.haarFeatures)))
        print(" -- compute all scores --")
        for i, im in enumerate(self.integralImages):
            scores[i, :] = [hf.evaluate(im) for hf in self.haarFeatures]

        weights_pos = np.ones(len(self.posImages), dtype='float') * 1.0 / (
                           2*len(self.posImages))
        weights_neg = np.ones(len(self.negImages), dtype='float') * 1.0 / (
                           2*len(self.negImages))
        weights = np.hstack((weights_pos, weights_neg))

        print(" -- select classifiers --")
        for i in range(num_classifiers):
            weights = weights / weights.sum()
            clf = VJ_Classifier(scores, self.labels, weights)
            clf.train()
            res = np.array([clf.predict(x) for x in scores])
            eps = clf.error
            self.classifiers.append(clf)
            beta = eps / (1.-eps)
            exp = 1 - ((1 - self.labels * res)/2)  # e=0 if correct, 1 otherwise
            weights = weights * (beta ** exp)
            alpha = np.log(1./beta)
            self.alphas.append(alpha)

    def predict(self, images):
        """Return predictions for a given list of images.

        Args:
            images (list of element of type numpy.array): list of images (observations).

        Returns:
            list: Predictions, one for each element in images.
        """
        ii = convert_images_to_integral_images(images)
        scores = np.zeros((len(ii), len(self.haarFeatures)))

        for i, im in enumerate(ii):
            scores[i, :] = [hf.evaluate(im) for hf in self.haarFeatures]

        result = []
        for clf in self.classifiers:
            result.append(np.array([clf.predict(x) for x in scores]))
        result = np.array(result)
        return np.sign(np.sum(np.array(self.alphas).reshape(len(self.alphas), 1) * (result - 0.5), axis=0))

    def faceDetection(self, image, filename):
        """Scans for faces in a given image.

        Complete this function following the instructions in the problem set
        document.

        Use this function to also save the output image.

        Args:
            image (numpy.array): Input image.
            filename (str): Output image file name.

        Returns:
            None.
        """
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        H, W = img.shape
        windows = []
        ind = []
        for h in range(H-24):
            for w in range(W-24):
                windows.append(img[h:h+24, w:w+24])
                ind.append([h, w])
        res = self.predict(windows)
        ind = np.array(ind)
        col, row = ind[res == 1].mean(axis=0).astype(int)

        cv2.rectangle(image, (row, col), (row+24, col+24), (255, 0, 0), 1)
        cv2.imwrite("output/{}.png".format(filename), image)

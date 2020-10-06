import numpy as np


class LabelEncoder:
    """
    Class that encodes and decodes labels according to a given encoding.
    """
    def __init__(self, labels_encoding, encoding_score_type='quadratic'):
        """
        labels_encoding (Dictionary): Dictionary of {label:encoding} where the encodings are arrays of -1, 0 or 1.
        encoding_score_type (String, optional, default='quadratic'): Description of the type of encoding score used to decode the labels. Choices are 'quadratic' and 'scalar'.
        """
        self.labels_encoding = labels_encoding
        self.labels = sorted([label for label in self.labels_encoding])

        self.labels_to_idx = {label:idx for idx, label in enumerate(self.labels)}
        
        self.n_classes = len(self.labels)
        self.encoding_dim = len(self.labels_encoding[self.labels[0]])

        self.encoding_matrix = np.array([self.labels_encoding[label] for label in self.labels])
        self.weights_matrix = np.abs(self.encoding_matrix)/np.sum(np.abs(self.encoding_matrix), axis=1).reshape(-1,1)

        self.encoding_score_type = encoding_score_type


    def encode_labels(self, Y):
        """
        Y (Iterable of length 'n_examples'): Labels of the examples.
        Returns the encoded labels as an array of shape (n_examples, encoding_dim) and the associated encoding weights as an array of the same shape.
        """
        encoded_Y = np.zeros((len(Y), self.encoding_dim))
        weights = np.zeros_like(encoded_Y)
        for i, label in enumerate(Y):
            label_idx = self.labels_to_idx[label]

            encoded_Y[i] = self.encoding_matrix[label_idx]
            weights[i] = self.weights_matrix[label_idx]
        
        return encoded_Y, weights


    def decode_labels(self, encoded_Y):
        """
        encoded_Y (Array of shape (n_examples, encoding_dim)): Array of encoded labels. It can contain real numbers, for instance in the case where encoded_Y are predictions.
        Decodes the labels by computing a score between encoded labels and the encoding matrix and taking the argmax as the class.
        Returns a list of decoded labels.
        """
        n_examples = encoded_Y.shape[0]
        scored_Y = self.encoding_score(encoded_Y) # Shape: (n_examples, n_classes)
        decoded_Y_idx = np.argmax(scored_Y, axis=1) # Shape: (n_examples,)
        decoded_Y = []
        for idx in decoded_Y_idx:
            decoded_Y.append(self.labels[idx])

        return decoded_Y


    def encoding_score(self, encoded_Y):
        """
        encoded_Y (Array of shape (n_examples, encoding_dim)): Array of encoded labels. It can contain real numbers, for instance in the case where encoded_Y are predictions.
        Two decoding score are offered: 'quadratic' and 'scalar'. This method can be overloaded to use an other type of encoding score.
        Returns an array of shape (n_examples, n_classes) of the scores where the max of each row represents the most likely class.
        'quadratic' type solves the following: -(f - Y)² * W . U
            where   
                f is the prediction of a label 'encoded_Y'. Shape: (n_examples, encoding_dim, 1)
                Y is the matrix of label encodings. Shape: (1, encoding_dim, n_classes)
                W is the matrix of weights of the encodings. Shape: (1, encoding_dim, n_classes)
                U is the vector of 1 everywhere used to perform a sum on the correct dimension. Shape: (1, encoding_dim, 1)
                * is the element-wise multiplication.
                ² is the element-wise multiplication by itself.
                . is the dot product
            For operation f - Y, 'f' and 'Y' need to be broadcasted (by copy) to so that it is well defined.
            Some simplifications allow to find an alternate expression which keeps the shape the matrix small:
                - (f².W - 2*f.(Y*W) + 1)
            Since we take the argmax, we can drop the + 1.
        'scalar' type solves the following: f.(Y*W)
            No broadcasting needed.
            For one-hot vectors encoding, it is equivalent to do nothing since the encoding matrix is the identity.
        """
        weighted_encoding = self.encoding_matrix * self.weights_matrix
        scalar_prod = encoded_Y.dot(weighted_encoding.T)

        if self.encoding_score_type == 'quadratic':
            score = 2* scalar_prod - (encoded_Y**2).dot(self.weights_matrix.T)
        elif self.encoding_score_type == 'scalar':
            score = scalar_prod
        return score
    

class OneHotEncoder(LabelEncoder):
    """
    Class the provides a one-hot encoding. For example, if we have 3 classes, the labels_encodings dictionary sould look like this:
        {
            1:np.array([1, 0, 0]),
            2:np.array([0, 1, 0]),
            3:np.array([0, 0, 1])
        }
    """
    def __init__(self, Y, encoding_score_type='quadratic'):
        labels = sorted(set(Y))
        one_hot_encoding = lambda idx: np.eye(1, len(labels), k=idx, dtype=int)[0]
        labels_encoding = {label:one_hot_encoding(idx) for idx, label in enumerate(labels)}

        super().__init__(labels_encoding)


class AllPairsEncoder(LabelEncoder):
    """
    Class the provides a all-pairs encoding. For example, if we have 4 classes, we have 6 pairs: (1,2), (1,3), (1,4), (2,3), (2,4), (3,4).
    The labels_encodings dictionary sould look like this:
        {
            1:np.array([ 1, 1, 1, 0, 0, 0]),
            2:np.array([-1, 0, 0, 1, 1, 0]),
            3:np.array([ 0,-1, 0,-1, 0, 1])
            4:np.array([ 0, 0,-1, 0,-1,-1])
        }
    """
    def __init__(self, Y, encoding_score_type='quadratic'):
        labels = sorted(set(Y))
        n_classes = len(labels)
        encoding_dim = int(n_classes*(n_classes-1)/2)
        labels_encoding = {label:np.zeros(encoding_dim) for label in labels}
        idx_to_pairs = [(i,j) for i in range(n_classes) for j in range(i+1, n_classes)]
        
        for idx, (i, j) in enumerate(idx_to_pairs):
            labels_encoding[labels[i]][idx] = 1
            labels_encoding[labels[j]][idx] = -1

        super().__init__(labels_encoding)
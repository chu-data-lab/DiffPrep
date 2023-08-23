class Identity(object):
    """Identity Transformer"""
    def __init__(self):
        self.method = "identity"
        pass

    def fit(self, X):
        pass

    def transform(self, X):
        return X

    def fit_transform(self, X):
        return X
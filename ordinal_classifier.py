from sklearn.base import BaseEstimator, ClassifierMixin, clone


class OrdinalClassifier(ClassifierMixin, BaseEstimator):
    """It implements a versitile naive ordinal classifier.
        This implementation is based on:
        
        Eibe Frank and Mark Hal, lECML 2001. 12th European Conference)
        https://www.cs.waikato.ac.nz/~eibe/pubs/ordinal_tech_report.pdf
        
        Parameters
        ----------
        classifier_type: ClassifierMixin
        An instantiatied classifier object.
        For example `LogisticRegressionCV`.
        
        ordered_labels: Optional, Iterable
        The order of the labels to be imposed as a list or np.ndarray
        It can be omitted if labels are already integers with no-gaps encoding the order.
        
        verbose: bool, default=True
        Whether to print output during the fit
        
        Attributes
        ----------
        classifiers: List[ClassifierMixin]
        A list of the binary classifiers for the ordered bisections.
        
        classifier_type: ClassifierMixin
        The generic of classifier instance used for each of the binary classifiers.
        
        
        Example
        -------
        ```
        clf = LogisticRegressionCV(Cs=35, penalty='l1', solver="liblinear", fit_intercept=False, max_iter=500, cv=6)
        ord_clf = OrdinalClassifier(clf, ordered_labels=['label_early',  'label_mid', 'label_late', 'label_verylater'])
        ord_clf.fit(X_train, y_train)
        probabilites = ord_clf.predict_proba(X)
        predicted_labels = ord_clf.predict(X)
        
        ```
        """
    def __init__(self, classifier_type, ordered_labels = None, verbose = False):
        self.classifier_type = classifier_type
        self.ordered_labels = np.array(ordered_labels) if ordered_labels else ordered_labels
        self.verbose = verbose
        self._ordered_dict = None
        self._n_classes = None
        self.classifiers = None
    
    @property
    def classes_(self):
        return self.ordered_labels
    
    def fit(self, X, y):
        """Fit the model according to the given training data.
            
            Parameters
            ----------
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
            
            y : array-like of shape (n_samples,)
            Target vector relative to X.
            
            Returns
            -------
            self : object
            """
        uq_labels = np.unique(y)
        self._n_classes = len(uq_labels)
        self.classifiers = [clone(self.classifier_type) for i in range(self._n_classes - 1)]
        
        if self.ordered_labels is None:
            self.ordered_labels = np.arange(self._n_classes)
    
        p = set(self.ordered_labels)
        assert not (p ^ set(uq_labels)), f"Found the label/s: {p} for which no order was specified"
        self._ordered_dict = dict(zip(self.ordered_labels, np.arange(self._n_classes)))
        self._rev_ordered_dict = {v:k for k, v in self._ordered_dict.items()}
        y_ordered = np.array([self._ordered_dict[i] for i in y])
        
        for i in range(1, self._n_classes):
            if self.verbose:
                print(f"Fitting bisection between {self.classes_[:i]} and {set(self.classes_) - set(self.classes_[:i])}")
            y_bin = (y_ordered >= i).astype(int)
            self.classifiers[i-1].fit(X, y_bin)
return self
    
    def predict(self, X):
        """Predict class labels for samples in X.
            
            Parameters
            ----------
            X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.
            
            Returns
            -------
            C : array, shape [n_samples]
            Predicted class label per sample.
            """
        probabilities = self.predict_proba(X)
        predicted_labels = probabilities.argmax(1)
        predicted_labels_names = self.ordered_labels[predicted_labels]
        return predicted_labels_names
    
    def predict_proba(self, X):
        """Probability estimates.
            
            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.
            
            Returns
            -------
            T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
            """
                P_tmp = np.zeros((X.shape[0], self._n_classes+1))
                P_tmp[:, 0] = 1
                for i in range(1, self._n_classes):
                    clf = self.classifiers[i-1]
                    bin_prob = clf.predict_proba(X)
                    ix_lbl_of_1 = np.where(clf.classes_ == 1)[0][0]
                    P_tmp[:, i] = bin_prob[:, ix_lbl_of_1]
                        probabilities = P_tmp[:, :-1] - P_tmp[:, 1:]
                        #patch
                        probabilities = np.clip(probabilities, 0, 1)
                        probabilities = probabilities / probabilities.sum(1)[:, None]
                            return probabilities

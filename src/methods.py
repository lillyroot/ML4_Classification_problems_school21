import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

def fs_by_nan_ratio_and_corr(x: pd.DataFrame, y: pd.Series | pd.DataFrame, nan_threshold=0.5, top_k=10):
    with_nan = x.isnull().mean()
    edge_cols = x.columns[(with_nan < nan_threshold).values]

    top_cols = x[edge_cols].corrwith(y).abs().sort_values(ascending=False).head(top_k).index.to_list()
    return top_cols

def roc_auc_curve(y_true: pd.DataFrame | pd.Series | np.array, y_probs: pd.DataFrame | pd.Series | np.array):
    y_true = np.asarray(y_true)
    unq = np.unique(y_probs)[::-1]
    coordinates = []

    unq = np.insert(unq, 0, 1.1)

    for threshold in unq:
        y_pred = (y_probs >= threshold).astype(int)

        tpr = recall_score(y_true, y_pred)

        tn = ((y_true == y_pred) & (y_pred == 0)).astype(int).mean()
        fp = ((y_true != y_pred) & (y_pred == 1)).astype(int).mean()

        fpr = fp / (fp + tn)
        coordinates.append((tpr, fpr))

    tpr, fpr = zip(*coordinates)
    return tpr, fpr

def calculate_auc(tpr, fpr):
    auc = 0
    for i in range(len(tpr) - 1):
        width = fpr[i + 1] - fpr[i]
        h = tpr[i + 1] + tpr[i]
        auc += width * h / 2
    return auc

def draw_roc_curve(tpr, fpr):
    plt.figure(figsize=(5, 5))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], linestyle='--', color='b')
    plt.grid(True)
    plt.legend()
    plt.show()

def draw_pr_curve(rec, prec, y_true: pd.Series | np.array):
    y_true = np.asarray(y_true)
    positive = (y_true == 1).astype(int).mean()
    plt.figure(figsize=(5, 5))
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot(rec, prec, color='r', lw=2, label='PR curve')
    plt.axhline(y=positive, linestyle='--', color='b')
    plt.grid(True)
    plt.legend()
    plt.show()


class myLogisticRegression():
    def __init__(self, random_state=21, epoch=50, tol=10**-3, learning_rate=0.001):
        self.w = None
        self.epoch = epoch
        self.tol = tol
        self.learning_rate = learning_rate
        self.gnr = np.random.default_rng()
        if random_state is not None:
            self.gnr = np.random.default_rng(random_state)

    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def fit_SGD(self, X: pd.DataFrame, y: pd.Series): #SGD, где batch - одна строка
        X_mat = np.asarray(X) # только лишь для того, чтобы избежать ошибки "matrices are not aligned" при матричном
        y_arr = np.asarray(y) # перемножении pandas dataframe в 79 строке (да, костыль)
        len_f = X.shape[1]
        w = self.gnr.uniform(-0.01, 0.01, len_f)

        for epoch in range(self.epoch):
            for index, row in X.iterrows():
                D = row @ w
                y_pred = self.sigmoid(D)
                grad = (y_pred - y[index]) * row
                w = w - self.learning_rate * grad

            pred_all = self.sigmoid(X_mat @ w)
            grad_all = ((pred_all - y_arr) @ X_mat) / len_f
            error = np.sum(abs(grad_all))

            if error <= self.tol:
                print('Convergence achieved')
                break

            indices = self.gnr.permutation(np.arange(len_f))
            X_mat = X_mat[indices]
            y_arr = y_arr[indices]

        self.w = w

    def fit(self, X: pd.DataFrame | np.array, y: pd.Series | np.array):
        X_mat = np.asarray(X)
        y_arr = np.asarray(y)
        len_f = X.shape[1]
        w = self.gnr.uniform(-0.01, 0.01, len_f)

        for epoch in range(self.epoch):
            D = X_mat @ w
            y_pred = self.sigmoid(D)
            grad = ((y_pred - y_arr) @ X_mat) / len_f
            w = w - self.learning_rate * grad


            error = np.sum(abs(grad))
            if error < self.tol:
                print('Convergence achieved')
                break

        self.w = w

    def predict(self, X: pd.DataFrame):
        probs = self.sigmoid(X @ self.w)
        return (probs > 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame):
        X_mat = np.asarray(X)
        probs_1 = self.sigmoid(X_mat @ self.w)
        probs_0 = 1 - probs_1
        return np.column_stack((probs_0, probs_1))


class myKNNClassifier():
    def __init__(self, n_neighbours: int = 5, metric: str = 'euclidean'):
        self.metric = metric
        self.k = n_neighbours

    def fit(self, X: pd.DataFrame | np.array, y: pd.Series | np.array):
        self.classes = np.unique(y)
        self.X = np.asarray(X)
        self.y = np.asarray(y)

    def nearest_neighbors(self, X: pd.DataFrame | np.array):
        X = np.asarray(X)
        for i in range(X.shape[0]):
            distances = np.sum(np.abs(self.X - X[i, :]), axis=1)
            if self.metric == 'euclidean':
                distances = np.sum((self.X - X[i, :]) ** 2, axis=1)
            nearest = self.y[np.argsort(distances)[:self.k]]
            unique, counts = np.unique(nearest, return_counts=True)
            yield unique, counts

    def predict(self, X: pd.DataFrame | np.array):
        y_pred = np.zeros(X.shape[0])
        for i, (unique, counts) in enumerate(self.nearest_neighbors(X)):
            max = unique[np.argmax(counts)]
            y_pred[i] = max
        return y_pred

    def predict_proba(self, X: pd.DataFrame | np.array):
        probs = np.zeros((X.shape[0], len(self.classes)))
        for i, (unique, counts) in enumerate(self.nearest_neighbors(X)):
            for label, counts in zip(unique, counts):
                pos = np.where(self.classes == label)
                probs[i, pos] = counts / self.k
        return probs


class GaussianNB():
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame | np.array, y: pd.Series | np.array):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes, self.counts = np.unique(y, return_counts=True)
        self.means = np.zeros((len(self.classes), X.shape[1]))
        self.vars = np.zeros((len(self.classes), X.shape[1]))
        self.prior_prob = np.zeros(len(self.classes))

        for i in range(len(self.classes)):
            temp_X = X[y == self.classes[i]]
            self.means[i, :] = temp_X.mean(axis=0)
            self.vars[i, :] = temp_X.var(axis=0)
            self.prior_prob[i] = self.counts[i] / X.shape[0]

        return self

    def gaussian_log_likelihood(self, X: pd.DataFrame | np.array, clss_idx: int):
        X = np.asarray(X)
        clss_probs =  np.sum((- 0.5 * np.log(2 * np.pi * self.vars[clss_idx, :]) -
                                ((X - self.means[clss_idx, :]) ** 2) / (2 * self.vars[clss_idx, :])), axis=1)
        return clss_probs

    def predict(self, X: pd.DataFrame | np.array):
        clss_pred = np.zeros((X.shape[0], len(self.classes)))
        for i in range(len(self.classes)):
            numerator = np.log(self.prior_prob[i]) + self.gaussian_log_likelihood(X, i)
            clss_pred[:, i] = numerator

        final_classes = np.argmax(clss_pred, axis=1)
        return final_classes

    def predict_proba(self, X: pd.DataFrame | np.array):
        probs = np.zeros((X.shape[0], len(self.classes)))
        for i in range(len(self.classes)):
            numerator = np.log(self.prior_prob[i]) + self.gaussian_log_likelihood(X, i)
            probs[:, i] = np.exp(numerator)

        probs = probs / np.sum(probs, axis=1, keepdims=True)
        return probs


#вспомогательная функция без особой логики в реализации
def evaluate_model(x_train, x_val, x_test, y_train, y_val, y_test, model_name, ginies_df=None,
                   log_reg_approach='standard', return_coefs=False, return_estimator=False, return_model=False):
    if 'log_reg' in model_name:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)
        model = LogisticRegression()

        param_grid = {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'solver': ['saga'],
            'max_iter': [3000, 4000, 7000, 10000, 25000, 40000],
            'class_weight': ['balanced'],
            'l1_ratio': [1]
        }

        grid = GridSearchCV(
            model,
            param_grid,
            cv=5,
            n_jobs=-1,
            scoring='accuracy'
        )

        grid.fit(x_train, y_train)

        tr_pred = grid.predict(x_train)
        tr_probs = grid.predict_proba(x_train)[:, 1]
        val_pred = grid.predict(x_val)
        val_probs = grid.predict_proba(x_val)[:, 1]
        tst_pred = grid.predict(x_test)
        tst_probs = grid.predict_proba(x_test)[:, 1]

    elif 'knn' in model_name:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)
        model = KNeighborsClassifier()

        param_grid = {
            'n_neighbors': [5, 11, 15, 21, 51, 101],
            'weights': ['uniform'],
            'p': [1, 2],
            'algorithm': ['auto']
        }

        grid = GridSearchCV(
            model,
            param_grid,
            cv=5,
            n_jobs=-1,
            scoring='accuracy'
        )

        grid.fit(x_train, y_train)

        tr_pred = grid.predict(x_train)
        tr_probs = grid.predict_proba(x_train)[:, 1]
        val_pred = grid.predict(x_val)
        val_probs = grid.predict_proba(x_val)[:, 1]
        tst_pred = grid.predict(x_test)
        tst_probs = grid.predict_proba(x_test)[:, 1]

    elif 'gaussian_nb' in model_name:
        model = GaussianNB()
        model.fit(x_train, y_train)
        tr_pred = model.predict(x_train)
        tr_probs = model.predict_proba(x_train)[:, 1]
        val_pred = model.predict(x_val)
        val_probs = model.predict_proba(x_val)[:, 1]
        tst_pred = model.predict(x_test)
        tst_probs = model.predict_proba(x_test)[:, 1]

    elif 'my_log_reg' in model_name:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)
        model = myLogisticRegression()
        if log_reg_approach == 'standard':
            model_name = model_name + '_standard'
            model.fit(x_train, y_train)
        elif log_reg_approach == 'SGD':
            model_name = model_name + '_SGD'
            x_train = pd.DataFrame(x_train)
            model.fit_SGD(x_train, y_train)
        else:
            raise Exception('Unknown log_reg approach')

        tr_pred = model.predict(x_train)
        tr_probs = model.predict_proba(x_train)[:, 1]
        val_pred = model.predict(x_val)
        val_probs = model.predict_proba(x_val)[:, 1]
        tst_pred = model.predict(x_test)
        tst_probs = model.predict_proba(x_test)[:, 1]

    elif 'my_knn' in model_name:
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)
        model = myKNNClassifier()
        model.fit(x_train, y_train)

        tr_pred = model.predict(x_train)
        tr_probs = model.predict_proba(x_train)[:, 1]
        val_pred = model.predict(x_val)
        val_probs = model.predict_proba(x_val)[:, 1]
        tst_pred = model.predict(x_test)
        tst_probs = model.predict_proba(x_test)[:, 1]

    elif 'my_gaussian_nb' in model_name:
        model = GaussianNB()
        model.fit(x_train, y_train)

        tr_pred = model.predict(x_train)
        tr_probs = model.predict_proba(x_train)[:, 1]
        val_pred = model.predict(x_val)
        val_probs = model.predict_proba(x_val)[:, 1]
        tst_pred = model.predict(x_test)
        tst_probs = model.predict_proba(x_test)[:, 1]

    else:
        raise Exception(f"Unknown model: {model_name}")

    if ginies_df is not None:
        ginies = [model_name]
        trues = [y_train, y_val, y_test]
        preds = [tr_pred, val_pred, tst_pred]
        probs = [tr_probs, val_probs, tst_probs]

        for i, (true, pred, prob) in enumerate(zip(trues, preds, probs)):
            tpr, fpr = roc_auc_curve(true, prob)
            draw_roc_curve(tpr, fpr)
            print(classification_report(true, pred))
            ginies.append(2 * calculate_auc(tpr, fpr) - 1)

        ginies_df.loc[len(ginies_df)] = ginies

    if return_coefs and 'log_reg' in model_name and not return_estimator:
        return ginies_df, grid.best_estimator_.coef_

    if return_coefs and return_estimator and 'log_reg' in model_name:
        return ginies_df, grid.best_estimator_.coef_, grid.best_estimator_

    if return_estimator and not return_coefs and not return_model:
        return grid.best_estimator_

    if return_model:
        return model

    return ginies_df

def recall(y_true: pd.Series | np.array, y_pred: pd.Series | np.array):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = ((y_true == y_pred) & (y_true == 1)).astype(int).mean()
    fn = ((y_true != y_pred) & (y_pred == 0)).astype(int).mean()

    if (tp + fn) == 0: return 0
    return tp / (tp + fn)

def precision(y_true: pd.Series | np.array, y_pred: pd.Series | np.array):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = ((y_true == y_pred) & (y_true == 1)).astype(int).mean()
    fp = ((y_true != y_pred) & (y_pred == 1)).astype(int).mean()

    if (tp + fp) == 0: return 1
    return tp / (tp + fp)

def f1_score(y_true: pd.Series | np.array, y_pred: pd.Series | np.array):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * prec * rec / (prec + rec)

def coordinates_PR(y_true: pd.Series | np.array, y_probs: pd.Series | np.array):
    y_true, y_probs = np.asarray(y_true), np.asarray(y_probs)
    uniq_probs = np.unique(y_probs)[::-1]
    uniq_probs = np.insert(uniq_probs, 0, 1.1)
    coordinates = []
    for threshhold in uniq_probs:
        y_pred = (y_probs >= threshhold).astype(int)
        rec = recall(y_true, y_pred)
        prec = precision(y_true, y_pred)
        coordinates.append((rec, prec))

    rec, prec = zip(*coordinates)
    return rec, prec

def auc_pr_comparison(x_test: pd.Series | np.array, y_test: pd.Series | np.array, model):
    x_test, y_test = np.asarray(x_test), np.asarray(y_test)
    probs = model.predict_proba(x_test)[:, 1]
    rec, prec = coordinates_PR(y_test, probs)
    draw_pr_curve(rec, prec, y_test)
    return calculate_auc(prec, rec)
















"""Training-only preprocessing; fixed Ridge head across representations."""
import copy
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-2, 5, 8)


def fit_ridge(X, y, seed=0):
    components = min(256, X.shape[1], len(X) - 1)
    model = Pipeline([('scale', StandardScaler()),
                      ('pca', PCA(n_components=components, svd_solver='randomized', random_state=seed)),
                      ('ridge', RidgeCV(alphas=ALPHAS))])
    model.fit(X, y)
    return model


class BaselineNN:
    def predict(self, X):
        with torch.inference_mode():
            return self.net(torch.as_tensor(self.scaler.transform(X), dtype=torch.float32)).numpy().ravel()


def fit_baseline_nn(X_train, y_train, seed=0, groups=None, max_epochs=150):
    """256/64 ReLU network; inner validation groups never enter the scaler fit."""
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    y_train = np.asarray(y_train, dtype=np.float32)
    if groups is not None:
        tr, va = next(GroupShuffleSplit(n_splits=1, test_size=.15, random_state=seed).split(X_train, groups=groups))
    else:
        tr, va = train_test_split(np.arange(len(X_train)), test_size=.15, random_state=seed)
    model = BaselineNN()
    model.scaler = StandardScaler().fit(X_train[tr])
    x = torch.tensor(model.scaler.transform(X_train), dtype=torch.float32)
    y = torch.tensor(y_train[:, None])
    model.net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 256), torch.nn.ReLU(),
                                    torch.nn.Linear(256, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
    opt = torch.optim.AdamW(model.net.parameters(), lr=.001, weight_decay=.01)
    rng = np.random.default_rng(seed)
    best, stale, best_state = np.inf, 0, None
    for epoch in range(max_epochs):
        model.net.train()
        order = rng.permutation(tr)
        for start in range(0, len(order), 256):
            idx = order[start:start + 256]
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model.net(x[idx]), y[idx])
            loss.backward()
            opt.step()
        model.net.eval()
        with torch.inference_mode():
            value = torch.nn.functional.mse_loss(model.net(x[va]), y[va]).item()
        if value < best - 1e-5:
            best, stale, best_state = value, 0, copy.deepcopy(model.net.state_dict())
            model.best_epoch_ = epoch + 1
        else:
            stale += 1
        if stale >= 15:
            break
    model.net.load_state_dict(best_state)
    model.net.eval()
    model.inner_train_size_ = len(tr)
    model.inner_validation_size_ = len(va)
    return model

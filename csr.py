import xgboost as xgb
import numpy as np
import fastavro as fa
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils.class_weight import compute_sample_weight
import os
import shutil

def mean(l):
    return sum(l) / len(l)

def std(l):
    m = mean(l)
    return mean([(x - m) ** 2 for x in l]) ** (1 / 2)

class CSRDataset:
    def __init__(self, filename):
        with open(filename, 'rb') as f:
            self.df = pd.DataFrame.from_records([record for record in fa.reader(f)])
        
        # replace missing ages with the average age of 27
        self.df['age'].replace(0, 27, inplace=True)

    # [1, 2, ... len(self.df) - 1]
    def db_index(self):
        return np.arange(len(self.df))
    
    # gets the given indexes and returns those applications and whether or not they got instantly approved
    def separate_input_target(self, index):
        sub_df = self.df.iloc[index]
        return sub_df.drop(['instant_approval', 'credit_limit'], axis=1), sub_df['instant_approval']

    # k fold cross validation
    def train_eval_splits(self, n_splits):
        kf = KFold(n_splits=n_splits, shuffle=True)
        for train_index, eval_index in kf.split(self.df):
            yield self.separate_input_target(train_index), self.separate_input_target(eval_index)

class CSRGridSearcher:
    def __init__(self, default_params, grid, dataset):
        self.clf = xgb.XGBClassifier()
        self.default_params = default_params
        self.grid = grid
        self.dataset = dataset
        self._grid_size = 1
        # grid size is exponential
        for param in grid:
            self._grid_size *= len(grid[param])
        self._metrics = []

    def grid_search(self, n_splits):
        self._metrics = []
        print(f'Searching through {self._grid_size} parameter sets')
        for i in range(self._grid_size):
            print(f'Evaluating parameter set {i + 1}')
            params = self._get_params(i)
            eval_metrics = self._train(params, n_splits)
            self._metrics.append((params, mean(eval_metrics), std(eval_metrics)))
        print('Done searching')

    # after search
    def best_params(self):
        return max(self._metrics, key=lambda m: m[1])[0]

    # after search, query a parameter to see how each value for it performed
    def param_metrics(self, param):
        value_to_metric = {}
        value_to_metric_std = {}
        for metric in self._metrics:
            value = metric[0][param]
            if not value_to_metric.get(value):
                value_to_metric[value] = []
                value_to_metric_std[value] = []
            value_to_metric[value].append(metric[1])
            value_to_metric_std[value].append(metric[2])

        for value in value_to_metric:
            value_to_metric[value] = (mean(value_to_metric[value]), mean(value_to_metric_std[value]))
        return value_to_metric
    
    # we would need 1 for loop for every parameter, so this reduces all those for loops to one loop that goes through every possible parameter set
    def _get_params(self, i):
        running_grid_size = self._grid_size
        params = {}
        for param in self.grid:
            running_grid_size //= len(self.grid[param])
            params[param] = self.grid[param][i // running_grid_size]
            i %= running_grid_size
        return params

    def _train(self, params, n_splits):
        eval_metrics = []
        for (train_X, train_Y), (eval_X, eval_Y) in self.dataset.train_eval_splits(n_splits):
            params = self.default_params | params
            self.clf.set_params(**params)
            self.clf.fit(train_X, train_Y, eval_set=[(train_X, train_Y), (eval_X, eval_Y)], verbose=False)
            eval_metrics.append(self.clf.evals_result()['validation_1'][self.default_params['eval_metric'][0]][-1])
        return eval_metrics

class CSRModel:
    def __init__(self, dataset, n_splits, ensemble=True):
        self.clfs = [xgb.XGBClassifier() for _ in range(n_splits if ensemble else 1)]
        self.dataset = dataset
        self.n_splits = n_splits
        self.ensemble = ensemble

    def save(self, directory):
        try:
            os.makedirs(directory)
        except FileExistsError:
            shutil.rmtree(directory)
            self.save(directory)
        for i, clf in enumerate(self.clfs):
            clf.save_model(os.path.join(directory, f'{i}.json'))

    def load(self, directory):
        for i, clf in enumerate(self.clfs):
            clf.load_model(os.path.join(directory, f'{i}.json'))

    def train(self, params):
        if self.ensemble:
            for clf_idx, ((train_X, train_Y), (eval_X, eval_Y)) in enumerate(self.dataset.train_eval_splits(self.n_splits)):
                self.clfs[clf_idx].set_params(**params)
                self.clfs[clf_idx].fit(train_X, train_Y, eval_set=[(train_X, train_Y), (eval_X, eval_Y)], verbose=False)
            return None, None
        else:
            (train_X, train_Y), (eval_X, eval_Y) = next(self.dataset.train_eval_splits(self.n_splits))
            self.clfs[0].set_params(**params)
            self.clfs[0].fit(train_X, train_Y, eval_set=[(train_X, train_Y), (eval_X, eval_Y)], verbose=False)
            return eval_X, eval_Y

    def predict_proba(self, x):
        return mean([clf.predict_proba(x) for clf in self.clfs])

    def predict(self, x):
        return self.predict_proba(x).argmax(axis=1)

    # given an application, find the the closest modifications that give a better instant approval score
    def find_improvements(self, x):
        improvements = []
        baseline = self.predict_proba(x)

        def add_improvement(i_x, field):
            proba = self.predict_proba(i_x)
            if proba[0][2] > baseline[0][2]: # only append this modification if it is actually an improvement
                improvements.append(({field: (x[field][0], i_x[field][0])}, proba))
                return True
            return False

        def search_many(field, limit, step):
            i_x = x.copy()
            improvement_found = False
            while not improvement_found and i_x[field][0] < limit:
                i_x[field][0] += step
                improvement_found = add_improvement(i_x, field)

                
        search_many('credit_score', 850, 25)
        search_many('income', 200_000, 10_000)
        search_many('age', 30, 1)

        if x['exact_5/24'][0] > 0:
            i_x = x.copy()
            i_x['exact_5/24'][0] = min(i_x['exact_5/24'][0] - 1, 4)
            i_x['relative_5/24'][0] = 0
            add_improvement(i_x, 'exact_5/24')

        i_x = x.copy()
        new_genders = [i for i in range(3) if i != i_x['gender'][0]]
        for gender in new_genders:
            i_x = x.copy()
            i_x['gender'][0] = gender
            add_improvement(i_x, 'gender')

        i_x = x.copy()
        i_x['years_with_chase'][0] += 1
        add_improvement(i_x, 'years_with_chase')

        return baseline, sorted(improvements, key=lambda improvement: -improvement[1][0][2])

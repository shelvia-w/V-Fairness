import copy
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.datasets import make_moons

class FairnessDataset():
    def __init__(self, dataset):
        self.dataset = dataset
        np.random.seed(12345678)
        
        if self.dataset == 'Adult_Gender':
            self.get_adult_data_1()
        elif self.dataset == 'Adult_Gender_Race':
            self.get_adult_data_2()
        elif self.dataset == 'Health':
            self.get_health_data()
        elif self.dataset == 'Credit':
            self.get_credit_data()
        elif self.dataset == 'Law':
            self.get_law_data()
        elif self.dataset == 'Moon_1':
            self.get_moon_data_1()
        elif self.dataset == 'Moon_2':
            self.get_moon_data_2()
        else:
            raise ValueError('Your argument {} for dataset name is invalid.'.format(self.dataset))
            
    def get_adult_data_1(self):
        
        raw_train_data = pd.read_csv('./datasets/adult/adult.data.txt', header=None, sep=', ').values
        raw_test_data = pd.read_csv('./datasets/adult/adult.test.txt', header=None, sep=', ').values
        
        train_data = remove_question(raw_train_data)
        test_data = remove_dot(remove_question(raw_test_data))
        labels = gather_labels(train_data)
        
        train_data = transform_to_binary_adult_1(train_data, labels)
        test_data = transform_to_binary_adult_1(test_data, labels)
        
        self.trn_data = tuple([d.astype(np.float32) for d in train_data])
        test_data = tuple([d.astype(np.float32) for d in test_data])
        self.val_data = tuple([a[:][:7530] for a in test_data])
        self.tst_data = tuple([a[:][7530:] for a in test_data])
            
    def get_adult_data_2(self):
        
        raw_train_data = pd.read_csv('./datasets/adult/adult.data.txt', header=None, sep=', ').values
        raw_test_data = pd.read_csv('./datasets/adult/adult.test.txt', header=None, sep=', ').values
        
        train_data = remove_question(raw_train_data)
        test_data = remove_dot(remove_question(raw_test_data))
        labels = gather_labels(train_data)
        
        train_data = transform_to_binary_adult_2(train_data, labels)
        test_data = transform_to_binary_adult_2(test_data, labels)
        
        self.trn_data = tuple([d.astype(np.float32) for d in train_data])
        test_data = tuple([d.astype(np.float32) for d in test_data])
        self.val_data = tuple([a[:][:7530] for a in test_data])
        self.tst_data = tuple([a[:][7530:] for a in test_data])
        
    def get_health_data(self):
        d = pd.read_csv('./datasets/health/health.csv')
        d = d[d['YEAR_t'] == 'Y3']
        sex = d['sexMISS'] == 0
        age = d['age_MISS'] == 0
        d = d.drop(['DaysInHospital', 'MemberID_t', 'YEAR_t'], axis=1)
        d = d[sex & age]

        ages = d[['age_%d5' % (i) for i in range(0, 9)]]
        sexs = d[['sexMALE', 'sexFEMALE']]
        charlson = d['CharlsonIndexI_max']

        x = d.drop(
            ['age_%d5' % (i) for i in range(0, 9)] + ['sexMALE', 'sexFEMALE', 'CharlsonIndexI_max', 'CharlsonIndexI_min',
                                                      'CharlsonIndexI_ave', 'CharlsonIndexI_range', 'CharlsonIndexI_stdev',
                                                      'trainset'], axis=1).values
        labels = gather_labels(x)
        xs = np.zeros_like(x)
        for i in range(len(labels)):
            xs[:, i] = x[:, i] > labels[i]
        x = xs[:, np.nonzero(np.mean(xs, axis=0) > 0.05)[0]].astype(np.float32)

        u = np.expand_dims(sexs.values[:, 0], 1)
        v = ages.values
        u = np.concatenate([v, u], axis=1).astype(np.float32)
        u = np.argmax(u[:,:9],axis=1) + 9 * (u[:,9] == 1).astype(np.float32)
        u = np.expand_dims(u, axis=1)
        enc = OneHotEncoder().fit(u)
        u = enc.transform(u).toarray().astype(np.float32)
        y = np.expand_dims((charlson.values > 0), 1).astype(np.float32)

        cf = int(0.8 * d.shape[0])
        self.trn_data = (x[:cf], u[:cf], y[:cf])
        self.val_data = (x[cf:cf+5000], u[cf:cf+5000], y[cf:cf+5000])
        self.tst_data = (x[cf+5000:], u[cf+5000:], y[cf+5000:])
        
    def get_credit_data(self):
        rawdata = pd.read_excel('./datasets/credit/default_clients.xls', header=1)
        rawdata = rawdata.sample(frac=1.0, random_state=12345678).reset_index(drop=True)

        columns = list(rawdata.columns)
        categ_cols = []
        for column in columns:
            if 2 < len(set(rawdata[column])) < 10:
                categ_cols.append((column, len(set(rawdata[column]))))

        preproc_data = copy.deepcopy(rawdata)
        for categ_col, n_items in categ_cols:
            for i in range(n_items):
                preproc_data[categ_col + str(i)] = (preproc_data[categ_col] == i).astype(float)
        preproc_data = preproc_data.drop(['EDUCATION', 'MARRIAGE'], axis=1)

        X = preproc_data.drop(['ID', 'SEX', 'default payment next month'], axis=1)
        Y = preproc_data['default payment next month']
        U = 2 - preproc_data['SEX']
        
        X = X.to_numpy(dtype=np.float64)
        X = normalize(X)
        U = np.expand_dims(U.to_numpy(dtype=np.float64), axis=1)
        Y = np.expand_dims(Y.to_numpy(dtype=np.float64), axis=1)

        X_trn = X[:20000]
        U_trn = U[:20000]
        Y_trn = Y[:20000]

        X_val = X[20000:25000]
        U_val = U[20000:25000]
        Y_val = Y[20000:25000]
        
        X_tst = X[25000:30000]
        U_tst = U[25000:30000]
        Y_tst = Y[25000:30000]
        
        self.trn_data = (X_trn, U_trn, Y_trn)
        self.val_data = (X_val, U_val, Y_val)
        self.tst_data = (X_tst, U_tst, Y_tst)
        
    def get_law_data(self):
        rawdata = pd.read_sas('./datasets/law/lawschs1_1.sas7bdat')
        rawdata = rawdata.drop(['college', 'Year', 'URM', 'enroll'], axis=1)
        rawdata = rawdata.dropna(axis=0)
        rawdata = rawdata.sample(frac=1.0, random_state=12345678).reset_index(drop=True)

        X = rawdata[['LSAT', 'GPA', 'Gender', 'resident']]
        U = rawdata['White']
        Y = rawdata['admit']
        
        X = X.to_numpy(dtype=np.float64)
        X = normalize(X)
        U = np.expand_dims(U.to_numpy(dtype=np.float64), axis=1)
        Y = np.expand_dims(Y.to_numpy(dtype=np.float64), axis=1)

        X_trn = X[:77267]
        U_trn = U[:77267]
        Y_trn = Y[:77267]

        X_val = X[77267:85000]
        U_val = U[77267:85000]
        Y_val = Y[77267:85000]
        
        X_tst = X[85000:96584]
        U_tst = U[85000:96584]
        Y_tst = Y[85000:96584]
        
        self.trn_data = (X_trn, U_trn, Y_trn)
        self.val_data = (X_val, U_val, Y_val)
        self.tst_data = (X_tst, U_tst, Y_tst)
    
    def get_moon_data_1(self):
        n_train = 10000
        n_val = 5000
        n_test = 5000
        X, Y = make_moons(n_samples=n_train+n_val+n_test, noise=0.2, random_state=0)
        U = np.zeros((n_train+n_val+n_test,1))
        
        np.random.seed(0)
        for i in range(n_train + n_val + n_test):
            if Y[i] == 0:
                if -0.734 < X[i][0] < 0.734:
                    U[i] = np.random.binomial(1, 0.9)
                else:
                    U[i] = np.random.binomial(1, 0.35)
            elif Y[i] == 1:
                if 0.262 < X[i][0] < 1.734:
                    U[i] = np.random.binomial(1, 0.55)
                else:
                    U[i] = np.random.binomial(1, 0.1)
        
        X = pd.DataFrame(X, columns=['x_1', 'x_2'])
        Y = pd.DataFrame(Y, columns=['y'])
        U = pd.DataFrame(U, columns=['u'])
        
        X_trn = X.loc[list(range(n_train)), :]
        Y_trn = Y.loc[list(range(n_train)), :]
        U_trn = U.loc[list(range(n_train)), :]
        
        X_val = X.loc[list(range(n_train,n_train+n_val)), :]
        Y_val = Y.loc[list(range(n_train,n_train+n_val)), :]
        U_val = U.loc[list(range(n_train,n_train+n_val)), :]
        
        X_tst = X.loc[list(range(n_train+n_val,n_train+n_val+n_test)), :]
        Y_tst = Y.loc[list(range(n_train+n_val,n_train+n_val+n_test)), :]
        U_tst = U.loc[list(range(n_train+n_val,n_train+n_val+n_test)), :]
        
        self.trn_data = (X_trn, U_trn, Y_trn)
        self.val_data = (X_val, U_val, Y_val)
        self.tst_data = (X_tst, U_tst, Y_tst)
        
    def get_moon_data_2(self):
        n_train = 10000
        n_val = 5000
        n_test = 5000
        X, Y = make_moons(n_samples=n_train+n_val+n_test, noise=0.2, random_state=0)
        U = np.zeros((n_train+n_val+n_test,3))

        np.random.seed(0)
        for i in range(n_train + n_val + n_test):
            if Y[i] == 0:
                U[i] = np.random.multinomial(1, [0.2, 0.3, 0.5])
            elif Y[i] == 1:
                U[i] = np.random.multinomial(1, [0.5, 0.2, 0.3])

        X = pd.DataFrame(X, columns=['x_1', 'x_2'])
        Y = pd.DataFrame(Y, columns=['y'])
        U = pd.DataFrame(U, columns=['u_1', 'u_2', 'u_3'])
        
        X_trn = X.loc[list(range(n_train)), :]
        Y_trn = Y.loc[list(range(n_train)), :]
        U_trn = U.loc[list(range(n_train)), :]
        
        X_val = X.loc[list(range(n_train,n_train+n_val)), :]
        Y_val = Y.loc[list(range(n_train,n_train+n_val)), :]
        U_val = U.loc[list(range(n_train,n_train+n_val)), :]
        
        X_tst = X.loc[list(range(n_train+n_val,n_train+n_val+n_test)), :]
        Y_tst = Y.loc[list(range(n_train+n_val,n_train+n_val+n_test)), :]
        U_tst = U.loc[list(range(n_train+n_val,n_train+n_val+n_test)), :]
        
        self.trn_data = (X_trn, U_trn, Y_trn)
        self.val_data = (X_val, U_val, Y_val)
        self.tst_data = (X_tst, U_tst, Y_tst)
        
    def get_dataset_in_tensor(self, batch_size=64):

        trn = tf.data.Dataset.from_tensor_slices(self.trn_data).batch(batch_size).prefetch(batch_size)
        val = tf.data.Dataset.from_tensor_slices(self.val_data).batch(batch_size).prefetch(batch_size)
        tst = tf.data.Dataset.from_tensor_slices(self.tst_data).batch(batch_size).prefetch(batch_size)

        return trn, val, tst
        
def remove_question(df):
        idx = np.ones([df.shape[0]], dtype=np.int32)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                try:
                    if '?' in df[i, j]:
                        idx[i] = 0
                except TypeError:
                    pass
        df = df[np.nonzero(idx)]
        return df

def remove_dot(df):
    for i in range(df.shape[0]):
        df[i, -1] = df[i, -1][:-1]
    return df

def gather_labels(df):
    labels = []
    for j in range(df.shape[1]):
        if type(df[0, j]) is str:
            labels.append(np.unique(df[:, j]).tolist())
        else:
            labels.append(np.median(df[:, j]))
    return labels

def transform_to_binary_adult_1(df, labels):
    d = np.zeros([df.shape[0], 102])
    u = np.zeros([df.shape[0], 1])
    y = np.zeros([df.shape[0], 1])
    idx = 0
    for j in range(len(labels)):
        if type(labels[j]) is list:
            if labels[j][0] == 'Female':
                for i in range(df.shape[0]):
                    u[i][0] = int(labels[j].index(df[i, j]))
            elif len(labels[j]) > 2:
                for i in range(df.shape[0]):
                    d[i, idx + int(labels[j].index(df[i, j]))] = 1
                idx += len(labels[j])
            else:
                for i in range(df.shape[0]):
                    y[i] = int(labels[j].index(df[i, j]))
        else:
            for i in range(df.shape[0]):
                d[i, idx] = float(df[i, j] > labels[j])
            idx += 1
    return d.astype(np.float32), u.astype(np.float32), y.astype(np.float32)

def transform_to_binary_adult_2(df, labels):
    d = np.zeros([df.shape[0], 97])
    u = np.zeros([df.shape[0], 2])
    y = np.zeros([df.shape[0], 1])
    idx = 0
    for j in range(len(labels)):
        if type(labels[j]) is list:
            if labels[j][0] == 'Female':
                for i in range(df.shape[0]):
                    u[i][0] = int(labels[j].index(df[i, j]))
            elif labels[j][0] == 'Amer-Indian-Eskimo':
                for i in range(df.shape[0]):
                    if df[i, j] in ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other']:
                        u[i][1] = 0
                    elif df[i, j] == 'White':
                        u[i][1] = 1
            elif len(labels[j]) > 2:
                for i in range(df.shape[0]):
                    d[i, idx + int(labels[j].index(df[i, j]))] = 1
                idx += len(labels[j])
            else:
                for i in range(df.shape[0]):
                    y[i] = int(labels[j].index(df[i, j]))
        else:
            for i in range(df.shape[0]):
                d[i, idx] = float(df[i, j] > labels[j])
            idx += 1

    u = u[:,0] + 2 * (u[:,1] == 1).astype(np.float32)
    u = np.expand_dims(u, axis=1)
    enc = OneHotEncoder().fit(u)
    u = enc.transform(u).toarray()
    return d.astype(np.float32), u.astype(np.float32), y.astype(np.float32)

def normalize(X):
    scaler_X = StandardScaler()
    return scaler_X.fit_transform(X)
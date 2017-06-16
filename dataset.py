import pandas as pd

class Quora:

    def __init__(self, fdata='./datas/train.csv', test_ratio=.2):
        self.fdata = fdata
        self.test_ratio = test_ratio
        self.train_x = None
        self.train_t = None
        self.test_x = None
        self.test_t = None

    def load(self):
        '''
        x = (sen1, sen2)
        y = 0 or 1
        '''
        data = pd.read_csv(self.fdata).dropna()
        test_ratio = self.test_ratio
        duplicated = [tuple(x) for x in data[(data.is_duplicate)==1][['question1', 'question2']].values]
        not_duplicated = [tuple(x) for x in data[(data.is_duplicate)==0][['question1', 'question2']].values]
        
        train_idx_dup = int(len(duplicated)*(1-test_ratio))
        train_idx_notdup = int(len(not_duplicated)*(1-test_ratio))
        
        self.train_x = duplicated[:train_idx_dup] + not_duplicated[:train_idx_notdup]
        self.train_t = [1] * len(duplicated[:train_idx_dup]) + [0] * len(not_duplicated[:train_idx_notdup])
        
        self.test_x = duplicated[train_idx_dup:] + not_duplicated[train_idx_notdup:]
        self.test_t = [1] * len(duplicated[train_idx_dup:]) + [0] * len(not_duplicated[train_idx_notdup:])
        
    def get(self, small=False):
        if self.train_x is None:
            self.load()
        if small:
            return (self.train_x[:100], self.train_t[:100], self.test_x[:20], self.test_t[:20])
        return (self.train_x, self.train_t, self.test_x, self.test_t)

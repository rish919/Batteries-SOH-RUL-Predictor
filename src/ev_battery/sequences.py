"""Optional CPU PyTorch LSTM; windows never bridge battery IDs or missing cycles."""
from copy import deepcopy
import random
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SETTINGS = {'window':5, 'hidden_size':16, 'max_epochs':50, 'patience':7,
            'batch_size':64, 'learning_rate':.003, 'min_windows':20}


def sequence_indices(frame: pd.DataFrame, window: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Positional indices for contiguous causal windows and their endpoint labels."""
    if window<1: raise ValueError('Window length must be positive')
    positions = frame.reset_index(drop=True); windows = []; ends = []
    for _, group in positions.groupby('battery_id', sort=True):
        group = group.sort_values('cycle_index')
        ix=group.index.to_numpy(); cycles=group.cycle_index.to_numpy()
        for end in range(window-1,len(group)):
            start=end-window+1
            if np.all(np.diff(cycles[start:end+1])==1):
                windows.append(ix[start:end+1]); ends.append(ix[end])
    return np.array(windows,dtype=int).reshape(-1,window), np.array(ends,dtype=int)


def torch_runtime(seed: int):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    return torch


def network(n_features: int, hidden: int):
    import torch
    class LSTMRegressor(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.lstm=torch.nn.LSTM(n_features,hidden,batch_first=True)
            self.head=torch.nn.Linear(hidden,1)
        def forward(self,x): return self.head(self.lstm(x)[0][:,-1,:]).squeeze(-1)
    return LSTMRegressor()


def train_network(X: np.ndarray, y: np.ndarray, seed: int, epochs: int,
                  validation: tuple[np.ndarray,np.ndarray] | None = None):
    torch=torch_runtime(seed); model=network(X.shape[2],SETTINGS['hidden_size'])
    optimizer=torch.optim.Adam(model.parameters(),lr=SETTINGS['learning_rate'])
    loss_fn=torch.nn.MSELoss(); tx=torch.tensor(X,dtype=torch.float32); ty=torch.tensor(y,dtype=torch.float32)
    rng=np.random.default_rng(seed); best=float('inf'); best_state=None; best_epoch=epochs; stale=0
    for epoch in range(1,epochs+1):
        model.train()
        for batch in np.array_split(rng.permutation(len(X)),max(1,int(np.ceil(len(X)/SETTINGS['batch_size'])))):
            optimizer.zero_grad(); loss=loss_fn(model(tx[batch]),ty[batch]); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
        if validation is not None:
            model.eval()
            with torch.no_grad():
                vx,vy=validation
                score=float(loss_fn(model(torch.tensor(vx,dtype=torch.float32)),torch.tensor(vy,dtype=torch.float32)))
            if score<best-1e-6: best=score; best_state=deepcopy(model.state_dict()); best_epoch=epoch; stale=0
            else: stale+=1
            if stale>=SETTINGS['patience']: break
    if best_state is not None: model.load_state_dict(best_state)
    return model,best_epoch


class SequenceModel:
    """Predict raw windows; preprocessing is learned on training endpoints/windows only."""
    def __init__(self, seed: int = 42):
        self.seed=seed
        self.preprocessor=Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),('scaler',StandardScaler())])
        self.target_scaler=StandardScaler()

    def fit(self, frame: pd.DataFrame, features: list[str], label: str, epochs: int):
        windows,ends=sequence_indices(frame,SETTINGS['window'])
        if len(ends)<SETTINGS['min_windows']: raise ValueError('Fewer than 20 training windows')
        self.features=features
        # Unique rows used by windows, no held-out battery and no padded data.
        self.preprocessor.fit(frame.iloc[np.unique(windows)][features])
        X=self.preprocessor.transform(frame[features])[windows]
        y=self.target_scaler.fit_transform(frame.iloc[ends][[label]]).ravel()
        self.model,_=train_network(X,y,self.seed,epochs)
        return self

    def predict_windows(self, raw: np.ndarray) -> np.ndarray:
        import torch
        shape=raw.shape
        X=self.preprocessor.transform(pd.DataFrame(raw.reshape(-1,shape[-1]),columns=self.features)).reshape(shape)
        self.model.eval()
        with torch.no_grad(): p=self.model(torch.tensor(X,dtype=torch.float32)).numpy()
        return self.target_scaler.inverse_transform(p.reshape(-1,1)).ravel()


def choose_epochs(frame: pd.DataFrame, features: list[str], label: str, seed: int) -> tuple[int,dict]:
    counts={bid:len(sequence_indices(g)[1]) for bid,g in frame.groupby('battery_id')}
    eligible=sorted(k for k,v in counts.items() if v>=5)
    if len(eligible)<3:
        return 25, {'method':'fixed_epochs','reason':'fewer than three training batteries with >=5 windows','epochs':25}
    # Deterministic choice from OUTER TRAINING batteries only.
    val_id=eligible[seed % len(eligible)]
    tr=frame[frame.battery_id!=val_id].reset_index(drop=True); va=frame[frame.battery_id==val_id].reset_index(drop=True)
    tw,te=sequence_indices(tr); vw,ve=sequence_indices(va)
    if len(te)<SETTINGS['min_windows']: return 25, {'method':'fixed_epochs','reason':'too few inner training windows','epochs':25}
    prep=Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),('scaler',StandardScaler())])
    prep.fit(tr.iloc[np.unique(tw)][features]); ys=StandardScaler().fit(tr.iloc[te][[label]])
    _,epochs=train_network(prep.transform(tr[features])[tw],ys.transform(tr.iloc[te][[label]]).ravel(),seed,SETTINGS['max_epochs'],
                          (prep.transform(va[features])[vw],ys.transform(va.iloc[ve][[label]]).ravel()))
    return epochs, {'method':'training_battery_early_stopping_then_refit','validation_groups':[val_id],
                    'training_groups':sorted(tr.battery_id.unique().tolist()),'epochs':epochs}

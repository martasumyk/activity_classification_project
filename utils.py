import numpy as np


def create_windows(df, window_size, step_size, feature_cols, label_col="activity_id"):
    """
    Create overlapping windows per subject.
    For each window:
      - compute mean and std for every feature
      - assign the majority activity in the window as label
    """
    X = []
    y = []

    for subj_id, df_sub in df.groupby("subject"):
        data = df_sub[feature_cols].values
        labels = df_sub[label_col].values

        n = len(df_sub)
        start = 0
        while start + window_size <= n:
            end = start + window_size
            window = data[start:end]
            window_labels = labels[start:end]

            label = np.bincount(window_labels).argmax()

            feat_mean = window.mean(axis=0)
            feat_std = window.std(axis=0)
            feats = np.concatenate([feat_mean, feat_std])

            X.append(feats)
            y.append(label)

            start += step_size

    X = np.stack(X)
    y = np.array(y, dtype=int)
    return X, y
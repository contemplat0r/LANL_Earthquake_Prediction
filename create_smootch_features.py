import numpy as np
import pandas as pd



def features_maker(df, first_index=None, last_index=None, smootch_windows_size = (3, 5, 7)):
    if first_index == None or last_index == None:
        first_index = 0
        last_index = df.shape[0] - 1

    smooth_feature_names = ['smooth_feature_{}_ws_{}'.format(i, window_size) for i, window_size in enumerate(smootch_windows_size)]
    for feature_name in smooth_feature_names:
        df[feature_name] = 0
    for i in df.index:
        for smooth_feature_name, window_size in zip(smooth_feature_names, smootch_windows_size):
            half_window_size = window_size // 2
            data_series = df['acoustic_data']
            if i < first_index + half_window_size:
                smooth_feature_value = data_series.iloc[first_index:first_index + window_size].mean()
            elif i < last_index - half_window_size:
                smooth_feature_value = data_series.iloc[last_index - window_size:last_index].mean()
            else:
                smooth_feature_value = data_series.iloc[i - half_window_size:i + half_window_size].mean()
            df.iloc[i][feature_name] = data_series[i] - smooth_feature_value
    return df


def add_features(
        df,
        first_index=None,
        last_index=None,
        sample_size=150000,
        holdout_size=50000,
        smootch_windows_size = (3, 5, 7)
    ):

    smootch_feature_names = ['smootch_feature_{}_ws_{}'.format(i, window_size) for i, window_size in enumerate(smootch_windows_size)]
    half_windows_size = [ws // 2 for ws in smootch_windows_size]
    sample_indexes = np.random.randint(first_index, last_index, sample_size)
    data_series = df.iloc[sample_indexes]['acoustic_data']
    sample_df = df.iloc[sample_indexes]
    for feature_name in smootch_feature_names:
        sample_df[feature_name] = 0
    begin_smootch_features_value = []
    end_smootch_features_value = []
    for h_w_size, feature_name in zip(half_windows_size, smootch_feature_names):
        begin_indexes = sample_indexes[:h_w_size]
        end_indexes = sample_indexes[-h_w_size:]
        if begin_indexes:
            for i, b_idx in enumerate(begin_indexes):
                begin_smootch_features_value.append(df[b_idx]['acoustic_data'] - df.iloc[first_index:first_index + window_size]['acoustic_data'].mean())
                sample_df.iloc[i][feature_name] = df[b_idx]['acoustic_data'] - df.iloc[first_index:first_index + window_size]['acoustic_data'].mean()
        if end_indexes:
            for i, e_idx in end_indexes:
                end_smootch_features_value.append(df[e_idx]['acoustic_data'] - df.iloc[last_index - window_size:last_index]['acoustic_data'].mean())
                sample_df.iloc[sample_df.shape[0] - i][feature_name] = df[e_idx]['acoustic_data'] - df.iloc[first_index:first_index + window_size]['acoustic_data'].mean()
    return sample_df

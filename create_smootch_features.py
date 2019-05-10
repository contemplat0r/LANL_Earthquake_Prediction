def features_maker(df, first_index=None, last_index=None, smootch_windows_size = (3, 5, 7)):
    if first_index == None or last_index == None:
        first_index = 0
        last_index = df.shape[0] - 1

    smooth_feature_names = ['smooth_feature_{}_ws_{}'.format(i, window_size) for i, window_size in enumerate(smootch_windows_size)]
    for current_index in df.indexes.tolist():
        for smooth_feature_name, window_size in zip(smooth_feature_names, smootch_windows_size):
            half_window_size = window_size % 2
            data_series = df['acoustic_data']
            if current_index < first_index + half_window_size:
                smooth_feature_value = data_series.iloc[first_index:first_index + window_size].mean()
            elif current_index < last_index - half_window_size:
                smooth_feature_value = data_series.iloc[last_index - window_size:last_index].mean()
            else:
                smooth_feature_value = data_series.iloc[i - half_window_size:i + half_window_size].mean():
            df.iloc[i][feature_name] = data_series[i] - smooth_feature_value
    return df

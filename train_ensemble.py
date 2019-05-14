
def train_models(earthquake_margin_indexes, complete_earthquakes_length, sample_size, holdout_size):
    for i in complete_earthquakes_length:
        earthquake_df = pd.read_csv(
                '../input/train/train.csv',
                #nrows=100000000,
                names=['acoustic_data', 'time_to_failure'],
                dtype={'acoustic_data': np.float32, 'time_to_failure': np.float32},
                skiprows=earthquake_margin_indexes[i],
                nrows=complete_earthquakes_length[i]
            )
        earthquake_add_features_df, holdout_add_features_df = add_features(
                earthquake_df,
                sample_size=sample_size,
                holdout_size=holdout_size
            )
        X_all = earthquake_add_features_df[earthquake_add_features_df.columns.drop('time_to_failure')]
        y_all = earthquake_add_features_df['time_to_failure']

        X_train, X_valid, y_train, y_valid = train_test_split(X_all, y_all, test_size=0.2, random_state=0)

        model = lgb.LGBMRegressor(**params, n_estimators = 20000, n_jobs = 10, num_iterations=40000)

	model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                eval_metric='mae',
                verbose=1000,
                early_stopping_rounds=4000
            )
        X_holdout = holdout_df[holdout_df.columns.drop('time_to_failure')]
        y_holdout = holdout_df['time_to_failure']
        
        y_holdout_predict = model.predict(X_holdout)
        print("earthquake {} mae {}".format(i, mean_absolute_error(y_holdout, y_holdout_predict)))

        not_seen_data_predict = model.predict(not_seen_data_df)
        not_seen_data_predict_df = pd.DataFrame({'time_to_failure': not_seen_data_predict})
        not_seen_data_predict_df.to_csv('not_seend_data_earthquake_{}_model_predict.csv', index=False)

        model.save_model('earthquake_{}_model'.format(i))

    return

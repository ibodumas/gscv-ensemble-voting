def preprocess_train():
    """
    Load and pre-process train data.
    Implemented the following:
        - removed redundant variables
        - removed variables with high missing ratio
        - handle missing values by imputation
        - processed and incorporated the text codes columns
            - it was extended to multiple variables
        - One-hot encoded the categorical variables (ignored ordering)
        - implemented [0, 1] normalization

    :return: X_data, y_data
    """

    ######### TRAIN #########
    X_data = pd.read_csv("data.csv")
    y_data = X_data.class_label

    # id and diagnosis_date will not be used for modeling
    X_data.drop(["id", "date", "class_label"], axis=1, inplace=True)

    # - calculate % of missing values
    perc_missing = pd.DataFrame(
        {"perc_missing_values": X_data.isnull().mean() * 100}
    ).sort_values("perc_missing_values")

    perc_missing.query("perc_missing_values > 0", inplace=True)
    perc_missing

    perc_missing.iloc[:, 0].plot(
        kind="bar", title="% of missing values per variables", figsize=(15, 10)
    )

    ## The following variables will be dropped,
    # because over 50% of them are missing X6 and K6
    X_data.drop(["X6", "K6"], axis=1, inplace=True)

    # Filling missing values by imputation via most_frequent
    imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    imputer.fit(X_data)

    X_data = pd.DataFrame(imputer.transform(X_data), columns=X_data.columns)

    # working on coded variable (semi structured in nature)
    symp_data = pd.DataFrame(X_data.codes.values, columns=["code"])
    X_data.drop(["codes"], axis=1, inplace=True)
    symp_unique = sorted(list(set(chain(*symp_data.sym.str.split(",").tolist()))))
    processed_codes = process_text_feature(symp_data, symp_unique)

    ## One-hot encoding for the categorical variables
    cat_var = ["t_score", "n_score", "m_score", "stage", "race", "side"]
    data_encode = X_data[cat_var]
    encoder = OneHotEncoder(handle_unknown="ignore")
    encoder.fit(data_encode)

    data_onehot = pd.DataFrame(
        encoder.transform(data_encode).toarray(),
        columns=encoder.get_feature_names()
    )

    ## drop encoded variables from X_data and add data_onehot
    ## and processed codes X_data to it
    X_data.drop(cat_var, axis=1, inplace=True)
    X_data = pd.concat([X_data, data_onehot, processed_codes], axis=1)

    scaler = MinMaxScaler()
    scaler.fit(X_data)
    X_data = scaler.fit_transform(X_data)

    return X_data, y_data

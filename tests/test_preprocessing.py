def test_data_cleaning():
    from src.data_preprocessing import DataPreprocessor
    dp = DataPreprocessor('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    dp.clean_data()
    assert 'customerID' not in dp.data.columns

def test_feature_engineering():
    from src.data_preprocessing import DataPreprocessor
    dp = DataPreprocessor('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    dp.clean_data().feature_engineering()
    assert 'gender_Male' in dp.data.columns

# More tests for API validity, model loading, etc.

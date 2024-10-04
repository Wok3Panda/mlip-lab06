import pytest
import pandas as pd
import numpy as np
from prediction_demo import data_preparation,data_split,train_model,eval_model

@pytest.fixture
def housing_data_sample():
    return pd.DataFrame(
      data ={
      'price':[13300000,12250000],
      'area':[7420,8960],
    	'bedrooms':[4,4],	
      'bathrooms':[2,4],	
      'stories':[3,4],	
      'mainroad':["yes","yes"],	
      'guestroom':["no","no"],	
      'basement':["no","no"],	
      'hotwaterheating':["no","no"],	
      'airconditioning':["yes","yes"],	
      'parking':[2,3],
      'prefarea':["yes","no"],	
      'furnishingstatus':["furnished","unfurnished"]}
    )

def test_data_preparation(housing_data_sample):
    feature_df, target_series = data_preparation(housing_data_sample)
    # Target and datapoints has same length
    assert feature_df.shape[0]==len(target_series)

    #Feature only has numerical values
    assert feature_df.shape[1] == feature_df.select_dtypes(include=(np.number,np.bool_)).shape[1]

@pytest.fixture
def feature_target_sample(housing_data_sample):
    feature_df, target_series = data_preparation(housing_data_sample)
    return (feature_df, target_series)

def test_data_split(feature_target_sample):
    # Unpack the features and target from the fixture
    features, target = feature_target_sample

    # Call the data_split function
    X_train, X_test, y_train, y_test = data_split(features, target)

    # Check that four objects are returned
    assert len((X_train, X_test, y_train, y_test)) == 4

    # Check that the returned objects have the correct type
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

    # Check that the total number of samples is consistent
    total_samples = len(features)
    assert len(X_train) + len(X_test) == total_samples
    assert len(y_train) + len(y_test) == total_samples

    # Check the split proportions
    test_size = 0.33  # As specified in data_split function
    expected_test_samples = int(total_samples * test_size)
    expected_train_samples = total_samples - expected_test_samples

    # Allow for possible rounding differences
    assert abs(len(X_test) - expected_test_samples) <= 1
    assert abs(len(X_train) - expected_train_samples) <= 1
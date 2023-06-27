
!git clone https://github.com/ipeirotis/autoencoders_census.git

# Commented out IPython magic to ensure Python compatibility.
# %cd autoencoders_census

"""## Code that transforms a dataframe to vector format and vice versa

Transform and reverse transform the data, allowing for preprocessing and postprocessing steps in pipelines. It provides functionality to handle missing values, encode categorical variables, and scale numeric variables.
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from pandas.api.types import is_numeric_dtype

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from pandas.api.types import is_numeric_dtype
import numpy as np
import pandas as pd

class DataTransformer:
    """
    Class for transforming data for machine learning.

    This class handles transformations like one-hot encoding for categorical data,
    min-max scaling for numerical data, and handling missing data.
    """

    def __init__(self, variable_types):
        """Initialize the transformer with the variable types dictionary."""
        self.variable_types = variable_types
        self.one_hot_encoders = {}
        self.min_max_scalers = {}

    def transform_dataframe(self, original_df):
        """
        Transform the dataframe according to the variable types.

        Categorical variables are one-hot encoded, numeric variables are min-max scaled,
        and missing values are replaced with dummy variables.

        Returns:
        - The transformed dataframe.
        - Dictionaries with fitted OneHotEncoders and MinMaxScalers for each column.
        """
        df = original_df.copy()

        for column, variable_type in self.variable_types.items():
            if variable_type == 'categorical':
                one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                df_encoded = pd.DataFrame(one_hot_encoder.fit_transform(df[[column]]))
                df_encoded.columns = [f"{column}_{cat}" for cat in one_hot_encoder.categories_[0]]
                df = pd.concat([df, df_encoded], axis=1)
                df = df.drop(column, axis=1)
                self.one_hot_encoders[column] = one_hot_encoder
            elif variable_type == 'numeric' and is_numeric_dtype(df[column]):
                min_max_scaler = MinMaxScaler()
                non_na_rows = df[column].notna()
                df.loc[non_na_rows, column] = min_max_scaler.fit_transform(df.loc[non_na_rows, [column]]).ravel()
                self.min_max_scalers[column] = min_max_scaler

        return df

    @staticmethod
    def add_missing_indicators(df):
        """
        Adds binary columns to the dataframe indicating the presence of missing values.

        For each column in the dataframe, this function adds a corresponding column
        with a binary indicator of whether the value in that row is missing (NaN).
        These new columns are named 'missing_<column_name>' and are appended to the dataframe.

        Args:
            df (pd.DataFrame): The input pandas DataFrame.

        Returns:
            result (pd.DataFrame): The DataFrame with added missing value indicator columns.
        """

        column_prefix = 'missing_'

        # Create DataFrame with indicator of missing values
        df_missing = pd.concat([df[c].isnull().astype(int) for c in df.columns], axis = 1)
        df_missing.columns = [f'{column_prefix}{c}' for c in df.columns]

        # Concatenate the original DataFrame with the missing indicator DataFrame
        result = pd.concat([df, df_missing], axis='columns')

        return result

    @staticmethod
    def proba_to_onehot(proba):
        """Convert a vector of probabilities into a max-likelihood one-hot vector."""
        onehot = np.zeros_like(proba)
        onehot[np.arange(len(proba)), np.argmax(proba, axis=1)] = 1
        return onehot

    def reverse_transform_dataframe(self, transformed_df):
        """
        Reverse the transformations applied to the dataframe.

        One-hot encoded categorical variables are decoded and min-max scaled numeric variables
        are inverse scaled.

        Returns the original dataframe.
        """
        df = transformed_df.copy()

        for column, variable_type in self.variable_types.items():
            if variable_type == 'categorical':
                one_hot_encoder = self.one_hot_encoders[column]
                original_cols = [col for col in df.columns if col.startswith(f"{column}_")]
                df_proba = df[original_cols].values
                onehot = self.proba_to_onehot(df_proba)
                df_original = pd.DataFrame(one_hot_encoder.inverse_transform(onehot))
                df_original.columns = [column]
                df = pd.concat([df.drop(original_cols, axis=1), df_original], axis=1)
            elif variable_type == 'numeric' and is_numeric_dtype(df[column]):
                min_max_scaler = self.min_max_scalers[column]
                non_na_rows = df[column].notna()
                #df.loc[non_na_rows, column] = min_max_scaler.inverse_transform(df.loc[non_na_rows, [column]]).ravel()
                #df.loc[non_na_rows, column] = min_max_scaler.inverse_transform(df.loc[non_na_rows, [column]]).ravel().reshape(-1, 1)
                inverse_transformed = min_max_scaler.inverse_transform(df.loc[non_na_rows, [column]])
                df.loc[non_na_rows, column] = inverse_transformed.flatten()


        return df

import unittest

class TestDataTransformer(unittest.TestCase):
    def setUp(self):
        self.variable_types = {
            'age': 'numeric',
            'gender': 'categorical',
            'income': 'numeric'
        }
        self.transformer = DataTransformer(self.variable_types)
        self.data = pd.DataFrame({
            'age': [25, 30, 35, np.nan],
            'gender': ['male', 'female', 'male', 'female'],
            'income': [50000.0, 60000.0, 70000.0, 80000.0]
        })

    def test_add_missing_indicators(self):
        df = pd.DataFrame({
            'age': [25, 30, np.nan, 35],
            'gender': ['male', 'female', None, 'male'],
        })
        transformed_df = self.transformer.add_missing_indicators(df)

        # Check that the output DataFrame has twice as many columns as the input DataFrame
        self.assertEqual(transformed_df.shape[1], 2 * df.shape[1])

        # Check that the output DataFrame contains all of the columns of the input DataFrame
        self.assertTrue(set(df.columns).issubset(set(transformed_df.columns)))

        # Check that the added columns in the output DataFrame start with 'missing_'
        missing_cols = [col for col in transformed_df.columns if col.startswith('missing_')]
        self.assertEqual(len(missing_cols), df.shape[1])

        # Check that 'missing_' columns contain only 0s and 1s
        for col in missing_cols:
            self.assertTrue(set(transformed_df[col].unique()).issubset({0, 1}))

        # Check that the number of 1s in 'missing_' columns matches the number of NaN values in the original DataFrame
        for original_col in df.columns:
            missing_col = f'missing_{original_col}'
            self.assertEqual(transformed_df[missing_col].sum(), df[original_col].isnull().sum())


    def test_transform_dataframe(self):

        data_missing = self.transformer.add_missing_indicators(self.data)
        transformed_df = self.transformer.transform_dataframe(data_missing)

        # Check that original DataFrame has been transformed properly
        self.assertNotIn('gender', transformed_df.columns)
        self.assertIn('gender_male', transformed_df.columns)
        self.assertIn('gender_female', transformed_df.columns)

        # Check that missing values have been handled correctly
        self.assertEqual(transformed_df.loc[3, 'missing_age'], 1)
        self.assertEqual(transformed_df.loc[0, 'missing_age'], 0)

        # Check that numeric columns have been scaled correctly
        self.assertEqual(transformed_df.loc[0, 'age'], 0)
        self.assertEqual(transformed_df.loc[1, 'age'], 0.5)
        self.assertEqual(transformed_df.loc[2, 'age'], 1)
        self.assertTrue(np.isnan(transformed_df.loc[3, 'age']))

    def test_proba_to_onehot(self):
        proba = np.array([[0.1, 0.9], [0.7, 0.3]])
        expected_onehot = np.array([[0, 1], [1, 0]])

        np.testing.assert_array_equal(self.transformer.proba_to_onehot(proba), expected_onehot)

    def test_reverse_transform_dataframe(self):
        transformed_df = self.transformer.transform_dataframe(self.data)
        reversed_df = self.transformer.reverse_transform_dataframe(transformed_df)

        # Check that DataFrame has been reversed correctly
        pd.testing.assert_frame_equal(reversed_df, self.data, check_like=True)

        # Check that the missing data has been reversed correctly
        self.assertTrue(pd.isnull(reversed_df.loc[3, 'age']))

        # Check that the numeric scaling has been reversed correctly
        self.assertTrue('age' in reversed_df.columns)
        self.assertTrue('income' in reversed_df.columns)
        self.assertListEqual(list(self.data['age'].dropna()), list(reversed_df['age'].dropna()))
        self.assertListEqual(list(self.data['income']), list(reversed_df['income']))

        # Check that the categorical encoding has been reversed correctly
        self.assertTrue('gender' in reversed_df.columns)
        self.assertListEqual(list(self.data['gender']), list(reversed_df['gender']))

def run_tests(test_class):
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    unittest.TextTestRunner().run(suite)

run_tests(TestDataTransformer)


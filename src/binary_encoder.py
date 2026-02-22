import json
import pandas as pd
import numpy as np
import os


class BinaryEncoder:
    def __init__(self, encoder_file):
        self.encoder_file = encoder_file

    def fit(self, df:pd.DataFrame, cols:list):
        """
            Generate or update the file that contains the binary encoding information
        """
        data = dict()
        for col in cols:
            unique = sorted(df[col].unique())
            n_unique = len(unique)

            # Necessary number of bits to fit all data
            n_bits = 1
            while 2 ** n_bits < n_unique: n_bits += 1

            # Map each unique item into a binary representation
            mapper = dict()
            mapper["num_bits"] = n_bits

            for i, item in enumerate(unique):
                i_bin = format(i, f'0{n_bits}b')
                i_bin = list(i_bin)
                i_bin = [float(i) for i in i_bin]
                mapper[item] = i_bin

            mapper["unknown"] = [1.0] * n_bits
            
            data[col] = mapper

        # Write the mapping into the file
        with open(self.encoder_file, "w") as file:
            json.dump(data, file, indent=4)

    def transform(self, df:pd.DataFrame, cols, drop=False) -> pd.DataFrame:
        """
            Encode the column with the information from the encoder file
        """
        with open(self.encoder_file, "r") as file:
            data = json.load(file)

        # Creating new columns in the dataframe
        for col in cols:
            for i in range(data[col]["num_bits"]):
                df[f"{col}_{i}"] = 0


        for index, row in df.iterrows():
            for target_col in cols:
                encoded_data = data[target_col][row[target_col]]

                for i, bit in enumerate(encoded_data):
                    df.at[index, f'{target_col}_{i}'] = bit

        if drop:
            df = df.drop(cols, axis=1)

        return df



    def fit_transform(self, df:pd.DataFrame, cols:str):
        """
            Executes the fit and transform at the same time
        """
        self.fit(df, cols)
        df = self.transform(df, cols)
        return df

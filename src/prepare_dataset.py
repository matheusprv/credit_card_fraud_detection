import pandas as pd
import numpy as np
from value_mapper import ValueMapper
import random

class PrepareDataset:
    def __init__(self, data_file):
        self.df = self._read_csv(data_file)
        self.df = self._drop_dolumns(self.df)
        self.df = self._continuos_values_normalization(self.df)
        self.df = self._gender_encoding(self.df)
        self.df = self._normalize_time(self.df)
        self.df = self._normalize_date(self.df)
        self.df = self._categorics_to_numeric(self.df)
        self.df = self._distribution_stats(self.df)

    def _read_csv(self, data_file):
        df = pd.read_csv(data_file, delimiter="|")
        return df
    
    def _drop_dolumns(self, df):
        """
            Rows that do not have date or time will be deleted. Many fraudulent transactions happen in night periods.
        """
        nan_rows = df[df["trans_date"].isna() | df["trans_time"].isna()].index
        df = df.drop(nan_rows)


        cols_to_drop = [
            "profile",                      # Information not provided
            "trans_num",                    # unique value
            "first", "last",                # name of the person
            "ssn", "acct_num",              # High correlation - 200 unique values on training 
            "street", "dob",
            "unix_time"
        ]
        # "cc_num" will be kept to create a history of the credic card

        """
            zip, merch_lat and merch_long appear as NaN in a significant amount of rows
            Only 1468 rows are not NaN|
        """
        cols_to_drop += ["zip", "merch_lat", "merch_long"]
        df = df.drop(columns=cols_to_drop)

        return df
    
    def _continuos_values_normalization(self, df):
        df['amt'] = df['amt'] / 50_000.0
        df['city_pop'] = df['city_pop'] / 3_000_000.0
        df["lat"] = df["lat"] / 90.0
        df["long"] = df["long"] / 180.0
        return df

    def _gender_encoding(self, df):
        df["gender"] = df["gender"].map({"M": 1, "F": 0})
        return df
    
    def _normalize_time(self, df):
        """
            Changing time so it goes from [0, 1440[, representing the time of the day
            Fraudulent transactions happens more frequently on night periods
            Transform these numeric values into a cyclic value with sin and cos
            This will prevents continuity problems, such as ending a day with high value and start with low
        """
        df["trans_time"] = pd.to_datetime(df["trans_time"], format="%H:%M:%S")
        df["trans_time"] = df["trans_time"].dt.hour * 60 + df["trans_time"].dt.minute

        df["time_sin"] = np.sin(2 * np.pi * df["trans_time"] / 1440)
        df["time_cos"] = np.cos(2 * np.pi * df["trans_time"] / 1440)

        df = df.drop(columns=["trans_time"])

        return df
    
    def _normalize_date(self, df):
        """
            Changing the date to a numeric value in range [0, 372)
            All months are considered to have 31 days for code simplicity
            Commemorative days may have more fraudulent transactions due to the increase in the total number of transactions
        """
        df["trans_date"] = pd.to_datetime(df["trans_date"], format="%Y-%m-%d")
        df["trans_date"] = (df["trans_date"].dt.month - 1) * 31 + df["trans_date"].dt.day

        df["date_sin"] = np.sin(2 * np.pi * df["trans_date"] / 372)
        df["date_cos"] = np.cos(2 * np.pi * df["trans_date"] / 372)

        df = df.drop(columns=["trans_date"])

        return df
    
    def _categorics_to_numeric(self, df):
        columns = ['category', 'job', 'merchant', 'state', 'city']
        
        for col in columns:
            mapping = ValueMapper.read_mapping(col)
            df[col] = df[col].map(mapping)
        
        return df


    # Código otimizado pelo ChatGPT utiizando vetorização para acelerar o processamento
    def _distribution_stats(self, df):
        df = df.copy()
        df["cc_num"] = df["cc_num"].astype(str)

        cols = ['lat', 'long', 'amt', 'time_sin', 'time_cos', 'date_sin', 'date_cos']

        # IMPORTANTE: se "cumulativo" depende do tempo, ordene antes
        # df = df.sort_values(["cc_num", "trans_date_trans_time"])  # exemplo

        g = df.groupby("cc_num", sort=False)

        # contador cumulativo por grupo (1,2,3,...)
        n = g.cumcount() + 1

        for col in cols:
            csum  = g[col].cumsum()
            csum2 = g[col].apply(lambda s: (s.astype(float)**2).cumsum()).reset_index(level=0, drop=True)

            mean = csum / n
            var  = csum2 / n - mean**2
            var  = var.clip(lower=0)  # evita negativo por erro numérico

            df[f"{col}_mean"] = mean
            df[f"{col}_std"]  = np.sqrt(var)   # ddof=0 (igual np.std default)

        # df = df.drop(columns=["cc_num"])

        return df

    # Código original
    # def _distribution_stats(self, df):
    #     df["cc_num"] = df["cc_num"].astype(str)

    #     cols = ['lat', 'long', 'amt', 'time_sin', 'time_cos', 'date_sin', 'date_cos']
        
    #     for col in cols:
    #         df[f"{col}_mean"] = np.nan
    #         df[f"{col}_std"] = np.nan

    #     cc_infos = {
    #         k: {
    #             c: []
    #             for c in cols
    #         }
    #         for k in df["cc_num"].unique().tolist()
    #     }

    #     print(len(list(cc_infos.keys())))
    #     print(list(cc_infos.keys()))
        
    #     for index, row in df.iterrows():
    #         cc = row["cc_num"]
    #         for col in cols:
    #             cc_infos[cc][col].append(row[col])
    #             temp_arr = np.array(cc_infos[cc][col])
    #             df.loc[index, f"{col}_mean"] = np.mean(temp_arr)
    #             df.loc[index, f"{col}_std"] = np.std(temp_arr)

    #     return df

    def train_val_test_split(self, df):
        """
            Split based on the cc_num to prevent data leakage between train and evaluation steps
        """

        cc_nums = df["cc_num"].unique().tolist()
    
        rng = random.Random("inteligência computacional") 
        rng.shuffle(cc_nums)
        
        cc_nums_test = cc_nums[:20]
        cc_nums_val = cc_nums[20:40]
        cc_nums_train = cc_nums[40:]


        df_train = df[df["cc_num"].isin(cc_nums_train)]
        df_val = df[df["cc_num"].isin(cc_nums_val)]
        df_test = df[df["cc_num"].isin(cc_nums_test)]


        df_train = df_train.drop(columns=["cc_num"])
        df_val = df_val.drop(columns=["cc_num"])
        df_test = df_test.drop(columns=["cc_num"])


        return df_train, df_val, df_test

import pandas as pd
from binary_encoder import BinaryEncoder

def drop_columns(df):
    """
        Rows that do not have date or time will be deleted. Many fraudulent transactions happen in night periods.
    """
    nan_rows = df[df["trans_date"].isna() | df["trans_time"].isna()].index
    df = df.drop(nan_rows)


    cols_to_drop = [
        "profile",                      # Information not provided
        "trans_num",                    # unique value
        "first", "last",                # name of the person
        "ssn", "acct_num", "cc_num",    # High correlation - 200 unique values on training - 
        "street", "dob",
        "unix_time"
    ]

    """
        zip, merch_lat and merch_long appear as NaN in a significant amount of rows
        Only 1468 rows are not NaN|
    """
    cols_to_drop += ["zip", "merch_lat", "merch_long"]
    df = df.drop(columns=cols_to_drop)

    return df


def categorize_city(df):
    """
        Instead of using the city name or the exact population number, we will classify cities into three categories based on their population size:

        Small: < 100_000       A
        Medium: < 1_000_000    B
        Big: >= 1_000_000      C

        Using the city name and defining an unique numerical value to each, some would have higher weights than others. 
        One-Hot Encoding would create too many columns, making the dataset sparse.
        The city population has a great amplitute, and its use could ofuscate some values
    """

    df["city_class"] = "B"
    df.loc[df['city_pop']  < 100_000, 'city_class'] = 'A'
    df.loc[df['city_pop'] >= 1_000_000, 'city_class'] = 'C'

    df['city_class'] = pd.Categorical(df["city_class"], categories=["A", "B", "C"])
    df = pd.get_dummies(df, columns=["city_class"], prefix=["city_class"], drop_first=True, dtype=int)

    df = df.drop(["city_pop", "city"], axis=1)

    return df


def normalize_time(df):
    """
        Changing time so it goes from [0, 1440[, representing the time of the day
        Fraudulent transactions may happen on smaller and higher values
    """
    df["trans_time"] = pd.to_datetime(df["trans_time"], format="%H:%M:%S")
    df["trans_time"] = df["trans_time"].dt.hour * 60 + df["trans_time"].dt.minute

    df["trans_time"] = df["trans_time"] / 1440.0

    return df


def normalize_date(df):
    """
        Changing the date to a numeric value in range [0, 372)
        All months are considered to have 31 days for code simplicity
        Commemorative days may have more fraudulent transactions due to the increase in the total number of transactions
    """
    df["trans_date"] = pd.to_datetime(df["trans_date"], format="%Y-%m-%d")
    df["trans_date"] = (df["trans_date"].dt.month - 1) * 31 + df["trans_date"].dt.day

    df["trans_date"] = df["trans_date"] / 372.0

    return df

def continuous_values_normalization(df):
    df["lat"] = df["lat"] / 180.0
    df["long"] = df["long"] / 180.0
    df["amt"] = df["amt"] / 50_000.0
    return df

def gender_encoding(df):
    df["gender"] = df["gender"].map({"M": 1, "F": 0})
    return df

def input_output_split(df:pd.DataFrame):
    Y = df["is_fraud"]
    X = df.drop(["is_fraud"], axis=1, inplace=False)
    return X.to_numpy(), Y.to_numpy()

def prepare(data_file, encoder_file="../data/data_encoder.json", train=True):
    df: pd.DataFrame = pd.read_csv(data_file, delimiter="|")

    df = drop_columns(df)
    df = categorize_city(df)
    df = normalize_time(df)
    df = normalize_date(df)
    df = continuous_values_normalization(df)
    df = gender_encoding(df)

    be = BinaryEncoder(encoder_file)
    if train:
        be.fit(df, ['category', 'job', 'merchant', 'state'])

    df = be.transform(df, ['category', 'job', 'merchant', 'state'], drop=True)

    return input_output_split(df), df
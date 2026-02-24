import json
import pandas as pd

try:
    import google.colab
    google_colab = True
except ImportError:
    google_colab = False


class ValueMapper:

    def generate_file_name(column):
        if google_colab:
            return f"./data/categoric/{column}.json" 
        else:
            return f"../data/categoric/{column}.json"

    def class2num(df:pd.DataFrame, column:str):
        unique_values = df[column].unique().tolist()
        unique_values = sorted(unique_values)
        
        data = {
            k: i
            for i, k in enumerate(unique_values)
        }
        data["UNKOWN"] = len(unique_values)

        file = ValueMapper.generate_file_name(column)
        with open(file, "w") as file:
            json.dump(data, file, indent=4)

    def read_mapping(column:str):
        file = ValueMapper.generate_file_name(column)
        with open(file, "r") as file:
            data = json.load(file)
        return data
import json
import pandas as pd
from os import listdir
from os.path import isfile, join


def read_file(dir_path=''):
    """"
        The purpose of this code is to read json files from a given path and
        return the information in a pandas data frame
    """

    # Read the files from the directory path provided
    only_files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    data = {}
    for file in only_files:
        with open(f"{dir_path}{file}") as f:
            bible = json.load(f)
        data.update(bible)

    # Take the files from the dictionary and turn it into a dataframe
    list_of_df = []
    for version in data.keys():
        content = []
        for item in data[version].items():
            ref, text = item

            book = ' '.join(ref.split()[:-1]).lower().replace(' ', '_')
            chapter = ref.split()[-1].split(":")[0]
            verse = ref.split()[-1].split(":")[1]

            content.append((version.lower(), book, chapter, verse, text))

        df = pd.DataFrame(data=content, columns=['version', 'book', 'chapter', 'verse', 'text'])
        list_of_df.append(df)

    df = pd.concat(list_of_df)

    return df
from env import user, password, host
import os   # to check for .csv file on my device (local)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Goals:
# We want to create a wrangle file for mall customer data
# Wrangle includes both acquisition and preparation,
# it would also be nice to have summarization of our
# data in there as well.
# so lets do that!

def acquire_mall(
    username = username,
    password = password,
    host = host,
    db = 'mall_customers'
    ) -> pd.DataFrame:
    """
    acquire_mall will make a request to our msql server associated
    with the credentials takeb from an imported env.py
    please check credentials and access to the mall_customers
    schema before interacting with the acquisition script
    
    return: a single pandas datqframe with all mall customer data
    """
    if os.path.exists('./mall_customers.csv'):
        return pd.read_csv('mall_customers.csv', index_col=0)     # returns csv file, if found
# index_col allow you to set which columns to be used as the index of the dataframe. 
# if index_col is 0, it means that "1" will be the index of the first column, "2" will be the index for the second column 
    else:
        connection = f'mysql+pymysql://{username}:{password}@{host}/{db}'
        query = 'SELECT * from customers'
        df = pd.read_sql(query, connection) # this read SQL query or database table into a DataFrame
                                            # using query and connection parameters defined above
        df.to_csv('mall_customers.csv')     # this writes object to a csv file
        return df                           # returns SQL query as a dataframe object

    
def missing_by_row(df):
    """
    missing_by_row will take in a pandas dataframe (df) and tell us 
    how many missing values in each row instead of each column
    """
    return pd.concat(                                  # chains pandas objects together (along a particular axis)
        [
            df.isna().sum(axis=1),                     # detect missing values along our vertical axis (columns) and sums them
            (df.isna().sum(axis=1) / df.shape[1])
        ], axis=1).rename(                             # renames columns as missing_cells and percent_missing
        columns={0:'missing_cells', 1:'percent_missing'}     
    ).groupby(                              # .groupby is grouping the data points (i.e. rows) based on the column values
        ['missing_cells',
         'percent_missing']
    ).count().reset_index().rename(columns = {'index': 'num_mising'})

# .count sums how many non-NA (includes None, NaN, NaT) cells for each column or row
# .reset_index reverts the dataframe to the default index


def summarize(df) -> None:
    """
    Summarize will take in a dataframe and report out statistics
    regarding the dataframe to the console.
    
    this will include:
     - the shape of the dataframe
     - the info reporting on the dataframe
     - the descriptive stats on the dataframe
     - missing values by column
     - missing values by row
    
    """
    print('--------------------------------')
    print('--------------------------------')
    print('Information on DataFrame: ')
    print(f'Shape of Dataframe: {df.shape}')
    print('--------------------------------')
    print(f'Basic DataFrame info:')
    print(df.info())
    print('--------------------------------')
    # print out continuous descriptive stats
    print(f'Continuous Column Stats:')
    print(df.describe().to_markdown())                      
    print('--------------------------------')
    # print out objects/categorical stats:
    print(f'Categorical Column Stats:')
    print(df.select_dtypes('O').describe().to_markdown())     # prints dataframe in Markdown-friendly format
    print('--------------------------------')
    print('Missing Values by Column: ')
    print(df.isna().sum())
    print('Missing Values by Row: ')
    print(missing_by_row(df).to_markdown())
    print('--------------------------------')
    print('--------------------------------')


def prep_mall(df):
    """
    prep_mall will take in a single pandas dataframe and perform the
    necessary preparation steps in order to turn it into a clean df
    
    """
    #capture any missing values and handle them (impute, drop, etc)
    # conveniently no missing values on this specific one
    # rename some columns:
    # we have the arbitrary customer id field that appears to be an index
    # so lets mark it as such:
    df = df.set_index('customer_id')     # changes the dataframe index from default to the customer_id column
    return df
   
    
def split_data(df):
    """
    based on an input dataframe,
    we will split the information into train,
    validate and test data sets,
    and return all three dataframes
    """
    train_val, test = train_test_split(          # split arrays or matrices into random train (train_val) and test subsets.
        df,
        train_size=0.8,                          # represents the proportion of the dataset to include in the train split.
        random_state=1349)                       # controls the shuffling of data before it is split.
    train, validate = train_test_split(
        train_val,                               # train_val contains the train and validate data together before we split them
        train_size=0.7,                          # represents the proportion of the dataset to include in the train split. If
        random_state=1349)
    return train, validate, test
    
    
def preprocess_mall(df):                         # combines split_data and preprocess_mall functions? scaling too?
    '''                                          
    preprocess_mall will take in values in form of a single pandas dataframe
    and make the data ready for spatial modeling,
    including:
     - splitting the data
     - encoding categorical features
     - scaling information (continuous columns)

    return: three pandas dataframes, ready for modeling structures.
    '''
    # capture any missing values and handle them (impute, drop, etc)
    # conveniently no missing values on this specific one
    # rename some columns:
    # we have the arbitrary customer id field that appears to be an index
    # so lets mark it as such:
    # encode categoricals:
    df = df.assign(                                # must reassign as df=
        is_male= pd.get_dummies(                   # .assign adds new columns to the dataframe
            df['gender'], drop_first=True          # drop_first = True helps in reducing the extra column created during dummy                                                          # variable creation.
        ).astype(int).values)
    # drop original gender col:
    df = df.drop(columns='gender')                 # must reassign as df=
    # split data:
    train, validate, test = split_data(df)
    # scale continuous features:
    scaler = MinMaxScaler()
    train = pd.DataFrame(
        scaler.fit_transform(train),
        index=train.index,
        columns=train.columns)
    validate = pd.DataFrame(
        scaler.transform(validate),
        index=validate.index,
        columns=validate.columns
    )
    test = pd.DataFrame(
        scaler.transform(test),
        index=test.index,
        columns=test.columns)
    return train, validate, test
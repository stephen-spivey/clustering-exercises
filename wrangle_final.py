from env import user, password, host
import os   # to check for .csv file on my device (local)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# CHECK FOR ADDITIONAL IMPORTS


# Acquire Functions
# SOME FUNCTIONS (e.g. acquire_mall, prep_mall) NEED TO BE MODIFIED FOR EACH DATASET

def get_db_url(db, username=env.username, host=env.host, password=env.password):
    return f'mysql+pymysql://{username}:{password}@{host}/{db}'


def acquire_mall(
    username = username,
    password = password,
    host = host,
    db = 'mall_customers'
    ) -> pd.DataFrame:
    """
    acquire_mall will make a request to our msql server associated
    with the credentials taken from an imported env.py
    please check credentials and access to the mall_customers
    schema before interacting with the acquisition script
    
    return: a single pandas dataframe with all mall customer data
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
        return df        


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
    

# Prepare Functions

def prep_mall(df):
    """
    prep_mall will take in a single pandas dataframe and perform the
    necessary preparation steps in order to turn it into a clean df
    
    """
    df = df.set_index('customer_id')     # changes the dataframe index from default to the customer_id column
    return df
    
# Interquartile Range (Outliers)
# NOTE: ADD FUNCTIONS FOR LOWER OUTLIERS
# DO I DO THIS BEFORE OR AFTER SPLITTING DATA

def get_upper_outliers(s, k):
    '''
    Given a series and a cutoff value, k, returns the upper outliers for the
    series.

    The values returned will be either 0 (if the point is not an outlier), or a
    number that indicates how far away from the upper bound the observation is.
    '''
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    upper_bound = q3 + k * iqr
    return s.apply(lambda x: max([x - upper_bound, 0]))


def add_upper_outlier_columns(df, k):
    '''
    Add a column with the suffix _outliers for all the numeric columns
    in the given dataframe.
    '''
    # outlier_cols = {col + '_outliers': get_upper_outliers(df[col], k)
    #                 for col in df.select_dtypes('number')}
    # return df.assign(**outlier_cols)

    for col in df.select_dtypes('number'):
        df[col + '_outliers'] = get_upper_outliers(df[col], k)

    return df

add_upper_outlier_columns(df, k=1.5)

df.head()
        
    
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
    
    
# Interquartile Range (Outliers)
# NOTE: ADD FUNCTIONS FOR LOWER OUTLIERS
# DO I DO THIS BEFORE OR AFTER SPLITTING DATA

def get_upper_outliers(s, k):
    '''
    Given a series and a cutoff value, k, returns the upper outliers for the
    series.

    The values returned will be either 0 (if the point is not an outlier), or a
    number that indicates how far away from the upper bound the observation is.
    '''
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    upper_bound = q3 + k * iqr
    return s.apply(lambda x: max([x - upper_bound, 0]))

def add_upper_outlier_columns(df, k):
    '''
    Add a column with the suffix _outliers for all the numeric columns
    in the given dataframe.
    '''
    # outlier_cols = {col + '_outliers': get_upper_outliers(df[col], k)
    #                 for col in df.select_dtypes('number')}
    # return df.assign(**outlier_cols)

    for col in df.select_dtypes('number'):
        df[col + '_outliers'] = get_upper_outliers(df[col], k)

    return df

add_upper_outlier_columns(df, k=1.5)

df.head()
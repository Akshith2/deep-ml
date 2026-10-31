import pandas as pd

def clean_text(text):
    text = text.strip()
    text = text[0].upper() + text[1:].lower()
    return text

def solution(df):

    """Trim surrounding whitespace from name and convert each name to title case (first letter of each word uppercase, the rest lowercase)

    Convert date to the standardized string format YYYY-MM-DD.

    Remove duplicate rows, keeping the first occurrence, and reset the index to a clean 0-based range.

    Fill any remaining missing entries in value with the mean of the non-missing value entries that remain after step 3.
    
    """

    df['name'] = df['name'].apply(clean_text)
    #print(df)
    
    df['date'] = pd.to_datetime(df['date'].str.strip())
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    df['date'] = df['date'].astype(str)
    #print(df)
    
    df = df.drop_duplicates()
    #print(df)
    
    df.loc[:,'value'] = df['value'].fillna(df['value'].mean())
    #print(df)

    return df

    pass
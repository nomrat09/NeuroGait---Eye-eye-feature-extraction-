import pandas as pd

csv_file = 'blink_features_clean.csv'

df_raw = pd.read_csv(csv_file)
print('Total rows in raw CSV:', len(df_raw))
print('Subjects before dropna:', df_raw['subject'].nunique())

# Rows per subject before dropna
print('\nRows per subject before dropna:')
print(df_raw['subject'].value_counts())

df = df_raw.dropna()
print('\nTotal rows after dropna:', len(df))
print('Subjects after dropna:', df['subject'].nunique())

# Rows per subject after dropna
print('\nRows per subject after dropna:')
print(df['subject'].value_counts())

print('\nNumber of unique subjects per label (after dropna):')
print(df.groupby('label')['subject'].nunique())

missing_subjects = set(df_raw['subject'].unique()) - set(df['subject'].unique())
print('\nSubjects lost due to NaNs or filtering:', missing_subjects)
if missing_subjects:
    print('Rows for lost subjects (before dropna):')
    print(df_raw[df_raw['subject'].isin(missing_subjects)]) 
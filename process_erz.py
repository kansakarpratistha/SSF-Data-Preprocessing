import pandas as pd
import owncloud
from io import StringIO
import numpy as np
import argparse

def get_logger_config(oc, trench_name):
    if trench_name == 'T1':
        logger_link = '/2. Erzgebirge/Data/TrenchOutflow/Configuration files/TR1_Trench.log'
    elif trench_name == 'T2':
        logger_link = '/2. Erzgebirge/Data/TrenchOutflow/Configuration files/TR2_Trench.log'
    elif trench_name == 'T3':
        logger_link = '/2. Erzgebirge/Data/TrenchOutflow/Configuration files/TR3_Trench.log'
    loggerFile = (oc.get_file_contents(logger_link))
    logger_data = StringIO(loggerFile.decode("ISO-8859-1"))
    log_df = pd.read_csv(logger_data, header=None, on_bad_lines='skip', encoding='iso-8859-1', skiprows=26)
    log_df.columns = ['S.n', 'Measure', 'Flag1', 'Flag2', 'ID', 'Task']
    log_df = log_df[log_df['Measure'] == "Yes"]
    log_df["ActivityID"] = log_df["ID"] + "_" + log_df["Task"]
    return log_df

def get_trench_ts_link(trench_name):
    if trench_name == 'T1':
        return '/2. Erzgebirge/Data/TrenchOutflow/Trench 1'
    elif trench_name == 'T2':
        return '/2. Erzgebirge/Data/TrenchOutflow/Trench 2'
    elif trench_name == 'T3':
        return '/2. Erzgebirge/Data/TrenchOutflow/Trench 3'

def get_timeseries_df(oc, trench_ts_link, activity_id):
    folderContents = (oc.list(trench_ts_link))

    timeseries_df = pd.DataFrame()
    for file in folderContents:
        data = oc.get_file_contents(file)
        obs_data = StringIO(data.decode("ISO-8859-1"))
        obs_df = pd.read_csv(obs_data, decimal=",", header=None, sep='\t', on_bad_lines='skip')
        timeseries_df = pd.concat([timeseries_df, obs_df])
        dups = (timeseries_df.duplicated().sum())
        timeseries_df = timeseries_df.drop_duplicates()
        # print("dup rows: " + str(dups))
        # print("obs df: " + str(obs_df.shape))
        # print("merged df: " + str(processed_df.shape))

    column_names = ['Timestamp'] + activity_id
    timeseries_df.columns =  column_names
    timeseries_df[timeseries_df.columns.difference(['Timestamp'])] = timeseries_df[timeseries_df.columns.difference(['Timestamp'])].apply(
        pd.to_numeric, errors="coerce"
    )

    return timeseries_df

def handle_missing_flow_data(df, col_name):
    # for f90, f91: short interval and consistent bef-after values fill with that value
    # gaps upto 1 hour or upto 6 intervals: linear interpolation
    mask = df[col_name].isna()
    # print(mask)
    gaps = []

    filled_indices = []
    interpolated_indices = []

    start_index = None
    for i in df.index:
        if mask.iloc[i]: #missing index found
            if start_index is None: #gap starts
                start_index = i
        else: #index has value
            if start_index is not None: #gap ends
                gaps.append((start_index, i))
                start_index = None
    if start_index is not None: #gap till the end
        gaps.append((start_index, df.index[-1]))

    for start, end in gaps:
        gap_size = end - start
        prev_index, next_index = start-1, end

        if prev_index >= 0 and next_index < len(df):
            prev_val, next_val = df.iloc[prev_index][col_name], df.iloc[next_index][col_name]
            prev_time, next_time = df.iloc[prev_index]['Timestamp'], df.iloc[next_index]['Timestamp']
            if gap_size == 1 and prev_val == next_val:
                df.loc[start:end, col_name] = prev_val
                filled_indices.extend(range(start, end))
            elif gap_size <= 6 or (next_time - prev_time).seconds <= 3600:
                df.loc[start:end, col_name] = df.loc[start-1:end, col_name].interpolate()
                interpolated_indices.extend(range(start, end))
    #return df, filled gaps indices, interpolated gaps indices
    return {'updated-df': df, 'filled-gaps': filled_indices, 'interpolated-gaps': interpolated_indices}

def interpolate_missing_data(df, col_name):
    mask = df[col_name].isna()
    gaps = []
    interpolated_indices = []
    start_index = None
    for i in df.index:
        if mask.iloc[i]: #missing index found
            if start_index is None: #gap starts
                start_index = i
        else: #index has value
            if start_index is not None: #gap ends
                gaps.append((start_index, i))
                start_index = None
    if start_index is not None: #gap till the end
        gaps.append((start_index, df.index[-1]))

    for start, end in gaps:
        gap_size = end - start
        prev_index, next_index = start-1, end

        if prev_index >= 0 and next_index < len(df):
            prev_val, next_val = df.iloc[prev_index][col_name], df.iloc[next_index][col_name]
            prev_time, next_time = df.iloc[prev_index]['Timestamp'], df.iloc[next_index]['Timestamp']
            if gap_size <= 6 or (next_time - prev_time).seconds <= 3600:
                df.loc[start:end, col_name] = df.loc[start-1:end, col_name].interpolate()
                interpolated_indices.extend(range(start, end))
    return {'updated-df': df, 'interpolated-gaps': interpolated_indices}

def calibrate_flow(trench_name, df):
    if trench_name == 'T1':
        df['f90_Flow(ml)'] = df['f90_GetCounts!'] * 84
        df['f91_Flow(ml)'] = df['f91_GetCounts!'] * 94
    elif trench_name == 'T2':
        df['f90_Flow(ml)'] = df['f90_GetCounts!'] * 105
        df['f91_Flow(ml)'] = df['f91_GetCounts!'] * 82
        df['f92_Flow(ml)'] = df['f92_GetCounts!'] * 78
    elif trench_name == 'T3':
        df['f90_Flow(ml)'] = df['f90_GetCounts!'] * 72
        df['f91_Flow(ml)'] = df['f91_GetCounts!'] * 72
        df['f92_Flow(ml)'] = df['f92_GetCounts!'] * 86
        df['f93_Flow(ml)'] = df['f93_GetCounts!'] * 72
    return df

def remove_trench_flow_anomalies(df, flow_columns):
    df[flow_columns] = df[flow_columns].where((df[flow_columns] >= 0) & (df[flow_columns]/1000 <= 1000)) #acceptable range is 0 to 1000 litres. Column values are in ml.
    return df


def remove_ec_anomalies(df, ec_columns):
    df[ec_columns] = df[ec_columns].where((df[ec_columns] >= 0) & (df[ec_columns] <= 10000)) #upper range is 10000 or 1000? on readme: for flag 1: 1000 but in removing: 10000
    return df

def remove_temp_anomalies(df, temp_columns):
    df[temp_columns] = df[temp_columns].where((df[temp_columns] >= 0) & (df[temp_columns] <= 30)) #acceptable range is 0-30 degrees
    return df


def flag_trench_flow(timeseries_df, flag_df, flow_columns):
    for col in flow_columns:
        flag_df.loc[timeseries_df[col].isna(), col] = -999
    return flag_df

def flag_ec(timeseries_df, flag_df, ec_columns):
    for col in ec_columns:
        flag_df.loc[timeseries_df[col].isna(), col] = -999
    return flag_df

def flag_temp(timeseries_df, flag_df, temp_columns):
    for col in temp_columns:
        #Temp within acceptable range but corresponding EC value less than 10 then flag 0
        corresponding_ec_col = col.replace('GetTemperature', 'GetEC')
        flag_df.loc[(timeseries_df[corresponding_ec_col] < 10) & (timeseries_df[col]>=0) & (timeseries_df[col]<=30), col] = 0

        flag_df.loc[timeseries_df[col].isna(), col] = -999

    return flag_df

def flag_interval_flow(df_flags, flow_columns, trench_name):
    for col in flow_columns:
        if trench_name == 'T2':
            # T2: 2024-08-06 11:30:00 to 2024-08-06 15:30:00 flag 0
            start_time = pd.to_datetime('2024-08-06 11:30:00')
            end_time = pd.to_datetime('2024-08-06 15:30:00')
            df_flags.loc[(df_flags['Timestamp'] >= start_time) & (df_flags['Timestamp'] <= end_time), col] = 0
        elif trench_name == 'T3':
            # T3: 2024-08-05 13:00:00 to 2024-08-05 18:00:00 flag 0
            start_time = pd.to_datetime('2024-08-05 13:00:00')
            end_time = pd.to_datetime('2024-08-05 18:00:00')
            df_flags.loc[(df_flags['Timestamp'] >= start_time) & (df_flags['Timestamp'] <= end_time), col] = 0
    return df_flags

def main():
    parser = argparse.ArgumentParser(description="Process temperature data.")
    parser.add_argument('--trench', type=str, required=True, help="Trench name (eg. T1 for Trench 1)")
    args = parser.parse_args()
    trench_name = args.trench.upper()

    if trench_name not in ['T1', 'T2', 'T3']:
        print("Invalid trench name. Trench name should be one of T1, T2 or T3.")
        return

    trench_ts_link = get_trench_ts_link(trench_name)

    nc_link = "https://nextcloud.gfz.de/s/Ywp7jG9P2xbYaNY"
    oc = owncloud.Client.from_public_link(nc_link)
    # get logger configuration for the site
    print('Fetching logger configuration')
    log_df = get_logger_config(oc, trench_name)
    
    print('Reading and merging timeseries data')
    # get the merged timeseries data for the site with headers and timestamps  
    timeseries_df= get_timeseries_df(oc, trench_ts_link, log_df['ActivityID'].tolist()) 
    timeseries_df['Timestamp'] = pd.to_datetime(timeseries_df['Timestamp'], dayfirst=True)   
    timeseries_df.index = range(len(timeseries_df)) 
    timeseries_df.to_csv(f'Erzgebirge_{trench_name}_merged.csv', index=False)    

    
    # Flagging columns
    tips_columns = [c for c in timeseries_df.columns if 'GetCounts' in c]
    flow_columns = [c.replace('GetCounts!', 'Flow(ml)') for c in tips_columns]
    EC_columns = [c for c in timeseries_df.columns if 'GetEC' in c or 'GetEC25' in c]
    Temperature_columns = [c for c in timeseries_df.columns if 'GetTemperature' in c and c.startswith('ec')]

    # Create a flag dataframe
    flag_columns = flow_columns + EC_columns + Temperature_columns
    df_flags = timeseries_df[['Timestamp'] + EC_columns + Temperature_columns].copy()
    df_flags[flow_columns] = 1 #initialize all flow flags to 1
    df_flags = df_flags[['Timestamp'] + flag_columns] #reorder columns
    df_flags[flag_columns] = 1 #initialize all flags to 1

    # Handle missing data
    print('Handling missing data')
    for col in tips_columns:
        mod_obj = handle_missing_flow_data(timeseries_df, col)
        timeseries_df = mod_obj['updated-df']
        flow_filled_indices = mod_obj['filled-gaps']
        flow_interpolated_indices = mod_obj['interpolated-gaps']

        #flag filled data as 1 and interpolated data as 0
        flow_col_name = col.replace('GetCounts!', 'Flow(ml)')
        df_flags.loc[flow_filled_indices, flow_col_name] = 1
        df_flags.loc[flow_interpolated_indices, flow_col_name] = 0
    
    for col in EC_columns:
        mod_obj = interpolate_missing_data(timeseries_df, col)
        timeseries_df = mod_obj['updated-df']
        ec_interpolated_indices = mod_obj['interpolated-gaps']

        #flag interpolated EC data as 0
        df_flags.loc[ec_interpolated_indices, col] = 0
    
    for col in Temperature_columns:
        mod_obj = interpolate_missing_data(timeseries_df, col)
        timeseries_df = mod_obj['updated-df']
        temp_interpolated_indices = mod_obj['interpolated-gaps']

        #flag interpolated Temperature data as 0
        df_flags.loc[temp_interpolated_indices, col] = 0

    # Convert tips to mm
    calibrate_flow(trench_name, timeseries_df)     
    
    # Removing anomalies
    print('Removing anomalies')
    timeseries_df = remove_trench_flow_anomalies(timeseries_df, flow_columns)
    timeseries_df = remove_ec_anomalies(timeseries_df, EC_columns)
    timeseries_df = remove_temp_anomalies(timeseries_df, Temperature_columns) 

    print('Processing complete')
    timeseries_df.to_csv(f'Erzgebirge_{trench_name}_processed.csv', index=False)
    print(f'Processed data saved to Erzgebirge_{trench_name}_processed.csv')

    #do flagging
    print('Flagging data')
    
    # Trench flow flagging
    flag_df = flag_trench_flow(timeseries_df, df_flags, flow_columns)
    if (trench_name == 'T2' or trench_name == 'T3'): #for Trench 2 and 3, trench flow for data between specific timeframes are flagged 0 due to influence of artificial irrigation experiments, which affected flow magnitude. 
        flag_df = flag_interval_flow(df_flags, flow_columns, trench_name)
    # EC flagging
    flag_df = flag_ec(timeseries_df, flag_df, EC_columns)
    # Temperature flagging
    flag_df = flag_temp(timeseries_df, flag_df, Temperature_columns)

    print('Flagging complete')
    flag_df.to_csv(f'Erzgebirge_{trench_name}_flag.csv', index=False)
    print(f'Flagged data saved to Erzgebirge_{trench_name}_flag.csv')

if __name__ == "__main__":
    main()

    
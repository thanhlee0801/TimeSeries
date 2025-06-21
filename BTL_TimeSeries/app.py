
# --- DEBUGGING INITIAL DATA ---
print(f"DEBUG: Shape of df_original (after loading): {df_original.shape if 'df_original' in locals() else 'Not found'}")
print(f"DEBUG: Shape of df_cleaned_overall (after initial cleaning): {df_cleaned_overall.shape if 'df_cleaned_overall' in locals() else 'Not found'}")
if 'df_cleaned_overall' in locals() and not df_cleaned_overall.empty:
    print(f"DEBUG: df_cleaned_overall min date: {df_cleaned_overall.index.min()}")
    print(f"DEBUG: df_cleaned_overall max date: {df_cleaned_overall.index.max()}")
    print(f"DEBUG: df_cleaned_overall columns: {df_cleaned_overall.columns.tolist()}")
    if intensity_column_name not in df_cleaned_overall.columns:
        print(f"ERROR: intensity_column_name '{intensity_column_name}' not found in df_cleaned_overall columns!")
else:
    print("ERROR: df_cleaned_overall is not available or is empty early in the script.")

print(f"DEBUG: last_historical_date: {last_historical_date if 'last_historical_date' in locals() else 'Not found'}")
print(f"DEBUG: forecast_end_date_overall: {forecast_end_date_overall if 'forecast_end_date_overall' in locals() else 'Not found'}")
# --- END DEBUGGING INITIAL DATA ---

# --- 1. Import Libraries ---

# Thư viện cơ bản và xử lý dữ liệu
import pandas as pd
import numpy as np
import os
import sys # Để thoát chương trình
import re # Để xử lý regex, nếu cần trong tương lai
import pickle # Để lưu và tải scaler/model
from datetime import datetime, timedelta, date # date được thêm từ phần sau
from PIL import Image # Dành cho việc xử lý ảnh, ví dụ để tạo GIF

# Thư viện vẽ biểu đồ
import matplotlib.pyplot as plt

# Thư viện cho Prophet
from prophet import Prophet

# Thư viện cho LSTM (TensorFlow/Keras)
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping # Import EarlyStopping ở đây
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report, accuracy_score, roc_auc_score # Thêm các metrics từ phần sau

# Thư viện cho PatchTST (PyTorch)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset # TensorDataset được thêm từ phần sau
import torch.optim as optim
from sklearn.preprocessing import StandardScaler # StandardScaler được dùng cho PatchTST

print("Libraries imported successfully!")

# THAY ĐỔI: Định nghĩa device ở phạm vi toàn cục
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {device}")


# --- 1. Import Libraries ---

# Thư viện cơ bản và xử lý dữ liệu
import pandas as pd
import numpy as np
import os
import sys # Để thoát chương trình
import re # Để xử lý regex, nếu cần trong tương lai
import pickle # Để lưu và tải scaler/model
from datetime import datetime, timedelta, date # date được thêm từ phần sau
from PIL import Image # Dành cho việc xử lý ảnh, ví dụ để tạo GIF

# Thư viện vẽ biểu đồ
import matplotlib.pyplot as plt

# Thư viện cho Prophet
from prophet import Prophet

# Thư viện cho LSTM (TensorFlow/Keras)
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping # Import EarlyStopping ở đây
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report, accuracy_score, roc_auc_score # Thêm các metrics từ phần sau

# Thư viện cho PatchTST (PyTorch)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset # TensorDataset được thêm từ phần sau
import torch.optim as optim
from sklearn.preprocessing import StandardScaler # StandardScaler được dùng cho PatchTST

print("Libraries imported successfully!")

# THAY ĐỔI: Định nghĩa device ở phạm vi toàn cục
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {device}")


# --- 3. Load and Preprocess Data ---
# Đường dẫn đến bộ dữ liệu Kaggle đã thêm vào
dataset_path = "/kaggle/input/digital-typhoon-dataset-wp-gifs/"
output_dir = "/kaggle/working/" # Thư mục để lưu file CSV tổng

# --- CẤU HÌNH CÁC NĂM MUỐN DỰ BÁO VÀ HIỂN THỊ TRONG BÁO CÁO ---
forecast_years = list(range(2026, 2031))
print(f"Dự báo sẽ được tạo cho các năm: {forecast_years}")
# forecast_end_date_overall now defines the end of the *test set*
forecast_end_date_overall = pd.to_datetime(f'{max(forecast_years)}-12-31')

# --- LOGIC MỚI: HỢP NHẤT TẤT CẢ CÁC FILE CSV VÀ TẠO FILE TỔNG VỚI CỘT THỜI GIAN ĐÃ GHÉP ---
# We'll now have a historical combined CSV and potentially a future test CSV
combined_historical_csv_path = os.path.join(output_dir, "combined_typhoon_historical_data.csv")
combined_future_test_csv_path = os.path.join(output_dir, "combined_typhoon_future_test_data.csv")

all_dataframes_raw = [] # Will store raw dataframes before any resampling
common_intensity_cols = ['usa_wind', 'wind', 'vmax', 'intensity_kt', 'wmo_wind', 'max_sustained_wind']

print(f"\nSearching and combining all suitable CSV files from: {dataset_path}")

found_any_data = False

for root, _, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(root, file)
            file_name_only = os.path.basename(file)
            print(f"  Processing: {file_path}")
            try:
                df_temp = pd.read_csv(file_path, skipinitialspace=True, low_memory=False)
                df_temp.columns = df_temp.columns.str.strip().str.lower()
                
                missing_time_cols = [col for col in ['year', 'month', 'day', 'hour'] if col not in df_temp.columns]
                if missing_time_cols:
                    print(f"    -> Skipped {file_name_only}: Missing one or more of required time columns: {missing_time_cols}")
                else:
                    df_temp['time'] = pd.to_datetime(df_temp[['year', 'month', 'day', 'hour']], errors='coerce')
                    initial_rows = len(df_temp)
                    df_temp = df_temp.dropna(subset=['time'])
                    rows_after_time_dropna = len(df_temp)

                    current_intensity_col = None
                    for col in common_intensity_cols:
                        if col in df_temp.columns:
                            current_intensity_col = col
                            break
                    
                    if df_temp.empty:
                        print(f"    -> Skipped {file_name_only}: All rows dropped due to invalid 'time' data (initial rows: {initial_rows}, after dropping invalid time: {rows_after_time_dropna}).")
                    elif not current_intensity_col:
                        print(f"    -> Skipped {file_name_only}: No suitable intensity column found among {common_intensity_cols}.")
                    else:
                        df_temp['source_file'] = file_name_only
                        
                        if current_intensity_col != 'intensity_kt':
                            df_temp.rename(columns={current_intensity_col: 'intensity_kt'}, inplace=True)
                        
                        cols_to_drop_time = [col for col in ['year', 'month', 'day', 'hour'] if col in df_temp.columns]
                        
                        # Keep all columns (excluding original time columns)
                        df_to_append = df_temp.drop(columns=cols_to_drop_time, errors='ignore').copy()

                        all_dataframes_raw.append(df_to_append)
                        found_any_data = True
                        print(f"    -> Added {len(df_to_append)} rows from {file_name_only} (Combined year/month/day/hour, using '{current_intensity_col}', kept all other columns and 'source_file').")

            except Exception as e:
                print(f"    -> Could not read or process {file_path}: {e}")

df_combined_historical = pd.DataFrame() # This will be the full historical data

if all_dataframes_raw:
    print("\nConcatenating all collected raw dataframes for historical data...")
    # Concatenate all raw dataframes. Do NOT drop duplicates by 'time' as per requirement.
    df_combined_historical = pd.concat(all_dataframes_raw, ignore_index=True)
    df_combined_historical = df_combined_historical.sort_values(by='time')
    
    # Set 'time' as index
    df_combined_historical = df_combined_historical.set_index('time')

    # Convert numeric columns and handle NaNs for historical data
    # Identify numeric columns including the renamed 'intensity_kt'
    numeric_cols_historical = df_combined_historical.select_dtypes(include=np.number).columns.tolist()
    
    # Apply to_numeric on identified numeric columns
    for col in numeric_cols_historical:
        if col in df_combined_historical.columns:
            df_combined_historical[col] = pd.to_numeric(df_combined_historical[col], errors='coerce')
    
    # Drop rows where critical numeric columns are NaN (e.g., intensity_kt)
    # You might want to be more specific here if certain NaNs are acceptable in historical data
    df_combined_historical = df_combined_historical.dropna(subset=['intensity_kt'])
    
    # Fill remaining NaNs for other numeric columns if desired (e.g., mean, median, 0, or ffill/bfill)
    # For now, let's keep them if they exist, or drop them if they are not critical.
    # If other numeric columns are important for models, consider more robust imputation.
    # df_combined_historical = df_combined_historical.ffill().bfill() # Be careful, this can spread data

    # Ensure source_file column is handled (it's already string/object)
    if 'source_file' in df_combined_historical.columns:
        # If there are any NaNs in source_file after concat (unlikely if added per row)
        # We can fill them with a placeholder if needed, but no resampling here.
        if df_combined_historical['source_file'].isnull().any():
             print("INFO: Found NaNs in 'source_file' in historical data. Filling with 'unknown_source_hist'.")
             df_combined_historical['source_file'].fillna('unknown_source_hist', inplace=True)

    # Save combined historical DataFrame
    if not df_combined_historical.empty:
        df_combined_historical.to_csv(combined_historical_csv_path, index=True)
        print(f"Successfully combined and saved historical data to: {combined_historical_csv_path}")
        print(f"Total rows in combined historical CSV: {len(df_combined_historical)}")
        print(f"Columns in combined historical CSV: {df_combined_historical.columns.tolist()}")
        print(f"Data types in combined historical CSV:\n{df_combined_historical.dtypes}")
    else:
        print("Combined historical DataFrame is empty after processing. Skipping saving to CSV.")
        found_any_data = False

else:
    print("\nNo suitable CSV files found or processed from the dataset for historical data. Cannot proceed without data.")
    found_any_data = False
    sys.exit(1) # Exit if no historical data

# --- Tải dữ liệu từ file CSV tổng đã tạo (historical) ---
df_for_patchtst = pd.DataFrame()
target_series = pd.Series()
intensity_column_name = 'intensity_kt'

if found_any_data and os.path.exists(combined_historical_csv_path) and not df_combined_historical.empty:
    try:
        print(f"\nLoading data from combined historical CSV: {combined_historical_csv_path}")
        df_loaded_historical = pd.read_csv(combined_historical_csv_path, index_col='time', parse_dates=True, skipinitialspace=True, low_memory=False)
        
        # Ensure intensity_kt column exists (should be handled during initial merge)
        if 'intensity_kt' not in df_loaded_historical.columns:
            # This case should ideally not happen if initial processing was correct
            # Re-check for common intensity cols and rename if necessary
            found_intensity_col = None
            for col in common_intensity_cols:
                if col in df_loaded_historical.columns:
                    found_intensity_col = col
                    break
            if found_intensity_col:
                intensity_column_name = found_intensity_col
                print(f"Warning: 'intensity_kt' not found in historical CSV after loading. Using '{intensity_column_name}'.")
                df_loaded_historical.rename(columns={found_intensity_col: 'intensity_kt'}, inplace=True)
                intensity_column_name = 'intensity_kt'
            else:
                raise ValueError("No 'intensity_kt' or other suitable intensity column found in historical CSV after loading.")
        
        df_for_patchtst = df_loaded_historical.copy() # This is your primary historical data source
        target_series = df_for_patchtst[intensity_column_name].dropna()

        print(f"\nSuccessfully loaded and processed historical data from combined CSV.")
        print(f"Using '{intensity_column_name}' as the primary intensity column.")
        print(f"Processed historical data for PatchTST (first 5 rows):")
        print(df_for_patchtst.head())
        print(f"Total historical data points for PatchTST: {len(df_for_patchtst)}")
        print(f"Total historical data points for Prophet/LSTM ({intensity_column_name}): {len(target_series)}")

    except Exception as e:
        print(f"\nError loading or processing combined historical CSV from {combined_historical_csv_path}: {e}")
        print("Fatal error: Cannot proceed without valid historical data.")
        sys.exit(1)
else:
    print("\nFatal error: No combined historical CSV found or it is empty after initial processing. Cannot proceed without data.")
    sys.exit(1)

# --- CHUẨN BỊ DỮ LIỆU CHUNG CHO DỰ BÁO DÀI HẠN VÀ TẠO TEST SET ---

# Define the current date (end of historical data to be used for training)
current_datetime = pd.Timestamp('2024-12-31 23:59:59') # Fixed current date for consistent results

# Ensure historical data doesn't go into the future
last_historical_date_in_data = df_for_patchtst.index.max()
if last_historical_date_in_data > current_datetime:
    print(f"Clipping historical data. Last data point was {last_historical_date_in_data}, but setting end to {current_datetime}")
    df_for_patchtst = df_for_patchtst[df_for_patchtst.index <= current_datetime].copy()
    target_series = target_series[target_series.index <= current_datetime].copy()
    # Re-check max date after clipping
    last_historical_date_in_data = df_for_patchtst.index.max()


print(f"\nLast historical data point for training/validation: {last_historical_date_in_data}")

# --- TẠO TẬP DỮ LIỆU TEST TỪ 2024 ĐẾN 2030 (6H MỘT LẦN) ---
# Start from 2024-01-01 if historical data goes earlier, or from last_historical_date_in_data + 6h
# If current_datetime is 2024-12-31, then the test set starts from 2025-01-01 00:00:00
test_start_date = current_datetime + pd.Timedelta(hours=6) # Start immediately after historical data ends

print(f"\nGenerating future test set from {test_start_date} to {forecast_end_date_overall} at 6-hour intervals.")

# Create the date range for the future test set
future_dates = pd.date_range(start=test_start_date, end=forecast_end_date_overall, freq='6h')

# Initialize future_test_df with just the 'time' index
df_future_test = pd.DataFrame(index=future_dates)

# Add all relevant columns found in historical data to the future test set
# These will be NaNs initially, as they are for future predictions
for col in df_for_patchtst.columns:
    if col != 'source_file': # source_file is not a feature for forecasting
        df_future_test[col] = np.nan # Initialize with NaN for prediction

# If 'source_file' is desired in the future test set, you'd need a rule for it.
# For now, it's typically not a feature used in forecasting the future.

# Save the future test set (can be empty values for now, will be filled by models)
if not df_future_test.empty:
    df_future_test.to_csv(combined_future_test_csv_path, index=True)
    print(f"Successfully generated and saved future test data to: {combined_future_test_csv_path}")
    print(f"Total rows in future test CSV: {len(df_future_test)}")
    print(f"Columns in future test CSV: {df_future_test.columns.tolist()}")
    print(f"Data types in future test CSV:\n{df_future_test.dtypes}")
else:
    print("Future test DataFrame is empty. Skipping saving to CSV.")

print("\nData loading and preprocessing complete.")
print("Historical data (df_for_patchtst) is ready for training/validation (no 6h resampling).")
print("Future test data (df_future_test) is generated at 6h intervals for predictions.")

# --- 5. Run Prophet Model ---
print("\n--- Running Prophet Model ---")
df_prophet = pd.DataFrame({'ds': target_series.index, 'y': target_series.values})
m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
m.fit(df_prophet)

# Tạo future dataframe kéo dài đến ngày kết thúc dự báo tổng thể
future_prophet_overall = m.make_future_dataframe(periods=int((forecast_end_date_overall - last_historical_date).total_seconds() / (12 * 3600)) + 1, freq='12h')
forecast_prophet_overall = m.predict(future_prophet_overall)
print("Prophet overall forecast completed.")

# --- 6. Run LSTM Model ---
print("\n--- Running LSTM Model ---")

# Khởi tạo các biến dự báo và cờ thực thi
lstm_forecast = pd.DataFrame() # Đảm bảo lstm_forecast luôn được định nghĩa
lstm_executed = False # Khởi tạo cờ điều khiển

try:
    # Dữ liệu cho LSTM phải là chuỗi thời gian liên tục và chỉ bao gồm cột cường độ
    # Đảm bảo target_series đã được chuẩn bị và không có NaN
    target_series_lstm = target_series.copy().dropna()

    # Giới hạn dữ liệu lịch sử cho LSTM để tránh quá tải và tăng tốc độ huấn luyện
    # Rút gọn số lần thử: Giảm số điểm dữ liệu đầu vào
    # Ví dụ: 50,000 điểm dữ liệu cuối cùng (khoảng 34 năm)
    # Hoặc 20,000 điểm để chạy nhanh hơn nữa
    max_lstm_data_points = 50000 

    if len(target_series_lstm) > max_lstm_data_points:
        print(f"Limiting LSTM data from {len(target_series_lstm)} to {max_lstm_data_points} points for faster training.")
        target_series_lstm = target_series_lstm.tail(max_lstm_data_points)

    if target_series_lstm.empty:
        raise ValueError("LSTM: Target series is empty after preprocessing or limiting data. Cannot run LSTM model.")

    # Scale data
    scaler_lstm = MinMaxScaler(feature_range=(0, 1))
    scaled_data_lstm = scaler_lstm.fit_transform(target_series_lstm.values.reshape(-1, 1))

    # Prepare data for LSTM (create sequences)
    # Giảm time_step để giảm độ phức tạp và tăng tốc độ
    time_step_lstm = 45 # Number of previous time steps to look at (e.g., 45 means 45 * 6 hours = 270 hours = ~11 days)
    
    X_lstm, y_lstm = [], []
    for i in range(len(scaled_data_lstm) - time_step_lstm):
        X_lstm.append(scaled_data_lstm[i:(i + time_step_lstm), 0])
        y_lstm.append(scaled_data_lstm[i + time_step_lstm, 0])
    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

    # Reshape input to be [samples, time_steps, features]
    X_lstm = X_lstm.reshape(X_lstm.shape[0], X_lstm.shape[1], 1)

    # Train-test split (use a specific split point for time series)
    train_size_lstm = int(len(X_lstm) * 0.8)
    X_train_lstm, X_test_lstm = X_lstm[0:train_size_lstm,:], X_lstm[train_size_lstm:len(X_lstm),:]
    y_train_lstm, y_test_lstm = y_lstm[0:train_size_lstm], y_lstm[train_size_lstm:len(y_lstm)]

    print(f"LSTM Train data shape: {X_train_lstm.shape}")
    print(f"LSTM Test data shape: {X_test_lstm.shape}")

    # Build LSTM Model with Dropout
    model_lstm = tf.keras.models.Sequential([
        tf.keras.Input(shape=(time_step_lstm, 1)), # Explicit Input layer
        tf.keras.layers.LSTM(50, return_sequences=True),
        tf.keras.layers.Dropout(0.2), # Dropout layer 1 (20% of neurons are dropped)
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dropout(0.2), # Dropout layer 2
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])

    model_lstm.compile(optimizer='adam', loss='mean_squared_error')

    # Add callbacks for early stopping and reducing learning rate
    callbacks_lstm = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True), # Tăng patience một chút để mô hình có thêm cơ hội cải thiện
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.00001) # Tăng patience một chút
    ]

    # Train LSTM Model
    # Rút gọn số lần thử: Giảm số epochs tối đa
    num_epochs_lstm = 50 # Đặt số epochs tối đa. EarlyStopping sẽ tự dừng sớm hơn nếu không cải thiện.
    batch_size_lstm = 128 # Tăng batch_size để đẩy nhanh thời gian huấn luyện trên GPU

    print("Starting LSTM training...")
    history_lstm = model_lstm.fit(X_train_lstm, y_train_lstm, 
                                  validation_data=(X_test_lstm, y_test_lstm), 
                                  epochs=num_epochs_lstm, 
                                  batch_size=batch_size_lstm, 
                                  verbose=1,
                                  callbacks=callbacks_lstm)
    print("LSTM training finished.")
    
    # Make predictions
    train_predict_lstm = model_lstm.predict(X_train_lstm)
    
    # Kiểm tra nếu X_test_lstm rỗng trước khi dự đoán
    test_predict_lstm = np.array([]) # Khởi tạo để tránh lỗi nếu test set rỗng
    y_test_lstm_inv = np.array([])
    rmse_lstm = np.nan
    mae_lstm = np.nan

    if X_test_lstm.shape[0] > 0:
        test_predict_lstm = model_lstm.predict(X_test_lstm)
        # Inverse transform predictions and actual values
        y_test_lstm_inv = scaler_lstm.inverse_transform(y_test_lstm.reshape(-1, 1))
        test_predict_lstm = scaler_lstm.inverse_transform(test_predict_lstm)
        
        # Evaluate model
        rmse_lstm = np.sqrt(mean_squared_error(y_test_lstm_inv, test_predict_lstm))
        mae_lstm = mean_absolute_error(y_test_lstm_inv, test_predict_lstm)
        print(f"LSTM RMSE: {rmse_lstm:.2f}")
        print(f"LSTM MAE: {mae_lstm:.2f}")
    else:
        print("LSTM: Test set is empty, skipping evaluation metrics.")


    # Future forecasting with LSTM (Iterative prediction)
    last_input_lstm = scaled_data_lstm[-time_step_lstm:]
    
    # Cần đảm bảo forecast_end_date_overall và last_historical_date đã được định nghĩa
    # (Chúng thường được định nghĩa ở các phần trước trong notebook)
    future_steps = int((forecast_end_date_overall - last_historical_date).total_seconds() / (6 * 3600)) + 1 
    if future_steps <= 0:
        print("WARNING: Future steps for LSTM calculated as 0 or negative. No future forecast will be made.")
        future_steps = 0 # Đảm bảo vòng lặp không chạy nếu không có bước nào để dự báo

    lstm_forecast_scaled = []
    current_input = last_input_lstm.reshape(1, time_step_lstm, 1) # Reshape for model input

    for _ in range(future_steps):
        next_pred_scaled = model_lstm.predict(current_input, verbose=0)[0, 0]
        lstm_forecast_scaled.append(next_pred_scaled)
        # Update current_input by removing the first element and adding the new prediction
        current_input = np.append(current_input[:, 1:, :], np.array(next_pred_scaled).reshape(1, 1, 1), axis=1)

    lstm_forecast_inv = scaler_lstm.inverse_transform(np.array(lstm_forecast_scaled).reshape(-1, 1))

    # Generate future dates for LSTM forecast
    last_date_lstm = target_series_lstm.index.max()
    future_dates_lstm = pd.date_range(start=last_date_lstm + pd.Timedelta(hours=6), 
                                      periods=len(lstm_forecast_inv), 
                                      freq='6h')

    lstm_forecast = pd.DataFrame({'time': future_dates_lstm, 'intensity_kt': lstm_forecast_inv.flatten()})
    lstm_forecast = lstm_forecast.set_index('time')
    print(f"LSTM forecast generated for {len(lstm_forecast)} steps.")
    
    lstm_executed = True # Đặt cờ thành True khi mọi thứ hoàn tất

except ValueError as ve:
    print(f"ERROR: LSTM setup/data error: {ve}")
    print("Skipping LSTM model execution due to data issues.")
except Exception as e:
    print(f"ERROR: An unexpected error occurred during LSTM execution: {e}")
    # lstm_forecast vẫn là DataFrame rỗng và lstm_executed vẫn là False

# --- 6. Run PatchTST Model ---
print("\n--- Running PatchTST Model ---")

# Khởi tạo biến dự báo và cờ thực thi
forecast_patchtst_overall_df = pd.DataFrame() # Đảm bảo biến này luôn được định nghĩa
patchtst_executed = False # Cờ kiểm soát việc thực thi PatchTST

# Kiểm tra GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEBUG: Using device: {device}")
if device == torch.device("cpu"):
    print("WARNING: CUDA (GPU) not available. PatchTST training will be significantly slower on CPU. This might be why it's not 'running'.")

try:
    if df_for_patchtst.empty or intensity_column_name not in df_for_patchtst.columns:
        raise ValueError("df_for_patchtst is empty or missing the target intensity column. Cannot run PatchTST.")
    print(f"DEBUG: df_for_patchtst initial shape: {df_for_patchtst.shape}")
    print(f"DEBUG: df_for_patchtst columns: {df_for_patchtst.columns.tolist()}")

    # Convert DataFrame index to datetime if not already
    df_for_patchtst.index = pd.to_datetime(df_for_patchtst.index)

    # Lấy cột cường độ để scaling riêng cho việc dự đoán (cần thiết cho inverse_transform)
    intensity_data_for_scaler = df_for_patchtst[[intensity_column_name]].values
    scaler_intensity_patchtst = StandardScaler()
    scaler_intensity_patchtst.fit(intensity_data_for_scaler)
    print(f"DEBUG: Scaler for intensity_column_name '{intensity_column_name}' created.")

    # Scale toàn bộ dữ liệu (cả features và target)
    scaler_patchtst = StandardScaler()
    scaled_data_patchtst = scaler_patchtst.fit_transform(df_for_patchtst.values)
    print(f"DEBUG: Scaled_data_patchtst shape: {scaled_data_patchtst.shape}")
    
    # Tìm chỉ số cột cường độ trong mảng đã scale
    # Điều này quan trọng vì sau khi scale, chúng ta làm việc với mảng NumPy, không phải DataFrame
    intensity_col_idx_scaled = df_for_patchtst.columns.get_loc(intensity_column_name)
    print(f"DEBUG: Intensity column index in scaled_data_patchtst: {intensity_col_idx_scaled}")


    # PatchTST hyperparameters
    # Cân nhắc giảm các giá trị này để chạy nhanh hơn trong quá trình debug
    seq_len = 96 # Length of the input sequence (e.g., 96 data points = 24 days at 6h intervals)
    pred_len = 24 # Length of prediction (e.g., 24 data points = 6 days at 6h intervals)
    patch_len = 16 # Length of each patch
    stride_len = 8 # Stride between patches

    # Calculate number of patches
    # For a robust calculation:
    num_patches = (seq_len - patch_len) // stride_len + 1 
    if (seq_len - patch_len) % stride_len != 0: 
        num_patches += 1
    if num_patches <= 0: # Sanity check for num_patches
        raise ValueError(f"Calculated num_patches is {num_patches}. Ensure seq_len > patch_len.")


    d_model = 128 # Hidden dimension of the model
    n_heads = 8 # Number of attention heads
    e_layers = 3 # Number of encoder layers
    dropout = 0.1 # Dropout rate

    input_dim = scaled_data_patchtst.shape[1] # Number of features (columns)
    output_dim = 1 # We are predicting only the intensity column

    print(f"DEBUG: PatchTST Config: input_dim={input_dim}, output_dim={output_dim}, seq_len={seq_len}, pred_len={pred_len}")
    print(f"DEBUG: Patching Config: patch_len={patch_len}, stride_len={stride_len}, num_patches={num_patches}")
    print(f"DEBUG: Model Config: d_model={d_model}, n_heads={n_heads}, e_layers={e_layers}, dropout={dropout}")

    # --- PatchTST Model Definition ---
    class PatchTST(nn.Module):
        def __init__(self, input_dim, output_dim, seq_len, pred_len, patch_len, stride_len, d_model, n_heads, e_layers, dropout, num_patches):
            super().__init__()
            self.seq_len = seq_len
            self.pred_len = pred_len
            self.patch_len = patch_len
            self.stride_len = stride_len
            self.num_patches = num_patches
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.d_model = d_model # Store d_model for reshape in forward

            # Patch embedding: Transforms each patch (patch_len) to d_model
            # Applies to each feature (N_vars) separately across N_patches
            self.patch_embedding = nn.Linear(patch_len, d_model) 

            # Positional encoding for patches: Shape (1, num_patches * input_dim, d_model)
            self.positional_encoding = nn.Parameter(torch.randn(1, num_patches * input_dim, d_model))

            # Encoder (Transformer Encoder)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

            # Head for prediction: From flattened encoder output to pred_len * output_dim
            self.head = nn.Linear(d_model * num_patches * input_dim, pred_len * output_dim) 

        def forward(self, x): # x shape: (batch_size, seq_len, input_dim)
            batch_size, seq_len, input_dim = x.shape
            
            # 1. Patching and Embedding
            # Create patches: unfold along dimension 1 (sequence length)
            # Resulting shape: (batch_size, num_patches, patch_len) for EACH feature
            # x.unfold returns (batch_size, num_patches, patch_len) if input is (batch_size, seq_len)
            
            # Need to unfold each feature separately if input_dim > 1
            x_patched_list = []
            for i in range(input_dim):
                # Apply unfold to each feature column: (batch_size, seq_len) -> (batch_size, num_patches, patch_len)
                patches_feature_i = x[:, :, i].unfold(dimension=1, size=self.patch_len, step=self.stride_len)
                x_patched_list.append(patches_feature_i)
            
            # Stack patches from all features: (batch_size, num_patches, patch_len, input_dim)
            x_patched = torch.stack(x_patched_list, dim=-1)
            
            # Reshape for embedding: (batch_size * num_patches * input_dim, patch_len)
            x_for_embedding = x_patched.reshape(-1, self.patch_len)
            
            # Apply patch embedding: (batch_size * num_patches * input_dim, d_model)
            x_embedding = self.patch_embedding(x_for_embedding)

            # Reshape back to (batch_size, num_patches * input_dim, d_model)
            x_embedding = x_embedding.reshape(batch_size, self.num_patches * input_dim, self.d_model)

            # 2. Add Positional Encoding
            # positional_encoding is (1, num_patches * input_dim, d_model)
            x_embedding = x_embedding + self.positional_encoding.repeat(batch_size, 1, 1).to(x.device)

            # 3. Transformer Encoder
            encoder_output = self.transformer_encoder(x_embedding) # (batch_size, num_patches * input_dim, d_model)

            # 4. Head for Prediction
            # Flatten the encoder output for the linear head: (batch_size, num_patches * input_dim * d_model)
            flattened_output = encoder_output.reshape(batch_size, -1) 
            
            # Predict the future sequence: (batch_size, pred_len * output_dim)
            prediction = self.head(flattened_output) 
            # Reshape to (batch_size, pred_len, output_dim)
            prediction = prediction.reshape(batch_size, self.pred_len, self.output_dim) 

            return prediction

    # --- Dataset and DataLoader ---
    class PatchTSTDataset(Dataset):
        def __init__(self, data, seq_len, pred_len):
            self.data = data # scaled_data_patchtst (NumPy array)
            self.seq_len = seq_len
            self.pred_len = pred_len

        def __len__(self):
            # Make sure there are enough data points to form at least one sequence and its target
            if len(self.data) < self.seq_len + self.pred_len:
                return 0
            return len(self.data) - self.seq_len - self.pred_len + 1

        def __getitem__(self, idx):
            x = self.data[idx : idx + self.seq_len]
            y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    # --- Prepare training data ---
    # Split data into train/validation
    train_size_patchtst = int(len(scaled_data_patchtst) * 0.8)
    train_data_patchtst = scaled_data_patchtst[:train_size_patchtst]
    val_data_patchtst = scaled_data_patchtst[train_size_patchtst:]

    print(f"DEBUG: train_data_patchtst shape: {train_data_patchtst.shape}")
    print(f"DEBUG: val_data_patchtst shape: {val_data_patchtst.shape}")

    train_dataset_patchtst = PatchTSTDataset(train_data_patchtst, seq_len, pred_len)
    val_dataset_patchtst = PatchTSTDataset(val_data_patchtst, seq_len, pred_len)

    # Check if datasets are empty
    if len(train_dataset_patchtst) == 0:
        raise ValueError(f"PatchTST training dataset is empty ({len(train_dataset_patchtst)} samples). Adjust seq_len/pred_len or ensure enough data.")
    if len(val_dataset_patchtst) == 0:
        print(f"WARNING: PatchTST validation dataset is empty ({len(val_dataset_patchtst)} samples). Using train dataset for validation (not ideal but prevents crash).")
        val_dataset_patchtst = train_dataset_patchtst # Fallback: Use train for validation if val is empty

    batch_size_patchtst = 64 # You can adjust this. Try smaller if OOM, larger for speed on GPU
    if batch_size_patchtst > len(train_dataset_patchtst) and len(train_dataset_patchtst) > 0:
        batch_size_patchtst = len(train_dataset_patchtst) # Avoid batch_size > dataset size
        print(f"DEBUG: Adjusted batch_size_patchtst to {batch_size_patchtst} as dataset is smaller.")


    train_loader_patchtst = DataLoader(train_dataset_patchtst, batch_size=batch_size_patchtst, shuffle=True)
    val_loader_patchtst = DataLoader(val_dataset_patchtst, batch_size=batch_size_patchtst, shuffle=False)

    print(f"DEBUG: Train DataLoader has {len(train_loader_patchtst)} batches.")
    if len(val_loader_patchtst) > 0:
        print(f"DEBUG: Val DataLoader has {len(val_loader_patchtst)} batches.")

    # --- Initialize Model, Loss, Optimizer ---
    model_patchtst = PatchTST(input_dim, output_dim, seq_len, pred_len, patch_len, stride_len, d_model, n_heads, e_layers, dropout, num_patches).to(device)
    criterion_patchtst = nn.MSELoss()
    optimizer_patchtst = torch.optim.Adam(model_patchtst.parameters(), lr=0.001)

    print(f"DEBUG: PatchTST model initialized and moved to {device}.")

    # --- Training Loop ---
    num_epochs_patchtst = 30 # Reduce for faster testing, increase for better performance
    
    print(f"Starting PatchTST training for {num_epochs_patchtst} epochs...")
    min_val_loss = float('inf')
    patience_patchtst = 10 # Early stopping patience
    patience_counter = 0

    for epoch in range(num_epochs_patchtst):
        model_patchtst.train()
        train_loss = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader_patchtst):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # y_batch_target should be the specific target column from y_batch
            # y_batch shape: (batch_size, pred_len, input_dim)
            # We want: (batch_size, pred_len, output_dim) where output_dim=1 (intensity)
            y_batch_target = y_batch[:, :, intensity_col_idx_scaled].unsqueeze(-1) 

            optimizer_patchtst.zero_grad()
            outputs = model_patchtst(X_batch) # (batch_size, pred_len, output_dim)
            
            # print(f"DEBUG: Epoch {epoch}, Batch {batch_idx} - X_batch shape: {X_batch.shape}")
            # print(f"DEBUG: Epoch {epoch}, Batch {batch_idx} - y_batch_target shape: {y_batch_target.shape}")
            # print(f"DEBUG: Epoch {epoch}, Batch {batch_idx} - outputs shape: {outputs.shape}")

            loss = criterion_patchtst(outputs, y_batch_target)
            loss.backward()
            optimizer_patchtst.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader_patchtst)

        # Validation
        model_patchtst.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch_val, y_batch_val in val_loader_patchtst:
                X_batch_val, y_batch_val = X_batch_val.to(device), y_batch_val.to(device)
                y_batch_val_target = y_batch_val[:, :, intensity_col_idx_scaled].unsqueeze(-1)

                outputs_val = model_patchtst(X_batch_val)
                loss_val = criterion_patchtst(outputs_val, y_batch_val_target)
                val_loss += loss_val.item()
        avg_val_loss = val_loss / len(val_loader_patchtst) if len(val_loader_patchtst) > 0 else 0

        print(f"Epoch {epoch+1}/{num_epochs_patchtst}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early stopping logic
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model_patchtst.state_dict(), os.path.join(output_dir, 'best_patchtst_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience_patchtst:
                print(f"Early stopping at epoch {epoch+1} as validation loss did not improve for {patience_patchtst} epochs.")
                break
    
    print("PatchTST training finished.")
    # Load best model for prediction
    model_patchtst.load_state_dict(torch.load(os.path.join(output_dir, 'best_patchtst_model.pth')))
    model_patchtst.eval() # Set to evaluation mode

    # --- Forecasting with PatchTST ---
    print("Starting PatchTST forecasting...")
    # Prepare last `seq_len` data points for initial forecast
    last_input_patchtst = scaled_data_patchtst[-seq_len:]
    print(f"DEBUG: Last input for forecasting shape: {last_input_patchtst.shape}")
    
    # Calculate future steps
    future_steps_patchtst = int((forecast_end_date_overall - last_historical_date).total_seconds() / (6 * 3600)) + 1
    if future_steps_patchtst <= 0:
        print("WARNING: Future steps calculated for PatchTST is 0 or negative. No future forecast will be made.")
        # Ensure future_steps_patchtst is at least 1 for the loop to run if needed
        future_steps_patchtst = 1 if future_steps_patchtst <= 0 else future_steps_patchtst
        
    print(f"DEBUG: Total future steps for PatchTST: {future_steps_patchtst}")
    
    # Store predictions
    patchtst_predictions_scaled = []
    current_input_sequence = torch.tensor(last_input_patchtst, dtype=torch.float32).unsqueeze(0).to(device) # (1, seq_len, input_dim)
    print(f"DEBUG: Initial current_input_sequence shape for forecast: {current_input_sequence.shape}")

    # Iteratively predict for `future_steps_patchtst`
    predicted_count = 0
    while predicted_count < future_steps_patchtst:
        with torch.no_grad():
            next_preds_batch = model_patchtst(current_input_sequence) # (1, pred_len, output_dim)
            print(f"DEBUG: Inside forecast loop - next_preds_batch shape: {next_preds_batch.shape}")
            
            # Only take the predicted intensity column
            # Squeeze(0) removes batch dim (1, pred_len, 1) -> (pred_len, 1)
            # Squeeze(-1) removes last dim (pred_len, 1) -> (pred_len,)
            next_preds_intensity_scaled = next_preds_batch.squeeze(0).squeeze(-1).cpu().numpy() 
            print(f"DEBUG: Inside forecast loop - next_preds_intensity_scaled shape (numpy): {next_preds_intensity_scaled.shape}")

            patchtst_predictions_scaled.extend(next_preds_intensity_scaled)
            predicted_count += len(next_preds_intensity_scaled)
            
            # --- Update the input sequence for the next prediction ---
            # Create new row(s) to append. Shape (pred_len, input_dim)
            new_data_points_scaled = torch.zeros(pred_len, input_dim).to(device)
            # Place the predicted intensity into the correct column
            new_data_points_scaled[:, intensity_col_idx_scaled] = next_preds_batch.squeeze(0)[:, 0] # Take first (intensity) output
            
            # Update current_input_sequence: shift left by pred_len, append new predictions
            current_input_sequence = torch.cat((current_input_sequence[:, pred_len:, :], new_data_points_scaled.unsqueeze(0)), dim=1)
            print(f"DEBUG: Inside forecast loop - updated current_input_sequence shape: {current_input_sequence.shape}")
            
            # Sanity check: ensure current_input_sequence always has seq_len
            # This is implicitly handled by `cat` and `pred_len` being the step
            if current_input_sequence.shape[1] != seq_len:
                print(f"WARNING: current_input_sequence has unexpected length {current_input_sequence.shape[1]}, expected {seq_len}. Adjusting.")
                current_input_sequence = current_input_sequence[:, -seq_len:, :]


    # Truncate predictions if we predicted more than needed due to pred_len batching
    if len(patchtst_predictions_scaled) > future_steps_patchtst:
        patchtst_predictions_scaled = patchtst_predictions_scaled[:future_steps_patchtst]

    # Inverse transform predictions
    # Need to create a dummy array with the correct number of features for inverse_transform
    # Fill with mean/zeros for other features, place predicted intensity in its column
    predicted_full_scaled_for_inv = np.zeros((len(patchtst_predictions_scaled), input_dim))
    predicted_full_scaled_for_inv[:, intensity_col_idx_scaled] = patchtst_predictions_scaled

    # Use the scaler for the entire data (scaler_patchtst) and then pick the intensity column,
    # OR use the specific scaler for intensity if you only scaled that column.
    # Given your current code, `scaler_patchtst` scales all columns.
    patchtst_forecast_inv_all_features = scaler_patchtst.inverse_transform(predicted_full_scaled_for_inv)
    patchtst_forecast_inv = patchtst_forecast_inv_all_features[:, intensity_col_idx_scaled]

    print(f"DEBUG: Final patchtst_forecast_inv shape: {patchtst_forecast_inv.shape}")

    # Generate future dates for PatchTST forecast
    last_date_patchtst = df_for_patchtst.index.max()
    future_dates_patchtst = pd.date_range(start=last_date_patchtst + pd.Timedelta(hours=6), 
                                         periods=len(patchtst_forecast_inv), 
                                         freq='6H')

    forecast_patchtst_overall_df = pd.DataFrame({'time': future_dates_patchtst, 'predicted_intensity': patchtst_forecast_inv.flatten()})
    forecast_patchtst_overall_df = forecast_patchtst_overall_df.set_index('time')
    print(f"PatchTST forecast generated for {len(forecast_patchtst_overall_df)} steps.")

    patchtst_executed = True # Đặt cờ thành True khi mọi thứ hoàn tất

except ValueError as ve:
    print(f"ERROR: PatchTST setup/data error: {ve}")
    print("Skipping PatchTST model execution due to data issues.")
except Exception as e:
    print(f"ERROR: An unexpected error occurred during PatchTST execution: {e}")
    # Đảm bảo forecast_patchtst_overall_df vẫn là DataFrame rỗng và patchtst_executed vẫn là False

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 7. Generate HTML Report and Comparison ---
# Đảm bảo output_dir được định nghĩa (ví dụ: output_dir = "/kaggle/working/")
output_dir = "/kaggle/working/"
os.makedirs(output_dir, exist_ok=True)

print(f"\nSaving plots to {output_dir}")

# Store plots for each year
model_plots_by_year = {}
comparison_plots_by_year = {}

# --- Chuẩn bị dữ liệu lịch sử để vẽ (Tập trung) ---
# Đảm bảo các biến này được định nghĩa và có dữ liệu từ các phần trước đó trong notebook của bạn:
# target_series, df_for_patchtst, intensity_column_name, forecast_years
# m (mô hình Prophet đã được fit), lstm_executed, patchtst_executed, forecast_prophet_overall, lstm_forecast, forecast_patchtst_overall_df

# Mốc thời gian bắt đầu vẽ dữ liệu lịch sử (ví dụ: 2 năm trước năm dự báo đầu tiên)
if 'forecast_years' in locals() and forecast_years:
    plot_historical_start_date = pd.to_datetime(f'{min(forecast_years)-2}-01-01')
else:
    print("ERROR: 'forecast_years' is not defined or empty. Cannot determine historical plot start date.")
    # Set a default if forecast_years is not available, but this indicates a problem earlier
    plot_historical_start_date = pd.to_datetime('2020-01-01')


# Sử dụng target_series cho dữ liệu lịch sử (Prophet, LSTM, Comparison)
recent_historical_data = pd.Series()
if 'target_series' in locals() and not target_series.empty and pd.api.types.is_datetime64_any_dtype(target_series.index):
    recent_historical_data = target_series[target_series.index >= plot_historical_start_date].copy()
    if recent_historical_data.empty or recent_historical_data.isnull().all():
        print(f"WARNING: recent_historical_data for plotting is empty or all NaN based on target_series from {plot_historical_start_date}. Plots might not show historical data.")
else:
    print("WARNING: 'target_series' is not available, empty, or its index is not datetime. Historical data for plots might be missing.")

# Sử dụng df_for_patchtst cho PatchTST's historical reference (nếu nó dùng nhiều features)
recent_df_patchtst_cleaned = pd.DataFrame()
if 'df_for_patchtst' in locals() and not df_for_patchtst.empty \
    and intensity_column_name in df_for_patchtst.columns \
    and pd.api.types.is_datetime64_any_dtype(df_for_patchtst.index):
    
    recent_df_patchtst_cleaned = df_for_patchtst[df_for_patchtst.index >= plot_historical_start_date].copy()
    if recent_df_patchtst_cleaned.empty or recent_df_patchtst_cleaned[intensity_column_name].isnull().all():
        print(f"WARNING: recent_df_patchtst_cleaned for plotting is empty or intensity column all NaN from {plot_historical_start_date}. PatchTST historical data might be missing.")
else:
    print("WARNING: 'df_for_patchtst' is not available, empty, or missing intensity column/datetime index. PatchTST historical data might be missing.")


for year in forecast_years:
    year_start = pd.Timestamp(year=year, month=1, day=1)
    year_end = pd.Timestamp(year=year, month=12, day=31, hour=23, minute=59, second=59) 

    # Lọc dữ liệu dự báo cho năm hiện tại
    prophet_yearly_forecast = pd.DataFrame()
    if 'forecast_prophet_overall' in locals() and not forecast_prophet_overall.empty:
        prophet_yearly_forecast = forecast_prophet_overall[(forecast_prophet_overall['ds'] >= year_start) & (forecast_prophet_overall['ds'] <= year_end)].copy()
    else:
        print(f"WARNING: 'forecast_prophet_overall' not available or empty for year {year}. Prophet plot will be skipped for this year.")

    lstm_yearly_forecast = pd.DataFrame()
    if 'lstm_forecast' in locals() and 'lstm_executed' in locals() and lstm_executed and not lstm_forecast.empty:
        lstm_yearly_forecast = lstm_forecast[(lstm_forecast.index >= year_start) & (lstm_forecast.index <= year_end)].copy()
    else:
        print(f"WARNING: 'lstm_forecast' not available or empty for year {year} or LSTM not executed. LSTM plot will be skipped for this year.")

    patchtst_yearly_forecast = pd.DataFrame()
    if 'forecast_patchtst_overall_df' in locals() and 'patchtst_executed' in locals() and patchtst_executed and not forecast_patchtst_overall_df.empty:
        patchtst_yearly_forecast = forecast_patchtst_overall_df[(forecast_patchtst_overall_df.index >= year_start) & (forecast_patchtst_overall_df.index <= year_end)].copy()
    else:
        print(f"WARNING: 'forecast_patchtst_overall_df' not available or empty for year {year} or PatchTST not executed. PatchTST plot will be skipped for this year.")


    # --- Lưu biểu đồ riêng lẻ cho từng mô hình theo năm ---

    # Biểu đồ Prophet
    if not prophet_yearly_forecast.empty and 'yhat' in prophet_yearly_forecast.columns and not prophet_yearly_forecast['yhat'].isnull().all():
        if 'm' in locals(): # Ensure prophet model 'm' exists
            fig_prophet_year = m.plot(forecast_prophet_overall) # Vẽ toàn bộ dự báo để có ngữ cảnh
            ax = fig_prophet_year.gca()
            ax.set_title(f'Prophet Forecast of Typhoon Intensity ({year})')
            ax.set_xlabel('Date')
            ax.set_ylabel(f'{intensity_column_name} (kt)')
            
            # Thêm đường thẳng để chỉ rõ ranh giới giữa lịch sử và dự báo
            if not recent_historical_data.empty and pd.notna(recent_historical_data.index.max()):
                ax.axvline(recent_historical_data.index.max(), color='grey', linestyle='--', linewidth=1, label='End of Historical Data')

            # Highlight khu vực năm dự báo hiện tại
            ax.axvspan(year_start, year_end, color='gray', alpha=0.2, label=f'{year} Forecast Period')
            ax.legend()
            plt.tight_layout()
            prophet_plot_path = os.path.join(output_dir, f'prophet_forecast_{year}.png')
            fig_prophet_year.savefig(prophet_plot_path)
            plt.close(fig_prophet_year)
            if year not in model_plots_by_year: model_plots_by_year[year] = {}
            model_plots_by_year[year]['prophet'] = prophet_plot_path
            print(f"Prophet plot for {year} saved.")
        else:
            print(f"WARNING: Prophet model 'm' not found. Cannot plot Prophet forecast for {year}.")
    else:
        print(f"Prophet forecast for {year} is empty or all NaN in 'yhat' column. Skipping plot.")

    # Biểu đồ LSTM
    if not lstm_yearly_forecast.empty and 'intensity_kt' in lstm_yearly_forecast.columns and not lstm_yearly_forecast['intensity_kt'].isnull().all():
        plt.figure(figsize=(14, 7))
        if not recent_historical_data.empty and not recent_historical_data.isnull().all():
            plt.plot(recent_historical_data.index, recent_historical_data.values, label='Actual Intensity (Historical)', color='blue', alpha=0.8)
        
        plt.plot(lstm_yearly_forecast.index, lstm_yearly_forecast['intensity_kt'], label='LSTM Forecast', linestyle='--', color='red') 
        
        plt.title(f'LSTM Forecast of Typhoon Intensity ({year})')
        plt.xlabel('Date')
        plt.ylabel(f'{intensity_column_name} (kt)')
        plt.legend()
        plt.grid(True)

        # --- Xử lý giới hạn trục X mạnh mẽ cho Biểu đồ LSTM ---
        # Định nghĩa các giới hạn fallback vững chắc là các đối tượng Timestamp thực sự
        fallback_left_xlim = pd.Timestamp(year=max(2000, year_start.year - 2), month=1, day=1) 
        fallback_right_xlim = pd.Timestamp(year=year_end.year + 1, month=1, day=1) 

        left_xlim = pd.NaT 
        right_xlim = pd.NaT

        # Cố gắng đặt giới hạn bên trái
        if not recent_historical_data.empty and pd.notna(recent_historical_data.index.min()):
            potential_left_hist = recent_historical_data.index.min()
            if isinstance(potential_left_hist, pd.Timestamp):
                left_xlim = max(potential_left_hist, year_start - pd.Timedelta(days=30))
        
        if pd.isna(left_xlim) and not lstm_yearly_forecast.empty and pd.notna(lstm_yearly_forecast.index.min()):
            potential_left_forecast = lstm_yearly_forecast.index.min()
            if isinstance(potential_left_forecast, pd.Timestamp):
                left_xlim = potential_left_forecast - pd.Timedelta(days=30)
        
        # Nếu vẫn không hợp lệ, sử dụng fallback
        if pd.isna(left_xlim) or not isinstance(left_xlim, pd.Timestamp):
            left_xlim = fallback_left_xlim
            print(f"DEBUG: LSTM Plot {year} - Using fallback_left_xlim: {left_xlim} as primary sources failed.")

        # Cố gắng đặt giới hạn bên phải
        if not lstm_yearly_forecast.empty and pd.notna(lstm_yearly_forecast.index.max()):
            potential_right_forecast = lstm_yearly_forecast.index.max()
            if isinstance(potential_right_forecast, pd.Timestamp):
                right_xlim = potential_right_forecast + pd.Timedelta(days=30)
        
        if pd.isna(right_xlim) or not isinstance(right_xlim, pd.Timestamp):
            if pd.notna(year_end):
                right_xlim = year_end + pd.Timedelta(days=30)
            elif 'forecast_end_date_overall' in locals() and pd.notna(forecast_end_date_overall) and isinstance(forecast_end_date_overall, pd.Timestamp):
                right_xlim = forecast_end_date_overall + pd.Timedelta(days=30)
            
        if pd.isna(right_xlim) or not isinstance(right_xlim, pd.Timestamp):
            right_xlim = fallback_right_xlim
            print(f"DEBUG: LSTM Plot {year} - Using fallback_right_xlim: {right_xlim} as primary sources failed.")

        # Xác thực cuối cùng và đặt xlim
        print(f"DEBUG: LSTM Plot {year} - Attempting plt.xlim with left_xlim={left_xlim}, right_xlim={right_xlim}")
        if pd.notna(left_xlim) and pd.notna(right_xlim) and isinstance(left_xlim, pd.Timestamp) and isinstance(right_xlim, pd.Timestamp):
            if left_xlim < right_xlim:
                plt.xlim(left_xlim, right_xlim)
            else:
                print(f"WARNING: Invalid x-axis limits (left_xlim >= right_xlim) for LSTM plot {year} ({left_xlim} vs {right_xlim}). Skipping xlim setting.")
        else:
            print(f"ERROR: Final x-axis limits for LSTM plot {year} are still invalid type/NaN/Inf ({left_xlim}, {right_xlim}). Skipping xlim setting.")

        plt.tight_layout()
        lstm_plot_path = os.path.join(output_dir, f'lstm_forecast_{year}.png')
        plt.savefig(lstm_plot_path)
        plt.close(plt.gcf())
        if year not in model_plots_by_year: model_plots_by_year[year] = {}
        model_plots_by_year[year]['lstm'] = lstm_plot_path
        print(f"LSTM plot for {year} saved.")
    else:
        print(f"LSTM forecast for {year} is empty or all NaN in 'intensity_kt' column, or LSTM was not executed. Skipping plot.")

    # Biểu đồ PatchTST
    # Đảm bảo bạn đã kiểm tra tên cột dự đoán của PatchTST, ở đây tôi đang dùng 'predicted_intensity'
    if not patchtst_yearly_forecast.empty and 'predicted_intensity' in patchtst_yearly_forecast.columns and not patchtst_yearly_forecast['predicted_intensity'].isnull().all():
        plt.figure(figsize=(14, 7))
        if not recent_df_patchtst_cleaned.empty and intensity_column_name in recent_df_patchtst_cleaned.columns and not recent_df_patchtst_cleaned[intensity_column_name].isnull().all():
            plt.plot(recent_df_patchtst_cleaned.index, recent_df_patchtst_cleaned[intensity_column_name].values, label='Actual Intensity (Historical)', color='blue', alpha=0.8)
        
        plt.plot(patchtst_yearly_forecast.index, patchtst_yearly_forecast['predicted_intensity'], label='PatchTST Forecast', linestyle=':', color='purple')
        
        plt.title(f'PatchTST Forecast of Typhoon Intensity ({year})')
        plt.xlabel('Date')
        plt.ylabel(f'{intensity_column_name} (kt)')
        plt.legend()
        plt.grid(True)
        
        # --- Xử lý giới hạn trục X mạnh mẽ cho Biểu đồ PatchTST ---
        fallback_left_xlim_patchtst = pd.Timestamp(year=max(2000, year_start.year - 2), month=1, day=1) 
        fallback_right_xlim_patchtst = pd.Timestamp(year=year_end.year + 1, month=1, day=1) 

        left_xlim_patchtst = pd.NaT 
        right_xlim_patchtst = pd.NaT

        # Cố gắng đặt giới hạn bên trái
        if not recent_df_patchtst_cleaned.empty and pd.notna(recent_df_patchtst_cleaned.index.min()):
            potential_left_hist_patchtst = recent_df_patchtst_cleaned.index.min()
            if isinstance(potential_left_hist_patchtst, pd.Timestamp):
                left_xlim_patchtst = max(potential_left_hist_patchtst, year_start - pd.Timedelta(days=30))
        
        if pd.isna(left_xlim_patchtst) and not patchtst_yearly_forecast.empty and pd.notna(patchtst_yearly_forecast.index.min()):
            potential_left_forecast_patchtst = patchtst_yearly_forecast.index.min()
            if isinstance(potential_left_forecast_patchtst, pd.Timestamp):
                left_xlim_patchtst = potential_left_forecast_patchtst - pd.Timedelta(days=30)
        
        if pd.isna(left_xlim_patchtst) or not isinstance(left_xlim_patchtst, pd.Timestamp):
            left_xlim_patchtst = fallback_left_xlim_patchtst
            print(f"DEBUG: PatchTST Plot {year} - Using fallback_left_xlim: {left_xlim_patchtst} as primary sources failed.")

        # Cố gắng đặt giới hạn bên phải
        if not patchtst_yearly_forecast.empty and pd.notna(patchtst_yearly_forecast.index.max()):
            potential_right_forecast_patchtst = patchtst_yearly_forecast.index.max()
            if isinstance(potential_right_forecast_patchtst, pd.Timestamp):
                right_xlim_patchtst = potential_right_forecast_patchtst + pd.Timedelta(days=30)
        
        if pd.isna(right_xlim_patchtst) or not isinstance(right_xlim_patchtst, pd.Timestamp):
            if pd.notna(year_end):
                right_xlim_patchtst = year_end + pd.Timedelta(days=30)
            elif 'forecast_end_date_overall' in locals() and pd.notna(forecast_end_date_overall) and isinstance(forecast_end_date_overall, pd.Timestamp):
                right_xlim_patchtst = forecast_end_date_overall + pd.Timedelta(days=30)
            
        if pd.isna(right_xlim_patchtst) or not isinstance(right_xlim_patchtst, pd.Timestamp):
            right_xlim_patchtst = fallback_right_xlim_patchtst
            print(f"DEBUG: PatchTST Plot {year} - Using fallback_right_xlim: {right_xlim_patchtst} as primary sources failed.")

        # Xác thực cuối cùng và đặt xlim
        print(f"DEBUG: PatchTST Plot {year} - Attempting plt.xlim with left_xlim={left_xlim_patchtst}, right_xlim={right_xlim_patchtst}")
        if pd.notna(left_xlim_patchtst) and pd.notna(right_xlim_patchtst) and isinstance(left_xlim_patchtst, pd.Timestamp) and isinstance(right_xlim_patchtst, pd.Timestamp):
            if left_xlim_patchtst < right_xlim_patchtst:
                plt.xlim(left_xlim_patchtst, right_xlim_patchtst)
            else:
                print(f"WARNING: Invalid x-axis limits (left_xlim >= right_xlim) for PatchTST plot {year} ({left_xlim_patchtst} vs {right_xlim_patchtst}). Skipping xlim setting.")
        else:
            print(f"ERROR: Final x-axis limits for PatchTST plot {year} are still invalid type/NaN/Inf ({left_xlim_patchtst}, {right_xlim_patchtst}). Skipping xlim setting.")

        plt.tight_layout()
        patchtst_plot_path = os.path.join(output_dir, f'patchtst_forecast_{year}.png')
        plt.savefig(patchtst_plot_path)
        plt.close(plt.gcf())
        if year not in model_plots_by_year: model_plots_by_year[year] = {}
        model_plots_by_year[year]['patchtst'] = patchtst_plot_path
        print(f"PatchTST plot for {year} saved.")
    else:
        print(f"PatchTST forecast for {year} is empty or all NaN in 'predicted_intensity' column, or PatchTST was not executed. Skipping plot.")

    # --- Lưu Biểu đồ So sánh theo năm ---
    plt.figure(figsize=(18, 9))
    # Chỉ vẽ dữ liệu lịch sử nếu có và không phải tất cả là NaN
    if not recent_historical_data.empty and not recent_historical_data.isnull().all():
        plt.plot(recent_historical_data.index, recent_historical_data.values, label='Actual Intensity (Historical)', color='blue', alpha=0.8)

    if not prophet_yearly_forecast.empty and 'yhat' in prophet_yearly_forecast.columns and not prophet_yearly_forecast['yhat'].isnull().all():
        plt.plot(prophet_yearly_forecast['ds'], prophet_yearly_forecast['yhat'], label='Prophet Forecast', linestyle='-', color='green', alpha=0.7)
    if not lstm_yearly_forecast.empty and 'intensity_kt' in lstm_yearly_forecast.columns and not lstm_yearly_forecast['intensity_kt'].isnull().all():
        plt.plot(lstm_yearly_forecast.index, lstm_yearly_forecast['intensity_kt'], label='LSTM Forecast', linestyle='--', color='red', alpha=0.7)
    if not patchtst_yearly_forecast.empty and 'predicted_intensity' in patchtst_yearly_forecast.columns and not patchtst_yearly_forecast['predicted_intensity'].isnull().all():
        plt.plot(patchtst_yearly_forecast.index, patchtst_yearly_forecast['predicted_intensity'], label='PatchTST Forecast', linestyle=':', color='purple', alpha=0.7)

    plt.title(f'Comparison of Typhoon Intensity Forecasts ({year})')
    plt.xlabel('Date')
    plt.ylabel(f'{intensity_column_name} (kt)')
    plt.legend()
    plt.grid(True)
    
    # --- Xử lý giới hạn trục X mạnh mẽ cho Biểu đồ So sánh ---
    # Sử dụng lại logic mạnh mẽ tương tự như trên để nhất quán
    left_xlim_comp = pd.NaT 
    right_xlim_comp = pd.NaT

    # Cố gắng đặt giới hạn bên trái
    if not recent_historical_data.empty and pd.notna(recent_historical_data.index.min()):
        potential_left_hist_comp = recent_historical_data.index.min()
        if isinstance(potential_left_hist_comp, pd.Timestamp):
            left_xlim_comp = max(potential_left_hist_comp, year_start - pd.Timedelta(days=30))
    
    if pd.isna(left_xlim_comp) and not prophet_yearly_forecast.empty and pd.notna(prophet_yearly_forecast['ds'].min()):
        potential_left_forecast_comp = prophet_yearly_forecast['ds'].min()
        if isinstance(potential_left_forecast_comp, pd.Timestamp):
            left_xlim_comp = potential_left_forecast_comp - pd.Timedelta(days=30)
    
    if pd.isna(left_xlim_comp) or not isinstance(left_xlim_comp, pd.Timestamp):
        left_xlim_comp = fallback_left_xlim # Sử dụng fallback từ tính toán trước
        print(f"DEBUG: Comparison Plot {year} - Using fallback_left_xlim: {left_xlim_comp} as primary sources failed.")

    # Cố gắng đặt giới hạn bên phải (xem xét tất cả các dự báo)
    potential_right_forecasts = []
    if not prophet_yearly_forecast.empty and pd.notna(prophet_yearly_forecast['ds'].max()):
        if isinstance(prophet_yearly_forecast['ds'].max(), pd.Timestamp):
            potential_right_forecasts.append(prophet_yearly_forecast['ds'].max())
    if not lstm_yearly_forecast.empty and pd.notna(lstm_yearly_forecast.index.max()):
        if isinstance(lstm_yearly_forecast.index.max(), pd.Timestamp):
            potential_right_forecasts.append(lstm_yearly_forecast.index.max())
    if not patchtst_yearly_forecast.empty and pd.notna(patchtst_yearly_forecast.index.max()):
        if isinstance(patchtst_yearly_forecast.index.max(), pd.Timestamp):
            potential_right_forecasts.append(patchtst_yearly_forecast.index.max())

    if potential_right_forecasts:
        right_xlim_comp = max(potential_right_forecasts) + pd.Timedelta(days=30)
    else:
        # Fallback nếu không có dữ liệu dự báo nào cho năm này
        right_xlim_comp = year_end + pd.Timedelta(days=30)
        if pd.isna(right_xlim_comp) and 'forecast_end_date_overall' in locals() and pd.notna(forecast_end_date_overall):
            right_xlim_comp = forecast_end_date_overall + pd.Timedelta(days=30)
            
    if pd.isna(right_xlim_comp) or not isinstance(right_xlim_comp, pd.Timestamp):
        right_xlim_comp = fallback_right_xlim # Sử dụng fallback từ tính toán trước
        print(f"DEBUG: Comparison Plot {year} - Using fallback_right_xlim: {right_xlim_comp} as primary sources failed.")

    # Xác thực cuối cùng và đặt xlim
    print(f"DEBUG: Comparison Plot {year} - Attempting plt.xlim with left_xlim={left_xlim_comp}, right_xlim={right_xlim_comp}")
    if pd.notna(left_xlim_comp) and pd.notna(right_xlim_comp) and isinstance(left_xlim_comp, pd.Timestamp) and isinstance(right_xlim_comp, pd.Timestamp):
        if left_xlim_comp < right_xlim_comp:
            plt.xlim(left_xlim_comp, right_xlim_comp)
        else:
            print(f"WARNING: Invalid x-axis limits (left_xlim >= right_xlim) for Comparison plot {year} ({left_xlim_comp} vs {right_xlim_comp}). Skipping xlim setting.")
    else:
        print(f"ERROR: Final x-axis limits for Comparison plot {year} are still invalid type/NaN/Inf ({left_xlim_comp}, {right_xlim_comp}). Skipping xlim setting.")


    plt.tight_layout()
    comparison_plot_path = os.path.join(output_dir, f'all_models_forecast_comparison_{year}.png')
    plt.savefig(comparison_plot_path)
    plt.close(plt.gcf())
    comparison_plots_by_year[year] = comparison_plot_path
    print(f"Comparison plot for {year} saved.")

# --- 8. Generate HTML Report with JavaScript Interaction ---
html_report_content = f"""
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dự báo Diễn biến Bão theo Năm</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; background-color: #f8f9fa; color: #343a40; }}
        .container {{ max-width: 1000px; margin: auto; background: #ffffff; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
        h1, h2 {{ color: #007bff; border-bottom: 2px solid #e9ecef; padding-bottom: 10px; margin-bottom: 25px; }}
        img {{ max-width: 100%; height: auto; display: block; margin: 25px auto; border: 1px solid #dee2e6; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
        .model-section {{ margin-bottom: 50px; background-color: #f0f8ff; padding: 20px; border-radius: 10px; border: 1px solid #cfe2ff; }}
        .model-section p {{ margin-bottom: 15px; font-size: 1.05em; }}
        .note {{ background-color: #e6f7ff; border-left: 5px solid #2196f3; padding: 18px; margin-top: 25px; border-radius: 8px; font-style: italic; color: #1a527c; }}
        .footer {{ text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #e9ecef; color: #6c757d; font-size: 0.9em; }}
        .plot-container {{ display: none; }} /* Default hidden */
        .plot-container.active {{ display: block; }} /* Show when active */
        .year-selector {{ margin-bottom: 30px; padding: 15px; background-color: #e9ecef; border-radius: 8px; display: flex; align-items: center; justify-content: center; }}
        .year-selector label {{ margin-right: 15px; font-weight: bold; font-size: 1.1em; color: #343a40; }}
        .year-selector select {{ padding: 10px 15px; border-radius: 5px; border: 1px solid #ced4da; font-size: 1.0em; cursor: pointer; background-color: #ffffff; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Báo cáo Dự báo Diễn biến Bão</h1>
        <p>Báo cáo này trình bày kết quả dự báo cường độ bão sử dụng các mô hình chuỗi thời gian khác nhau trên bộ dữ liệu Digital Typhoon của Kaggle.</p>

        <div class="year-selector">
            <label for="yearSelect">Chọn năm dự báo:</label>
            <select id="yearSelect" onchange="showYearCharts()">
                {"".join(f"<option value='{year}'>{year}</option>" for year in forecast_years)}
            </select>
        </div>

        {"".join(f'''
            <div class="year-section plot-container" id="year_{year}_section">
                <h2>Kết quả Dự báo cho Năm {year}</h2>

                <div class="model-section">
                    <h3>Prophet Forecast ({year})</h3>
                    <img src="prophet_forecast_{year}.png" alt="Prophet Forecast {year}">
                </div>

                <div class="model-section">
                    <h3>LSTM Forecast ({year})</h3>
                    <img src="lstm_forecast_{year}.png" alt="LSTM Forecast {year}">
                </div>

                <div class="model-section">
                    <h3>PatchTST Forecast ({year})</h3>
                    <img src="patchtst_forecast_{year}.png" alt="PatchTST Forecast {year}">
                </div>

                <div class="model-section">
                    <h2>So sánh Dự báo của Các Mô hình cho Năm {year}</h2>
                    <img src="all_models_forecast_comparison_{year}.png" alt="Comparison Forecast {year}">
                    <h3>Nhận xét so sánh:</h3>
                    <ul>
                        <li><b>Prophet:</b> Thường cho thấy xu hướng mượt mà, nắm bắt tốt các thành phần mùa vụ và xu hướng dài hạn. Nó có thể ít phản ứng với các biến động ngắn hạn.</li>
                        <li><b>LSTM:</b> Có khả năng học các phụ thuộc phức tạp và biến động phi tuyến tính. Dự báo của nó có thể phản ánh tốt hơn các thay đổi đột ngột nếu có trong dữ liệu lịch sử.</li>
                        <li><b>PatchTST:</b> Là một mô hình dựa trên Transformer, nó có khả năng nắm bắt các phụ thuộc dài hạn và các mối quan hệ phức tạp giữa các điểm dữ liệu. Nó có thể tạo ra các dự báo chi tiết và nhạy cảm với cấu trúc dữ liệu hơn.</li>
                    </ul>
                    <p>Trong trường hợp không có dữ liệu thực tế cho năm {year}, việc so sánh định lượng là không thể. Tuy nhiên, qua biểu đồ, chúng ta có thể đánh giá mức độ đồng thuận giữa các mô hình về xu hướng tổng thể và biên độ cường độ bão dự kiến. Sự khác biệt đáng kể giữa các mô hình có thể chỉ ra sự không chắc chắn hoặc sự nhạy cảm của chúng đối với các đặc điểm dữ liệu khác nhau.</p>
                </div>
            </div>
        ''' for year in forecast_years)}

        <p class="footer">Báo cáo được tạo tự động bởi mã Python trên Kaggle Notebooks.</p>
    </div>

    <script>
        function showYearCharts() {{
            var selectedYear = document.getElementById('yearSelect').value;
            var yearSections = document.getElementsByClassName('year-section');
            for (var i = 0; i < yearSections.length; i++) {{
                yearSections[i].classList.remove('active');
            }}
            var activeSection = document.getElementById('year_' + selectedYear + '_section');
            if (activeSection) {{
                activeSection.classList.add('active');
            }}
        }}
        // Show charts for the first year by default when page loads
        document.addEventListener('DOMContentLoaded', function() {{
            showYearCharts();
        }});
    </script>
</body>
</html>
"""

with open(os.path.join(output_dir, "typhoon_forecast_report.html"), "w", encoding="utf-8") as f:
    f.write(html_report_content)

print("\n--- HTML Report Generated ---")
print(f"Bạn có thể tìm thấy báo cáo tại: {output_dir}typhoon_forecast_report.html")
print("Tải xuống từ phần 'Output' của Kaggle Notebook của bạn.")


# --- 9. Triển khai Flask để phục vụ báo cáo HTML ---
print("\n--- Deploying Flask to serve HTML report ---")

# 1. Cài đặt Flask (nếu chưa có)
try:
    import flask
except ImportError:
    print("Flask not found, installing...")
    import flask
print("Flask ready.")

from flask import Flask, render_template_string, send_from_directory
import threading
import socket

# 2. Tạo thư mục 'templates' và di chuyển file HTML vào đó
# Đảm bảo file HTML đã được tạo ở bước trước đó.
html_filename = "typhoon_forecast_report.html"
source_html_path = os.path.join(output_dir, html_filename)
templates_dir = os.path.join(output_dir, "templates") # Tạo thư mục templates bên trong output_dir
os.makedirs(templates_dir, exist_ok=True)
destination_html_path = os.path.join(templates_dir, html_filename)

# Kiểm tra nếu file HTML đã tồn tại trước khi cố gắng copy
if os.path.exists(source_html_path):
    # Copy file HTML vào thư mục templates
    os.rename(source_html_path, destination_html_path)
    print(f"Moved {html_filename} to {templates_dir}")
else:
    print(f"Warning: {html_filename} not found at {source_html_path}. Please ensure previous steps ran successfully.")
    # Exit or handle error gracefully if HTML isn't there
    # For now, we'll let Flask try to run and it will likely fail to find the template.


app = Flask(__name__, template_folder=templates_dir)

@app.route('/')
def home():
    try:
        with open(destination_html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return render_template_string(html_content)
    except FileNotFoundError:
        return "Error: HTML report not found. Please run the previous steps to generate it."

# Route để phục vụ các hình ảnh
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(output_dir, filename)


# Hàm để tìm một cổng trống
def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

# Hàm để chạy Flask trong một luồng riêng
def run_flask_app():
    global flask_port
    flask_port = find_free_port()
    print(f"Attempting to run Flask app on port {flask_port}")
    try:
        app.run(host='0.0.0.0', port=flask_port, debug=False, use_reloader=False)
    except Exception as e:
        print(f"Error starting Flask app: {e}")

# Chạy Flask app trong một luồng nền
if __name__ == '__main__':
    if os.path.exists(destination_html_path):
        flask_thread = threading.Thread(target=run_flask_app)
        flask_thread.daemon = True
        flask_thread.start()

        import time
        time.sleep(5) # Đợi 5 giây để server khởi động

        print("\n--- Flask app is running! ---")
        print(f"Bạn có thể truy cập ứng dụng từ trình duyệt của mình tại:")
        print(f"  http://localhost:{flask_port}/ (trong môi trường Kaggle)")
        print("\nLưu ý: Ứng dụng này chỉ chạy trong phiên Kaggle Notebook hiện tại.")
        print("Đóng notebook sẽ dừng ứng dụng Flask.")
        print("\nĐể giữ notebook chạy và Flask hoạt động, hãy đảm bảo ô này không bị dừng.")
    else:
        print("Flask app not started because HTML file was not found or moved successfully.")

  import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 7. Generate HTML Report and Comparison ---
# Đảm bảo output_dir được định nghĩa (ví dụ: output_dir = "/kaggle/working/")
output_dir = "/kaggle/working/"
os.makedirs(output_dir, exist_ok=True)

print(f"\nSaving plots to {output_dir}")

# Store plots for each year
model_plots_by_year = {}
comparison_plots_by_year = {}

# --- Chuẩn bị dữ liệu lịch sử để vẽ (Tập trung) ---
# Đảm bảo các biến này được định nghĩa và có dữ liệu từ các phần trước đó trong notebook của bạn:
# target_series, df_for_patchtst, intensity_column_name, forecast_years
# m (mô hình Prophet đã được fit), lstm_executed, patchtst_executed, forecast_prophet_overall, lstm_forecast, forecast_patchtst_overall_df

# Mốc thời gian bắt đầu vẽ dữ liệu lịch sử (ví dụ: 2 năm trước năm dự báo đầu tiên)
if 'forecast_years' in locals() and forecast_years:
    plot_historical_start_date = pd.to_datetime(f'{min(forecast_years)-2}-01-01')
else:
    print("ERROR: 'forecast_years' is not defined or empty. Cannot determine historical plot start date.")
    # Set a default if forecast_years is not available, but this indicates a problem earlier
    plot_historical_start_date = pd.to_datetime('2020-01-01')


# Sử dụng target_series cho dữ liệu lịch sử (Prophet, LSTM, Comparison)
recent_historical_data = pd.Series()
if 'target_series' in locals() and not target_series.empty and pd.api.types.is_datetime64_any_dtype(target_series.index):
    recent_historical_data = target_series[target_series.index >= plot_historical_start_date].copy()
    if recent_historical_data.empty or recent_historical_data.isnull().all():
        print(f"WARNING: recent_historical_data for plotting is empty or all NaN based on target_series from {plot_historical_start_date}. Plots might not show historical data.")
else:
    print("WARNING: 'target_series' is not available, empty, or its index is not datetime. Historical data for plots might be missing.")

# Sử dụng df_for_patchtst cho PatchTST's historical reference (nếu nó dùng nhiều features)
recent_df_patchtst_cleaned = pd.DataFrame()
if 'df_for_patchtst' in locals() and not df_for_patchtst.empty \
    and intensity_column_name in df_for_patchtst.columns \
    and pd.api.types.is_datetime64_any_dtype(df_for_patchtst.index):
    
    recent_df_patchtst_cleaned = df_for_patchtst[df_for_patchtst.index >= plot_historical_start_date].copy()
    if recent_df_patchtst_cleaned.empty or recent_df_patchtst_cleaned[intensity_column_name].isnull().all():
        print(f"WARNING: recent_df_patchtst_cleaned for plotting is empty or intensity column all NaN from {plot_historical_start_date}. PatchTST historical data might be missing.")
else:
    print("WARNING: 'df_for_patchtst' is not available, empty, or missing intensity column/datetime index. PatchTST historical data might be missing.")


for year in forecast_years:
    year_start = pd.Timestamp(year=year, month=1, day=1)
    year_end = pd.Timestamp(year=year, month=12, day=31, hour=23, minute=59, second=59) 

    # Lọc dữ liệu dự báo cho năm hiện tại
    prophet_yearly_forecast = pd.DataFrame()
    if 'forecast_prophet_overall' in locals() and not forecast_prophet_overall.empty:
        prophet_yearly_forecast = forecast_prophet_overall[(forecast_prophet_overall['ds'] >= year_start) & (forecast_prophet_overall['ds'] <= year_end)].copy()
    else:
        print(f"WARNING: 'forecast_prophet_overall' not available or empty for year {year}. Prophet plot will be skipped for this year.")

    lstm_yearly_forecast = pd.DataFrame()
    if 'lstm_forecast' in locals() and 'lstm_executed' in locals() and lstm_executed and not lstm_forecast.empty:
        lstm_yearly_forecast = lstm_forecast[(lstm_forecast.index >= year_start) & (lstm_forecast.index <= year_end)].copy()
    else:
        print(f"WARNING: 'lstm_forecast' not available or empty for year {year} or LSTM not executed. LSTM plot will be skipped for this year.")

    patchtst_yearly_forecast = pd.DataFrame()
    if 'forecast_patchtst_overall_df' in locals() and 'patchtst_executed' in locals() and patchtst_executed and not forecast_patchtst_overall_df.empty:
        patchtst_yearly_forecast = forecast_patchtst_overall_df[(forecast_patchtst_overall_df.index >= year_start) & (forecast_patchtst_overall_df.index <= year_end)].copy()
    else:
        print(f"WARNING: 'forecast_patchtst_overall_df' not available or empty for year {year} or PatchTST not executed. PatchTST plot will be skipped for this year.")


    # --- Lưu biểu đồ riêng lẻ cho từng mô hình theo năm ---

    # Biểu đồ Prophet
    if not prophet_yearly_forecast.empty and 'yhat' in prophet_yearly_forecast.columns and not prophet_yearly_forecast['yhat'].isnull().all():
        if 'm' in locals(): # Ensure prophet model 'm' exists
            fig_prophet_year = m.plot(forecast_prophet_overall) # Vẽ toàn bộ dự báo để có ngữ cảnh
            ax = fig_prophet_year.gca()
            ax.set_title(f'Prophet Forecast of Typhoon Intensity ({year})')
            ax.set_xlabel('Date')
            ax.set_ylabel(f'{intensity_column_name} (kt)')
            
            # Thêm đường thẳng để chỉ rõ ranh giới giữa lịch sử và dự báo
            if not recent_historical_data.empty and pd.notna(recent_historical_data.index.max()):
                ax.axvline(recent_historical_data.index.max(), color='grey', linestyle='--', linewidth=1, label='End of Historical Data')

            # Highlight khu vực năm dự báo hiện tại
            ax.axvspan(year_start, year_end, color='gray', alpha=0.2, label=f'{year} Forecast Period')
            ax.legend()
            plt.tight_layout()
            prophet_plot_path = os.path.join(output_dir, f'prophet_forecast_{year}.png')
            fig_prophet_year.savefig(prophet_plot_path)
            plt.close(fig_prophet_year)
            if year not in model_plots_by_year: model_plots_by_year[year] = {}
            model_plots_by_year[year]['prophet'] = prophet_plot_path
            print(f"Prophet plot for {year} saved.")
        else:
            print(f"WARNING: Prophet model 'm' not found. Cannot plot Prophet forecast for {year}.")
    else:
        print(f"Prophet forecast for {year} is empty or all NaN in 'yhat' column. Skipping plot.")

    # Biểu đồ LSTM
    if not lstm_yearly_forecast.empty and 'intensity_kt' in lstm_yearly_forecast.columns and not lstm_yearly_forecast['intensity_kt'].isnull().all():
        plt.figure(figsize=(14, 7))
        if not recent_historical_data.empty and not recent_historical_data.isnull().all():
            plt.plot(recent_historical_data.index, recent_historical_data.values, label='Actual Intensity (Historical)', color='blue', alpha=0.8)
        
        plt.plot(lstm_yearly_forecast.index, lstm_yearly_forecast['intensity_kt'], label='LSTM Forecast', linestyle='--', color='red') 
        
        plt.title(f'LSTM Forecast of Typhoon Intensity ({year})')
        plt.xlabel('Date')
        plt.ylabel(f'{intensity_column_name} (kt)')
        plt.legend()
        plt.grid(True)

        # --- Xử lý giới hạn trục X mạnh mẽ cho Biểu đồ LSTM ---
        # Định nghĩa các giới hạn fallback vững chắc là các đối tượng Timestamp thực sự
        fallback_left_xlim = pd.Timestamp(year=max(2000, year_start.year - 2), month=1, day=1) 
        fallback_right_xlim = pd.Timestamp(year=year_end.year + 1, month=1, day=1) 

        left_xlim = pd.NaT 
        right_xlim = pd.NaT

        # Cố gắng đặt giới hạn bên trái
        if not recent_historical_data.empty and pd.notna(recent_historical_data.index.min()):
            potential_left_hist = recent_historical_data.index.min()
            if isinstance(potential_left_hist, pd.Timestamp):
                left_xlim = max(potential_left_hist, year_start - pd.Timedelta(days=30))
        
        if pd.isna(left_xlim) and not lstm_yearly_forecast.empty and pd.notna(lstm_yearly_forecast.index.min()):
            potential_left_forecast = lstm_yearly_forecast.index.min()
            if isinstance(potential_left_forecast, pd.Timestamp):
                left_xlim = potential_left_forecast - pd.Timedelta(days=30)
        
        # Nếu vẫn không hợp lệ, sử dụng fallback
        if pd.isna(left_xlim) or not isinstance(left_xlim, pd.Timestamp):
            left_xlim = fallback_left_xlim
            print(f"DEBUG: LSTM Plot {year} - Using fallback_left_xlim: {left_xlim} as primary sources failed.")

        # Cố gắng đặt giới hạn bên phải
        if not lstm_yearly_forecast.empty and pd.notna(lstm_yearly_forecast.index.max()):
            potential_right_forecast = lstm_yearly_forecast.index.max()
            if isinstance(potential_right_forecast, pd.Timestamp):
                right_xlim = potential_right_forecast + pd.Timedelta(days=30)
        
        if pd.isna(right_xlim) or not isinstance(right_xlim, pd.Timestamp):
            if pd.notna(year_end):
                right_xlim = year_end + pd.Timedelta(days=30)
            elif 'forecast_end_date_overall' in locals() and pd.notna(forecast_end_date_overall) and isinstance(forecast_end_date_overall, pd.Timestamp):
                right_xlim = forecast_end_date_overall + pd.Timedelta(days=30)
            
        if pd.isna(right_xlim) or not isinstance(right_xlim, pd.Timestamp):
            right_xlim = fallback_right_xlim
            print(f"DEBUG: LSTM Plot {year} - Using fallback_right_xlim: {right_xlim} as primary sources failed.")

        # Xác thực cuối cùng và đặt xlim
        print(f"DEBUG: LSTM Plot {year} - Attempting plt.xlim with left_xlim={left_xlim}, right_xlim={right_xlim}")
        if pd.notna(left_xlim) and pd.notna(right_xlim) and isinstance(left_xlim, pd.Timestamp) and isinstance(right_xlim, pd.Timestamp):
            if left_xlim < right_xlim:
                plt.xlim(left_xlim, right_xlim)
            else:
                print(f"WARNING: Invalid x-axis limits (left_xlim >= right_xlim) for LSTM plot {year} ({left_xlim} vs {right_xlim}). Skipping xlim setting.")
        else:
            print(f"ERROR: Final x-axis limits for LSTM plot {year} are still invalid type/NaN/Inf ({left_xlim}, {right_xlim}). Skipping xlim setting.")

        plt.tight_layout()
        lstm_plot_path = os.path.join(output_dir, f'lstm_forecast_{year}.png')
        plt.savefig(lstm_plot_path)
        plt.close(plt.gcf())
        if year not in model_plots_by_year: model_plots_by_year[year] = {}
        model_plots_by_year[year]['lstm'] = lstm_plot_path
        print(f"LSTM plot for {year} saved.")
    else:
        print(f"LSTM forecast for {year} is empty or all NaN in 'intensity_kt' column, or LSTM was not executed. Skipping plot.")

    # Biểu đồ PatchTST
    # Đảm bảo bạn đã kiểm tra tên cột dự đoán của PatchTST, ở đây tôi đang dùng 'predicted_intensity'
    if not patchtst_yearly_forecast.empty and 'predicted_intensity' in patchtst_yearly_forecast.columns and not patchtst_yearly_forecast['predicted_intensity'].isnull().all():
        plt.figure(figsize=(14, 7))
        if not recent_df_patchtst_cleaned.empty and intensity_column_name in recent_df_patchtst_cleaned.columns and not recent_df_patchtst_cleaned[intensity_column_name].isnull().all():
            plt.plot(recent_df_patchtst_cleaned.index, recent_df_patchtst_cleaned[intensity_column_name].values, label='Actual Intensity (Historical)', color='blue', alpha=0.8)
        
        plt.plot(patchtst_yearly_forecast.index, patchtst_yearly_forecast['predicted_intensity'], label='PatchTST Forecast', linestyle=':', color='purple')
        
        plt.title(f'PatchTST Forecast of Typhoon Intensity ({year})')
        plt.xlabel('Date')
        plt.ylabel(f'{intensity_column_name} (kt)')
        plt.legend()
        plt.grid(True)
        
        # --- Xử lý giới hạn trục X mạnh mẽ cho Biểu đồ PatchTST ---
        fallback_left_xlim_patchtst = pd.Timestamp(year=max(2000, year_start.year - 2), month=1, day=1) 
        fallback_right_xlim_patchtst = pd.Timestamp(year=year_end.year + 1, month=1, day=1) 

        left_xlim_patchtst = pd.NaT 
        right_xlim_patchtst = pd.NaT

        # Cố gắng đặt giới hạn bên trái
        if not recent_df_patchtst_cleaned.empty and pd.notna(recent_df_patchtst_cleaned.index.min()):
            potential_left_hist_patchtst = recent_df_patchtst_cleaned.index.min()
            if isinstance(potential_left_hist_patchtst, pd.Timestamp):
                left_xlim_patchtst = max(potential_left_hist_patchtst, year_start - pd.Timedelta(days=30))
        
        if pd.isna(left_xlim_patchtst) and not patchtst_yearly_forecast.empty and pd.notna(patchtst_yearly_forecast.index.min()):
            potential_left_forecast_patchtst = patchtst_yearly_forecast.index.min()
            if isinstance(potential_left_forecast_patchtst, pd.Timestamp):
                left_xlim_patchtst = potential_left_forecast_patchtst - pd.Timedelta(days=30)
        
        if pd.isna(left_xlim_patchtst) or not isinstance(left_xlim_patchtst, pd.Timestamp):
            left_xlim_patchtst = fallback_left_xlim_patchtst
            print(f"DEBUG: PatchTST Plot {year} - Using fallback_left_xlim: {left_xlim_patchtst} as primary sources failed.")

        # Cố gắng đặt giới hạn bên phải
        if not patchtst_yearly_forecast.empty and pd.notna(patchtst_yearly_forecast.index.max()):
            potential_right_forecast_patchtst = patchtst_yearly_forecast.index.max()
            if isinstance(potential_right_forecast_patchtst, pd.Timestamp):
                right_xlim_patchtst = potential_right_forecast_patchtst + pd.Timedelta(days=30)
        
        if pd.isna(right_xlim_patchtst) or not isinstance(right_xlim_patchtst, pd.Timestamp):
            if pd.notna(year_end):
                right_xlim_patchtst = year_end + pd.Timedelta(days=30)
            elif 'forecast_end_date_overall' in locals() and pd.notna(forecast_end_date_overall) and isinstance(forecast_end_date_overall, pd.Timestamp):
                right_xlim_patchtst = forecast_end_date_overall + pd.Timedelta(days=30)
            
        if pd.isna(right_xlim_patchtst) or not isinstance(right_xlim_patchtst, pd.Timestamp):
            right_xlim_patchtst = fallback_right_xlim_patchtst
            print(f"DEBUG: PatchTST Plot {year} - Using fallback_right_xlim: {right_xlim_patchtst} as primary sources failed.")

        # Xác thực cuối cùng và đặt xlim
        print(f"DEBUG: PatchTST Plot {year} - Attempting plt.xlim with left_xlim={left_xlim_patchtst}, right_xlim={right_xlim_patchtst}")
        if pd.notna(left_xlim_patchtst) and pd.notna(right_xlim_patchtst) and isinstance(left_xlim_patchtst, pd.Timestamp) and isinstance(right_xlim_patchtst, pd.Timestamp):
            if left_xlim_patchtst < right_xlim_patchtst:
                plt.xlim(left_xlim_patchtst, right_xlim_patchtst)
            else:
                print(f"WARNING: Invalid x-axis limits (left_xlim >= right_xlim) for PatchTST plot {year} ({left_xlim_patchtst} vs {right_xlim_patchtst}). Skipping xlim setting.")
        else:
            print(f"ERROR: Final x-axis limits for PatchTST plot {year} are still invalid type/NaN/Inf ({left_xlim_patchtst}, {right_xlim_patchtst}). Skipping xlim setting.")

        plt.tight_layout()
        patchtst_plot_path = os.path.join(output_dir, f'patchtst_forecast_{year}.png')
        plt.savefig(patchtst_plot_path)
        plt.close(plt.gcf())
        if year not in model_plots_by_year: model_plots_by_year[year] = {}
        model_plots_by_year[year]['patchtst'] = patchtst_plot_path
        print(f"PatchTST plot for {year} saved.")
    else:
        print(f"PatchTST forecast for {year} is empty or all NaN in 'predicted_intensity' column, or PatchTST was not executed. Skipping plot.")

    # --- Lưu Biểu đồ So sánh theo năm ---
    plt.figure(figsize=(18, 9))
    # Chỉ vẽ dữ liệu lịch sử nếu có và không phải tất cả là NaN
    if not recent_historical_data.empty and not recent_historical_data.isnull().all():
        plt.plot(recent_historical_data.index, recent_historical_data.values, label='Actual Intensity (Historical)', color='blue', alpha=0.8)

    if not prophet_yearly_forecast.empty and 'yhat' in prophet_yearly_forecast.columns and not prophet_yearly_forecast['yhat'].isnull().all():
        plt.plot(prophet_yearly_forecast['ds'], prophet_yearly_forecast['yhat'], label='Prophet Forecast', linestyle='-', color='green', alpha=0.7)
    if not lstm_yearly_forecast.empty and 'intensity_kt' in lstm_yearly_forecast.columns and not lstm_yearly_forecast['intensity_kt'].isnull().all():
        plt.plot(lstm_yearly_forecast.index, lstm_yearly_forecast['intensity_kt'], label='LSTM Forecast', linestyle='--', color='red', alpha=0.7)
    if not patchtst_yearly_forecast.empty and 'predicted_intensity' in patchtst_yearly_forecast.columns and not patchtst_yearly_forecast['predicted_intensity'].isnull().all():
        plt.plot(patchtst_yearly_forecast.index, patchtst_yearly_forecast['predicted_intensity'], label='PatchTST Forecast', linestyle=':', color='purple', alpha=0.7)

    plt.title(f'Comparison of Typhoon Intensity Forecasts ({year})')
    plt.xlabel('Date')
    plt.ylabel(f'{intensity_column_name} (kt)')
    plt.legend()
    plt.grid(True)
    
    # --- Xử lý giới hạn trục X mạnh mẽ cho Biểu đồ So sánh ---
    # Sử dụng lại logic mạnh mẽ tương tự như trên để nhất quán
    left_xlim_comp = pd.NaT 
    right_xlim_comp = pd.NaT

    # Cố gắng đặt giới hạn bên trái
    if not recent_historical_data.empty and pd.notna(recent_historical_data.index.min()):
        potential_left_hist_comp = recent_historical_data.index.min()
        if isinstance(potential_left_hist_comp, pd.Timestamp):
            left_xlim_comp = max(potential_left_hist_comp, year_start - pd.Timedelta(days=30))
    
    if pd.isna(left_xlim_comp) and not prophet_yearly_forecast.empty and pd.notna(prophet_yearly_forecast['ds'].min()):
        potential_left_forecast_comp = prophet_yearly_forecast['ds'].min()
        if isinstance(potential_left_forecast_comp, pd.Timestamp):
            left_xlim_comp = potential_left_forecast_comp - pd.Timedelta(days=30)
    
    if pd.isna(left_xlim_comp) or not isinstance(left_xlim_comp, pd.Timestamp):
        left_xlim_comp = fallback_left_xlim # Sử dụng fallback từ tính toán trước
        print(f"DEBUG: Comparison Plot {year} - Using fallback_left_xlim: {left_xlim_comp} as primary sources failed.")

    # Cố gắng đặt giới hạn bên phải (xem xét tất cả các dự báo)
    potential_right_forecasts = []
    if not prophet_yearly_forecast.empty and pd.notna(prophet_yearly_forecast['ds'].max()):
        if isinstance(prophet_yearly_forecast['ds'].max(), pd.Timestamp):
            potential_right_forecasts.append(prophet_yearly_forecast['ds'].max())
    if not lstm_yearly_forecast.empty and pd.notna(lstm_yearly_forecast.index.max()):
        if isinstance(lstm_yearly_forecast.index.max(), pd.Timestamp):
            potential_right_forecasts.append(lstm_yearly_forecast.index.max())
    if not patchtst_yearly_forecast.empty and pd.notna(patchtst_yearly_forecast.index.max()):
        if isinstance(patchtst_yearly_forecast.index.max(), pd.Timestamp):
            potential_right_forecasts.append(patchtst_yearly_forecast.index.max())

    if potential_right_forecasts:
        right_xlim_comp = max(potential_right_forecasts) + pd.Timedelta(days=30)
    else:
        # Fallback nếu không có dữ liệu dự báo nào cho năm này
        right_xlim_comp = year_end + pd.Timedelta(days=30)
        if pd.isna(right_xlim_comp) and 'forecast_end_date_overall' in locals() and pd.notna(forecast_end_date_overall):
            right_xlim_comp = forecast_end_date_overall + pd.Timedelta(days=30)
            
    if pd.isna(right_xlim_comp) or not isinstance(right_xlim_comp, pd.Timestamp):
        right_xlim_comp = fallback_right_xlim # Sử dụng fallback từ tính toán trước
        print(f"DEBUG: Comparison Plot {year} - Using fallback_right_xlim: {right_xlim_comp} as primary sources failed.")

    # Xác thực cuối cùng và đặt xlim
    print(f"DEBUG: Comparison Plot {year} - Attempting plt.xlim with left_xlim={left_xlim_comp}, right_xlim={right_xlim_comp}")
    if pd.notna(left_xlim_comp) and pd.notna(right_xlim_comp) and isinstance(left_xlim_comp, pd.Timestamp) and isinstance(right_xlim_comp, pd.Timestamp):
        if left_xlim_comp < right_xlim_comp:
            plt.xlim(left_xlim_comp, right_xlim_comp)
        else:
            print(f"WARNING: Invalid x-axis limits (left_xlim >= right_xlim) for Comparison plot {year} ({left_xlim_comp} vs {right_xlim_comp}). Skipping xlim setting.")
    else:
        print(f"ERROR: Final x-axis limits for Comparison plot {year} are still invalid type/NaN/Inf ({left_xlim_comp}, {right_xlim_comp}). Skipping xlim setting.")


    plt.tight_layout()
    comparison_plot_path = os.path.join(output_dir, f'all_models_forecast_comparison_{year}.png')
    plt.savefig(comparison_plot_path)
    plt.close(plt.gcf())
    comparison_plots_by_year[year] = comparison_plot_path
    print(f"Comparison plot for {year} saved.")


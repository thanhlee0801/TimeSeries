import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX # SARIMAX cho Volume
from statsmodels.tsa.arima.model import ARIMA # ARIMA cho các cột giá
import numpy as np
import warnings

warnings.filterwarnings("ignore") # Tắt cảnh báo để output dễ đọc hơn

# --- Cấu hình đường dẫn file ---
input_file_path = 'GOOGL_2006-01-01_to_2018-01-01.csv'
output_file_path_arima_sarima_final_imputation = 'GOOGL_2006-01-01_to_2018-01-01_ARIMA_SARIMA_Final_Imputation.csv'

# --- Bước 1: Đọc và tiền xử lý dữ liệu gốc ---
try:
    data_original = pd.read_csv(input_file_path, parse_dates=['Date'], index_col='Date').sort_index()
    print("Dữ liệu gốc 5 dòng đầu tiên:")
    print(data_original.head())
    print("\nThông tin dữ liệu gốc:")
    data_original.info()

except Exception as e:
    print(f"Lỗi khi đọc file: {e}")
    exit()

# --- Bước 2: Tạo chuỗi với các ngày thiếu (bao gồm cuối tuần/ngày lễ) cho TẤT CẢ CÁC CỘT ---
start_date = data_original.index.min()
end_date = data_original.index.max()
full_date_range_daily = pd.date_range(start=start_date, end=end_date, freq='D')

# Reindex toàn bộ DataFrame để chèn các ngày bị thiếu, tạo ra NaN trong tất cả các cột
data_with_nans = data_original.reindex(full_date_range_daily)

# --- Tạo biến ngoại sinh is_weekend ---
# 5=Thứ Bảy, 6=Chủ Nhật. Đây là một biến quan trọng để mô hình học được hành vi khác biệt của ngày nghỉ.
data_with_nans['is_weekend'] = data_with_nans.index.dayofweek.isin([5, 6]).astype(int)

print("\nDataFrame với các ngày thiếu (NaNs) - 5 dòng đầu:")
print(data_with_nans.head(10))
print(f"\nSố lượng NaNs mỗi cột trước khi điền:\n{data_with_nans.isnull().sum()}")

# --- Bước 3: Nội suy dữ liệu thiếu bằng SARIMA cho Volume và ARIMA cho các cột giá ---

# Các cột giá
price_cols = ['Open', 'High', 'Low', 'Close']
# Cột Volume
volume_col = 'Volume'

# Tạo một DataFrame mới để lưu kết quả đã điền
data_filled_all_cols = data_with_nans.copy()

# --- Xử lý cột Volume bằng SARIMA ---
print(f"\n--- Xử lý cột: {volume_col} bằng SARIMA ---")
current_volume_series = data_with_nans[volume_col]
exog_volume = data_with_nans[['is_weekend']] # Biến ngoại sinh cho Volume

if current_volume_series.isnull().all():
    print(f"Cột '{volume_col}' rỗng hoàn toàn, bỏ qua SARIMA và giữ nguyên NaNs.")
else:
    # Điền tạm thời NaNs bằng interpolate để đảm bảo chuỗi liên tục cho mô hình.
    # CHÚ Ý: Tại đây chúng ta sẽ không dùng ffill/bfill thô bạo.
    # Thay vào đó, chúng ta sẽ dựa vào interpolate và fallback.
    volume_series_temp_filled = current_volume_series.interpolate(method='linear', limit_direction='both')
    # Nếu vẫn còn NaN (ví dụ: khoảng trống rất lớn ở đầu/cuối), điền bằng giá trị trung bình/trung vị
    if volume_series_temp_filled.isnull().any():
        print(f"Cảnh báo: interpolate không điền hết NaNs cho {volume_col} trong bước điền tạm thời. Sử dụng giá trị trung bình.")
        volume_series_temp_filled = volume_series_temp_filled.fillna(current_volume_series.mean())
        if volume_series_temp_filled.isnull().any(): # Nếu chuỗi rỗng hoàn toàn ban đầu
            volume_series_temp_filled = volume_series_temp_filled.fillna(0) # Đảm bảo không có NaN khi truyền vào mô hình

    # --- Phân tích tính dừng và mùa vụ cho Volume (sau khi điền tạm thời) ---
    # Các biểu đồ này giúp bạn tự xác định các bậc p,d,q,P,D,Q,S
    print(f"\n--- Phân tích ACF/PACF cho {volume_col} (sau khi điền tạm thời) ---")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Phân tích tính dừng và mùa vụ cho {volume_col}', fontsize=16)
    axes[0, 0].plot(volume_series_temp_filled)
    axes[0, 0].set_title(f'Chuỗi thời gian {volume_col}')
    plot_acf(volume_series_temp_filled, lags=40, ax=axes[0, 1], title=f'ACF - {volume_col}')
    plot_pacf(volume_series_temp_filled, lags=40, ax=axes[1, 0], title=f'PACF - {volume_col}', method='ywm')
    volume_diff = volume_series_temp_filled.diff().dropna()
    axes[1, 1].plot(volume_diff)
    axes[1, 1].set_title(f'Sai phân bậc 1 của {volume_col}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    volume_seasonal_diff = volume_series_temp_filled.diff(periods=7).dropna()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Phân tích ACF/PACF cho sai phân mùa vụ của {volume_col} (S=7)', fontsize=16)
    plot_acf(volume_seasonal_diff, lags=40, ax=axes[0], title=f'ACF - Sai phân mùa vụ {volume_col}')
    plot_pacf(volume_seasonal_diff, lags=40, ax=axes[1], title=f'PACF - Sai phân mùa vụ {volume_col}', method='ywm')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- ĐỀ XUẤT BẬC SARIMA DỰA TRÊN QUAN SÁT CÁC BIỂU ĐỒ TRÊN ---
    # Dựa trên phân tích của bạn và quan sát biểu đồ, đây là điểm khởi đầu hợp lý.
    # Tuy nhiên, bạn vẫn cần xem lại các biểu đồ ACF/PACF cuối cùng để tinh chỉnh.
    # d=0, D=1 (chu kỳ 7) cho Volume là hợp lý nếu có tính mùa vụ mạnh
    order_volume = (1, 0, 1) # Ví dụ: Thử AR(1) và MA(1) phi-mùa vụ
    seasonal_order_volume = (1, 1, 0, 7) # Ví dụ: AR(1) mùa vụ và sai phân mùa vụ bậc 1, chu kỳ 7
    print(f"\nĐề xuất bậc SARIMA: {order_volume}{seasonal_order_volume} cho cột '{volume_col}' (Cần điều chỉnh sau khi xem biểu đồ)")

    try:
        model_sarima_volume = SARIMAX(volume_series_temp_filled,
                                      exog=exog_volume,
                                      order=order_volume,
                                      seasonal_order=seasonal_order_volume,
                                      enforce_stationarity=False,
                                      enforce_invertibility=False)
        model_sarima_volume_fit = model_sarima_volume.fit(disp=False) # disp=False để không hiển thị quá trình tối ưu hóa
        print(model_sarima_volume_fit.summary())

        fitted_volume_values = model_sarima_volume_fit.fittedvalues
        nan_indices_volume = data_with_nans[volume_col].isnull()
        if nan_indices_volume.any():
            data_filled_all_cols.loc[nan_indices_volume, volume_col] = fitted_volume_values.loc[nan_indices_volume]

    except Exception as e:
        print(f"Lỗi khi huấn luyện/dự đoán SARIMA cho cột '{volume_col}': {e}.")
        print(f"Không thể điền bằng SARIMA. Đang thử interpolate(method='linear') làm phương án dự phòng.")
        # Fallback to linear interpolation if SARIMA fails
        data_filled_all_cols[volume_col] = data_with_nans[volume_col].interpolate(method='linear', limit_direction='both')
        if data_filled_all_cols[volume_col].isnull().any():
            print(f"Interpolate cũng không thể điền hết. Cuối cùng điền bằng mean.")
            data_filled_all_cols.loc[data_filled_all_cols[volume_col].isnull(), volume_col] = current_volume_series.mean()
            if data_filled_all_cols[volume_col].isnull().any():
                 print(f"Vẫn còn NaNs sau khi điền bằng mean. Giữ nguyên NaN cho cột '{volume_col}'.")


# --- Xử lý các cột giá (Open, High, Low, Close) bằng ARIMA ---
for col in price_cols:
    print(f"\n--- Xử lý cột: {col} bằng ARIMA ---")
    current_series = data_with_nans[col]
    exog_price = data_with_nans[['is_weekend']] # Biến ngoại sinh cho giá

    if current_series.isnull().all():
        print(f"Cột '{col}' rỗng hoàn toàn, bỏ qua ARIMA và giữ nguyên NaNs.")
        continue

    # Điền tạm thời NaNs bằng interpolate
    series_temp_filled = current_series.interpolate(method='linear', limit_direction='both')
    if series_temp_filled.isnull().any():
        print(f"Cảnh báo: interpolate không điền hết NaNs cho {col} trong bước điền tạm thời. Sử dụng giá trị trung bình.")
        series_temp_filled = series_temp_filled.fillna(current_series.mean())
        if series_temp_filled.isnull().any():
            series_temp_filled = series_temp_filled.fillna(current_series.dropna().iloc[0]) # Fallback to first valid value

    # --- Phân tích tính dừng cho cột giá (sau khi điền tạm thời) ---
    # Các biểu đồ này giúp bạn tự xác định các bậc p,d,q
    print(f"\n--- Phân tích ACF/PACF cho {col} (sau khi điền tạm thời) ---")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Phân tích tính dừng cho {col}', fontsize=16)
    axes[0, 0].plot(series_temp_filled)
    axes[0, 0].set_title(f'Chuỗi thời gian {col}')
    plot_acf(series_temp_filled, lags=40, ax=axes[0, 1], title=f'ACF - {col}')
    plot_pacf(series_temp_filled, lags=40, ax=axes[1, 0], title=f'PACF - {col}', method='ywm')
    price_diff = series_temp_filled.diff().dropna()
    axes[1, 1].plot(price_diff)
    axes[1, 1].set_title(f'Sai phân bậc 1 của {col}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- ĐỀ XUẤT BẬC ARIMA DỰA TRÊN QUAN SÁT CÁC BIỂU ĐỒ TRÊN ---
    # Theo phân tích của bạn, ARIMA(0,1,0) (Random Walk) là điểm khởi đầu tốt.
    # Tuy nhiên, bạn vẫn cần xem lại các biểu đồ ACF/PACF cuối cùng để tinh chỉnh.
    order_price = (0, 1, 0) # d=1 (sai phân bậc 1), p=0, q=0 (Random Walk)
    print(f"\nĐề xuất bậc ARIMA: {order_price} cho cột '{col}' (Cần điều chỉnh sau khi xem biểu đồ)")

    try:
        model_arima_price = ARIMA(series_temp_filled,
                                  exog=exog_price,
                                  order=order_price)
        model_arima_price_fit = model_arima_price.fit(disp=False)
        print(model_arima_price_fit.summary())

        fitted_price_values = model_arima_price_fit.fittedvalues
        nan_indices_price = data_with_nans[col].isnull()
        if nan_indices_price.any():
            data_filled_all_cols.loc[nan_indices_price, col] = fitted_price_values.loc[nan_indices_price]

    except Exception as e:
        print(f"Lỗi khi huấn luyện/dự đoán ARIMA cho cột '{col}': {e}.")
        print(f"Không thể điền bằng ARIMA. Đang thử interpolate(method='linear') làm phương án dự phòng.")
        # Fallback to linear interpolation if ARIMA fails
        data_filled_all_cols[col] = data_with_nans[col].interpolate(method='linear', limit_direction='both')
        if data_filled_all_cols[col].isnull().any():
            print(f"Interpolate cũng không thể điền hết. Cuối cùng điền bằng mean.")
            data_filled_all_cols.loc[data_filled_all_cols[col].isnull(), col] = current_series.mean()
            if data_filled_all_cols[col].isnull().any():
                 print(f"Vẫn còn NaNs sau khi điền bằng mean. Giữ nguyên NaN cho cột '{col}'.")


# --- Bước 4: Kiểm tra và điều chỉnh ràng buộc Low/High/Open/Close ---
print("\nKiểm tra và điều chỉnh ràng buộc Low/High/Open/Close sau imputation...")
for index, row in data_filled_all_cols.iterrows():
    # Sử dụng np.isclose để so sánh số thực, tránh lỗi làm tròn
    # Đảm bảo Low <= Open, Close <= High
    # Low không được lớn hơn Open hoặc Close
    if not np.isclose(row['Low'], row['Open']) and row['Low'] > row['Open']:
        data_filled_all_cols.loc[index, 'Low'] = row['Open']
    if not np.isclose(row['Low'], row['Close']) and row['Low'] > row['Close']:
        data_filled_all_cols.loc[index, 'Low'] = min(data_filled_all_cols.loc[index, 'Low'], row['Close'])

    # High không được nhỏ hơn Open hoặc Close
    if not np.isclose(row['High'], row['Open']) and row['High'] < row['Open']:
        data_filled_all_cols.loc[index, 'High'] = row['Open']
    if not np.isclose(row['High'], row['Close']) and row['High'] < row['Close']:
        data_filled_all_cols.loc[index, 'High'] = max(data_filled_all_cols.loc[index, 'High'], row['Close'])

    # Đảm bảo Open và Close nằm trong khoảng Low-High sau điều chỉnh Low/High
    data_filled_all_cols.loc[index, 'Open'] = np.clip(row['Open'], data_filled_all_cols.loc[index, 'Low'], data_filled_all_cols.loc[index, 'High'])
    data_filled_all_cols.loc[index, 'Close'] = np.clip(row['Close'], data_filled_all_cols.loc[index, 'Low'], data_filled_all_cols.loc[index, 'High'])

print(f"\nSố lượng NaNs mỗi cột sau khi điền và điều chỉnh:\n{data_filled_all_cols.isnull().sum()}")
print("\nDataFrame đã điền hoàn chỉnh (5 dòng đầu):")
print(data_filled_all_cols.head(10))
print("\nDataFrame đã điền hoàn chỉnh (5 dòng cuối):")
print(data_filled_all_cols.tail(10))


# --- Bước 5: Lưu dữ liệu đã điền vào một file CSV mới ---
data_filled_all_cols.to_csv(output_file_path_arima_sarima_final_imputation, index=True)
print(f"\nDữ liệu đã điền bằng ARIMA/SARIMA (có biến ngoại sinh và không ffill) đã được lưu vào: {output_file_path_arima_sarima_final_imputation}")


# --- Bước 6: Vẽ đồ thị chuỗi thời gian, ACF và PACF của dữ liệu đã điền ---
# Vẽ đồ thị cho cột Close
plt.figure(figsize=(14, 7))
plt.plot(data_filled_all_cols['Close'])
plt.title('Đồ thị Chuỗi Thời Gian Giá Đóng Cửa GOOGL (Đã điền ARIMA có Exog)', fontsize=16)
plt.xlabel('Thời gian', fontsize=12)
plt.ylabel('Giá đóng cửa (USD)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
plot_acf(data_filled_all_cols['Close'], lags=40, ax=axes[0])
axes[0].set_title('ACF - Giá Đóng Cửa GOOGL (Đã điền ARIMA có Exog)', fontsize=14)
axes[0].set_xlabel('Độ trễ', fontsize=12)
axes[0].set_ylabel('Hệ số tự tương quan', fontsize=12)
axes[0].grid(True, linestyle='--', alpha=0.7)
plot_pacf(data_filled_all_cols['Close'], lags=40, ax=axes[1], method='ywm')
axes[1].set_title('PACF - Giá Đóng Cửa GOOGL (Đã điền ARIMA có Exog)', fontsize=14)
axes[1].set_xlabel('Độ trễ', fontsize=12)
axes[1].set_ylabel('Hệ số tự tương quan riêng phần', fontsize=12)
axes[1].grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Vẽ đồ thị cho cột Volume
plt.figure(figsize=(14, 7))
plt.plot(data_filled_all_cols['Volume'])
plt.title('Đồ thị Chuỗi Thời Gian Khối Lượng Giao Dịch GOOGL (Đã điền SARIMA có Exog)', fontsize=16)
plt.xlabel('Thời gian', fontsize=12)
plt.ylabel('Khối lượng', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
plot_acf(data_filled_all_cols['Volume'], lags=40, ax=axes[0])
axes[0].set_title('ACF - Khối Lượng Giao Dịch GOOGL (Đã điền SARIMA có Exog)', fontsize=14)
axes[0].set_xlabel('Độ trễ', fontsize=12)
axes[0].set_ylabel('Hệ số tự tương quan', fontsize=12)
axes[0].grid(True, linestyle='--', alpha=0.7)
plot_pacf(data_filled_all_cols['Volume'], lags=40, ax=axes[1], method='ywm')
axes[1].set_title('PACF - Khối Lượng Giao Dịch GOOGL (Đã điền SARIMA có Exog)', fontsize=14)
axes[1].set_xlabel('Độ trễ', fontsize=12)
axes[1].set_ylabel('Hệ số tự tương quan riêng phần', fontsize=12)
axes[1].grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# --- Bước 1: Đọc dữ liệu từ file CSV ---
file_path = 'GOOGL_2006-01-01_to_2018-01-01.csv' # Đảm bảo file này nằm cùng thư mục hoặc cung cấp đường dẫn đầy đủ

try:
    # Đọc file CSV, phân tích cột 'Date' thành định dạng ngày tháng và đặt làm chỉ mục
    data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    print("Dữ liệu 5 dòng đầu tiên:")
    print(data.head())
    print("\nThông tin dữ liệu:")
    data.info()

    # Chọn cột 'Close' (giá đóng cửa) để phân tích chuỗi thời gian
    price_series = data['Close']

except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file tại đường dẫn '{file_path}'. Vui lòng kiểm tra lại tên file hoặc đường dẫn.")
    exit() # Thoát chương trình nếu không tìm thấy file
except KeyError:
    print("Lỗi: Không tìm thấy cột 'Close' trong file CSV. Vui lòng kiểm tra lại tên cột.")
    exit()
except Exception as e:
    print(f"Đã xảy ra lỗi khi đọc file CSV: {e}")
    exit()

# --- Bước 2: Vẽ đồ thị chuỗi thời gian của giá đóng cửa ---
plt.figure(figsize=(14, 7))
plt.plot(price_series)
plt.title('Đồ thị Chuỗi Thời Gian Giá Đóng Cửa Cổ Phiếu GOOGL', fontsize=16)
plt.xlabel('Thời gian', fontsize=12)
plt.ylabel('Giá đóng cửa (USD)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- Bước 3: Vẽ đồ thị ACF và PACF của giá đóng cửa ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Vẽ ACF
plot_acf(price_series, lags=40, ax=axes[0]) # lags=40 để xem xét 40 độ trễ đầu tiên
axes[0].set_title('Biểu đồ Hàm Tự Tương Quan (ACF) - Giá GOOGL', fontsize=14)
axes[0].set_xlabel('Độ trễ', fontsize=12)
axes[0].set_ylabel('Hệ số tự tương quan', fontsize=12)
axes[0].grid(True, linestyle='--', alpha=0.7)

# Vẽ PACF
plot_pacf(price_series, lags=40, ax=axes[1], method='ywm') # 'ywm' là một phương pháp tính PACF
axes[1].set_title('Biểu đồ Hàm Tự Tương Quan Riêng Phần (PACF) - Giá GOOGL', fontsize=14)
axes[1].set_xlabel('Độ trễ', fontsize=12)
axes[1].set_ylabel('Hệ số tự tương quan riêng phần', fontsize=12)
axes[1].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
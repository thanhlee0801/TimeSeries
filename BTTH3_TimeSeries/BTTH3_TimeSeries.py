import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd


# --- 1. Tạo Dữ liệu Giả lập Hành vi Người đi bộ/Robot ---
# Giả sử chúng ta có 3 chế độ hành vi tiềm ẩn:
# Chế độ 0: Đi thẳng (tốc độ X cao, Y thấp)
# Chế độ 1: Rẽ phải (tốc độ X trung bình, Y dương)
# Chế độ 2: Rẽ trái (tốc độ X trung bình, Y âm)

def generate_behavior_data(num_sequences=200, seq_len=50, noise_std=0.1):
    data = []
    true_modes = []  # Để kiểm tra độ chính xác sau này
    for _ in range(num_sequences):
        current_seq = []
        current_modes = []
        current_mode = np.random.randint(3)  # Bắt đầu với một chế độ ngẫu nhiên

        for t in range(seq_len):
            # Quyết định có chuyển chế độ không (để tạo SLDS like data)
            if np.random.rand() < 0.1 and t > 0:  # 10% cơ hội chuyển đổi
                current_mode = np.random.randint(3)

            if current_mode == 0:  # Đi thẳng
                vx, vy = np.random.normal(1.0, 0.1), np.random.normal(0.0, 0.05)
            elif current_mode == 1:  # Rẽ phải
                vx, vy = np.random.normal(0.5, 0.1), np.random.normal(0.5, 0.1)
            else:  # Chế độ 2: Rẽ trái
                vx, vy = np.random.normal(0.5, 0.1), np.random.normal(-0.5, 0.1)

            # Thêm nhiễu quan sát
            observation = np.array([vx, vy]) + np.random.normal(0, noise_std, 2)
            current_seq.append(observation)
            current_modes.append(current_mode)

        data.append(np.array(current_seq))
        true_modes.append(np.array(current_modes))
    return np.array(data), np.array(true_modes)


num_sequences = 200
seq_len = 50
observation_dim = 2  # (vx, vy)
data_raw, true_modes = generate_behavior_data(num_sequences, seq_len)
print(f"Kích thước dữ liệu thô: {data_raw.shape}")  # (num_sequences, seq_len, observation_dim)


# --- 2. Định nghĩa một Transformer Encoder tùy chỉnh để tạo Embeddings ---
class CustomTransformerEncoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1):
        super(CustomTransformerEncoder, self).__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim

        # Input projection layer: map observation_dim to model_dim
        self.input_projection = nn.Linear(input_dim, model_dim)

        # Positional Encoding
        self.positional_encoding = self._generate_positional_encoding(seq_len, model_dim)

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True  # Quan trọng: batch_size là chiều đầu tiên
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _generate_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Thêm chiều batch: (1, max_len, d_model)
        return pe.to(device)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)

        # Project input to model_dim
        x = self.input_projection(x)  # (batch_size, seq_len, model_dim)

        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :]

        # Pass through Transformer Encoder
        output = self.transformer_encoder(x)  # (batch_size, seq_len, model_dim)
        return output


# --- Cấu hình Transformer ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {device}")

model_dim = 64  # Chiều của embeddings Transformer
num_heads = 4
num_layers = 2

transformer_model = CustomTransformerEncoder(observation_dim, model_dim, num_heads, num_layers).to(device)
print(f"\nKiến trúc Transformer: {transformer_model}")

# Chuyển đổi dữ liệu thô sang Tensor
data_tensor = torch.tensor(data_raw, dtype=torch.float32).to(device)

# --- 3. Tạo Embeddings từ Transformer ---
print("\n--- Đang tạo embeddings hành vi bằng Transformer... ---")
transformer_model.eval()  # Chuyển sang chế độ đánh giá (tắt dropout, batchnorm)
with torch.no_grad():
    embeddings = transformer_model(data_tensor).cpu().numpy()
print(f"Kích thước embeddings sau Transformer: {embeddings.shape}")
# (num_sequences, seq_len, model_dim)

# --- 4. Chuẩn bị Embeddings cho GMM ---
# Flatten embeddings để có thể áp dụng GMM trực tiếp
# Mỗi timestep trong mỗi chuỗi sẽ là một điểm dữ liệu riêng biệt
flat_embeddings = embeddings.reshape(-1, model_dim)
print(f"Kích thước embeddings làm phẳng: {flat_embeddings.shape}")

# Chuẩn hóa các embeddings đã làm phẳng
print("--- Đang chuẩn hóa embeddings đã làm phẳng cho GMM... ---")
scaler_gmm = StandardScaler()
scaled_flat_embeddings = scaler_gmm.fit_transform(flat_embeddings)
print("Embeddings đã được chuẩn hóa.")

# --- 5. Áp dụng và Huấn luyện Gaussian Mixture Model (GMM) ---
# Chọn số lượng thành phần Gaussian (số cụm)
# Dữ liệu giả lập của chúng ta có 3 chế độ tiềm ẩn
n_components = 3

print(f"\n--- Đang huấn luyện Gaussian Mixture Model với {n_components} cụm... ---")
gmm = GaussianMixture(n_components=n_components, random_state=42, covariance_type='full', n_init=10)
gmm.fit(scaled_flat_embeddings)
print("GMM đã được huấn luyện.")

# Lấy nhãn cụm dự đoán cho từng timestep
predicted_flat_modes = gmm.predict(scaled_flat_embeddings)
predicted_modes_reshaped = predicted_flat_modes.reshape(num_sequences, seq_len)

# --- 6. Đánh giá và Trực quan hóa Kết quả ---
print("\n--- Kết quả Phân cụm GMM ---")

# In ra BIC và AIC để đánh giá mô hình
print(f"\nBIC (Bayesian Information Criterion): {gmm.bic(scaled_flat_embeddings):.2f}")
print(f"AIC (Akaike Information Criterion): {gmm.aic(scaled_flat_embeddings):.2f}")

# Trực quan hóa GMM trên embeddings đã giảm chiều (sử dụng PCA)
print("\n--- Đang trực quan hóa kết quả... ---")
pca = PCA(n_components=2)
flat_embeddings_2d = pca.fit_transform(scaled_flat_embeddings)  # PCA trên dữ liệu đã chuẩn hóa

plt.figure(figsize=(15, 7))

# Biểu đồ 1: Các chế độ thực tế (cho một chuỗi ví dụ)
sample_idx = 0
plt.subplot(1, 2, 1)
plt.plot(true_modes[sample_idx], label='Chế độ thực tế')
plt.title(f'Chế độ hành vi thực tế (Chuỗi {sample_idx})')
plt.xlabel('Thời gian')
plt.ylabel('Chế độ')
plt.legend()
plt.grid(True)

# Biểu đồ 2: Phân cụm GMM trên không gian embedding đã giảm chiều (PCA)
plt.subplot(1, 2, 2)
sns.scatterplot(
    x=flat_embeddings_2d[:, 0],
    y=flat_embeddings_2d[:, 1],
    hue=predicted_flat_modes,
    palette='viridis',
    s=20,
    alpha=0.6,
    edgecolor='w',
    legend='full'
)
plt.title('Phân cụm hành vi bằng GMM trên Embeddings của Transformer (PCA 2D)')
plt.xlabel('Thành phần chính 1')
plt.ylabel('Thành phần chính 2')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

print("\n--- So sánh chế độ thực tế và chế độ GMM cho chuỗi ví dụ ---")
print(f"Chế độ thực tế của chuỗi {sample_idx}: {true_modes[sample_idx]}")
print(f"Chế độ GMM dự đoán của chuỗi {sample_idx}: {predicted_modes_reshaped[sample_idx]}")
print("\nLưu ý: GMM dự đoán các nhãn cụm (0, 1, 2, ...) nhưng thứ tự có thể khác với nhãn thực tế.")
print("Bạn cần ánh xạ các nhãn GMM để phù hợp với nhãn thực tế để đánh giá định lượng.")

print("\n--- Hoàn thành chương trình ---")
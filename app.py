import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Giả sử bạn đã huấn luyện mô hình hồi quy và lưu nó trong file 'model.pkl'
# Load data
try:
    df = pd.read_csv("insurance.csv")  # Đường dẫn tương đối đúng
except FileNotFoundError:
    st.error("Không tìm thấy file dữ liệu. Vui lòng kiểm tra lại đường dẫn.")
    st.stop()

# 1. Mã hóa các biến phân loại
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# 2. Chuyển đổi các cột boolean thành số (1 cho True, 0 cho False)
df_encoded[['sex_male', 'smoker_yes', 'region_northwest', 'region_southeast', 'region_southwest']] = df_encoded[['sex_male', 'smoker_yes', 'region_northwest', 'region_southeast', 'region_southwest']].astype(int)

# 3. Chuẩn hóa các biến số liên tục
scaler = StandardScaler()
X_continuous = df_encoded[['age', 'bmi', 'children']]  # Các biến liên tục
X_scaled = scaler.fit_transform(X_continuous)

# 4. Kết hợp dữ liệu đã chuẩn hóa vào DataFrame gốc
X_scaled_df = pd.DataFrame(X_scaled, columns=['age', 'bmi', 'children'])

# 5. Loại bỏ các cột liên quan đến các biến phân loại đã được mã hóa
df_encoded = df_encoded.drop(['age', 'bmi', 'children'], axis=1)

# 6. Thêm các cột đã chuẩn hóa vào DataFrame
df_final = pd.concat([df_encoded, X_scaled_df], axis=1)

# 7. Kiểm tra DataFrame cuối cùng
st.write(df_final.head())

# 8. Tải mô hình hồi quy đã huấn luyện
# (Nếu bạn chưa có mô hình, bạn cần huấn luyện và lưu mô hình trước)
# Giả sử mô hình của bạn đã được huấn luyện và lưu vào tệp 'model.pkl'
try:
    with open('model/model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Không tìm thấy mô hình. Vui lòng kiểm tra lại đường dẫn đến mô hình.")
    st.stop()

# 9. Biến mục tiêu 'charges'
y = df_encoded['charges']

# 10. Chia dữ liệu thành tập huấn luyện và tập kiểm tra (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 11. Kiểm tra kích thước của các tập
st.write(f"Kích thước tập huấn luyện: {X_train.shape}")
st.write(f"Kích thước tập kiểm tra: {X_test.shape}")

# 12. Tính IQR cho cột 'charges'
Q1 = df_encoded['charges'].quantile(0.25)
Q3 = df_encoded['charges'].quantile(0.75)
IQR = Q3 - Q1

# Xác định các giá trị ngoại lai trong cột 'charges'
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Tạo điều kiện để thay thế các giá trị ngoại lai
df_encoded['charges'] = df_encoded['charges'].apply(
    lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x)
)

# 13. Kiểm tra lại sau khi thay thế giá trị ngoại lai
st.write(f"Thống kê mô tả của 'charges' sau khi xử lý ngoại lai:\n{df_encoded['charges'].describe()}")

# --- Streamlit app ---
st.title('Dự đoán Chi Phí Bảo Hiểm Y Tế')

# Nhập thông tin người dùng
sex_male = st.number_input('Giới tính Nam (1 nếu có, 0 nếu không):', min_value=0, max_value=1)
smoker_yes = st.number_input('Hút thuốc (1 nếu có, 0 nếu không):', min_value=0, max_value=1)
age = st.number_input('Tuổi:', min_value=18, max_value=120)
bmi = st.number_input('Chỉ số BMI:', min_value=10.0, max_value=50.0)
children = st.number_input('Số trẻ em:', min_value=0, max_value=10)

# Khi người dùng nhấn nút "Dự đoán"
if st.button('Dự đoán'):
    # Tạo DataFrame từ dữ liệu người dùng
    new_data = pd.DataFrame({
        'sex_male': [sex_male],
        'smoker_yes': [smoker_yes],
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'region_northwest': [0],  # Các giá trị mặc định cho các vùng
        'region_southeast': [0],
        'region_southwest': [0]
    })

    # Thêm hằng số vào các biến độc lập
    new_data_sm = sm.add_constant(new_data)

    # Dự đoán chi phí bảo hiểm
    predicted_charges = model.predict(new_data_sm)
    
    # Hiển thị kết quả dự đoán
    st.write(f'Dự đoán chi phí bảo hiểm: ${predicted_charges[0]:.2f}')

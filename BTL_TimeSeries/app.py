import streamlit as st

# Cách 1: Nhúng trực tiếp chuỗi HTML
html_string = """

<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dự báo Diễn biến Bão theo Năm</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; background-color: #f8f9fa; color: #343a40; }
        .container { max-width: 1000px; margin: auto; background: #ffffff; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        h1, h2 { color: #007bff; border-bottom: 2px solid #e9ecef; padding-bottom: 10px; margin-bottom: 25px; }
        img { max-width: 100%; height: auto; display: block; margin: 25px auto; border: 1px solid #dee2e6; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
        .model-section { margin-bottom: 50px; background-color: #f0f8ff; padding: 20px; border-radius: 10px; border: 1px solid #cfe2ff; }
        .model-section p { margin-bottom: 15px; font-size: 1.05em; }
        .note { background-color: #e6f7ff; border-left: 5px solid #2196f3; padding: 18px; margin-top: 25px; border-radius: 8px; font-style: italic; color: #1a527c; }
        .footer { text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #e9ecef; color: #6c757d; font-size: 0.9em; }
        .plot-container { display: none; } /* Default hidden */
        .plot-container.active { display: block; } /* Show when active */
        .year-selector { margin-bottom: 30px; padding: 15px; background-color: #e9ecef; border-radius: 8px; display: flex; align-items: center; justify-content: center; }
        .year-selector label { margin-right: 15px; font-weight: bold; font-size: 1.1em; color: #343a40; }
        .year-selector select { padding: 10px 15px; border-radius: 5px; border: 1px solid #ced4da; font-size: 1.0em; cursor: pointer; background-color: #ffffff; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Báo cáo Dự báo Diễn biến Bão</h1>
        <p>Báo cáo này trình bày kết quả dự báo cường độ bão sử dụng các mô hình chuỗi thời gian khác nhau trên bộ dữ liệu Digital Typhoon của Kaggle.</p>

        <div class="year-selector">
            <label for="yearSelect">Chọn năm dự báo:</label>
            <select id="yearSelect" onchange="showYearCharts()">
                <option value='2026'>2026</option><option value='2027'>2027</option><option value='2028'>2028</option><option value='2029'>2029</option><option value='2030'>2030</option>
            </select>
        </div>

        
            <div class="year-section plot-container" id="year_2026_section">
                <h2>Kết quả Dự báo cho Năm 2026</h2>

                <div class="model-section">
                    <h3>Prophet Forecast (2026)</h3>
                    <img src="prophet_forecast_2026.png" alt="Prophet Forecast 2026">
                </div>

                <div class="model-section">
                    <h3>LSTM Forecast (2026)</h3>
                    <img src="lstm_forecast_2026.png" alt="LSTM Forecast 2026">
                </div>

                <div class="model-section">
                    <h3>PatchTST Forecast (2026)</h3>
                    <img src="patchtst_forecast_2026.png" alt="PatchTST Forecast 2026">
                </div>

                <div class="model-section">
                    <h2>So sánh Dự báo của Các Mô hình cho Năm 2026</h2>
                    <img src="all_models_forecast_comparison_2026.png" alt="Comparison Forecast 2026">
                    <h3>Nhận xét so sánh:</h3>
                    <ul>
                        <li><b>Prophet:</b> Thường cho thấy xu hướng mượt mà, nắm bắt tốt các thành phần mùa vụ và xu hướng dài hạn. Nó có thể ít phản ứng với các biến động ngắn hạn.</li>
                        <li><b>LSTM:</b> Có khả năng học các phụ thuộc phức tạp và biến động phi tuyến tính. Dự báo của nó có thể phản ánh tốt hơn các thay đổi đột ngột nếu có trong dữ liệu lịch sử.</li>
                        <li><b>PatchTST:</b> Là một mô hình dựa trên Transformer, nó có khả năng nắm bắt các phụ thuộc dài hạn và các mối quan hệ phức tạp giữa các điểm dữ liệu. Nó có thể tạo ra các dự báo chi tiết và nhạy cảm với cấu trúc dữ liệu hơn.</li>
                    </ul>
                    <p>Trong trường hợp không có dữ liệu thực tế cho năm 2026, việc so sánh định lượng là không thể. Tuy nhiên, qua biểu đồ, chúng ta có thể đánh giá mức độ đồng thuận giữa các mô hình về xu hướng tổng thể và biên độ cường độ bão dự kiến. Sự khác biệt đáng kể giữa các mô hình có thể chỉ ra sự không chắc chắn hoặc sự nhạy cảm của chúng đối với các đặc điểm dữ liệu khác nhau.</p>
                </div>
            </div>
        
            <div class="year-section plot-container" id="year_2027_section">
                <h2>Kết quả Dự báo cho Năm 2027</h2>

                <div class="model-section">
                    <h3>Prophet Forecast (2027)</h3>
                    <img src="prophet_forecast_2027.png" alt="Prophet Forecast 2027">
                </div>

                <div class="model-section">
                    <h3>LSTM Forecast (2027)</h3>
                    <img src="lstm_forecast_2027.png" alt="LSTM Forecast 2027">
                </div>

                <div class="model-section">
                    <h3>PatchTST Forecast (2027)</h3>
                    <img src="patchtst_forecast_2027.png" alt="PatchTST Forecast 2027">
                </div>

                <div class="model-section">
                    <h2>So sánh Dự báo của Các Mô hình cho Năm 2027</h2>
                    <img src="all_models_forecast_comparison_2027.png" alt="Comparison Forecast 2027">
                    <h3>Nhận xét so sánh:</h3>
                    <ul>
                        <li><b>Prophet:</b> Thường cho thấy xu hướng mượt mà, nắm bắt tốt các thành phần mùa vụ và xu hướng dài hạn. Nó có thể ít phản ứng với các biến động ngắn hạn.</li>
                        <li><b>LSTM:</b> Có khả năng học các phụ thuộc phức tạp và biến động phi tuyến tính. Dự báo của nó có thể phản ánh tốt hơn các thay đổi đột ngột nếu có trong dữ liệu lịch sử.</li>
                        <li><b>PatchTST:</b> Là một mô hình dựa trên Transformer, nó có khả năng nắm bắt các phụ thuộc dài hạn và các mối quan hệ phức tạp giữa các điểm dữ liệu. Nó có thể tạo ra các dự báo chi tiết và nhạy cảm với cấu trúc dữ liệu hơn.</li>
                    </ul>
                    <p>Trong trường hợp không có dữ liệu thực tế cho năm 2027, việc so sánh định lượng là không thể. Tuy nhiên, qua biểu đồ, chúng ta có thể đánh giá mức độ đồng thuận giữa các mô hình về xu hướng tổng thể và biên độ cường độ bão dự kiến. Sự khác biệt đáng kể giữa các mô hình có thể chỉ ra sự không chắc chắn hoặc sự nhạy cảm của chúng đối với các đặc điểm dữ liệu khác nhau.</p>
                </div>
            </div>
        
            <div class="year-section plot-container" id="year_2028_section">
                <h2>Kết quả Dự báo cho Năm 2028</h2>

                <div class="model-section">
                    <h3>Prophet Forecast (2028)</h3>
                    <img src="prophet_forecast_2028.png" alt="Prophet Forecast 2028">
                </div>

                <div class="model-section">
                    <h3>LSTM Forecast (2028)</h3>
                    <img src="lstm_forecast_2028.png" alt="LSTM Forecast 2028">
                </div>

                <div class="model-section">
                    <h3>PatchTST Forecast (2028)</h3>
                    <img src="patchtst_forecast_2028.png" alt="PatchTST Forecast 2028">
                </div>

                <div class="model-section">
                    <h2>So sánh Dự báo của Các Mô hình cho Năm 2028</h2>
                    <img src="all_models_forecast_comparison_2028.png" alt="Comparison Forecast 2028">
                    <h3>Nhận xét so sánh:</h3>
                    <ul>
                        <li><b>Prophet:</b> Thường cho thấy xu hướng mượt mà, nắm bắt tốt các thành phần mùa vụ và xu hướng dài hạn. Nó có thể ít phản ứng với các biến động ngắn hạn.</li>
                        <li><b>LSTM:</b> Có khả năng học các phụ thuộc phức tạp và biến động phi tuyến tính. Dự báo của nó có thể phản ánh tốt hơn các thay đổi đột ngột nếu có trong dữ liệu lịch sử.</li>
                        <li><b>PatchTST:</b> Là một mô hình dựa trên Transformer, nó có khả năng nắm bắt các phụ thuộc dài hạn và các mối quan hệ phức tạp giữa các điểm dữ liệu. Nó có thể tạo ra các dự báo chi tiết và nhạy cảm với cấu trúc dữ liệu hơn.</li>
                    </ul>
                    <p>Trong trường hợp không có dữ liệu thực tế cho năm 2028, việc so sánh định lượng là không thể. Tuy nhiên, qua biểu đồ, chúng ta có thể đánh giá mức độ đồng thuận giữa các mô hình về xu hướng tổng thể và biên độ cường độ bão dự kiến. Sự khác biệt đáng kể giữa các mô hình có thể chỉ ra sự không chắc chắn hoặc sự nhạy cảm của chúng đối với các đặc điểm dữ liệu khác nhau.</p>
                </div>
            </div>
        
            <div class="year-section plot-container" id="year_2029_section">
                <h2>Kết quả Dự báo cho Năm 2029</h2>

                <div class="model-section">
                    <h3>Prophet Forecast (2029)</h3>
                    <img src="prophet_forecast_2029.png" alt="Prophet Forecast 2029">
                </div>

                <div class="model-section">
                    <h3>LSTM Forecast (2029)</h3>
                    <img src="lstm_forecast_2029.png" alt="LSTM Forecast 2029">
                </div>

                <div class="model-section">
                    <h3>PatchTST Forecast (2029)</h3>
                    <img src="patchtst_forecast_2029.png" alt="PatchTST Forecast 2029">
                </div>

                <div class="model-section">
                    <h2>So sánh Dự báo của Các Mô hình cho Năm 2029</h2>
                    <img src="all_models_forecast_comparison_2029.png" alt="Comparison Forecast 2029">
                    <h3>Nhận xét so sánh:</h3>
                    <ul>
                        <li><b>Prophet:</b> Thường cho thấy xu hướng mượt mà, nắm bắt tốt các thành phần mùa vụ và xu hướng dài hạn. Nó có thể ít phản ứng với các biến động ngắn hạn.</li>
                        <li><b>LSTM:</b> Có khả năng học các phụ thuộc phức tạp và biến động phi tuyến tính. Dự báo của nó có thể phản ánh tốt hơn các thay đổi đột ngột nếu có trong dữ liệu lịch sử.</li>
                        <li><b>PatchTST:</b> Là một mô hình dựa trên Transformer, nó có khả năng nắm bắt các phụ thuộc dài hạn và các mối quan hệ phức tạp giữa các điểm dữ liệu. Nó có thể tạo ra các dự báo chi tiết và nhạy cảm với cấu trúc dữ liệu hơn.</li>
                    </ul>
                    <p>Trong trường hợp không có dữ liệu thực tế cho năm 2029, việc so sánh định lượng là không thể. Tuy nhiên, qua biểu đồ, chúng ta có thể đánh giá mức độ đồng thuận giữa các mô hình về xu hướng tổng thể và biên độ cường độ bão dự kiến. Sự khác biệt đáng kể giữa các mô hình có thể chỉ ra sự không chắc chắn hoặc sự nhạy cảm của chúng đối với các đặc điểm dữ liệu khác nhau.</p>
                </div>
            </div>
        
            <div class="year-section plot-container" id="year_2030_section">
                <h2>Kết quả Dự báo cho Năm 2030</h2>

                <div class="model-section">
                    <h3>Prophet Forecast (2030)</h3>
                    <img src="prophet_forecast_2030.png" alt="Prophet Forecast 2030">
                </div>

                <div class="model-section">
                    <h3>LSTM Forecast (2030)</h3>
                    <img src="lstm_forecast_2030.png" alt="LSTM Forecast 2030">
                </div>

                <div class="model-section">
                    <h3>PatchTST Forecast (2030)</h3>
                    <img src="patchtst_forecast_2030.png" alt="PatchTST Forecast 2030">
                </div>

                <div class="model-section">
                    <h2>So sánh Dự báo của Các Mô hình cho Năm 2030</h2>
                    <img src="all_models_forecast_comparison_2030.png" alt="Comparison Forecast 2030">
                    <h3>Nhận xét so sánh:</h3>
                    <ul>
                        <li><b>Prophet:</b> Thường cho thấy xu hướng mượt mà, nắm bắt tốt các thành phần mùa vụ và xu hướng dài hạn. Nó có thể ít phản ứng với các biến động ngắn hạn.</li>
                        <li><b>LSTM:</b> Có khả năng học các phụ thuộc phức tạp và biến động phi tuyến tính. Dự báo của nó có thể phản ánh tốt hơn các thay đổi đột ngột nếu có trong dữ liệu lịch sử.</li>
                        <li><b>PatchTST:</b> Là một mô hình dựa trên Transformer, nó có khả năng nắm bắt các phụ thuộc dài hạn và các mối quan hệ phức tạp giữa các điểm dữ liệu. Nó có thể tạo ra các dự báo chi tiết và nhạy cảm với cấu trúc dữ liệu hơn.</li>
                    </ul>
                    <p>Trong trường hợp không có dữ liệu thực tế cho năm 2030, việc so sánh định lượng là không thể. Tuy nhiên, qua biểu đồ, chúng ta có thể đánh giá mức độ đồng thuận giữa các mô hình về xu hướng tổng thể và biên độ cường độ bão dự kiến. Sự khác biệt đáng kể giữa các mô hình có thể chỉ ra sự không chắc chắn hoặc sự nhạy cảm của chúng đối với các đặc điểm dữ liệu khác nhau.</p>
                </div>
            </div>
        

        <p class="footer">Báo cáo được tạo tự động bởi mã Python trên Kaggle Notebooks.</p>
    </div>

    <script>
        function showYearCharts() {
            var selectedYear = document.getElementById('yearSelect').value;
            var yearSections = document.getElementsByClassName('year-section');
            for (var i = 0; i < yearSections.length; i++) {
                yearSections[i].classList.remove('active');
            }
            var activeSection = document.getElementById('year_' + selectedYear + '_section');
            if (activeSection) {
                activeSection.classList.add('active');
            }
        }
        // Show charts for the first year by default when page loads
        document.addEventListener('DOMContentLoaded', function() {
            showYearCharts();
        });
    </script>
</body>
</html>

"""
st.markdown(html_string, unsafe_allow_html=True)

st.write("---")

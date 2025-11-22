# 🕰️ Temporal Robustness in Face Recognition (AIFR Pipeline)

> **Dự án này xây dựng và đánh giá các pipeline nhận diện khuôn mặt tập trung vào tính bền vững theo thời gian (Temporal Robustness). Mục tiêu là giải quyết bài toán nhận diện khuôn mặt bất biến theo độ tuổi (Age-Invariant Face Recognition) bằng cách sử dụng kiến trúc Siamese Network kết hợp với chiến lược Age-Gap Hard Mining.**

Dự án so sánh hiệu năng của 3 kiến trúc mô hình từ cổ điển đến hiện đại:

1.  **Simple CNN** (Baseline - Mạng nông)
2.  **FaceNet** (Inception-ResNet-v1 - 2015)
3.  **ArcFace** (ResNet50 - SOTA)

---

## 📂 Cấu trúc Dự án (Project Structure)

```text
Face_Aging_Project/
│
├── data/                   # Thư mục chứa dữ liệu
│   └── CACD_Cropped_112/   # Ảnh đã qua xử lý (MTCNN Crop & Align)
│
├── models/                 # Định nghĩa các kiến trúc mạng
│   ├── __init__.py
│   ├── simple_cnn.py       # Mạng CNN 4 lớp tự xây dựng
│   ├── facenet_model.py    # Wrapper cho Inception-ResNet-v1
│   └── arcface_model.py    # Wrapper cho ResNet50
│
├── utils.py                # Các hàm cốt lõi: Dataset, Siamese Wrapper, Contrastive Loss
├── train.py                # Script chính để huấn luyện
├── preprocess.py           # (Optional) Script chạy MTCNN để cắt ảnh
├── requirements.txt        # Các thư viện cần thiết
└── README.md               # Hướng dẫn sử dụng
```

## 🚀 Cách chạy

1. **Cài đặt:** `pip install -r requirements.txt`
2. **Xử lý ảnh:** `python preprocess.py`
3. **Train Simple CNN:** `python train.py --model simple_cnn --subset 0.4`
4. **Train FaceNet:** `python train.py --model facenet --subset 0.4`
5. **Train ArcFace:** `python train.py --model arcface --subset 0.4`

Kết quả model lưu tại folder `checkpoints/`.

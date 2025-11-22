import os
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm

# --- CẤU HÌNH ---
INPUT_DIR = ''       # Thư mục gốc chứa ảnh lộn xộn
OUTPUT_DIR = '' # Thư mục mới sẽ chứa ảnh sạch
IMAGE_SIZE = 112                  # Kích thước chuẩn cho ArcFace
MARGIN = 0                        # Lề thêm vào quanh mặt (0 để lấy sát mặt)

def preprocess_dataset():
    # 1. Kiểm tra GPU (MTCNN chạy trên GPU nhanh hơn nhiều)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Đang chạy trên thiết bị: {device}')

    # 2. Khởi tạo MTCNN
    # keep_all=False: Chỉ lấy khuôn mặt có độ tin cậy cao nhất (tránh lấy nhầm người đi đường)
    # select_largest=False: Mặc định MTCNN chọn mặt xác suất cao nhất.
    mtcnn = MTCNN(
        image_size=IMAGE_SIZE, 
        margin=MARGIN, 
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], 
        factor=0.709, 
        post_process=True,
        device=device
    )

    # 3. Duyệt thư mục
    if not os.path.exists(INPUT_DIR):
        print(f"Lỗi: Không tìm thấy {INPUT_DIR}")
        return

    # Lấy danh sách các folder con (tên người)
    classes = sorted([d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))])
    
    print(f"Tìm thấy {len(classes)} thư mục người. Bắt đầu xử lý...")

    # Dùng tqdm để hiện thanh tiến trình
    processed_count = 0
    error_count = 0

    for cls_name in tqdm(classes):
        src_folder = os.path.join(INPUT_DIR, cls_name)
        dst_folder = os.path.join(OUTPUT_DIR, cls_name)
        
        # Tạo thư mục đích nếu chưa có
        os.makedirs(dst_folder, exist_ok=True)
        
        # Lấy danh sách ảnh
        images = [f for f in os.listdir(src_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        for img_name in images:
            src_path = os.path.join(src_folder, img_name)
            dst_path = os.path.join(dst_folder, img_name)
            
            # Nếu ảnh đã xử lý rồi thì bỏ qua (để có thể resume nếu code dừng)
            if os.path.exists(dst_path):
                continue

            try:
                # Load ảnh
                img = Image.open(src_path).convert('RGB')
                
                # --- PHÉP MÀU Ở ĐÂY ---
                # mtcnn(img, save_path) sẽ tự động:
                # 1. Detect
                # 2. Align (xoay mặt)
                # 3. Crop & Resize về 112x112
                # 4. Lưu thẳng vào file đích
                result = mtcnn(img, save_path=dst_path)
                
                if result is None:
                    # Không tìm thấy mặt nào trong ảnh
                    # (Rất bình thường với dataset lớn, có thể ảnh đó chụp lưng hoặc mờ)
                    error_count += 1
                else:
                    processed_count += 1
                    
            except Exception as e:
                print(f"Lỗi file {img_name}: {e}")
                error_count += 1

    print("\n--- HOÀN TẤT ---")
    print(f"✅ Đã xử lý thành công: {processed_count} ảnh")
    print(f"⚠️ Không tìm thấy mặt/Lỗi: {error_count} ảnh")
    print(f"📁 Dữ liệu sạch nằm tại: {OUTPUT_DIR}")

if __name__ == '__main__':
    preprocess_dataset()
import os
import torch
import sys

# Kiểm tra xem gdown đã được cài chưa
try:
    import gdown
except ImportError:
    print("❌ Lỗi: Thư viện 'gdown' chưa được cài đặt.")
    print("👉 Hãy chạy lệnh: pip install -r requirements.txt")
    sys.exit(1)

# ==========================================
# CẤU HÌNH (BẠN PHẢI SỬA ID Ở ĐÂY)
# ==========================================
# 👇 Dán ID file Google Drive của file .pth vào dòng dưới 👇
# Ví dụ link: drive.google.com/file/d/1A2B3C.../view -> ID là 1A2B3C...
GDRIVE_FILE_ID = '1FH81gkKbsLG1EOVn1WjoyfQlJVCpm8p3' 

MODEL_FILENAME = "ir_se_101_temporal_best.pth"

def verify_and_download():
    """Kiểm tra file model, nếu hỏng hoặc thiếu thì tải lại."""
    
    # 1. Kiểm tra xem người dùng đã điền ID chưa
    if 'ID_FILE_MODEL' in GDRIVE_FILE_ID or len(GDRIVE_FILE_ID) < 10:
        print("❌ CRITICAL ERROR: Bạn chưa điền ID file Google Drive vào file model_loader.py!")
        print("👉 Vui lòng mở file model_loader.py và sửa biến GDRIVE_FILE_ID.")
        # Không exit ngay để tránh crash app nếu chạy local, nhưng sẽ báo lỗi
        return

    # 2. Kiểm tra file trên ổ cứng
    if os.path.exists(MODEL_FILENAME):
        try:
            print(f"🔍 Đang kiểm tra tính toàn vẹn của {MODEL_FILENAME}...")
            # Thử load nhẹ header để xem file có bị lỗi magic number không
            # map_location='cpu' để test nhanh không cần GPU
            torch.load(MODEL_FILENAME, map_location='cpu')
            print("✅ File model hợp lệ (Integrity check passed)!")
            return
        except Exception as e:
            print(f"⚠️ File bị lỗi (Corrupt): {e}")
            print("🗑️ Đang xóa file hỏng để tải lại...")
            os.remove(MODEL_FILENAME)
    else:
        print(f"⚠️ Không tìm thấy file {MODEL_FILENAME} trên máy.")

    # 3. Tải xuống bằng gdown
    print(f"⬇️ Đang tải model từ Google Drive (ID: {GDRIVE_FILE_ID})...")
    url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
    
    # fuzzy=True giúp gdown tự tìm file kể cả khi link hơi khác
    try:
        output = gdown.download(url, MODEL_FILENAME, quiet=False, fuzzy=True)
        
        if not output:
            print("❌ Tải xuống thất bại. Kiểm tra lại ID hoặc kết nối mạng.")
            sys.exit(1)
            
        # Kiểm tra lại lần nữa sau khi tải
        torch.load(MODEL_FILENAME, map_location='cpu')
        print("✅ Tải xuống và kiểm tra thành công!")
        
    except Exception as e:
        print(f"❌ Lỗi khi tải hoặc kiểm tra file: {e}")
        print("👉 Vui lòng kiểm tra lại ID file Google Drive hoặc quyền truy cập (file phải là Public).")
        if os.path.exists(MODEL_FILENAME):
            os.remove(MODEL_FILENAME) # Xóa file lỗi để lần sau tải lại
        sys.exit(1)

# Tự động chạy hàm kiểm tra khi import file này
if __name__ == "__main__" or "streamlit" in sys.modules:
    verify_and_download()

# ==========================================
# BIẾN TOÀN CỤC (Import cái này vào app.py)
# ==========================================
MODEL_CHECKPOINT_PATH = MODEL_FILENAME
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Device set to: {DEVICE}")
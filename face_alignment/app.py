import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import torch
import torch.nn.functional as F
import os
import align

# =========================================================
# CẤU HÌNH
# =========================================================
import os
import torch
import requests
import sys

def download_from_gdrive(id, destination):
    """
    Tải file từ Google Drive (hỗ trợ file kích thước lớn)
    """
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = _get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    _save_response_content(response, destination)

def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def _save_response_content(response, destination):
    CHUNK_SIZE = 32768
    
    # Lấy tổng kích thước file (nếu có) để hiển thị progress (đơn giản)
    total_length = response.headers.get('content-length')
    
    print(f"⬇️ Downloading to {destination}...")
    
    with open(destination, "wb") as f:
        downloaded = 0
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: 
                f.write(chunk)
                downloaded += len(chunk)
                # Hiển thị dấu chấm để báo hiệu đang tải
                if total_length:
                    # Logic hiển thị % có thể thêm ở đây
                    pass
    print("\n✅ Download complete!")

# ==========================================
# CONFIGURATION
# ==========================================

# 1. Thay thế ID này bằng ID file thực tế trên Google Drive của bạn
# Ví dụ link: drive.google.com/file/d/1A2B3C.../view -> ID là 1A2B3C...
GDRIVE_FILE_ID = '1FH81gkKbsLG1EOVn1WjoyfQlJVCpm8p3' 

# 2. Tên file model sẽ lưu trên máy
MODEL_FILENAME = "ir_se_101_temporal_best.pth"

# 3. Kiểm tra và tải file
if not os.path.exists(MODEL_FILENAME):
    print(f"⚠️ Model file '{MODEL_FILENAME}' not found locally.")
    
    if GDRIVE_FILE_ID == 'YOUR_GDRIVE_FILE_ID_HERE':
        print("❌ Error: Please update 'GDRIVE_FILE_ID' in model_loader.py with your real Google Drive File ID.")
        sys.exit(1)
    try:
        download_from_gdrive(GDRIVE_FILE_ID, MODEL_FILENAME)
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        sys.exit(1)
else:
    print(f"✅ Found model file: {MODEL_FILENAME}")

# Trong app.py
from model_loader import MODEL_CHECKPOINT_PATH, DEVICE

print(f"🚀 Device set to: {DEVICE}")

# Import net.py
try:
    import net
except ImportError:
    st.error("⚠️ LỖI: Không tìm thấy file `net.py`. Hãy đặt cùng thư mục.")
    st.stop()

st.set_page_config(page_title="AdaFace Demo", layout="centered")

# ------------------------
# 1. Load Model
# ------------------------
@st.cache_resource
def load_system_model():
    model = net.build_model("ir_101").to(DEVICE)

    device = torch.device(DEVICE)
    
    if os.path.exists(MODEL_CHECKPOINT_PATH):
        print(f"📥 Loading model from {MODEL_CHECKPOINT_PATH}")
        checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            # Handle pretrained format
            new_state = {
                k[6:]: v for k, v in checkpoint['state_dict'].items() 
                if k.startswith('model.')
            }
            model.load_state_dict(new_state, strict=False)
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"⚠️  Model path not found: {MODEL_CHECKPOINT_PATH}")
        return None
    
    model.to(device)
    model.eval()
    return model

def extract_style_adaface(model, pil_img):
    """
    Hàm này mô phỏng lại logic:
    np_img -> BGR convert -> Normalize thủ công -> Tensor -> Model -> Normalize Feature
    """
    try:
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        np_img = np.array(pil_img) 
        
        bgr_img = ((np_img[:, :, ::-1] / 255.) - 0.5) / 0.5
        tensor = torch.tensor(
            bgr_img.transpose(2, 0, 1)
        ).float().unsqueeze(0).to(DEVICE)

        # 5. Forward Pass
        with torch.no_grad():
            out = model(tensor)
            
            if isinstance(out, (tuple, list)):
                feature = out[0]
            else:
                feature = out
            
            norm_val = torch.norm(feature, p=2, dim=1).item()
            feature = F.normalize(feature, dim=1)

            return feature.cpu().numpy()[0], norm_val

    except Exception as e:
        st.error(f"Lỗi xử lý ảnh: {e}")
        return None, 0.0

def compute_cosine(a, b):
    return float(np.dot(a, b))

st.title("🔍 AdaFace Verification")
model = load_system_model()

if not model:
    st.stop()

with st.sidebar:
    use_mtcnn = st.checkbox("Dùng MTCNN Crop", value=True)
    threshold = st.slider("Ngưỡng (Threshold)", 0.0, 1.0, 0.30, 0.01)

col1, col2 = st.columns(2)
f1 = col1.file_uploader("Ảnh 1", type=["jpg", "png", "jpeg"])
f2 = col2.file_uploader("Ảnh 2", type=["jpg", "png", "jpeg"])

if f1 and f2:
    st.write("---")
    c1, c2 = st.columns(2)
    
    # --- Xử lý Ảnh 1 ---
    img1 = Image.open(f1).convert("RGB")
    # Bước Alignment (Cắt mặt)
    if use_mtcnn:
        align1 = align.get_aligned_face(image_path=None, rgb_pil_image=img1)
        final_img1 = align1 if align1 else ImageOps.fit(img1, (112,112))
    else:
        final_img1 = ImageOps.fit(img1, (112,112)) # Center crop cơ bản
        
    c1.image(final_img1, caption="Input 1 (Aligned)", width=150)
    # Gọi hàm xử lý mới
    emb1, n1 = extract_style_adaface(model, final_img1)

    # --- Xử lý Ảnh 2 ---
    img2 = Image.open(f2).convert("RGB")
    if use_mtcnn:
        align2 = align.get_aligned_face(image_path=None, rgb_pil_image=img2)
        final_img2 = align2 if align2 else ImageOps.fit(img2, (112,112))
    else:
        final_img2 = ImageOps.fit(img2, (112,112))

    c2.image(final_img2, caption="Input 2 (Aligned)", width=150)
    emb2, n2 = extract_style_adaface(model, final_img2)

    # --- Kết quả ---
    if emb1 is not None and emb2 is not None:
        score = compute_cosine(emb1, emb2)
        
        st.markdown(f"<h2 style='text-align: center; color: #4CAF50;'>Sim: {score:.4f}</h2>", unsafe_allow_html=True)
        
        if score >= threshold:
            st.success("✅ SAME PERSON")
        else:
            st.error("❌ DIFFERENT PERSON")
            
        st.progress(max(0.0, min(1.0, float(score))))
        
        with st.expander("Debug Info"):
            st.write(f"Norm 1: {n1:.2f} | Norm 2: {n2:.2f}")
            st.caption("Nếu Norm thấp (<20) có thể ảnh mờ hoặc model chưa khớp.")
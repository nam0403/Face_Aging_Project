import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import torch
import torch.nn.functional as F
import os
import sys

import align

# =========================================================
# CẤU HÌNH
# =========================================================
MODEL_CHECKPOINT_PATH = "weights/adaface.ckpt" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    if not os.path.exists(MODEL_CHECKPOINT_PATH):
        st.error(f"⚠️ Không tìm thấy file: `{MODEL_CHECKPOINT_PATH}`")
        return None

    try:
        st.info(f"Loading model: `{MODEL_CHECKPOINT_PATH}` on `{DEVICE}`...")
        
        # Build Model Architecture
        model = net.build_model()
        
        # Load Weights
        checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE)
        state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
        
        # Fix DataParallel keys
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(new_state_dict, strict=False)
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Lỗi load model: {e}")
        return None

# ------------------------
# 2. XỬ LÝ ẢNH THEO CODE CỦA BẠN
# ------------------------

def extract_style_adaface(model, pil_img):
    """
    Hàm này mô phỏng lại logic:
    np_img -> BGR convert -> Normalize thủ công -> Tensor -> Model -> Normalize Feature
    """
    try:
        # 1. Đảm bảo Input là PIL RGB
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        
        # 2. Convert sang Numpy
        # Lưu ý: PIL mặc định là RGB.
        np_img = np.array(pil_img) 
        
        # 3. Preprocessing (Logic cũ của bạn)
        # ::-1 để đảo chiều kênh màu từ RGB sang BGR (quan trọng với model InsightFace/AdaFace)
        bgr_img = ((np_img[:, :, ::-1] / 255.) - 0.5) / 0.5
        
        # 4. Tạo Tensor: (H, W, C) -> (C, H, W)
        tensor = torch.tensor(
            bgr_img.transpose(2, 0, 1)
        ).float().unsqueeze(0).to(DEVICE)

        # 5. Forward Pass
        with torch.no_grad():
            # Model AdaFace thường trả về (feature, norm) hoặc chỉ feature
            out = model(tensor)
            
            if isinstance(out, (tuple, list)):
                feature = out[0]
            else:
                feature = out
            
            # Lấy Norm gốc để check chất lượng ảnh (Optional)
            norm_val = torch.norm(feature, p=2, dim=1).item()

            # 6. Normalize Feature (Quan trọng)
            feature = F.normalize(feature, dim=1)
            
            # 7. Convert sang Numpy
            return feature.cpu().numpy()[0], norm_val

    except Exception as e:
        st.error(f"Lỗi xử lý ảnh: {e}")
        return None, 0.0

def compute_cosine(a, b):
    # Dùng Numpy dot product cho an toàn
    return float(np.dot(a, b))

# ------------------------
# 3. Giao diện Streamlit
# ------------------------
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
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os
import random

# --- IMPORT CÁC MODULE TỪ PROJECT CỦA BẠN ---
from utils import SiameseWrapper
from models.simple_cnn import SimpleCNNBackbone
from models.facenet_model import FaceNetBackbone
from models.arcface_model import ArcFaceResNetBackbone

# ==========================================
# 1. TẠO TẬP TEST CỐ ĐỊNH (FAIR TESTING)
# ==========================================
def generate_test_pairs(root_dir, output_file="test_pairs.txt", num_pairs=6000):
    """
    Tạo file danh sách cặp ảnh test để đảm bảo các model đều test trên cùng 1 tập dữ liệu.
    Format: path1,path2,label,age_gap
    """
    if os.path.exists(output_file):
        print(f"ℹ️ File {output_file} đã tồn tại. Sẽ sử dụng file này.")
        return

    print(f"🔄 Đang tạo tập test mới từ {root_dir}...")
    
    # Index dữ liệu
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    data_map = {} # {class_index: [(path, age), ...]}
    
    for idx, cls_name in enumerate(classes):
        cls_folder = os.path.join(root_dir, cls_name)
        imgs = []
        for f in os.listdir(cls_folder):
            if f.lower().endswith(('.jpg', '.png')):
                try:
                    age = int(f.split('_')[0])
                    imgs.append((os.path.join(cls_folder, f), age))
                except: pass
        if len(imgs) >= 2:
            data_map[idx] = imgs

    valid_indices = list(data_map.keys())
    pairs = []
    
    # Tạo 50% cặp cùng người (Positive), 50% khác người (Negative)
    for i in tqdm(range(num_pairs), desc="Generating Pairs"):
        is_same = (i % 2 == 0)
        
        if is_same:
            # Positive Pair
            idx = random.choice(valid_indices)
            candidates = data_map[idx]
            img1 = random.choice(candidates)
            img2 = random.choice(candidates)
            while img1 == img2 and len(candidates) > 1:
                img2 = random.choice(candidates)
            
            label = 1 # 1 = Giống nhau
            gap = abs(img1[1] - img2[1])
            pairs.append(f"{img1[0]},{img2[0]},{label},{gap}\n")
            
        else:
            # Negative Pair
            idx1, idx2 = random.sample(valid_indices, 2)
            img1 = random.choice(data_map[idx1])
            img2 = random.choice(data_map[idx2])
            
            label = 0 # 0 = Khác nhau
            gap = abs(img1[1] - img2[1]) 
            pairs.append(f"{img1[0]},{img2[0]},{label},{gap}\n")

    with open(output_file, "w") as f:
        f.writelines(pairs)
    print(f"✅ Đã lưu {num_pairs} cặp vào {output_file}")

# ==========================================
# 2. CLASS ĐÁNH GIÁ (EVALUATOR)
# ==========================================
class FaceEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def predict(self, img_path):
        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                # Lấy vector đặc trưng (đã normalize)
                emb = self.model.forward_one(img)
            return emb
        except:
            return None

    def run_test(self, pair_file):
        y_true = []
        y_scores = []
        age_gaps = []

        with open(pair_file, 'r') as f:
            lines = f.readlines()

        print("   ▶ Đang chạy inference...")
        for line in tqdm(lines):
            p1, p2, label, gap = line.strip().split(',')
            
            emb1 = self.predict(p1)
            emb2 = self.predict(p2)
            
            if emb1 is not None and emb2 is not None:
                # Tính Cosine Similarity (-1 đến 1)
                score = F.cosine_similarity(emb1, emb2).item()
                
                y_true.append(int(label))
                y_scores.append(score)
                age_gaps.append(int(gap))

        return np.array(y_true), np.array(y_scores), np.array(age_gaps)

# ==========================================
# 3. HÀM VẼ BIỂU ĐỒ VÀ MAIN
# ==========================================
def main():
    # CẤU HÌNH
    DATA_DIR = "data/CACD_Cropped_112"
    TEST_FILE = "test_pairs.txt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Tạo đề thi (nếu chưa có)
    generate_test_pairs(DATA_DIR, TEST_FILE)
    
    # 2. Định nghĩa danh sách model cần chấm điểm
    # LƯU Ý: Đảm bảo đường dẫn 'path' trỏ đúng file .pth bạn đã train
    models_config = [
        {
            "name": "Simple CNN",
            "backbone": SimpleCNNBackbone(embedding_size=512),
            "path": "checkpoints/simple_cnn_epoch_20.pth",
            "color": "red"
        },
        {
            "name": "FaceNet",
            "backbone": FaceNetBackbone(embedding_size=512),
            "path": "checkpoints/facenet_epoch_20.pth",
            "color": "blue"
        },
        {
            "name": "ArcFace",
            "backbone": ArcFaceResNetBackbone(embedding_size=512),
            "path": "checkpoints/arcface_epoch_20.pth",
            "color": "green"
        }
    ]
    
    results = {} # Lưu kết quả để vẽ

    print("\n🚀 BẮT ĐẦU ĐÁNH GIÁ SO SÁNH...")

    for config in models_config:
        print(f"\n🧪 Đánh giá Model: {config['name']}")
        
        # Load Model
        try:
            # Bọc backbone vào SiameseWrapper
            full_model = SiameseWrapper(config['backbone']).to(device)
            # Load weights
            state_dict = torch.load(config['path'], map_location=device)
            full_model.load_state_dict(state_dict)
        except FileNotFoundError:
            print(f"⚠️ Không tìm thấy file weight: {config['path']}. Bỏ qua model này.")
            continue
        except Exception as e:
            print(f"⚠️ Lỗi khi load model: {e}")
            continue

        # Chạy đánh giá
        evaluator = FaceEvaluator(full_model, device)
        y_true, y_scores, gaps = evaluator.run_test(TEST_FILE)
        
        # Tính Metrics cơ bản
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Tìm Best Threshold (ngưỡng tối ưu)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        y_pred = (y_scores >= optimal_threshold).astype(int)
        accuracy = accuracy_score(y_true, y_pred)
        
        print(f"   ⭐ Kết quả: AUC = {roc_auc:.4f} | Best Acc = {accuracy:.4f} | Threshold = {optimal_threshold:.4f}")
        
        # --- PHÂN TÍCH TEMPORAL ROBUSTNESS (Age Gap) ---
        # Chỉ xét trên các cặp Positive (Cùng người)
        pos_mask = (y_true == 1)
        pos_scores = y_scores[pos_mask]
        pos_gaps = gaps[pos_mask]
        
        gap_bins = [(0, 5), (6, 10), (11, 100)]
        gap_accs = []
        
        for (low, high) in gap_bins:
            mask = (pos_gaps >= low) & (pos_gaps <= high)
            if mask.sum() > 0:
                # Tỷ lệ nhận đúng (True Positive Rate) tại threshold chung
                acc = (pos_scores[mask] >= optimal_threshold).sum() / mask.sum()
                gap_accs.append(acc)
            else:
                gap_accs.append(0)
        
        results[config['name']] = {
            "fpr": fpr, "tpr": tpr, "auc": roc_auc,
            "gap_accs": gap_accs, "color": config['color']
        }

    # ==========================================
    # 4. VẼ BIỂU ĐỒ SO SÁNH
    # ==========================================
    

    if not results:
        print("❌ Không có kết quả nào để vẽ.")
        return

    plt.figure(figsize=(14, 6))

    # Biểu đồ 1: ROC Curve
    plt.subplot(1, 2, 1)
    for name, res in results.items():
        plt.plot(res['fpr'], res['tpr'], label=f"{name} (AUC={res['auc']:.3f})", color=res['color'], linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('So sánh ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    # Biểu đồ 2: Temporal Robustness (Bar Chart)
    plt.subplot(1, 2, 2)
    x = np.arange(3)
    width = 0.25
    labels = ['0-5 năm', '6-10 năm', '> 10 năm']
    
    for i, (name, res) in enumerate(results.items()):
        # Vẽ cột, dịch chuyển vị trí x để không đè lên nhau
        offset = (i - 1) * width 
        plt.bar(x + offset, res['gap_accs'], width, label=name, color=res['color'], alpha=0.8)

    plt.xlabel('Age Gap (Độ lệch tuổi)')
    plt.ylabel('Accuracy (TPR)')
    plt.title('Tính bền vững theo thời gian (Temporal Robustness)')
    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = "comparison_result.png"
    plt.savefig(save_path)
    print(f"\n✅ Đã lưu biểu đồ so sánh tại: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()
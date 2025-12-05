import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, det_curve
from sklearn.calibration import calibration_curve
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os
import random
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns

# --- IMPORT MODELS ---
from models.model_ir_se50 import IR_SE50
from models.ada_face import AdaFaceNet

# --- OFFICIAL WRAPPER ---
try:
    import net
    HAS_OFFICIAL = True
except ImportError:
    HAS_OFFICIAL = False

class AdaFaceOriginalWrapper(nn.Module):
    def __init__(self, architecture='ir_101', device='cuda', ckpt_path="weights/adaface_ir101_ms1mv2.ckpt"):
        super(AdaFaceOriginalWrapper, self).__init__()
        if not HAS_OFFICIAL: raise ImportError("Missing net.py")
        self.model = net.build_model(architecture)
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')['state_dict']
            self.model.load_state_dict({k.replace('model.',''):v for k,v in ckpt.items() if k.startswith('model.')})
            self.model.eval().to(device)
        
    def forward(self, x):
        return F.normalize(self.model(torch.flip(x, [1]))[0], p=2, dim=1)

# --- MTCNN ---
try:
    from facenet_pytorch import MTCNN
    HAS_MTCNN = True
except ImportError: HAS_MTCNN = False

# =======================================================================
# 1. ADVANCED METRIC UTILS
# =======================================================================

def calculate_eer(y_true, y_scores):
    """Tính Equal Error Rate (EER)"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    # Tìm điểm FPR gần FNR nhất
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    return fpr[idx], thresholds[idx]

def get_bootstrap_ci(y_true, y_scores, metric_func, n_bootstraps=100, alpha=0.95):
    """Tính khoảng tin cậy (CI) 95% bằng Bootstrap"""
    bootstrapped_scores = []
    rng = np.random.RandomState(42)
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_scores), len(y_scores))
        if len(np.unique(y_true[indices])) < 2: continue
        score = metric_func(y_true[indices], y_scores[indices])
        bootstrapped_scores.append(score)
    sorted_scores = np.array(bootstrapped_scores); sorted_scores.sort()
    lower = sorted_scores[int((1.0 - alpha) / 2 * len(sorted_scores))]
    upper = sorted_scores[int((1.0 + alpha) / 2 * len(sorted_scores))]
    return lower, upper

def auc_metric(y_t, y_s):
    fpr, tpr, _ = roc_curve(y_t, y_s)
    return auc(fpr, tpr)

def get_tar_at_far(y_true, y_scores, target_far):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    try:
        idx = np.where(fpr <= target_far)[0][-1]
        return tpr[idx], thresholds[idx]
    except: return 0.0, 0.0

# =======================================================================
# 2. EVALUATOR CLASS (Update return data)
# =======================================================================
class FaceEvaluator:
    def __init__(self, model, device):
        self.model = model; self.device = device; self.model.eval()
        self.transform = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
        self.mtcnn = MTCNN(image_size=112, margin=0, device=device, post_process=True) if HAS_MTCNN else None

    def predict(self, img_path):
        try:
            img = Image.open(img_path).convert('RGB')
            face = self.mtcnn(img) if self.mtcnn else None
            if face is None: face = self.transform(img)
            with torch.no_grad():
                emb = self.model(face.unsqueeze(0).to(self.device))
                if not isinstance(self.model, AdaFaceOriginalWrapper):
                    if hasattr(self.model, 'forward_one'): emb = self.model.forward_one(face.unsqueeze(0).to(self.device))
            return F.normalize(emb, p=2, dim=1) if not isinstance(self.model, AdaFaceOriginalWrapper) else emb
        except: return None

    def run_test(self, pair_file):
        # Lưu thêm path và ID để phân tích lỗi
        data = {"y_true":[], "y_scores":[], "age_gaps":[], "paths":[], "ids":[]}
        with open(pair_file, 'r') as f: lines = f.readlines()
        print("   ▶ Running Inference...")
        for line in tqdm(lines, leave=False):
            try:
                p1, p2, label, gap = line.strip().split(',')
                emb1 = self.predict(p1); emb2 = self.predict(p2)
                if emb1 is not None and emb2 is not None:
                    data["y_true"].append(int(label))
                    data["y_scores"].append(F.cosine_similarity(emb1, emb2).item())
                    data["age_gaps"].append(int(gap))
                    data["paths"].append((p1, p2))
                    # Giả định path dạng: data/ID/img.jpg -> lấy ID
                    data["ids"].append(os.path.basename(os.path.dirname(p1)))
            except: continue
        
        for k in ["y_true", "y_scores", "age_gaps"]: data[k] = np.array(data[k])
        data["paths"] = np.array(data["paths"])
        data["ids"] = np.array(data["ids"])
        return data

# =======================================================================
# 3. VISUALIZATION & ANALYSIS
# =======================================================================

def visualize_hard_examples(data, model_name, output_dir, threshold, num=5):
    """Vẽ ảnh các ca lỗi nặng nhất (FP/FN)"""
    y_true = data['y_true']; scores = data['y_scores']; paths = data['paths']
    
    # False Negatives (Cùng người, điểm thấp)
    fn_idxs = np.where((y_true == 1) & (scores < threshold))[0]
    worst_fn = fn_idxs[np.argsort(scores[fn_idxs])[:num]]
    
    # False Positives (Khác người, điểm cao)
    fp_idxs = np.where((y_true == 0) & (scores > threshold))[0]
    worst_fp = fp_idxs[np.argsort(scores[fp_idxs])[::-1][:num]]

    def plot_grid(indices, title, fname):
        if len(indices) == 0: return
        fig, axes = plt.subplots(len(indices), 2, figsize=(6, 3*len(indices)))
        if len(indices)==1: axes=[axes]
        fig.suptitle(f"{model_name} - {title}", y=1.02)
        for i, idx in enumerate(indices):
            try:
                img1 = Image.open(paths[idx][0]); img2 = Image.open(paths[idx][1])
                axes[i][0].imshow(img1); axes[i][0].axis('off')
                axes[i][1].imshow(img2); axes[i][1].axis('off')
                axes[i][0].set_title(f"S:{scores[idx]:.3f} | Gap:{data['age_gaps'][idx]}", loc='left', fontsize=9)
            except: pass
        plt.tight_layout(); plt.savefig(os.path.join(output_dir, fname)); plt.close()

    plot_grid(worst_fn, "Worst False Negatives", "error_FN.png")
    plot_grid(worst_fp, "Worst False Positives", "error_FP.png")

def plot_advanced_metrics(data, model_name, output_dir):
    y_true, y_scores = data['y_true'], data['y_scores']
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. DET Curve (Log-Log)
    fpr, fnr, _ = det_curve(y_true, y_scores)
    axs[0,0].plot(fpr, fnr, lw=2)
    axs[0,0].set_xscale('log'); axs[0,0].set_yscale('log')
    axs[0,0].set_xlabel('FAR (False Accept Rate)'); axs[0,0].set_ylabel('FRR (False Reject Rate)')
    axs[0,0].set_title('DET Curve (Security)')
    axs[0,0].grid(True, which="both", alpha=0.5)
    
    # 2. Calibration Curve
    prob_true, prob_pred = calibration_curve(y_true, (y_scores+1)/2, n_bins=10)
    axs[0,1].plot(prob_pred, prob_true, marker='o', label='Model')
    axs[0,1].plot([0, 1], [0, 1], 'k--', label='Perfect')
    axs[0,1].set_xlabel('Predicted Probability'); axs[0,1].set_ylabel('Fraction of Positives')
    axs[0,1].set_title('Calibration (Reliability)')
    axs[0,1].legend()
    
    # 3. Score Dist
    sns.kdeplot(y_scores[y_true==0], fill=True, ax=axs[1,0], label='Neg', color='red')
    sns.kdeplot(y_scores[y_true==1], fill=True, ax=axs[1,0], label='Pos', color='green')
    axs[1,0].set_title('Score Distribution'); axs[1,0].legend()
    
    # 4. Age Gap Degradation Line
    gaps = sorted(list(set(data['age_gaps'])))
    # Binning cho gọn
    bins = [(0,2), (3,5), (6,10), (11,100)]
    bin_accs = []
    bin_labels = []
    fpr_g, tpr_g, threshs = roc_curve(y_true, y_scores)
    opt_th = threshs[np.argmax(tpr_g - fpr_g)]
    
    for l, h in bins:
        mask = (data['y_true']==1) & (data['age_gaps']>=l) & (data['age_gaps']<=h)
        if mask.sum() > 0:
            acc = (y_scores[mask] > opt_th).sum() / mask.sum()
            bin_accs.append(acc)
            bin_labels.append(f"{l}-{h}")
    
    axs[1,1].plot(bin_labels, bin_accs, marker='o', linestyle='-', color='purple')
    axs[1,1].set_ylim(0, 1.1); axs[1,1].set_title('Accuracy vs Age Gap')
    axs[1,1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "advanced_plots.png"))
    plt.close()

def analyze_identities(data, threshold, output_dir):
    """Phân tích Identity nào hay bị sai nhất"""
    df = pd.DataFrame({'id': data['ids'], 'label': data['y_true'], 'score': data['y_scores']})
    pos_df = df[df['label']==1].copy()
    pos_df['is_fn'] = pos_df['score'] < threshold
    
    stats = pos_df.groupby('id').agg(total=('id','count'), fn=('is_fn','sum')).reset_index()
    stats['fnr'] = stats['fn'] / stats['total']
    worst = stats[stats['total']>=3].sort_values('fnr', ascending=False).head(10)
    worst.to_csv(os.path.join(output_dir, "worst_identities.csv"), index=False)
    return worst

# =======================================================================
# MAIN
# =======================================================================
def main():
    DATA_DIR = "data/CACD_Cropped_112"
    TEST_FILE = "test_pairs.txt"
    REPORT_DIR = "eval_report_v4"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Models Config
    models_config = [
        {
            "name": "FaceNet_FineTuned",
            "model_init": lambda: AdaFaceNet(2000, 512),
            "path": "checkpoints/adaface_best.pth",
            "type": "custom"
        },
        {
            "name": "ArcFace_FineTuned",
            "model_init": lambda: IR_SE50(),
            "path": "weights/InsightFace_Pytorch%2Bmodel_ir_se50.pth",
            "type": "custom"
        },
        {
            "name": "AdaFace_FineTuned",
            "model_init": lambda: AdaFaceOriginalWrapper(),
            "path": None,
            "type": "official"
        }
    ]

    print(f"\n🚀 STARTING ADVANCED EVALUATION...")
    #generate_test_pairs(DATA_DIR, TEST_FILE, 6000) # Uncomment if needed

    for config in models_config:
        print(f"\n🧪 Model: {config['name']}")
        try:
            # 1. Load Model
            if config['type'] == 'official':
                if not HAS_OFFICIAL: continue
                model = config['model_init']().to(device)
            else:
                model = config['model_init']().to(device)
                if os.path.exists(config['path']):
                    ckpt = torch.load(config['path'], map_location=device)
                    # Smart Load
                    if 'state_dict' in ckpt: ckpt = ckpt['state_dict']
                    model_keys = list(model.state_dict().keys())
                    ckpt_keys = list(ckpt.keys())
                    needs_bb = any(k.startswith('backbone.') for k in model_keys)
                    has_bb = any(k.startswith('backbone.') for k in ckpt_keys)
                    new_dict = {}
                    if needs_bb and not has_bb: 
                        for k,v in ckpt.items(): new_dict['backbone.'+k] = v
                    elif not needs_bb and has_bb:
                        for k,v in ckpt.items(): new_dict[k.replace('backbone.', '')] = v
                    else: new_dict = ckpt
                    model.load_state_dict(new_dict, strict=False)
                else: print(f"⚠️ Missing path {config['path']}"); continue

            # 2. Run Test
            out_dir = os.path.join(REPORT_DIR, config['name'])
            os.makedirs(out_dir, exist_ok=True)
            
            evaluator = FaceEvaluator(model, device)
            data = evaluator.run_test(TEST_FILE)
            y_true, y_scores = data['y_true'], data['y_scores']

            # 3. Calc Core Metrics
            eer, eer_thresh = calculate_eer(y_true, y_scores)
            roc_auc = auc_metric(y_true, y_scores)
            auc_low, auc_high = get_bootstrap_ci(y_true, y_scores, auc_metric, n_bootstraps=50)
            
            # Print Console Summary
            print(f"   📊 CORE METRICS:")
            print(f"      - AUC: {roc_auc:.4f} (95% CI: {auc_low:.4f}-{auc_high:.4f})")
            print(f"      - EER: {eer:.2%} @ Thresh {eer_thresh:.4f}")

            # 4. Per-Gap Analysis (Advanced)
            gap_bins = [(0,2), (3,5), (6,10), (11,100)]
            gap_report = []
            print(f"   📅 PER-GAP PERFORMANCE:")
            print(f"      {'Gap':<8} | {'AUC':<8} | {'TAR@FAR=1e-3':<15} | {'Count'}")
            
            for l, h in gap_bins:
                # Mask: Positives in Gap + All Negatives (to keep FAR baseline stable)
                pos_mask = (y_true == 1) & (data['age_gaps'] >= l) & (data['age_gaps'] <= h)
                neg_mask = (y_true == 0)
                mask = pos_mask | neg_mask
                
                if pos_mask.sum() > 5:
                    y_sub = y_true[mask]
                    s_sub = y_scores[mask]
                    
                    g_auc = auc_metric(y_sub, s_sub)
                    g_tar, _ = get_tar_at_far(y_sub, s_sub, 1e-3)
                    
                    print(f"      {l}-{h:<5} | {g_auc:.4f}   | {g_tar*100:.2f}%          | {pos_mask.sum()}")
                    gap_report.append({'gap': f"{l}-{h}", 'auc': g_auc, 'tar_1e-3': g_tar})

            # 5. Save & Visualize
            pd.DataFrame(gap_report).to_csv(os.path.join(out_dir, "gap_metrics.csv"), index=False)
            
            # Vẽ 4 biểu đồ nâng cao (DET, Calibration, Hist, Degradation)
            plot_advanced_metrics(data, config['name'], out_dir)
            
            # Xuất ảnh lỗi (Hard Examples)
            visualize_hard_examples(data, config['name'], out_dir, eer_thresh)
            
            # Phân tích Identity
            analyze_identities(data, eer_thresh, out_dir)

        except Exception as e: print(f"❌ Error: {e}"); continue

    print("\n✅ Evaluation V4 Complete. Check 'eval_report_v4' folder.")

if __name__ == "__main__":
    main()
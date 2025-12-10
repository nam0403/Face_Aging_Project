import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, det_curve
from sklearn.calibration import calibration_curve
from tqdm import tqdm
import pandas as pd
import os

from dataset.dataset import AgeGapDataset
import net


CONFIG = {
    'pretrained_path': 'weights/adaface.ckpt',
    'finetuned_path': 'ir_se_101_temporal_best.pth',
    'csv_path': 'cacd_test.csv',  # Use test set for evaluation
    'root_dir': 'data/CACD_Cropped_112',
    'batch_size': 128,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': 'evaluation_results_comprehensive'
}


# ================= ADVANCED METRICS =================
def calculate_eer(y_true, y_scores):
    """Calculate Equal Error Rate (EER)"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    return fpr[idx], thresholds[idx]


def get_bootstrap_ci(y_true, y_scores, metric_func, n_bootstraps=100, alpha=0.95):
    """Calculate 95% confidence interval using Bootstrap"""
    bootstrapped_scores = []
    rng = np.random.RandomState(42)
    
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_scores), len(y_scores))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = metric_func(y_true[indices], y_scores[indices])
        bootstrapped_scores.append(score)
    
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    lower = sorted_scores[int((1.0 - alpha) / 2 * len(sorted_scores))]
    upper = sorted_scores[int((1.0 + alpha) / 2 * len(sorted_scores))]
    
    return lower, upper


def auc_metric(y_t, y_s):
    """Helper for bootstrap AUC"""
    fpr, tpr, _ = roc_curve(y_t, y_s)
    return auc(fpr, tpr)


def get_tar_at_far(y_true, y_scores, target_far):
    """Get True Accept Rate at specific False Accept Rate"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    try:
        idx = np.where(fpr <= target_far)[0][-1]
        return tpr[idx], thresholds[idx]
    except:
        return 0.0, 0.0


# ================= LOAD MODEL =================
def load_model(model_path, model_type="ir_101"):
    """Load fine-tuned or pretrained model"""
    device = torch.device(CONFIG['device'])
    model = net.build_model(model_type)
    
    if os.path.exists(model_path):
        print(f"📥 Loading model from {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        except Exception:
            print(f"⚠️  weights_only=True failed, trying weights_only=False...")
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            new_state = {
                k[6:]: v for k, v in checkpoint['state_dict'].items() 
                if k.startswith('model.')
            }
            model.load_state_dict(new_state, strict=False)
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"⚠️  Model path not found: {model_path}")
        return None
    
    model.to(device)
    model.eval()
    return model


# ================= EVALUATION =================
def extract_embeddings(model, loader, device):
    """Extract embeddings for all images"""
    all_emb_a, all_emb_p, all_emb_n = [], [], []
    all_gaps = []
    
    print("🔄 Extracting embeddings...")
    
    with torch.no_grad():
        for anchor, pos, neg, gap, _ in tqdm(loader):
            anchor = anchor.to(device)
            pos = pos.to(device)
            neg = neg.to(device)
            
            emb_a = model(anchor)
            emb_p = model(pos)
            emb_n = model(neg)
            
            if isinstance(emb_a, (tuple, list)):
                emb_a = emb_a[0]
                emb_p = emb_p[0]
                emb_n = emb_n[0]
            
            emb_a = F.normalize(emb_a, p=2, dim=1)
            emb_p = F.normalize(emb_p, p=2, dim=1)
            emb_n = F.normalize(emb_n, p=2, dim=1)
            
            all_emb_a.append(emb_a.cpu())
            all_emb_p.append(emb_p.cpu())
            all_emb_n.append(emb_n.cpu())
            all_gaps.append(gap.cpu())
    
    all_emb_a = torch.cat(all_emb_a, dim=0)
    all_emb_p = torch.cat(all_emb_p, dim=0)
    all_emb_n = torch.cat(all_emb_n, dim=0)
    all_gaps = torch.cat(all_gaps, dim=0)
    
    return all_emb_a, all_emb_p, all_emb_n, all_gaps


def compute_core_metrics(emb_a, emb_p, emb_n):
    """Compute core similarity metrics"""
    pos_sims = F.cosine_similarity(emb_a, emb_p, dim=1).numpy()
    neg_sims = F.cosine_similarity(emb_a, emb_n, dim=1).numpy()
    
    results = {
        'pos_sim_mean': np.mean(pos_sims),
        'pos_sim_std': np.std(pos_sims),
        'neg_sim_mean': np.mean(neg_sims),
        'neg_sim_std': np.std(neg_sims),
        'separation': np.mean(pos_sims) - np.mean(neg_sims),
        'pos_sims': pos_sims,
        'neg_sims': neg_sims
    }
    
    return results


def compute_verification_metrics(emb_a, emb_p, emb_n):
    """Compute verification accuracy at different thresholds"""
    pos_sims = F.cosine_similarity(emb_a, emb_p, dim=1).numpy()
    neg_sims = F.cosine_similarity(emb_a, emb_n, dim=1).numpy()
    
    y_true = np.concatenate([np.ones(len(pos_sims)), np.zeros(len(neg_sims))])
    y_scores = np.concatenate([pos_sims, neg_sims])
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # EER
    eer, eer_thresh = calculate_eer(y_true, y_scores)
    
    # Optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    
    # TAR@FAR
    tar_at_far = {}
    for target_far in [0.1, 0.01, 0.001, 0.0001]:
        tar, thresh = get_tar_at_far(y_true, y_scores, target_far)
        tar_at_far[f'TAR@FAR={target_far}'] = tar
        tar_at_far[f'Thresh@FAR={target_far}'] = thresh
    
    # Bootstrap CI for AUC
    auc_low, auc_high = get_bootstrap_ci(y_true, y_scores, auc_metric, n_bootstraps=100)
    
    return {
        'roc_auc': roc_auc,
        'auc_ci_low': auc_low,
        'auc_ci_high': auc_high,
        'eer': eer,
        'eer_threshold': eer_thresh,
        'optimal_threshold': optimal_threshold,
        'optimal_tpr': optimal_tpr,
        'optimal_fpr': optimal_fpr,
        'optimal_accuracy': (optimal_tpr + (1 - optimal_fpr)) / 2,
        'tar_at_far': tar_at_far,
        'roc_data': (fpr, tpr, thresholds),
        'y_true': y_true,
        'y_scores': y_scores
    }


def analyze_age_bins(emb_a, emb_p, emb_n, gaps, bin_edges=[0, 0.2, 0.4, 0.6, 0.8, 1.0]):
    """Detailed age bin analysis"""
    pos_sims = F.cosine_similarity(emb_a, emb_p, dim=1).numpy()
    neg_sims = F.cosine_similarity(emb_a, emb_n, dim=1).numpy()
    gaps_np = gaps.numpy()
    
    bin_results = []
    
    print(f"\n📊 Age Bin Analysis:")
    print(f"{'Bin Range':<15} | {'Pos Sim':<10} | {'Neg Sim':<10} | {'Sep':<8} | {'AUC':<8} | {'TAR@1e-3':<10} | {'Count'}")
    print("-" * 90)
    
    for i in range(len(bin_edges) - 1):
        low, high = bin_edges[i], bin_edges[i + 1]
        
        # Positive pairs in this age bin
        pos_mask = (gaps_np >= low) & (gaps_np < high)
        
        if pos_mask.sum() < 10:  # Skip bins with too few samples
            continue
        
        pos_in_bin = pos_sims[pos_mask]
        
        # For verification metrics, use all negatives with positives in this bin
        y_true_bin = np.concatenate([np.ones(len(pos_in_bin)), np.zeros(len(neg_sims))])
        y_scores_bin = np.concatenate([pos_in_bin, neg_sims])
        
        # Compute metrics for this bin
        bin_auc = auc_metric(y_true_bin, y_scores_bin)
        tar_1e3, _ = get_tar_at_far(y_true_bin, y_scores_bin, 1e-3)
        
        bin_result = {
            'bin_range': f"{low:.2f}-{high:.2f}",
            'bin_low': low,
            'bin_high': high,
            'count': pos_mask.sum(),
            'pos_sim_mean': np.mean(pos_in_bin),
            'pos_sim_std': np.std(pos_in_bin),
            'neg_sim_mean': np.mean(neg_sims),
            'separation': np.mean(pos_in_bin) - np.mean(neg_sims),
            'auc': bin_auc,
            'tar_at_1e-3': tar_1e3
        }
        
        bin_results.append(bin_result)
        
        print(f"{bin_result['bin_range']:<15} | "
              f"{bin_result['pos_sim_mean']:<10.4f} | "
              f"{bin_result['neg_sim_mean']:<10.4f} | "
              f"{bin_result['separation']:<8.4f} | "
              f"{bin_result['auc']:<8.4f} | "
              f"{bin_result['tar_at_1e-3']*100:<10.2f}% | "
              f"{bin_result['count']}")
    
    return bin_results


def temporal_degradation_analysis(emb_a, emb_p, gaps):
    """Analyze how similarity degrades with age gap"""
    pos_sims = F.cosine_similarity(emb_a, emb_p, dim=1).numpy()
    gaps = gaps.numpy()
    
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(gaps, pos_sims)
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'degradation_rate': abs(slope)
    }


# ================= VISUALIZATION =================
def plot_comprehensive_analysis(results, bin_results, degradation, model_name, save_dir):
    """Create comprehensive visualization plots"""
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # 1. ROC Curve
    ax1 = fig.add_subplot(gs[0, 0])
    fpr, tpr, _ = results['verification']['roc_data']
    ax1.plot(fpr, tpr, 'b-', lw=2, 
             label=f"AUC = {results['verification']['roc_auc']:.4f}")
    ax1.plot([0, 1], [0, 1], 'k--', lw=1)
    ax1.plot(results['verification']['optimal_fpr'], 
             results['verification']['optimal_tpr'], 
             'ro', markersize=8, label='Optimal Point')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'{model_name} - ROC Curve')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. DET Curve (Log-Log)
    ax2 = fig.add_subplot(gs[0, 1])
    fpr_det, fnr_det, _ = det_curve(results['verification']['y_true'], 
                                     results['verification']['y_scores'])
    ax2.plot(fpr_det, fnr_det, 'b-', lw=2)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('FAR (False Accept Rate)')
    ax2.set_ylabel('FRR (False Reject Rate)')
    ax2.set_title('DET Curve (Security Analysis)')
    ax2.grid(True, which="both", alpha=0.3)
    
    # 3. Similarity Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(results['core']['neg_sims'], bins=50, alpha=0.6, 
             label='Negative', color='red', density=True)
    ax3.hist(results['core']['pos_sims'], bins=50, alpha=0.6, 
             label='Positive', color='green', density=True)
    ax3.axvline(np.mean(results['core']['pos_sims']), color='green', 
                linestyle='--', lw=2)
    ax3.axvline(np.mean(results['core']['neg_sims']), color='red', 
                linestyle='--', lw=2)
    ax3.set_xlabel('Cosine Similarity')
    ax3.set_ylabel('Density')
    ax3.set_title('Score Distribution')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Calibration Curve
    ax4 = fig.add_subplot(gs[1, 0])
    prob_true, prob_pred = calibration_curve(
        results['verification']['y_true'], 
        (results['verification']['y_scores'] + 1) / 2, 
        n_bins=10
    )
    ax4.plot(prob_pred, prob_true, marker='o', lw=2, label='Model')
    ax4.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect')
    ax4.set_xlabel('Predicted Probability')
    ax4.set_ylabel('Fraction of Positives')
    ax4.set_title('Calibration Curve (Reliability)')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # 5. Age Bin Performance - Positive Similarity
    ax5 = fig.add_subplot(gs[1, 1])
    bin_df = pd.DataFrame(bin_results)
    x_pos = range(len(bin_df))
    ax5.plot(x_pos, bin_df['pos_sim_mean'], 'o-', lw=2, 
             markersize=8, color='green', label='Positive Sim')
    ax5.fill_between(x_pos, 
                     bin_df['pos_sim_mean'] - bin_df['pos_sim_std'],
                     bin_df['pos_sim_mean'] + bin_df['pos_sim_std'],
                     alpha=0.3, color='green')
    ax5.set_xlabel('Age Gap Bin')
    ax5.set_ylabel('Cosine Similarity')
    ax5.set_title('Positive Similarity vs Age Gap')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(bin_df['bin_range'], rotation=45)
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # 6. Age Bin Performance - Separation
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(x_pos, bin_df['separation'], 'o-', lw=2, 
             markersize=8, color='blue')
    ax6.fill_between(x_pos, 0, bin_df['separation'], alpha=0.3, color='blue')
    ax6.set_xlabel('Age Gap Bin')
    ax6.set_ylabel('Separation (Pos - Neg)')
    ax6.set_title('Separation vs Age Gap')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(bin_df['bin_range'], rotation=45)
    ax6.axhline(y=0.3, color='r', linestyle='--', label='Target: 0.3')
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    # 7. Age Bin Performance - AUC
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.bar(x_pos, bin_df['auc'], color='purple', alpha=0.7)
    ax7.set_xlabel('Age Gap Bin')
    ax7.set_ylabel('ROC AUC')
    ax7.set_title('AUC vs Age Gap')
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(bin_df['bin_range'], rotation=45)
    ax7.axhline(y=0.9, color='r', linestyle='--', label='Target: 0.9')
    ax7.set_ylim([0.8, 1.0])
    ax7.legend()
    ax7.grid(alpha=0.3, axis='y')
    
    # 8. Age Bin Performance - TAR@FAR
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(x_pos, bin_df['tar_at_1e-3'] * 100, 'o-', lw=2, 
             markersize=8, color='orange')
    ax8.set_xlabel('Age Gap Bin')
    ax8.set_ylabel('TAR@FAR=0.001 (%)')
    ax8.set_title('True Accept Rate @ FAR=0.1%')
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels(bin_df['bin_range'], rotation=45)
    ax8.axhline(y=80, color='r', linestyle='--', label='Target: 80%')
    ax8.legend()
    ax8.grid(alpha=0.3)
    
    # 9. Temporal Degradation Scatter
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.scatter(results['gaps'], results['core']['pos_sims'], 
                alpha=0.3, s=10, color='blue')
    x_line = np.linspace(0, 1, 100)
    y_line = degradation['slope'] * x_line + degradation['intercept']
    ax9.plot(x_line, y_line, 'r-', lw=2, 
             label=f"Slope: {degradation['slope']:.4f}")
    ax9.set_xlabel('Normalized Age Gap')
    ax9.set_ylabel('Cosine Similarity')
    ax9.set_title(f"Temporal Degradation (R²={degradation['r_squared']:.4f})")
    ax9.legend()
    ax9.grid(alpha=0.3)
    
    # 10. Sample Count per Bin
    ax10 = fig.add_subplot(gs[3, 0])
    ax10.bar(x_pos, bin_df['count'], color='gray', alpha=0.7)
    ax10.set_xlabel('Age Gap Bin')
    ax10.set_ylabel('Number of Samples')
    ax10.set_title('Sample Distribution Across Age Bins')
    ax10.set_xticks(x_pos)
    ax10.set_xticklabels(bin_df['bin_range'], rotation=45)
    ax10.grid(alpha=0.3, axis='y')
    
    # 11. Performance Metrics Summary (Text)
    ax11 = fig.add_subplot(gs[3, 1:])
    ax11.axis('off')
    
    summary_text = f"""
    PERFORMANCE SUMMARY - {model_name}
    
    Core Metrics:
    • ROC AUC: {results['verification']['roc_auc']:.4f} (95% CI: {results['verification']['auc_ci_low']:.4f} - {results['verification']['auc_ci_high']:.4f})
    • EER: {results['verification']['eer']:.4f} @ Threshold {results['verification']['eer_threshold']:.4f}
    • Optimal Accuracy: {results['verification']['optimal_accuracy']:.4f} @ Threshold {results['verification']['optimal_threshold']:.4f}
    
    Similarity Statistics:
    • Positive Mean: {results['core']['pos_sim_mean']:.4f} ± {results['core']['pos_sim_std']:.4f}
    • Negative Mean: {results['core']['neg_sim_mean']:.4f} ± {results['core']['neg_sim_std']:.4f}
    • Separation: {results['core']['separation']:.4f}
    
    Temporal Robustness:
    • Degradation Rate: {degradation['degradation_rate']:.4f} per unit gap
    • R²: {degradation['r_squared']:.6f}
    • p-value: {degradation['p_value']:.2e}
    
    TAR @ FAR:
    • TAR@FAR=0.1:    {results['verification']['tar_at_far']['TAR@FAR=0.1']:.4f}
    • TAR@FAR=0.01:   {results['verification']['tar_at_far']['TAR@FAR=0.01']:.4f}
    • TAR@FAR=0.001:  {results['verification']['tar_at_far']['TAR@FAR=0.001']:.4f}
    • TAR@FAR=0.0001: {results['verification']['tar_at_far']['TAR@FAR=0.0001']:.4f}
    """
    
    ax11.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
              verticalalignment='center', bbox=dict(boxstyle='round', 
              facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'{model_name} - Comprehensive Temporal Robustness Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    filename = f"comprehensive_analysis_{model_name}.png"
    plt.savefig(os.path.join(save_dir, filename), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {save_dir}/comprehensive_analysis_{model_name}.png")


def compare_models(pretrained_results, finetuned_results, save_dir):
    """Create side-by-side comparison plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    models = ['Pretrained', 'Fine-tuned']
    results_list = [pretrained_results, finetuned_results]
    colors = ['blue', 'red']
    
    # 1. ROC Curves Comparison
    ax = axes[0, 0]
    for i, (name, res, color) in enumerate(zip(models, results_list, colors)):
        fpr, tpr, _ = res['verification']['roc_data']
        ax.plot(fpr, tpr, lw=2, color=color,
                label=f"{name} (AUC={res['verification']['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Similarity Distributions
    ax = axes[0, 1]
    for i, (name, res, color) in enumerate(zip(models, results_list, colors)):
        ax.hist(res['core']['pos_sims'], bins=50, alpha=0.4, 
                label=f'{name} Pos', color=color, density=True)
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Density')
    ax.set_title('Positive Similarity Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Age Bin AUC Comparison
    ax = axes[0, 2]
    pretrained_bins = pd.DataFrame(pretrained_results['age_bins'])
    finetuned_bins = pd.DataFrame(finetuned_results['age_bins'])
    
    x = np.arange(len(pretrained_bins))
    width = 0.35
    
    ax.bar(x - width/2, pretrained_bins['auc'], width, 
           label='Pretrained', color='blue', alpha=0.7)
    ax.bar(x + width/2, finetuned_bins['auc'], width, 
           label='Fine-tuned', color='red', alpha=0.7)
    ax.set_xlabel('Age Gap Bin')
    ax.set_ylabel('ROC AUC')
    ax.set_title('AUC Across Age Bins')
    ax.set_xticks(x)
    ax.set_xticklabels(pretrained_bins['bin_range'], rotation=45)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # 4. Separation Comparison
    ax = axes[1, 0]
    ax.bar(x - width/2, pretrained_bins['separation'], width, 
           label='Pretrained', color='blue', alpha=0.7)
    ax.bar(x + width/2, finetuned_bins['separation'], width, 
           label='Fine-tuned', color='red', alpha=0.7)
    ax.set_xlabel('Age Gap Bin')
    ax.set_ylabel('Separation')
    ax.set_title('Separation Across Age Bins')
    ax.set_xticks(x)
    ax.set_xticklabels(pretrained_bins['bin_range'], rotation=45)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # 5. TAR@FAR Comparison
    ax = axes[1, 1]
    ax.plot(x, pretrained_bins['tar_at_1e-3'] * 100, 'o-', lw=2, 
            markersize=8, color='blue', label='Pretrained')
    ax.plot(x, finetuned_bins['tar_at_1e-3'] * 100, 'o-', lw=2, 
            markersize=8, color='red', label='Fine-tuned')
    ax.set_xlabel('Age Gap Bin')
    ax.set_ylabel('TAR@FAR=0.001 (%)')
    ax.set_title('TAR@FAR=0.1% Across Age Bins')
    ax.set_xticks(x)
    ax.set_xticklabels(pretrained_bins['bin_range'], rotation=45)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 6. Degradation Rate Comparison
    ax = axes[1, 2]
    metrics = ['Degradation\nRate', 'R²', 'Separation', 'AUC']
    pretrained_vals = [
        pretrained_results['degradation']['degradation_rate'],
        pretrained_results['degradation']['r_squared'],
        pretrained_results['core']['separation'],
        pretrained_results['verification']['roc_auc']
    ]
    finetuned_vals = [
        finetuned_results['degradation']['degradation_rate'],
        finetuned_results['degradation']['r_squared'],
        finetuned_results['core']['separation'],
        finetuned_results['verification']['roc_auc']
    ]
    
    x_metrics = np.arange(len(metrics))
    
    ax.bar(x_metrics - width/2, pretrained_vals, width, 
           label='Pretrained', color='blue', alpha=0.7)
    ax.bar(x_metrics + width/2, finetuned_vals, width, 
           label='Fine-tuned', color='red', alpha=0.7)
    
    ax.set_ylabel('Score')
    ax.set_title('Overall Metrics Comparison')
    ax.set_xticks(x_metrics)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Add relative improvement text
    for i, (p, f) in enumerate(zip(pretrained_vals, finetuned_vals)):
        diff = ((f - p) / p) * 100 if p != 0 else 0
        # For degradation, lower is better, so flip the sign for coloring
        is_good = diff > 0 if i != 0 else diff < 0 
        color = 'green' if is_good else 'red'
        sign = '+' if diff > 0 else ''
        
        ax.text(i, max(p, f) + 0.01, f"{sign}{diff:.1f}%", 
                ha='center', fontsize=9, color=color, fontweight='bold')

    plt.suptitle('Pretrained vs Fine-tuned: Comprehensive Comparison', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    save_path = os.path.join(save_dir, 'model_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved comparison plot: {save_path}")


def evaluate_single_model(model_path, model_name, dataloader, save_dir):
    """Pipeline to evaluate a single model"""
    print(f"\n{'='*20} EVALUATING: {model_name} {'='*20}")
    
    # 1. Load Model
    model = load_model(model_path)
    if model is None:
        return None
    
    # 2. Extract Embeddings
    emb_a, emb_p, emb_n, gaps= extract_embeddings(model, dataloader, CONFIG['device'])
    
    # 3. Compute Metrics
    print("Computing core metrics...")
    core_results = compute_core_metrics(emb_a, emb_p, emb_n)
    
    print("Computing verification metrics...")
    ver_results = compute_verification_metrics(emb_a, emb_p, emb_n)
    
    # 4. Analyze Age Bins
    bin_results = analyze_age_bins(emb_a, emb_p, emb_n, gaps)
    
    # 5. Analyze Degradation
    degradation = temporal_degradation_analysis(emb_a, emb_p, gaps)
    
    # 6. Aggregate Results
    full_results = {
        'core': core_results,
        'verification': ver_results,
        'age_bins': bin_results,
        'degradation': degradation,
        'gaps': gaps.numpy()
    }
    
    # 7. Plot
    plot_comprehensive_analysis(full_results, bin_results, degradation, model_name, save_dir)
    
    return full_results


# ================= MAIN EXECUTION =================
def main():
    # Setup
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    print(f"🚀 Starting Evaluation. Results will be saved to: {CONFIG['save_dir']}")
    print(f"💻 Device: {CONFIG['device']}")

    # Define Transform (Standard for ArcFace/AdaFace 112x112)
    val_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Load Dataset
    try:
        dataset = AgeGapDataset(
            csv_path=CONFIG['csv_path'],
            root_dir=CONFIG['root_dir'],
            transform=val_transform,
            samples_per_identity=5,  # Fewer samples per id for testing is fine
            negative_strategy='hard'  # Use hard negatives for strict evaluation
        )
        
        dataloader = DataLoader(
            dataset, 
            batch_size=CONFIG['batch_size'], 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
    except Exception as e:
        print(f"❌ Error initializing dataset: {e}")
        return

    # Evaluate Pretrained
    pretrained_res = evaluate_single_model(
        CONFIG['pretrained_path'], 
        "Pretrained_AdaFace", 
        dataloader, 
        CONFIG['save_dir']
    )

    # Evaluate Fine-tuned
    finetuned_res = evaluate_single_model(
        CONFIG['finetuned_path'], 
        "Finetuned_Temporal", 
        dataloader, 
        CONFIG['save_dir']
    )

    # Compare
    if pretrained_res and finetuned_res:
        print("\n⚔️  Generating Comparison Report...")
        compare_models(pretrained_res, finetuned_res, CONFIG['save_dir'])
        
        # Write summary text file
        with open(os.path.join(CONFIG['save_dir'], 'summary_report.txt'), 'w') as f:
            f.write("COMPARISON SUMMARY\n")
            f.write("==================\n\n")
            
            f.write(f"{'Metric':<25} | {'Pretrained':<15} | {'Fine-tuned':<15} | {'Diff'}\n")
            f.write("-" * 70 + "\n")
            
            metrics_to_compare = [
                ('ROC AUC', pretrained_res['verification']['roc_auc'], finetuned_res['verification']['roc_auc']),
                ('TAR@FAR=0.001', pretrained_res['verification']['tar_at_far']['TAR@FAR=0.001'], finetuned_res['verification']['tar_at_far']['TAR@FAR=0.001']),
                ('Separation', pretrained_res['core']['separation'], finetuned_res['core']['separation']),
                ('Degradation Rate', pretrained_res['degradation']['degradation_rate'], finetuned_res['degradation']['degradation_rate']),
            ]
            
            for name, p, ft in metrics_to_compare:
                diff = ft - p
                f.write(f"{name:<25} | {p:<15.4f} | {ft:<15.4f} | {diff:+.4f}\n")
        
        print("✅ Summary report saved.")
    else:
        print("⚠️  Skipping comparison as one or more models failed to load.")

    print("\n🎉 Evaluation pipeline completed!")


if __name__ == "__main__":
    main()
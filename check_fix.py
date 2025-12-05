#!/usr/bin/env python3
"""
diagnose_embeddings.py

Single-file diagnostic for face-recognition embeddings.
Compares original checkpoint vs fine-tuned checkpoint:
 - builds temporal positive/negative pairs from CACD
 - extracts embeddings, normalizes
 - computes cosine + L2 statistics (mean/std)
 - computes ROC/AUC, TAR@FAR
 - logs results to CSV and saves simple histograms (png)
 - optional t-SNE if sklearn available

Usage:
  python diagnose_embeddings.py --root_dir /path/to/CACD --orig_ckpt orig.pth --finetune_ckpt finetune.pth
"""

import os, sys, argparse, random, math, csv, json, time
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cudnn.benchmark = True

# ----------------------
# Utils
# ----------------------
def parse_age(fname):
    try:
        return int(os.path.basename(fname).split('_')[0])
    except:
        return -1

def is_image_file(n):
    return n.lower().endswith(('.jpg','.jpeg','.png'))

# ----------------------
# Dataset builder (identity-split)
# ----------------------
def build_id_map(root_dir, min_imgs=2, subset_ratio=1.0, shuffle=True, seed=42):
    ids = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,d))]
    ids = sorted(ids)
    if shuffle:
        random.seed(seed)
        random.shuffle(ids)
    nsub = max(1, int(len(ids) * subset_ratio))
    ids = ids[:nsub]
    id_map = {}
    for idn in ids:
        p = os.path.join(root_dir, idn)
        imgs = []
        for f in os.listdir(p):
            if not is_image_file(f): continue
            age = parse_age(f)
            if age < 0:
                continue
            imgs.append((os.path.join(p,f), age))
        if len(imgs) >= min_imgs:
            imgs.sort(key=lambda x: x[1])
            id_map[idn] = imgs
    return id_map

def build_pairs_from_idmap(id_map, min_age_gap=10, neg_ratio=1.0, max_pos=None):
    pos = []
    for idn, imgs in id_map.items():
        n = len(imgs)
        for i in range(n):
            for j in range(i+1, n):
                gap = abs(imgs[j][1] - imgs[i][1])
                if gap >= min_age_gap:
                    pos.append((imgs[i][0], imgs[j][0], 1, gap))
    all_imgs = [(idn, p, a) for idn, lst in id_map.items() for (p,a) in lst]
    neg = []
    num_neg = int(len(pos) * neg_ratio)
    attempts = 0
    while len(neg) < num_neg and attempts < max(10000, num_neg*5) and len(all_imgs) > 1:
        a = random.choice(all_imgs); b = random.choice(all_imgs)
        if a[0] != b[0]:
            neg.append((a[1], b[1], 0, abs(a[2]-b[2])))
        attempts += 1
    if max_pos and len(pos) > max_pos:
        pos = random.sample(pos, max_pos)
    return pos, neg

# ----------------------
# Model loader helper
# - If model is a full AdaFaceNet that returns logits when labels fed,
#   ensure we can call backbone to obtain embeddings.
# - Tries common imports for IR_SE50 models.
# ----------------------
def try_build_backbone(use_iresnet50, embedding_size=512, checkpoint=None, device='cpu'):
    """
    Returns (model_fn, is_backbone_instance)
      - model_fn(img_tensor) -> embeddings (BxD) normalized not necessarily
    If checkpoint provided, tries to load into backbone/state_dict where applicable.
    """
    if use_iresnet50:
        try:
            from models.model_ir_se50 import Backbone as IRBackbone
            backbone = IRBackbone(num_layers=50, drop_ratio=0.4, mode='ir_se')
        except Exception:
            try:
                from models.model_ir_se50 import IR_SE50
                backbone = IR_SE50(embedding_size)
            except Exception as e:
                print("ERROR: cannot import IR_SE50 from models.model_ir_se50:", e)
                raise
    else:
        # Fallback: tiny CNN linear (not recommended)
        import torch.nn as nn
        class Tiny(nn.Module):
            def __init__(self, emb=512):
                super().__init__()
                self.fc = nn.Linear(3*112*112, emb)
            def forward(self, x):
                return self.fc(x.view(x.size(0), -1))
        backbone = Tiny(embedding_size)

    # try load checkpoint into backbone or model
    if checkpoint:
        if not os.path.exists(checkpoint):
            print("Warning: checkpoint does not exist:", checkpoint)
        else:
            ck = torch.load(checkpoint, map_location='cpu')
            state = ck
            if isinstance(ck, dict) and ('state_dict' in ck or 'model' in ck):
                # try to extract
                if 'state_dict' in ck:
                    state = ck['state_dict']
                elif 'model' in ck:
                    state = ck['model']
            # Build cleaned dict
            new = {}
            for k,v in state.items():
                nk = k
                if nk.startswith('module.'): nk = nk[7:]
                # strip common prefixes
                nk = nk.replace('backbone.','').replace('model.','')
                new[nk] = v
            try:
                backbone.load_state_dict(new, strict=False)
                print("Loaded checkpoint into backbone (strict=False).")
            except Exception as e:
                print("Warning: cannot fully load ckpt into backbone:", e)
                # still continue. Might be full-model ckpt; user can supply backbone-only ckpt.
    backbone.to(device)
    backbone.eval()
    return backbone

# ----------------------
# Embedding extraction wrapper: robustly handle models that output different things
# - If model called returns tuple (logits, norms) or (logits,) or raw embedding.
# - We attempt to extract a 2D tensor (BxD) as embedding.
# ----------------------
def extract_embedding_from_model(model, images, device):
    """
    images: torch tensor BxCxHxW (on device)
    returns: normalized embeddings BxD (np array)
    """
    model.eval()
    with torch.no_grad():
        out = model(images)
    # out can be: tensor (BxD) OR (logits, norms) OR (logits,) or (logits, norms, feats)
    if isinstance(out, tuple) or isinstance(out, list):
        # try to find tensor with shape BxD where D ~ 512
        for item in out[::-1]:
            if isinstance(item, torch.Tensor) and item.dim() == 2:
                emb = item
                break
        else:
            # fallback to first tensor
            for item in out:
                if isinstance(item, torch.Tensor) and item.dim() == 2:
                    emb = item; break
            else:
                raise RuntimeError("Cannot find 2D tensor in model output tuple.")
    elif isinstance(out, torch.Tensor):
        if out.dim() == 2:
            emb = out
        else:
            # some models may return BxCxHxW; flatten last dims
            emb = out.view(out.size(0), -1)
    else:
        raise RuntimeError("Unknown model output type: %s" % type(out))

    # ensure float and normalized
    emb = emb.float()
    emb = F.normalize(emb, dim=1)
    return emb.cpu().numpy()

# ----------------------
# Batch embedding extractor given filepaths list
# ----------------------
def batch_extract_embeddings(model, paths, transform, device, batch_size=64):
    embeddings = []
    paths_out = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i+batch_size]
        imgs = []
        for p in batch:
            try:
                img = Image.open(p).convert('RGB')
                img = transform(img)
                imgs.append(img)
            except Exception as e:
                print("Warning load image", p, e)
        if len(imgs) == 0:
            continue
        x = torch.stack(imgs).to(device)
        embs = extract_embedding_from_model(model, x, device)
        embeddings.append(embs)
        paths_out.extend(batch[:len(embs)])
    if len(embeddings) == 0:
        return np.zeros((0,512)), []
    embeddings = np.vstack(embeddings)
    return embeddings, paths_out

# ----------------------
# Pair similarity metrics
# ----------------------
def compute_pair_sims(emb_dict, pairs):
    # emb_dict: {path: emb(np1d)}
    sims = []
    l2s = []
    labs = []
    gaps = []
    for a,b,label,gap in pairs:
        if a not in emb_dict or b not in emb_dict:
            continue
        ea = emb_dict[a]; eb = emb_dict[b]
        cos = float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-12))
        l2 = float(np.linalg.norm(ea - eb))
        sims.append(cos); l2s.append(l2); labs.append(label); gaps.append(gap)
    return np.array(sims), np.array(l2s), np.array(labs), np.array(gaps)

# ----------------------
# ROC/AUC and TAR@FAR
# ----------------------
def compute_roc_metrics(sims_pos, sims_neg, num_thr=200):
    scores = np.concatenate([sims_pos, sims_neg])
    labels = np.concatenate([np.ones_like(sims_pos), np.zeros_like(sims_neg)])
    thr = np.linspace(1.0, -1.0, num_thr)
    tpr = []; fpr = []
    for t in thr:
        tp = np.sum(sims_pos >= t)
        fn = np.sum(sims_pos < t)
        fp = np.sum(sims_neg >= t)
        tn = np.sum(sims_neg < t)
        tpr.append(tp / max(1, tp+fn))
        fpr.append(fp / max(1, fp+tn))
    # auc via trapezoid
    auc = 0.0
    for i in range(1,len(fpr)):
        auc += 0.5*(tpr[i]+tpr[i-1])*abs(fpr[i]-fpr[i-1])
    far_targets = [1e-1, 1e-2, 1e-3]
    tar = {}
    for ft in far_targets:
        vals = [tp for tp,f in zip(tpr,fpr) if f <= ft]
        tar[ft] = max(vals) if vals else 0.0
    return {'thr': thr, 'tpr': np.array(tpr), 'fpr': np.array(fpr), 'auc': auc, 'tar_at_far': tar}

# ----------------------
# Main diagnostic routine
# ----------------------
def run_diagnostic(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    os.makedirs(args.out_dir, exist_ok=True)

    print("Building id_map from:", args.root_dir)
    id_map = build_id_map(args.root_dir, min_imgs=2, subset_ratio=args.subset, shuffle=True, seed=args.seed)
    pos_pairs, neg_pairs = build_pairs_from_idmap(id_map, min_age_gap=args.min_age_gap, neg_ratio=1.0, max_pos=args.num_pairs//2)
    print(f"Built pos={len(pos_pairs)} neg={len(neg_pairs)} (min_gap={args.min_age_gap})")

    # sample balanced pairs for speed
    max_eval = args.num_pairs
    all_pairs = (pos_pairs[:max_eval//2] if len(pos_pairs)>max_eval//2 else pos_pairs) + \
                (neg_pairs[:max_eval//2] if len(neg_pairs)>max_eval//2 else neg_pairs)
    random.shuffle(all_pairs)
    # give separate pos/neg lists for metrics
    pos_eval = [p for p in all_pairs if p[2]==1]
    neg_eval = [p for p in all_pairs if p[2]==0]

    print("Eval pairs:", len(pos_eval), "pos;", len(neg_eval), "neg")

    # transform
    tf = transforms.Compose([transforms.Resize((112,112)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])

    results = []

    # helper to process a checkpoint/model name
    def process_model(tag, ckpt_path):
        print("\n=== Processing", tag, ckpt_path)
        # Build backbone/model
        model = try_build_backbone(args.use_iresnet50, embedding_size=args.embedding_size, checkpoint=ckpt_path, device=device)

        # Check outputs on single image to detect shape
        sample_img_path = None
        for idn, lst in id_map.items():
            if len(lst)>0:
                sample_img_path = lst[0][0]; break
        if sample_img_path is None:
            raise RuntimeError("No sample image found.")

        img = Image.open(sample_img_path).convert('RGB')
        x = tf(img).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            out = model(x)
        # detect output
        out_shapes = None
        if isinstance(out, (tuple,list)):
            out_shapes = [o.shape if isinstance(o, torch.Tensor) else None for o in out]
        elif isinstance(out, torch.Tensor):
            out_shapes = [out.shape]
        print("Model sample output shapes:", out_shapes)

        # Build unique image list to extract embeddings (from pairs)
        unique_paths = set()
        for a,b,lab,g in pos_eval+neg_eval:
            unique_paths.add(a); unique_paths.add(b)
        unique_paths = list(unique_paths)
        print("Unique images to embed:", len(unique_paths))

        # Extract embeddings batchwise
        emb_array, used_paths = batch_extract_embeddings(model, unique_paths, tf, device, batch_size=args.batch_size)
        print("Extracted embeddings:", emb_array.shape, "used:", len(used_paths))
        emb_dict = {p: emb for p, emb in zip(used_paths, emb_array)}

        # Compute pair sims
        sims, l2s, labs, gaps = compute_pair_sims(emb_dict, pos_eval+neg_eval)
        # split pos/neg
        sims_pos = sims[labs==1]; sims_neg = sims[labs==0]
        l2_pos = l2s[labs==1]; l2_neg = l2s[labs==0]
        gaps_pos = gaps[labs==1]

        # stats
        res = {
            'tag': tag,
            'ckpt': ckpt_path,
            'n_pos': len(sims_pos),
            'n_neg': len(sims_neg),
            'cos_pos_mean': float(np.mean(sims_pos)) if len(sims_pos)>0 else None,
            'cos_pos_std': float(np.std(sims_pos)) if len(sims_pos)>0 else None,
            'cos_neg_mean': float(np.mean(sims_neg)) if len(sims_neg)>0 else None,
            'cos_neg_std': float(np.std(sims_neg)) if len(sims_neg)>0 else None,
            'l2_pos_mean': float(np.mean(l2_pos)) if len(l2_pos)>0 else None,
            'l2_neg_mean': float(np.mean(l2_neg)) if len(l2_neg)>0 else None
        }

        # ROC metrics
        roc = compute_roc_metrics(sims_pos, sims_neg, num_thr=200)
        res['auc'] = float(roc['auc'])
        res['tar_at_far'] = roc['tar_at_far']

        print(f"  AUC: {res['auc']:.4f}")
        print(f"  POS mean/std: {res['cos_pos_mean']:.4f} / {res['cos_pos_std']:.4f}")
        print(f"  NEG mean/std: {res['cos_neg_mean']:.4f} / {res['cos_neg_std']:.4f}")
        print("  TAR@FAR:", res['tar_at_far'])

        # accuracy by gap bins (pos only)
        bins = [(0,2),(3,4),(5,8),(9,10),(11,1000)]
        threshold = None
        # choose threshold at FPR<=0.01 if available
        thr_list = roc['thr']; fpr_list = roc['fpr']; tpr_list = roc['tpr']
        best_thr = None; best_tpr = 0.0
        for t,f,tpr in zip(thr_list, fpr_list, tpr_list):
            if f <= 0.01 and tpr > best_tpr:
                best_tpr = tpr; best_thr = t
        if best_thr is None:
            combined = np.concatenate([sims_pos, sims_neg]) if (len(sims_pos)+len(sims_neg))>0 else np.array([0.0])
            best_thr = float(np.median(combined))
        res['used_threshold'] = float(best_thr)

        gap_stats = []
        for lo,hi in bins:
            idx = np.where((gaps_pos >= lo) & (gaps_pos <= hi))[0]
            if idx.size == 0:
                gap_stats.append({'range':f"{lo}-{hi}", 'count':0, 'tpr':None, 'avg_sim':None})
            else:
                sims_in = sims_pos[idx]
                tpr = float((sims_in >= best_thr).sum() / len(sims_in))
                gap_stats.append({'range':f"{lo}-{hi}", 'count':int(len(sims_in)), 'tpr':tpr, 'avg_sim':float(sims_in.mean())})
        res['gap_stats'] = gap_stats

        # Save histograms to CSV
        csv_hist = os.path.join(args.out_dir, f"hist_{tag}.csv")
        with open(csv_hist, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['type','value'])
            for v in sims_pos:
                writer.writerow(['pos', float(v)])
            for v in sims_neg:
                writer.writerow(['neg', float(v)])
        print("Saved histogram csv:", csv_hist)

        # Save a JSON result
        json_path = os.path.join(args.out_dir, f"report_{tag}.json")
        with open(json_path, 'w') as f:
            json.dump(res, f, indent=2)
        print("Saved JSON report:", json_path)

        # Try plotting histograms if matplotlib exists
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6,4))
            plt.hist(sims_neg, bins=80, alpha=0.6, label='neg')
            plt.hist(sims_pos, bins=80, alpha=0.6, label='pos')
            plt.legend()
            plt.title(f"Similarity hist {tag} (pos mean={res['cos_pos_mean']:.3f} neg mean={res['cos_neg_mean']:.3f})")
            plt.xlabel('cosine')
            plt.ylabel('count')
            png = os.path.join(args.out_dir, f"hist_{tag}.png")
            plt.savefig(png, dpi=150, bbox_inches='tight')
            plt.close()
            print("Saved histogram png:", png)
        except Exception as e:
            print("matplotlib not available or plotting failed:", e)

        # Optional t-SNE of a sample (try sklearn if available)
        try:
            from sklearn.manifold import TSNE
            sample_paths = used_paths[:500] if len(used_paths)>500 else used_paths
            sample_embs = np.vstack([emb_dict for emb_dict in [emb_dict.get(p) for p in sample_paths] if emb_dict is not None])
            if sample_embs.shape[0] >= 50:
                tsne = TSNE(n_components=2, perplexity=30, random_state=42)
                emb2d = tsne.fit_transform(sample_embs)
                # save csv
                out_tsne = os.path.join(args.out_dir, f"tsne_{tag}.csv")
                with open(out_tsne, 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['path','x','y'])
                    for p,(x,y) in zip(sample_paths, emb2d):
                        w.writerow([p, float(x), float(y)])
                print("Saved t-SNE csv:", out_tsne)
        except Exception:
            pass

        return res

    # process original & finetuned
    rows = []
    if args.orig_ckpt:
        r0 = process_model('orig', args.orig_ckpt)
        rows.append(r0)
    if args.finetune_ckpt:
        r1 = process_model('finetuned', args.finetune_ckpt)
        rows.append(r1)

    # save summary csv
    csv_out = os.path.join(args.out_dir, 'summary_report.csv')
    with open(csv_out, 'w', newline='') as f:
        fieldnames = ['tag','ckpt','n_pos','n_neg','cos_pos_mean','cos_pos_std','cos_neg_mean','cos_neg_std','auc','used_threshold','tar_at_far']
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for r in rows:
            writer.writerow([r.get(f) if not isinstance(r.get(f), dict) else json.dumps(r.get(f)) for f in fieldnames])
    print("Saved summary CSV:", csv_out)
    print("Done diagnostic. Reports in", args.out_dir)


# ----------------------
# CLI
# ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', required=True)
    parser.add_argument('--orig_ckpt', default=None, help="path to original checkpoint (backbone or full model)")
    parser.add_argument('--finetune_ckpt', default=None, help="path to finetuned checkpoint")
    parser.add_argument('--use_iresnet50', action='store_true', help="import IR_SE50 from models.model_ir_se50")
    parser.add_argument('--out_dir', default='diag_out')
    parser.add_argument('--num_pairs', type=int, default=4000)
    parser.add_argument('--min_age_gap', type=int, default=10)
    parser.add_argument('--subset', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--embedding_size', type=int, default=512)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_diagnostic(args)

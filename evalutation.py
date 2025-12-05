import os
import sys
import random
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Callable, Any
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageFile
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import net
# Configure PIL and PyTorch
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cudnn.benchmark = True

# Optional MTCNN import
try:
    from facenet_pytorch import MTCNN
    HAS_MTCNN = True
except ImportError:
    HAS_MTCNN = False

# Import custom models
from models.facenet_model import SiameseFaceNet
from models.model_ir_se50 import IR_SE50
from models.ada_face import AdaFaceNet


# ============================================================================
# Configuration and Data Classes
# ============================================================================
import net
from face_alignment import align

adaface_models = {
    'ir_101': "weights/adaface_ir101_ms1mv2.ckpt",
}

class AdaFaceOriginalWrapper(torch.nn.Module):
    def __init__(self, architecture='ir_101', device='cpu'):
        super().__init__()
        assert architecture in adaface_models
        self.device = device

        self.model = net.build_model(architecture)
        statedict = torch.load(adaface_models[architecture], map_location=device)['state_dict']

        model_statedict = {
            key[6:]: val for key, val in statedict.items()
            if key.startswith('model.')
        }

        self.model.load_state_dict(model_statedict)
        self.model.to(device)
        self.model.eval()
        
    def forward(self, x):
        """
        x: tensor [B, 3, 112, 112]
        Return: normalized embedding
        """
        with torch.no_grad():
            feature, _ = self.model(x)
            feature = F.normalize(feature, dim=1)
            return feature

    # ✅ hàm xử ảnh từ path (cho evaluation)
    def extract(self, image_path):
        try:
            aligned_img = align.get_aligned_face(image_path)
            if aligned_img is None:
                return None

            np_img = np.array(aligned_img)

            bgr_img = ((np_img[:, :, ::-1] / 255.) - 0.5) / 0.5
            tensor = torch.tensor(
                bgr_img.transpose(2, 0, 1)
            ).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                feature, _ = self.model(tensor)

            feature = F.normalize(feature, dim=1)
            return feature.squeeze(0).cpu().numpy()

        except Exception as e:
            print(f"AdaFace error {image_path} : {e}")
            return None
        
@dataclass
class EvaluationConfig:
    """Configuration for evaluation pipeline."""
    root_dir: str
    pairs_file: str = "test_pairs.txt"
    num_pairs: int = 6000
    min_imgs_per_id: int = 3
    seed: int = 42
    output_dir: str = "eval_reports"
    use_mtcnn: bool = True
    image_size: int = 112
    batch_size: int = 32

    @property
    def age_gap_bins(self) -> List[Tuple[int, int]]:
        """
        Finer-grained bins to analyze temporal degradation of face embeddings.
        """
        return [
            (0, 2),
            (3, 5),
            (6, 10)
        ]

@dataclass
class ModelConfig:
    """Configuration for a single model to evaluate."""
    name: str
    model_init: Callable[[], torch.nn.Module]
    checkpoint_path: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if self.checkpoint_path and not os.path.exists(self.checkpoint_path):
            print(f"⚠️  Checkpoint not found: {self.checkpoint_path}")


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    auc: float
    optimal_threshold: float
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    accuracy_by_age_gap: Dict[str, float]
    neg_score_by_age_gap: Dict[str, float]
    tar_at_far: Dict[float, Tuple[float, float]]  # FAR -> (TAR, threshold)


# ============================================================================
# Utility Functions
# ============================================================================

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def is_image_file(filename: str) -> bool:
    """Check if file is an image."""
    return filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))


def parse_age_from_filename(filename: str) -> int:
    """
    Extract age from filename.
    Expected format: {age}_{id}_{timestamp}.jpg
    """
    try:
        return int(Path(filename).stem.split('_')[0])
    except (ValueError, IndexError):
        return -1


# ============================================================================
# Pair Generation
# ============================================================================

class PairGenerator:
    """Generates balanced test pairs with age-gap diversity."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.id_map: Dict[str, List[Tuple[str, int]]] = {}
        
    def load_dataset(self) -> None:
        """Load and organize dataset by identity."""
        root = Path(self.config.root_dir)
        
        for id_dir in root.iterdir():
            if not id_dir.is_dir():
                continue
                
            images = []
            for img_file in id_dir.iterdir():
                if not is_image_file(img_file.name):
                    continue
                    
                age = parse_age_from_filename(img_file.name)
                if age < 0:
                    continue
                    
                images.append((str(img_file), age))
            
            # Filter identities with sufficient images
            if len(images) >= self.config.min_imgs_per_id:
                images.sort(key=lambda x: x[1])  # Sort by age
                self.id_map[id_dir.name] = images
        
        if len(self.id_map) < 2:
            raise ValueError(
                f"Insufficient identities with >= {self.config.min_imgs_per_id} images. "
                f"Found {len(self.id_map)} identities."
            )
        
        print(f"✅ Loaded {len(self.id_map)} identities")
    
    def generate_positive_pairs(self) -> List[Tuple[str, str, int, int]]:
        """Generate positive pairs (same identity)."""
        pairs = []
        
        for id_name, images in self.id_map.items():
            n = len(images)
            for i in range(n):
                for j in range(i + 1, n):
                    path1, age1 = images[i]
                    path2, age2 = images[j]
                    age_gap = abs(age2 - age1)
                    pairs.append((path1, path2, 1, age_gap))
        
        print(f"✅ Generated {len(pairs)} positive pairs")
        return pairs
    
    def generate_negative_pairs(self, 
                               num_pairs: int,
                               similar_age_ratio: float = 0.5) -> List[Tuple[str, str, int, int]]:
        """
        Generate negative pairs (different identities).
        
        Args:
            num_pairs: Number of negative pairs to generate
            similar_age_ratio: Ratio of similar-age negatives (<5 years)
        """
        all_images = [
            (id_name, path, age) 
            for id_name, images in self.id_map.items() 
            for path, age in images
        ]
        
        pairs = []
        num_similar = int(num_pairs * similar_age_ratio)
        num_large_gap = num_pairs - num_similar
        
        # Generate similar-age negatives (<5 years)
        pairs.extend(self._generate_negatives_with_constraint(
            all_images, num_similar, max_age_diff=5
        ))
        
        # Generate large age-gap negatives (>10 years)
        pairs.extend(self._generate_negatives_with_constraint(
            all_images, num_large_gap, min_age_diff=10
        ))
        
        # Fill remaining with random negatives if needed
        while len(pairs) < num_pairs:
            pairs.extend(self._generate_negatives_with_constraint(
                all_images, num_pairs - len(pairs), max_attempts=50
            ))
        
        print(f"✅ Generated {len(pairs)} negative pairs")
        return pairs[:num_pairs]
    
    def _generate_negatives_with_constraint(self,
                                          all_images: List[Tuple[str, str, int]],
                                          num_pairs: int,
                                          min_age_diff: int = 0,
                                          max_age_diff: int = 1000,
                                          max_attempts: int = None) -> List[Tuple[str, str, int, int]]:
        """Generate negative pairs with age difference constraints."""
        pairs = []
        max_attempts = max_attempts or num_pairs * 10
        attempts = 0
        
        while len(pairs) < num_pairs and attempts < max_attempts:
            img1 = random.choice(all_images)
            img2 = random.choice(all_images)
            
            # Different identities
            if img1[0] == img2[0]:
                attempts += 1
                continue
            
            age_diff = abs(img1[2] - img2[2])
            
            # Check age constraint
            if min_age_diff <= age_diff <= max_age_diff:
                pairs.append((img1[1], img2[1], 0, age_diff))
            
            attempts += 1
        
        return pairs
    
    def generate_and_save(self) -> str:
        """Generate pairs and save to file."""
        if os.path.exists(self.config.pairs_file):
            print(f"ℹ️  Using existing pairs file: {self.config.pairs_file}")
            return self.config.pairs_file
        
        set_seed(self.config.seed)
        self.load_dataset()
        
        # Generate pairs
        positive_pairs = self.generate_positive_pairs()
        
        num_pos = min(len(positive_pairs), self.config.num_pairs // 2)
        num_neg = self.config.num_pairs - num_pos
        
        if len(positive_pairs) > num_pos:
            positive_pairs = random.sample(positive_pairs, num_pos)
        
        negative_pairs = self.generate_negative_pairs(num_neg)
        
        # Combine and shuffle
        all_pairs = positive_pairs + negative_pairs
        random.shuffle(all_pairs)
        
        # Save to file
        with open(self.config.pairs_file, 'w', encoding='utf-8') as f:
            for path1, path2, label, age_gap in all_pairs:
                f.write(f"{path1},{path2},{label},{age_gap}\n")
        
        print(f"✅ Saved {len(all_pairs)} pairs to {self.config.pairs_file}")
        print(f"   Positive: {len(positive_pairs)}, Negative: {len(negative_pairs)}")
        
        return self.config.pairs_file


# ============================================================================
# Model Loading and Inference
# ============================================================================

class ModelLoader:
    """Robust model loader with checkpoint handling."""
    
    @staticmethod
    def load_checkpoint(model: torch.nn.Module,
                       checkpoint_path: str,
                       device: torch.device) -> Tuple[bool, Optional[List], Optional[List]]:
        """
        Load checkpoint into model with robust key handling.
        
        Returns:
            (success, missing_keys, unexpected_keys)
        """
        if not os.path.exists(checkpoint_path):
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            return False, None, None
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            return False, None, None
        
        # Extract state dict
        state_dict = ModelLoader._extract_state_dict(checkpoint)
        
        # Clean keys
        state_dict = ModelLoader._clean_state_dict_keys(state_dict)
        
        # Try loading with key adaptation
        return ModelLoader._load_with_adaptation(model, state_dict)
    
    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
        """Extract state dict from various checkpoint formats."""
        if isinstance(checkpoint, dict):
            for key in ['state_dict', 'model', 'model_state_dict']:
                if key in checkpoint:
                    return checkpoint[key]
            return checkpoint
        return checkpoint
    
    @staticmethod
    def _clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Remove common prefixes from state dict keys."""
        cleaned = {}
        for key, value in state_dict.items():
            new_key = key
            # Remove module prefix from DataParallel
            if new_key.startswith('module.'):
                new_key = new_key.replace('module.', '', 1)
            cleaned[new_key] = value
        return cleaned
    
    @staticmethod
    def _load_with_adaptation(model: torch.nn.Module,
                            state_dict: Dict[str, torch.Tensor]) -> Tuple[bool, List, List]:
        """Load state dict with automatic key adaptation."""
        # Try direct load
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print("✅ Loaded checkpoint (strict=False)")
            return True, missing, unexpected
        except Exception as e:
            print(f"⚠️  Direct load failed: {e}")
        
        # Try adapting keys
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        
        adapted_dict = ModelLoader._adapt_keys(model_keys, ckpt_keys, state_dict)
        
        try:
            missing, unexpected = model.load_state_dict(adapted_dict, strict=False)
            print("✅ Loaded checkpoint with key adaptation")
            return True, missing, unexpected
        except Exception as e:
            print(f"❌ Failed to load checkpoint after adaptation: {e}")
            return False, None, None
    
    @staticmethod
    def _adapt_keys(model_keys: set,
                   ckpt_keys: set,
                   state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Adapt checkpoint keys to match model keys."""
        adapted = {}
        
        # Check if we need to add/remove 'backbone.' prefix
        model_has_backbone = any(k.startswith('backbone.') for k in model_keys)
        ckpt_has_backbone = any(k.startswith('backbone.') for k in ckpt_keys)
        
        for key, value in state_dict.items():
            if model_has_backbone and not ckpt_has_backbone:
                # Add backbone prefix
                adapted[f'backbone.{key}'] = value
            elif not model_has_backbone and ckpt_has_backbone:
                # Remove backbone prefix
                adapted[key.replace('backbone.', '', 1)] = value
            else:
                adapted[key] = value
        
        return adapted


class EmbeddingExtractor:
    """Extract normalized embeddings from various model architectures."""
    
    @staticmethod
    def extract(model: torch.nn.Module,
               input_tensor: torch.Tensor,
               device: torch.device) -> torch.Tensor:
        """
        Extract and normalize embeddings.
        
        Handles various output formats:
        - Single tensor: (B, D)
        - Tuple: (logits, embeddings) or (embeddings,)
        - Dict: {'embedding': tensor}
        
        Returns:
            Normalized embedding tensor (B, D)
        """
        model.eval()
        
        with torch.no_grad():
            output = model(input_tensor)
        
        embedding = EmbeddingExtractor._find_embedding_tensor(output)
        
        if embedding is None:
            raise RuntimeError(
                f"Cannot extract embedding from model output. "
                f"Output type: {type(output)}, "
                f"Output shape: {output.shape if isinstance(output, torch.Tensor) else 'N/A'}"
            )
        
        # Ensure 2D and normalize
        if embedding.dim() != 2:
            embedding = embedding.view(embedding.size(0), -1)
        
        embedding = embedding.float().to(device)
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding
    
    @staticmethod
    def _find_embedding_tensor(output: Any) -> Optional[torch.Tensor]:
        """Find the embedding tensor from various output formats."""
        # Direct tensor
        if isinstance(output, torch.Tensor):
            return output
        
        # Dictionary
        if isinstance(output, dict):
            for key in ['embedding', 'embeddings', 'features']:
                if key in output and isinstance(output[key], torch.Tensor):
                    return output[key]
        
        # Tuple or list
        if isinstance(output, (tuple, list)):
            # Prefer last 2D tensor (common pattern: logits, embeddings)
            candidates = []
            for item in output:
                if isinstance(item, torch.Tensor) and item.dim() == 2:
                    candidates.append(item)
            
            if candidates:
                return candidates[-1]
        
        return None


# ============================================================================
# Face Processing Pipeline
# ============================================================================

class FaceProcessor:
    """Handle face detection, alignment, and preprocessing."""
    
    def __init__(self, image_size: int = 112, use_mtcnn: bool = True, device: torch.device = None):
        self.image_size = image_size
        self.device = device or torch.device('cpu')
        
        # Standard transform (fallback)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # MTCNN for face detection and alignment
        self.mtcnn = None
        if use_mtcnn and HAS_MTCNN:
            try:
                self.mtcnn = MTCNN(
                    image_size=image_size,
                    margin=0,
                    min_face_size=20,
                    device=self.device,
                    post_process=False
                )
                print("✨ MTCNN enabled for face alignment")
            except Exception as e:
                print(f"⚠️  MTCNN initialization failed: {e}")
                self.mtcnn = None
    
    def process_image(self, image_path: str) -> Optional[torch.Tensor]:
        """
        Load and process image.
        
        Returns:
            Processed tensor (C, H, W) or None if processing fails
        """
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"⚠️  Failed to load image {image_path}: {e}")
            return None
        
        # Try MTCNN first
        if self.mtcnn is not None:
            try:
                face = self.mtcnn(image)
                if face is not None:
                    return self._process_mtcnn_output(face)
            except Exception as e:
                pass  # Fall back to standard transform
        
        # Fallback to standard transform
        return self.transform(image)
    
    def _process_mtcnn_output(self, face: Any) -> torch.Tensor:
        """Process MTCNN output to standard format."""
        if isinstance(face, Image.Image):
            return self.transform(face)
        
        if isinstance(face, torch.Tensor):
            # Ensure correct range and normalization
            if face.max() > 2.0:  # In range [0, 255]
                face = face / 255.0
            elif face.min() < 0:  # In range [-1, 1]
                face = (face + 1) / 2.0
            
            # Resize if needed
            if face.shape[-2:] != (self.image_size, self.image_size):
                face = F.interpolate(
                    face.unsqueeze(0),
                    size=(self.image_size, self.image_size),
                    mode='bilinear',
                    align_corners=False
                )[0]
            
            # Normalize to [-1, 1]
            mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
            face = (face - mean) / std
            
            return face
        
        raise ValueError(f"Unexpected MTCNN output type: {type(face)}")


# ============================================================================
# Evaluation Pipeline
# ============================================================================

class FaceEvaluator:
    """Main evaluation pipeline."""
    
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 config: EvaluationConfig):
        self.model = model.to(device).eval()
        self.device = device
        self.config = config
        self.processor = FaceProcessor(
            image_size=config.image_size,
            use_mtcnn=config.use_mtcnn,
            device=device
        )
    
    def extract_embedding(self, image_path: str) -> Optional[np.ndarray]:

        # --- CASE 1: AdaFace original (paper) ---
        if isinstance(self.model, AdaFaceOriginalWrapper):
            return self.model.extract(image_path)

        # --- CASE 2: Other models ---
        tensor = self.processor.process_image(image_path)
        if tensor is None:
            return None
        
        batch = tensor.unsqueeze(0).to(self.device)

        embedding = EmbeddingExtractor.extract(
            self.model, batch, self.device
        )

        return embedding.squeeze(0).cpu().numpy()

    
    def evaluate_pairs(self, pairs_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate model on pairs.
        
        Returns:
            (labels, scores, age_gaps)
        """
        labels = []
        scores = []
        age_gaps = []
        
        with open(pairs_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in tqdm(lines, desc="Evaluating pairs", unit="pair"):
            try:
                path1, path2, label, age_gap = line.strip().split(',')
                label = int(label)
                age_gap = int(age_gap)
            except ValueError:
                continue
            
            emb1 = self.extract_embedding(path1)
            emb2 = self.extract_embedding(path2)
            
            if emb1 is None or emb2 is None:
                continue
            
            # Cosine similarity
            score = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-12))
            
            labels.append(label)
            scores.append(score)
            age_gaps.append(age_gap)
        
        return np.array(labels), np.array(scores), np.array(age_gaps)
    
    def compute_metrics(self,
                       labels: np.ndarray,
                       scores: np.ndarray,
                       age_gaps: np.ndarray) -> EvaluationMetrics:
        """Compute comprehensive evaluation metrics."""
        # ROC curve
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        # Optimal threshold
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        # Accuracy by age gap (positive pairs only)
        accuracy_by_gap = self._compute_accuracy_by_age_gap(
            labels, scores, age_gaps, optimal_threshold
        )
        
        # Negative score analysis by age gap
        neg_score_by_gap = self._compute_neg_score_by_age_gap(
            labels, scores, age_gaps
        )
        
        # TAR at FAR
        tar_at_far = {}
        for far_threshold in [0.01, 0.001, 0.0001]:
            tar, threshold = self._compute_tar_at_far(labels, scores, far_threshold)
            tar_at_far[far_threshold] = (tar, threshold)
        
        return EvaluationMetrics(
            auc=roc_auc,
            optimal_threshold=optimal_threshold,
            fpr=fpr,
            tpr=tpr,
            thresholds=thresholds,
            accuracy_by_age_gap=accuracy_by_gap,
            neg_score_by_age_gap=neg_score_by_gap,
            tar_at_far=tar_at_far
        )
    
    def _compute_accuracy_by_age_gap(self,
                                    labels: np.ndarray,
                                    scores: np.ndarray,
                                    age_gaps: np.ndarray,
                                    threshold: float) -> Dict[str, float]:
        """Compute accuracy for positive pairs in each age gap bin."""
        pos_mask = labels == 1
        pos_scores = scores[pos_mask]
        pos_gaps = age_gaps[pos_mask]
        
        accuracy_by_gap = {}
        
        for low, high in self.config.age_gap_bins:
            gap_mask = (pos_gaps >= low) & (pos_gaps <= high)
            
            if gap_mask.sum() == 0:
                accuracy_by_gap[f"{low}-{high}"] = np.nan
            else:
                accuracy = (pos_scores[gap_mask] >= threshold).mean()
                accuracy_by_gap[f"{low}-{high}"] = float(accuracy)
        
        return accuracy_by_gap
    
    def _compute_neg_score_by_age_gap(self,
                                     labels: np.ndarray,
                                     scores: np.ndarray,
                                     age_gaps: np.ndarray) -> Dict[str, float]:
        """Compute mean negative score for each age gap bin."""
        neg_mask = labels == 0
        neg_scores = scores[neg_mask]
        neg_gaps = age_gaps[neg_mask]
        
        neg_score_by_gap = {}
        
        for low, high in self.config.age_gap_bins:
            gap_mask = (neg_gaps >= low) & (neg_gaps <= high)
            
            if gap_mask.sum() == 0:
                neg_score_by_gap[f"{low}-{high}"] = np.nan
            else:
                mean_score = neg_scores[gap_mask].mean()
                neg_score_by_gap[f"{low}-{high}"] = float(mean_score)
        
        return neg_score_by_gap
    
    @staticmethod
    def _compute_tar_at_far(labels: np.ndarray,
                           scores: np.ndarray,
                           far_threshold: float) -> Tuple[float, float]:
        """Compute True Accept Rate at given False Accept Rate."""
        fpr, tpr, thresholds = roc_curve(labels, scores)
        
        valid_indices = np.where(fpr <= far_threshold)[0]
        
        if len(valid_indices) == 0:
            return 0.0, None
        
        idx = valid_indices[-1]
        return float(tpr[idx]), float(thresholds[idx])


# ============================================================================
# Visualization and Reporting
# ============================================================================

class ReportGenerator:
    """Generate evaluation reports and visualizations."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_model_report(self,
                            model_name: str,
                            labels: np.ndarray,
                            scores: np.ndarray,
                            age_gaps: np.ndarray,
                            metrics: EvaluationMetrics,
                            age_gap_bins: List[Tuple[int, int]]) -> None:
        """Generate complete report for a single model."""
        model_dir = self.output_dir / model_name.replace(' ', '_')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # ROC curve
        self._plot_roc_curve(model_name, metrics, model_dir)
        
        # Score distribution
        self._plot_score_distribution(model_name, labels, scores, metrics.optimal_threshold, model_dir)
        
        # Age gap robustness
        self._plot_age_gap_robustness(model_name, metrics.accuracy_by_age_gap, age_gap_bins, model_dir)
        
        # Summary CSV
        self._save_summary_csv(model_name, metrics, age_gap_bins, model_dir)
        self._save_full_report_csv(
            model_name,
            labels,
            scores,
            age_gaps,
            metrics,
            age_gap_bins,
            model_dir
        )


        print(f"📊 Generated report for {model_name}")
        print(f"   AUC: {metrics.auc:.4f}")
        print(f"   Optimal Threshold: {metrics.optimal_threshold:.4f}")
        for far, (tar, _) in metrics.tar_at_far.items():
            print(f"   TAR @ FAR={far}: {tar*100:.2f}%")
    
    def _plot_roc_curve(self, model_name: str, metrics: EvaluationMetrics, output_dir: Path) -> None:
        """Plot ROC curve."""
        plt.figure(figsize=(8, 6))
        plt.plot(metrics.fpr, metrics.tpr, linewidth=2, label=f'AUC = {metrics.auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {model_name}', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{model_name}_roc.png", dpi=150)
        plt.close()
    
    def _plot_score_distribution(self,
                                model_name: str,
                                labels: np.ndarray,
                                scores: np.ndarray,
                                threshold: float,
                                output_dir: Path) -> None:
        """Plot score distribution for positive and negative pairs."""
        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]
        
        plt.figure(figsize=(10, 6))
        plt.hist(neg_scores, bins=50, alpha=0.6, label='Negative Pairs', density=True, color='red')
        plt.hist(pos_scores, bins=50, alpha=0.6, label='Positive Pairs', density=True, color='green')
        plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.3f}')
        plt.xlabel('Similarity Score', fontsize=12)
        plt.ylabel('Density', fontsize=12)

        plt.title(f'Score Distribution - {model_name}', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{model_name}_scores.png", dpi=150)
        plt.close()
    
    def _plot_age_gap_robustness(self,
                                model_name: str,
                                accuracy_by_gap: Dict[str, float],
                                age_gap_bins: List[Tuple[int, int]],
                                output_dir: Path) -> None:
        """Plot accuracy by age gap for positive pairs."""
        labels = [f"{low}-{high}" for low, high in age_gap_bins]
        accuracies = [accuracy_by_gap.get(label, np.nan) for label in labels]
        
        plt.figure(figsize=(10, 6))
        x_pos = np.arange(len(labels))
        colors = ['green' if not np.isnan(acc) else 'gray' for acc in accuracies]
        
        bars = plt.bar(x_pos, np.nan_to_num(accuracies), color=colors, alpha=0.7)
        plt.xticks(x_pos, labels, fontsize=11)
        plt.ylim(0, 1.0)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xlabel('Age Gap (years)', fontsize=12)
        plt.title(f'Robustness by Age Gap (Positive Pairs) - {model_name}', fontsize=14)
        
        # Add value labels on bars
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            if not np.isnan(acc):
                plt.text(bar.get_x() + bar.get_width()/2, acc + 0.02,
                        f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_dir / f"{model_name}_robustness.png", dpi=150)
        plt.close()
    
    def _save_summary_csv(self,
                         model_name: str,
                         metrics: EvaluationMetrics,
                         age_gap_bins: List[Tuple[int, int]],
                         output_dir: Path) -> None:
        """Save summary statistics to CSV."""
        csv_path = output_dir / f"{model_name}_summary.csv"
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Overall metrics
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Model Name', model_name])
            writer.writerow(['AUC', f'{metrics.auc:.4f}'])
            writer.writerow(['Optimal Threshold', f'{metrics.optimal_threshold:.4f}'])
            writer.writerow([])
            
            # TAR @ FAR
            writer.writerow(['FAR', 'TAR', 'Threshold'])
            for far, (tar, thresh) in sorted(metrics.tar_at_far.items()):
                writer.writerow([f'{far:.4f}', f'{tar:.4f}', f'{thresh:.4f}' if thresh else 'N/A'])
            writer.writerow([])
            
            # Positive pair accuracy by age gap
            writer.writerow(['Age Gap', 'Accuracy (Positive Pairs)'])
            for low, high in age_gap_bins:
                label = f"{low}-{high}"
                acc = metrics.accuracy_by_age_gap.get(label, np.nan)
                writer.writerow([label, f'{acc:.4f}' if not np.isnan(acc) else 'N/A'])
            writer.writerow([])
            
            # Negative pair scores by age gap
            writer.writerow(['Age Gap', 'Mean Negative Score'])
            for low, high in age_gap_bins:
                label = f"{low}-{high}"
                score = metrics.neg_score_by_age_gap.get(label, np.nan)
                writer.writerow([label, f'{score:.4f}' if not np.isnan(score) else 'N/A'])
    
    def generate_comparison_report(self,
                                  results: Dict[str, EvaluationMetrics]) -> None:
        """Generate comparison report for multiple models."""
        if not results:
            return
        
        # ROC comparison
        plt.figure(figsize=(10, 8))
        for model_name, metrics in results.items():
            plt.plot(metrics.fpr, metrics.tpr, linewidth=2,
                    label=f'{model_name} (AUC={metrics.auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve Comparison', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "roc_comparison.png", dpi=150)
        plt.close()
        
        # Comparison table
        self._save_comparison_csv(results)
        
        print(f"\n📊 Comparison report saved to {self.output_dir}")
    
    def _save_comparison_csv(self, results: Dict[str, EvaluationMetrics]) -> None:
        """Save comparison table to CSV."""
        csv_path = self.output_dir / "model_comparison.csv"
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['Model', 'AUC', 'Optimal Threshold', 
                           'TAR@FAR=0.01', 'TAR@FAR=0.001', 'TAR@FAR=0.0001'])
            
            # Sort by AUC descending
            sorted_results = sorted(results.items(), key=lambda x: x[1].auc, reverse=True)
            
            for model_name, metrics in sorted_results:
                row = [
                    model_name,
                    f'{metrics.auc:.4f}',
                    f'{metrics.optimal_threshold:.4f}'
                ]
                
                for far in [0.01, 0.001, 0.0001]:
                    tar, _ = metrics.tar_at_far.get(far, (np.nan, None))
                    row.append(f'{tar:.4f}' if not np.isnan(tar) else 'N/A')
                
                writer.writerow(row)

    def _save_full_report_csv(self,
                          model_name: str,
                          labels: np.ndarray,
                          scores: np.ndarray,
                          age_gaps: np.ndarray,
                          metrics: EvaluationMetrics,
                          age_gap_bins: List[Tuple[int, int]],
                          output_dir: Path) -> None:
    
        csv_path = output_dir / f"{model_name}_FULL_REPORT.csv"

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # GENERAL
            writer.writerow(["=== GENERAL INFORMATION ==="])
            writer.writerow(["Model", model_name])
            writer.writerow(["Total pairs", len(labels)])
            writer.writerow(["Positive pairs", int((labels == 1).sum())])
            writer.writerow(["Negative pairs", int((labels == 0).sum())])
            writer.writerow([])

            # CORE METRICS
            writer.writerow(["=== CORE METRICS ==="])
            writer.writerow(["AUC", metrics.auc])
            writer.writerow(["Optimal threshold", metrics.optimal_threshold])
            writer.writerow([])

            # TAR @ FAR
            writer.writerow(["=== TAR @ FAR ==="])
            writer.writerow(["FAR", "TAR", "Threshold"])
            for far, (tar, th) in metrics.tar_at_far.items():
                writer.writerow([far, tar, th])
            writer.writerow([])

            # ROBUSTNESS (POSITIVE)
            writer.writerow(["=== TEMPORAL ROBUSTNESS (POS PAIRS) ==="])
            writer.writerow(["Age gap", "Accuracy"])
            for low, high in age_gap_bins:
                label = f"{low}-{high}"
                acc = metrics.accuracy_by_age_gap.get(label, np.nan)
                writer.writerow([label, acc])
            writer.writerow([])

            # NEGATIVE BEHAVIOR
            writer.writerow(["=== NEGATIVE SCORE BY AGE GAP ==="])
            writer.writerow(["Age gap", "Mean score"])
            for low, high in age_gap_bins:
                label = f"{low}-{high}"
                val = metrics.neg_score_by_age_gap.get(label, np.nan)
                writer.writerow([label, val])

        print(f"✅ FULL REPORT saved to: {csv_path}")

# ============================================================================
# t-SNE Visualization
# ============================================================================

class TSNEVisualizer:
    """Generate t-SNE visualizations for embedding space analysis."""
    
    def __init__(self, evaluator: FaceEvaluator):
        self.evaluator = evaluator
    
    def visualize(self,
                 root_dir: str,
                 output_path: str,
                 num_identities: int = 8,
                 images_per_identity: int = 12) -> None:
        """
        Create t-SNE visualization of embeddings.
        
        Args:
            root_dir: Root directory containing identity folders
            output_path: Path to save visualization
            num_identities: Number of identities to visualize
            images_per_identity: Max images per identity
        """
        print("🎨 Generating t-SNE visualization...")
        
        # Collect embeddings
        embeddings, labels, paths = self._collect_embeddings(
            root_dir, num_identities, images_per_identity
        )
        
        if len(embeddings) < 10:
            print("⚠️  Insufficient embeddings for t-SNE (need at least 10)")
            return
        
        # Compute t-SNE
        perplexity = min(30, len(embeddings) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Save CSV
        self._save_tsne_csv(paths, embeddings_2d, labels, output_path)
        
        # Plot
        self._plot_tsne(embeddings_2d, labels, output_path)
        
        print(f"✅ t-SNE visualization saved to {output_path}")
    
    def _collect_embeddings(self,
                           root_dir: str,
                           num_identities: int,
                           images_per_identity: int) -> Tuple[np.ndarray, list, list]:
        """Collect embeddings for visualization."""
        root = Path(root_dir)
        identity_dirs = [d for d in root.iterdir() if d.is_dir()]
        
        if len(identity_dirs) == 0:
            return np.array([]), [], []
        
        # Sample identities
        selected_identities = random.sample(
            identity_dirs,
            min(len(identity_dirs), num_identities)
        )
        
        embeddings = []
        labels = []
        paths = []
        
        for identity_idx, identity_dir in enumerate(selected_identities):
            image_files = [f for f in identity_dir.iterdir() if is_image_file(f.name)]
            
            if len(image_files) == 0:
                continue
            
            # Sample images
            selected_images = image_files[:images_per_identity] if len(image_files) <= images_per_identity else random.sample(image_files, images_per_identity)
            
            for img_path in selected_images:
                emb = self.evaluator.extract_embedding(str(img_path))
                if emb is not None:
                    embeddings.append(emb)
                    labels.append(identity_idx)
                    paths.append(str(img_path))
        
        return np.stack(embeddings) if embeddings else np.array([]), labels, paths
    
    def _save_tsne_csv(self,
                      paths: list,
                      embeddings_2d: np.ndarray,
                      labels: list,
                      output_path: str) -> None:
        """Save t-SNE results to CSV."""
        csv_path = Path(output_path).with_suffix('.csv')
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['path', 'x', 'y', 'label'])
            
            for path, (x, y), label in zip(paths, embeddings_2d, labels):
                writer.writerow([path, f'{x:.6f}', f'{y:.6f}', label])
    
    def _plot_tsne(self,
                  embeddings_2d: np.ndarray,
                  labels: list,
                  output_path: str) -> None:
        """Plot t-SNE visualization."""
        plt.figure(figsize=(12, 10))
        
        unique_labels = sorted(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for label_idx, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[label_idx]],
                label=f'Identity {label}',
                alpha=0.6,
                s=50
            )
        
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.title('t-SNE Visualization of Face Embeddings', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()


# ============================================================================
# Main Evaluation Runner
# ============================================================================

class EvaluationRunner:
    """Orchestrate the complete evaluation pipeline."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
    
    def run(self, model_configs: List[ModelConfig]) -> Dict[str, EvaluationMetrics]:
        """Run evaluation for all models."""
        # Set seed
        set_seed(self.config.seed)
        
        # Generate pairs
        pair_generator = PairGenerator(self.config)
        pairs_file = pair_generator.generate_and_save()
        
        # Initialize report generator
        report_gen = ReportGenerator(self.config.output_dir)
        
        # Evaluate each model
        results = {}
        
        for model_config in model_configs:
            print(f"\n{'='*70}")
            print(f"Evaluating: {model_config.name}")
            print(f"{'='*70}")
            
            try:
                model = model_config.model_init()
            except Exception as e:
                print(f"❌ Failed to initialize model: {e}")
                continue
            
            # Load checkpoint if provided
            if model_config.checkpoint_path:
                success, missing, unexpected = ModelLoader.load_checkpoint(
                    model, model_config.checkpoint_path, self.device
                )
                if not success:
                    print("⚠️  Continuing with current model weights")
            
            # Create evaluator
            evaluator = FaceEvaluator(model, self.device, self.config)
            
            # Evaluate on pairs
            labels, scores, age_gaps = evaluator.evaluate_pairs(pairs_file)
            
            if len(labels) == 0:
                print("❌ No valid predictions. Skipping model.")
                continue
            
            # Compute metrics
            metrics = evaluator.compute_metrics(labels, scores, age_gaps)
            # ===========================
            # EXTRA EVALUATION
            # ===========================
            import extra_eval

            eer = extra_eval.calc_eer(labels, scores)
            print("EER:", eer)

            per_gap_auc = extra_eval.per_gap_auc(labels, scores, age_gaps)
            print("AUC per gap:", per_gap_auc)

            tar_gap = extra_eval.per_gap_tar_at_far(labels, scores, age_gaps, [1e-3])
            print("TAR@FAR=1e-3 per gap:", tar_gap)

            neg_gap = extra_eval.negative_score_by_gap(labels, scores, age_gaps)
            print("Negative score by gap:", neg_gap)

            auc_ci = extra_eval.bootstrap_ci(labels, scores, lambda y, s: extra_eval.calc_auc(y, s))
            print("AUC 95% CI:", auc_ci)

            calibrated_scores = extra_eval.platt_scaling(scores, labels)

            # false_pos, false_neg, identity_errors = extra_eval.collect_fp_fn(
            #     labels, scores, pairs, optimal_threshold
            # )

            # Generate report
            report_gen.generate_model_report(
                model_config.name,
                labels,
                scores,
                age_gaps,
                metrics,
                self.config.age_gap_bins
            )
            
            # Store results
            results[model_config.name] = metrics
            
            # Optional: Generate t-SNE visualization
            try:
                tsne_viz = TSNEVisualizer(evaluator)
                tsne_path = os.path.join(
                    self.config.output_dir,
                    f"{model_config.name.replace(' ', '_')}_tsne.png"
                )
                tsne_viz.visualize(
                    self.config.root_dir,
                    tsne_path,
                    num_identities=8,
                    images_per_identity=12
                )
            except Exception as e:
                print(f"⚠️  t-SNE visualization failed: {e}")
        
        # Generate comparison report
        if results:
            report_gen.generate_comparison_report(results)
            
            # Print summary
            self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict[str, EvaluationMetrics]) -> None:
        """Print evaluation summary."""
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        
        # Sort by AUC
        sorted_results = sorted(results.items(), key=lambda x: x[1].auc, reverse=True)
        
        for rank, (model_name, metrics) in enumerate(sorted_results, 1):
            print(f"\n#{rank} {model_name}")
            print(f"   AUC: {metrics.auc:.4f}")
            print(f"   Optimal Threshold: {metrics.optimal_threshold:.4f}")
            
            for far, (tar, _) in sorted(metrics.tar_at_far.items()):
                print(f"   TAR @ FAR={far}: {tar*100:.2f}%")
            
            print("   Age Gap Robustness (Positive Pairs):")
            for gap_label, acc in metrics.accuracy_by_age_gap.items():
                if not np.isnan(acc):
                    print(f"      {gap_label} years: {acc*100:.1f}%")
        
        print("\n" + "="*70)
        best_model = sorted_results[0][0]
        best_auc = sorted_results[0][1].auc
        print(f"🏆 Best Model: {best_model} (AUC = {best_auc:.4f})")
        print("="*70)


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Comprehensive evaluation of face recognition models on age-invariant datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--root_dir', type=str, required=True,
                       help='Root directory containing identity folders')
    parser.add_argument('--pairs_file', type=str, default='test_pairs.txt',
                       help='Path to pairs file (will be generated if missing)')
    parser.add_argument('--num_pairs', type=int, default=6000,
                       help='Number of test pairs to generate')
    parser.add_argument('--min_imgs', type=int, default=3,
                       help='Minimum images per identity')
    
    # Evaluation arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='eval_reports',
                       help='Output directory for reports')
    parser.add_argument('--use_mtcnn', action='store_true',
                       help='Use MTCNN for face alignment')
    parser.add_argument('--image_size', type=int, default=112,
                       help='Input image size')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for inference')
    
    args = parser.parse_args()
    
    # Create configuration
    config = EvaluationConfig(
        root_dir=args.root_dir,
        pairs_file=args.pairs_file,
        num_pairs=args.num_pairs,
        min_imgs_per_id=args.min_imgs,
        seed=args.seed,
        output_dir=args.output_dir,
        use_mtcnn=args.use_mtcnn,
        image_size=args.image_size,
        batch_size=args.batch_size
    )

    model_configs = [
        ModelConfig(
            name="ArcFace Fine-tuned",
            model_init=lambda: IR_SE50(),
            checkpoint_path="weights/InsightFace_Pytorch%2Bmodel_ir_se50.pth"
        ),
        ModelConfig(
            name="Facenet Fine-tuned",
            model_init= lambda: AdaFaceNet(num_classes= 1000, embedding_size=512),
            checkpoint_path="checkpoints/adaface_best.pth"
        ),
        # ModelConfig(
        #     name="AdaFace Fine-tuned",
        #     model_init=lambda:  SiameseFaceNet(embedding_size=512),
        #     checkpoint_path="old_checkpoints/facenet_sub1.0_ep100.pth"
        # ),
        ModelConfig(
            name="AdaFace Fine-tuned",
            model_init= lambda: AdaFaceOriginalWrapper(architecture='ir_101', device='cuda'),
            checkpoint_path=None
        )
    ]
    
    # Run evaluation
    runner = EvaluationRunner(config)
    results = runner.run(model_configs)
    
    if not results:
        print("\n❌ No models were successfully evaluated")
        sys.exit(1)
    
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
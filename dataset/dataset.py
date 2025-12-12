import pandas as pd
import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict
import torch


class AgeGapDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None, samples_per_identity=10, 
                 negative_strategy='balanced'):
        
        self.root_dir = root_dir
        self.transform = transform
        self.samples_per_identity = samples_per_identity
        self.negative_strategy = negative_strategy
        
        print(f"\n📂 Loading dataset: {csv_path}")
        df = pd.read_csv(csv_path)

        # Build identity dictionary
        self.data_dict = defaultdict(list)
        self.all_images = []

        for _, row in df.iterrows():
            item = {
                'path': row['file_path'], 
                'age': int(row['age']),
                'id': str(row['identity'])
            }
            self.data_dict[item['id']].append(item)
            self.all_images.append(item)

        self.ids = list(self.data_dict.keys())
        self.ids = [id_ for id_ in self.ids if len(self.data_dict[id_]) >= 2]
        
        print(f"✅ Valid identities (≥2 images): {len(self.ids)}")
        print(f"✅ Total images: {len(self.all_images)}")
        self.identity_age_ranges = {}
        all_gaps = []
        
        for person_id in self.ids:
            imgs = self.data_dict[person_id]
            ages = [x['age'] for x in imgs]
            age_range = max(ages) - min(ages)
            self.identity_age_ranges[person_id] = age_range
            
            if age_range > 0:
                all_gaps.append(age_range)
        if len(all_gaps) > 0:
            self.max_age_gap = np.percentile(all_gaps, 100)
            print(f"✅ Age gap normalization (100th percentile): {self.max_age_gap:.1f} years")
            print(f"   Mean per-identity gap: {np.mean(all_gaps):.1f} years")
            print(f"   Median per-identity gap: {np.median(all_gaps):.1f} years")
        else:
            self.max_age_gap = 30.0  # Fallback
            print(f"⚠️  Using fallback normalization: 30 years")
        
        if self.max_age_gap < 5.0:
            print(f"⚠️  Max age gap too small ({self.max_age_gap:.1f}), using 30 years")
            self.max_age_gap = 30.0
        
        print("🔄 Pre-sorting negative candidates by age...")
        self.all_images_by_age = sorted(self.all_images, key=lambda x: x['age'])
        
        print(f"✅ Dataset ready: {len(self)} total samples\n")

    def __len__(self):
        """Return total number of training samples"""
        return len(self.ids) * self.samples_per_identity

    def _load_image(self, relative_path):
        """Load and transform image"""
        full_path = os.path.join(self.root_dir, relative_path)
        try:
            img = Image.open(full_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img
        except Exception as e:
            print(f"❌ Error loading: {full_path} - {e}")
            if self.transform:
                black_img = Image.new('RGB', (112, 112), color='black')
                return self.transform(black_img)
            else:
                return torch.zeros((3, 112, 112))
            
    def _get_positive_pair(self, anchor_id):
        person_imgs = self.data_dict[anchor_id]
        anchor = random.choice(person_imgs)
        
        positive_candidates = [x for x in person_imgs if x['path'] != anchor['path']]
        if not positive_candidates: return None, None, 0
        positive_candidates.sort(key=lambda x: abs(x['age'] - anchor['age']), reverse=True)
        
        if random.random() < 0.7:
            positive = positive_candidates[0] # Hardest
        else:
            positive = random.choice(positive_candidates) # Random
            
        age_gap = abs(anchor['age'] - positive['age'])
        return anchor, positive, age_gap

    def _get_negative_balanced(self, anchor_id, anchor_age):
        """Get negative with balanced difficulty"""
        neg_candidates = [x for x in self.all_images if x['id'] != anchor_id]
        neg_candidates.sort(key=lambda x: abs(x['age'] - anchor_age))
        
        n_candidates = len(neg_candidates)
        hard_end = max(1, int(n_candidates * 0.2))      # Top 20% (closest age)
        medium_end = max(hard_end + 1, int(n_candidates * 0.5))  # Next 30%
        hard_zone = neg_candidates[:hard_end]
        medium_zone = neg_candidates[hard_end:medium_end]
        easy_zone = neg_candidates[medium_end:]

        rand = random.random()
        
        if rand < 0.3 and len(easy_zone) > 0:
            negative = random.choice(easy_zone)
        elif rand < 0.7 and len(medium_zone) > 0:
            negative = random.choice(medium_zone)
        elif len(hard_zone) > 0:
            negative = random.choice(hard_zone)
        else:
            # Fallback to any negative
            negative = random.choice(neg_candidates)
        
        return negative

    def _get_negative_hard(self, anchor_id, anchor_age):
        """Get hard negative (close in age)"""
        neg_candidates = [x for x in self.all_images if x['id'] != anchor_id]
        
        # Sort by age distance
        neg_candidates.sort(key=lambda x: abs(x['age'] - anchor_age))
        
        # Sample from top 20% hardest
        n_hard = max(50, int(len(neg_candidates) * 0.2))
        hard_candidates = neg_candidates[:n_hard]
        
        return random.choice(hard_candidates)

    def __getitem__(self, idx):
        """Get training triplet: (anchor, positive, negative, age_gap)"""
        
        max_retries = 10
        
        for attempt in range(max_retries):
            try:
                # Map linear index to identity
                id_idx = idx % len(self.ids)
                anchor_id = self.ids[id_idx]
                
                # ===== 1. Get Anchor and Positive (Hard) =====
                anchor, positive, age_gap = self._get_positive_pair(anchor_id)
                
                if anchor is None:
                    # Failed to get positive pair, try different identity
                    id_idx = random.randint(0, len(self.ids) - 1)
                    anchor_id = self.ids[id_idx]
                    continue
                
                # ===== 2. Get Negative =====
                if self.negative_strategy == 'balanced':
                    negative = self._get_negative_balanced(anchor_id, anchor['age'])
                else:  # 'hard'
                    negative = self._get_negative_hard(anchor_id, anchor['age'])
                
                # ===== 3. Load Images =====
                anchor_img = self._load_image(anchor['path'])
                pos_img = self._load_image(positive['path'])
                neg_img = self._load_image(negative['path'])
                
                # ===== 4. Normalize Age Gap =====
                # FIXED: Better normalization using 95th percentile
                normalized_gap = min(age_gap / self.max_age_gap, 1.0)
                
                return (
                    anchor_img, 
                    pos_img, 
                    neg_img, 
                    torch.tensor(normalized_gap, dtype=torch.float32),
                    torch.tensor(id_idx, dtype=torch.long)
                )
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"❌ Failed to sample triplet after {max_retries} attempts: {e}")
                    # Return dummy sample
                    return self._get_dummy_sample()
                continue
        
        # Should not reach here, but just in case
        return self._get_dummy_sample()

    def _get_dummy_sample(self):
        """Return dummy sample when all attempts fail"""
        print("⚠️  Returning dummy sample")
        
        if self.transform:
            dummy_pil = Image.new('RGB', (112, 112), color='black')
            dummy = self.transform(dummy_pil)
        else:
            dummy = torch.zeros((3, 112, 112))
        
        return dummy, dummy, dummy, torch.tensor(0.5, dtype=torch.float32)

    def get_statistics(self):
        """Get dataset statistics"""
        print("\n" + "="*60)
        print("📊 DATASET STATISTICS")
        print("="*60)
        
        # Sample 1000 triplets to get statistics
        n_samples = min(1000, len(self))
        gaps = []
        pos_ages = []
        neg_age_diffs = []
        
        print(f"Sampling {n_samples} triplets for statistics...")
        
        for i in range(n_samples):
            try:
                _, _, _, gap = self[i]
                gaps.append(gap.item())
            except:
                continue
        
        if len(gaps) > 0:
            print(f"\nAge Gap Distribution:")
            print(f"  Mean:   {np.mean(gaps):.3f}")
            print(f"  Median: {np.median(gaps):.3f}")
            print(f"  Std:    {np.std(gaps):.3f}")
            print(f"  Min:    {min(gaps):.3f}")
            print(f"  Max:    {max(gaps):.3f}")
            print(f"  25th percentile: {np.percentile(gaps, 25):.3f}")
            print(f"  75th percentile: {np.percentile(gaps, 75):.3f}")
        
        print(f"\nDataset Size:")
        print(f"  Total samples: {len(self)}")
        print(f"  Identities: {len(self.ids)}")
        print(f"  Samples per identity: {self.samples_per_identity}")
        print(f"  Total images: {len(self.all_images)}")
        
        # Identity distribution
        imgs_per_id = [len(self.data_dict[id_]) for id_ in self.ids]
        print(f"\nImages per Identity:")
        print(f"  Mean:   {np.mean(imgs_per_id):.1f}")
        print(f"  Median: {np.median(imgs_per_id):.1f}")
        print(f"  Min:    {min(imgs_per_id)}")
        print(f"  Max:    {max(imgs_per_id)}")
        
        print("="*60 + "\n")

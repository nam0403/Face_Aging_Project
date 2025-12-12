import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def split_cacd_dataset(csv_path='cacd_mapping.csv', 
                       val_ratio=0.15,
                       test_ratio=0.15,
                       random_seed=42):
    """
    Split CACD dataset by identity (not by images!)
    
    Args:
        csv_path: Path to original CSV
        val_ratio: Validation set ratio (default 15%)
        test_ratio: Test set ratio (default 15%)
        random_seed: Random seed for reproducibility
    
    Creates:
        - cacd_train.csv (70% identities)
        - cacd_val.csv (15% identities)
        - cacd_test.csv (15% identities)
    """
    
    print(f"📂 Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"✅ Total images: {len(df)}")
    print(f"✅ Total identities: {df['identity'].nunique()}")
    
    # Group by identity
    identity_groups = df.groupby('identity').size()
    
    # Filter identities with at least 2 images (needed for positive pairs)
    valid_identities = identity_groups[identity_groups >= 2].index.tolist()
    
    print(f"✅ Valid identities (≥2 images): {len(valid_identities)}")
    
    # Calculate split sizes
    n_identities = len(valid_identities)
    n_test = int(n_identities * test_ratio)
    n_val = int(n_identities * val_ratio)
    n_train = n_identities - n_test - n_val
    
    print(f"\n📊 Split Plan:")
    print(f"   Train: {n_train} identities ({100*(1-val_ratio-test_ratio):.1f}%)")
    print(f"   Val:   {n_val} identities ({100*val_ratio:.1f}%)")
    print(f"   Test:  {n_test} identities ({100*test_ratio:.1f}%)")
    
    # Split identities (not images!)
    np.random.seed(random_seed)
    
    # First split: train+val vs test
    train_val_ids, test_ids = train_test_split(
        valid_identities,
        test_size=test_ratio,
        random_state=random_seed
    )
    
    # Second split: train vs val
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_ratio/(1-test_ratio),  # Adjust ratio
        random_state=random_seed
    )
    
    # Create dataframes
    train_df = df[df['identity'].isin(train_ids)]
    val_df = df[df['identity'].isin(val_ids)]
    test_df = df[df['identity'].isin(test_ids)]
    
    print(f"\n✅ Split Complete:")
    print(f"   Train: {len(train_df)} images from {len(train_ids)} identities")
    print(f"   Val:   {len(val_df)} images from {len(val_ids)} identities")
    print(f"   Test:  {len(test_df)} images from {len(test_ids)} identities")
    
    # Calculate age statistics per split
    def get_age_stats(split_df, split_name):
        print(f"\n📊 {split_name} Age Statistics:")
        print(f"   Age range: {split_df['age'].min()}-{split_df['age'].max()} years")
        print(f"   Mean age: {split_df['age'].mean():.1f} years")
        
        # Per-identity age gaps
        gaps = []
        for identity in split_df['identity'].unique():
            ages = split_df[split_df['identity'] == identity]['age'].values
            if len(ages) >= 2:
                gaps.append(ages.max() - ages.min())
        
        if gaps:
            print(f"   Mean age gap per identity: {np.mean(gaps):.1f} years")
            print(f"   Max age gap per identity: {max(gaps):.1f} years")
    
    get_age_stats(train_df, "Train")
    get_age_stats(val_df, "Validation")
    get_age_stats(test_df, "Test")
    
    # Save splits
    train_path = 'cacd_train.csv'
    val_path = 'cacd_val.csv'
    test_path = 'cacd_test.csv'
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n💾 Saved splits:")
    print(f"   {train_path}")
    print(f"   {val_path}")
    print(f"   {test_path}")
    
    # Verify no overlap
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)
    
    assert len(train_set & val_set) == 0, "Train/Val overlap detected!"
    assert len(train_set & test_set) == 0, "Train/Test overlap detected!"
    assert len(val_set & test_set) == 0, "Val/Test overlap detected!"
    
    print(f"\n✅ No identity overlap between splits - verified!")
    
    return train_path, val_path, test_path


def verify_split_quality(train_csv, val_csv, test_csv):
    """Verify the quality of the split"""
    
    print("\n" + "="*60)
    print("🔍 VERIFYING SPLIT QUALITY")
    print("="*60)
    
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    
    # Check identity overlap
    train_ids = set(train_df['identity'].unique())
    val_ids = set(val_df['identity'].unique())
    test_ids = set(test_df['identity'].unique())
    
    print(f"\n✅ Identity counts:")
    print(f"   Train: {len(train_ids)}")
    print(f"   Val:   {len(val_ids)}")
    print(f"   Test:  {len(test_ids)}")
    
    # Check age distribution similarity
    print(f"\n✅ Age distributions:")
    print(f"   Train mean: {train_df['age'].mean():.1f} ± {train_df['age'].std():.1f}")
    print(f"   Val mean:   {val_df['age'].mean():.1f} ± {val_df['age'].std():.1f}")
    print(f"   Test mean:  {test_df['age'].mean():.1f} ± {test_df['age'].std():.1f}")
    
    # Check images per identity
    print(f"\n✅ Images per identity:")
    print(f"   Train: {len(train_df)/len(train_ids):.1f} avg")
    print(f"   Val:   {len(val_df)/len(val_ids):.1f} avg")
    print(f"   Test:  {len(test_df)/len(test_ids):.1f} avg")
    
    print("\n" + "="*60)
    print("✅ Split verification complete!")
    print("="*60)


if __name__ == "__main__":
    print("🚀 Splitting CACD dataset by identity...\n")
    
    # Create splits
    train_csv, val_csv, test_csv = split_cacd_dataset(
        csv_path='cacd_mapping.csv',
        val_ratio=0.15,   # 15% for validation
        test_ratio=0.15,  # 15% for test
        random_seed=42
    )
    
    # Verify quality
    verify_split_quality(train_csv, val_csv, test_csv)
    
    print("\n✅ Done! Now update your training and evaluation scripts:")
    print("   - Training: use 'cacd_train.csv'")
    print("   - Validation during training: use 'cacd_val.csv'")
    print("   - Final evaluation: use 'cacd_test.csv'")
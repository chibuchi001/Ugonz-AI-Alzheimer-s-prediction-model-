"""
Alzheimer's Disease Prediction Pipeline
========================================
A comprehensive multimodal ML pipeline for AD prediction using:
- Clinical data
- Neuroimaging features
- Genetic data (with APOE and PRS support)

Features:
- XGBoost and PyTorch neural network models
- SHAP-based feature importance
- Ensemble methods
- Robust evaluation metrics
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    f1_score, roc_auc_score, balanced_accuracy_score, 
    confusion_matrix, classification_report, precision_recall_curve,
    average_precision_score
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import warnings
import pickle
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import json

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the entire pipeline"""
    # Directories
    data_dir: str = 'alzheimers_hackathon/data'
    model_dir: str = 'alzheimers_hackathon/models'
    output_dir: str = 'alzheimers_hackathon/outputs'
    
    # Data splits
    test_size: float = 0.15
    val_size: float = 0.15
    random_seed: int = 42
    
    # Genetic features
    n_genetic_features: int = 100
    use_prs: bool = True
    use_apoe: bool = True
    
    # Model parameters
    xgb_params: dict = None
    nn_hidden_dims: List[int] = None
    nn_epochs: int = 100
    nn_batch_size: int = 32
    nn_learning_rate: float = 0.001
    
    # Ensemble
    ensemble_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = {
                'objective': 'binary:logistic',
                'eval_metric': ['logloss', 'auc'],
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'scale_pos_weight': 1,
                'random_state': self.random_seed
            }
        if self.nn_hidden_dims is None:
            self.nn_hidden_dims = [256, 128, 64]
        if self.ensemble_weights is None:
            self.ensemble_weights = {'xgboost': 0.5, 'neural_net': 0.5}


def setup_directories(config: PipelineConfig):
    """Create necessary directories"""
    for directory in [config.data_dir, config.model_dir, config.output_dir]:
        os.makedirs(directory, exist_ok=True)
    print("✓ Directory structure created")


# ==============================================================================
# DATA LOADING WITH FIXES
# ==============================================================================

class MultimodalDataLoader:
    """Handles loading of all data modalities with proper alignment"""
    
    # Known AD-associated SNPs (rsIDs)
    AD_ASSOCIATED_SNPS = [
        'rs429358',   # APOE ε4 variant
        'rs7412',     # APOE ε2 variant
        'rs6656401',  # CR1
        'rs6733839',  # BIN1
        'rs35349669', # INPP5D
        'rs190982',   # MEF2C
        'rs2718058',  # NME8
        'rs10948363', # CD2AP
        'rs11771145', # EPHA1
        'rs9331896',  # CLU
        'rs983392',   # MS4A6A
        'rs10792832', # PICALM
        'rs4147929',  # ABCA7
        'rs3865444',  # CD33
        'rs28834970', # PTK2B
        'rs9271192',  # HLA-DRB5
        'rs10838725', # CELF1
        'rs17125944', # FERMT2
        'rs10498633', # SLC24A4
        'rs8093731',  # DSG2
    ]
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.sample_id_column = None
        self.clinical_df = None
        self.imaging_dict = None
        self.genotypes = None
        self.genetic_sample_ids = None
        self.snp_ids = None
        
    def load_clinical_data(self, filepath: str, sample_id_col: str = 'IID') -> pd.DataFrame:
        """
        Load clinical data with proper handling
        
        Args:
            filepath: Path to clinical data file (TSV/CSV)
            sample_id_col: Column name containing sample IDs
        """
        # Detect separator
        with open(filepath, 'r') as f:
            first_line = f.readline()
        sep = '\t' if '\t' in first_line else ','
        
        df = pd.read_csv(filepath, sep=sep)
        
        # Store sample ID column
        self.sample_id_column = sample_id_col
        if sample_id_col not in df.columns:
            # Try common alternatives
            for alt in ['ID', 'SampleID', 'Subject_ID', 'PTID', 'RID']:
                if alt in df.columns:
                    self.sample_id_column = alt
                    break
            else:
                # Use index as ID
                df['IID'] = df.index.astype(str)
                self.sample_id_column = 'IID'
        
        # Impute numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != self.sample_id_column]
        
        if len(numeric_cols) > 0:
            # Use KNN imputer for better imputation
            imputer = KNNImputer(n_neighbors=5)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        self.clinical_df = df
        print(f"✓ Loaded clinical data: {df.shape[0]} samples, {df.shape[1]} features")
        return df
    
    def load_npz_data(self, filepath: str) -> Dict[str, np.ndarray]:
        """Load imaging/array data from NPZ file"""
        data = np.load(filepath, allow_pickle=True)
        arrays_dict = {}
        
        for key in data.files:
            arr = data[key]
            # Handle object arrays
            if arr.dtype == object:
                arr = np.array(arr.tolist())
            arrays_dict[key] = arr
            print(f"  Loaded {key}: shape={arr.shape}, dtype={arr.dtype}")
        
        self.imaging_dict = arrays_dict
        print(f"✓ Loaded NPZ data with {len(arrays_dict)} arrays")
        return arrays_dict
    
    def load_bed_genetic_data(self, bedfile_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load genetic data from PLINK BED format with missing value handling
        
        Args:
            bedfile_path: Path to .bed file (without extension)
        """
        from bed_reader import open_bed
        
        # Handle path with or without extension
        if bedfile_path.endswith('.bed'):
            bedfile_path = bedfile_path[:-4]
        
        bed = open_bed(bedfile_path + '.bed')
        genotypes = bed.read()
        sample_ids = np.array(bed.iid)
        snp_ids = np.array(bed.sid)
        
        # FIX: Handle missing genotype calls
        # In PLINK BED format, missing values are typically encoded as -1 or NaN
        missing_mask = np.isnan(genotypes) | (genotypes < 0) | (genotypes > 2)
        missing_rate = missing_mask.sum() / genotypes.size
        print(f"  Missing genotype rate: {missing_rate:.4%}")
        
        if missing_rate > 0:
            # Impute missing values with column-wise mode (most common genotype)
            for col_idx in range(genotypes.shape[1]):
                col = genotypes[:, col_idx]
                col_missing = missing_mask[:, col_idx]
                if col_missing.any():
                    valid_vals = col[~col_missing]
                    if len(valid_vals) > 0:
                        # Use mode (most common value)
                        mode_val = np.round(np.median(valid_vals))
                        genotypes[col_missing, col_idx] = mode_val
                    else:
                        genotypes[col_missing, col_idx] = 1  # Heterozygous as default
        
        # Ensure values are in valid range [0, 1, 2]
        genotypes = np.clip(genotypes, 0, 2)
        
        del bed  # Free memory
        
        self.genotypes = genotypes
        self.genetic_sample_ids = sample_ids
        self.snp_ids = snp_ids
        
        print(f"✓ Loaded genetic data: {genotypes.shape[0]} samples, {genotypes.shape[1]} SNPs")
        return genotypes, sample_ids, snp_ids
    
    def extract_apoe_status(self) -> Optional[np.ndarray]:
        """
        Extract APOE ε4 carrier status from genetic data
        
        APOE genotype is determined by rs429358 and rs7412:
        - ε2: rs429358=T/T, rs7412=T/T
        - ε3: rs429358=T/T, rs7412=C/C
        - ε4: rs429358=C/C, rs7412=C/C
        
        Returns array with APOE ε4 allele count (0, 1, or 2)
        """
        if self.snp_ids is None or self.genotypes is None:
            print("  Warning: Genetic data not loaded, cannot extract APOE")
            return None
        
        snp_list = list(self.snp_ids)
        
        # Look for APOE SNPs
        apoe_e4_snp = 'rs429358'
        
        if apoe_e4_snp in snp_list:
            idx = snp_list.index(apoe_e4_snp)
            apoe_status = self.genotypes[:, idx].astype(int)
            print(f"✓ Extracted APOE ε4 status from {apoe_e4_snp}")
            return apoe_status
        else:
            # Try to find any APOE-related SNPs
            apoe_snps = [s for s in snp_list if 'APOE' in s.upper() or s in ['rs429358', 'rs7412']]
            if apoe_snps:
                idx = snp_list.index(apoe_snps[0])
                apoe_status = self.genotypes[:, idx].astype(int)
                print(f"✓ Extracted APOE status from {apoe_snps[0]}")
                return apoe_status
            
            print("  Warning: APOE SNPs not found in genetic data")
            return None
    
    def calculate_polygenic_risk_score(self, weights_file: Optional[str] = None) -> np.ndarray:
        """
        Calculate Polygenic Risk Score (PRS) for Alzheimer's disease
        
        Args:
            weights_file: Optional path to file with SNP weights
                         Format: SNP_ID, Weight (one per line)
        
        Returns:
            Array of PRS values for each sample
        """
        if self.genotypes is None or self.snp_ids is None:
            raise ValueError("Genetic data must be loaded first")
        
        snp_list = list(self.snp_ids)
        
        # Load weights if provided
        if weights_file and os.path.exists(weights_file):
            weights_df = pd.read_csv(weights_file)
            snp_weights = dict(zip(weights_df.iloc[:, 0], weights_df.iloc[:, 1]))
        else:
            # Use default weights based on AD GWAS studies
            # These are example effect sizes (log odds ratios)
            snp_weights = {
                'rs429358': 1.2,    # APOE ε4 - strongest effect
                'rs7412': -0.5,     # APOE ε2 - protective
                'rs6656401': 0.15,  # CR1
                'rs6733839': 0.18,  # BIN1
                'rs35349669': 0.08, # INPP5D
                'rs190982': 0.07,   # MEF2C
                'rs10948363': 0.10, # CD2AP
                'rs9331896': 0.14,  # CLU
                'rs10792832': 0.12, # PICALM
                'rs4147929': 0.15,  # ABCA7
            }
        
        # Calculate PRS
        prs = np.zeros(self.genotypes.shape[0])
        snps_used = 0
        
        for snp, weight in snp_weights.items():
            if snp in snp_list:
                idx = snp_list.index(snp)
                prs += weight * self.genotypes[:, idx]
                snps_used += 1
        
        if snps_used == 0:
            print("  Warning: No weighted SNPs found, using top variance SNPs")
            # Fall back to using SNPs with highest variance
            variances = np.var(self.genotypes, axis=0)
            top_snp_indices = np.argsort(variances)[-20:]
            for idx in top_snp_indices:
                prs += self.genotypes[:, idx] / 20
            snps_used = 20
        
        # Standardize PRS
        prs = (prs - prs.mean()) / (prs.std() + 1e-8)
        
        print(f"✓ Calculated PRS using {snps_used} SNPs")
        return prs
    
    def select_ad_associated_snps(self) -> Tuple[np.ndarray, List[str]]:
        """
        Select AD-associated SNPs for feature extraction
        
        Returns:
            Tuple of (genotype matrix for selected SNPs, list of SNP IDs)
        """
        if self.genotypes is None or self.snp_ids is None:
            raise ValueError("Genetic data must be loaded first")
        
        snp_list = list(self.snp_ids)
        selected_indices = []
        selected_snps = []
        
        # First, try to get known AD-associated SNPs
        for snp in self.AD_ASSOCIATED_SNPS:
            if snp in snp_list:
                selected_indices.append(snp_list.index(snp))
                selected_snps.append(snp)
        
        print(f"  Found {len(selected_indices)} known AD-associated SNPs")
        
        # If we don't have enough, add high-variance SNPs
        n_needed = self.config.n_genetic_features - len(selected_indices)
        if n_needed > 0:
            variances = np.var(self.genotypes, axis=0)
            # Exclude already selected
            variances[selected_indices] = -1
            top_var_indices = np.argsort(variances)[-n_needed:]
            
            for idx in top_var_indices:
                if idx not in selected_indices:
                    selected_indices.append(idx)
                    selected_snps.append(self.snp_ids[idx])
        
        selected_genotypes = self.genotypes[:, selected_indices]
        print(f"✓ Selected {len(selected_snps)} genetic features")
        
        return selected_genotypes, selected_snps
    
    def combine_multimodal_data(
        self,
        imaging_sample_ids: Optional[np.ndarray] = None,
        target_column: str = 'AD_status'
    ) -> pd.DataFrame:
        """
        FIX: Combine all modalities with proper sample alignment
        
        Args:
            imaging_sample_ids: Sample IDs for imaging data (if available)
            target_column: Name of target variable column
        
        Returns:
            Combined DataFrame with aligned samples
        """
        if self.clinical_df is None:
            raise ValueError("Clinical data must be loaded first")
        
        combined_df = self.clinical_df.copy()
        clinical_ids = combined_df[self.sample_id_column].astype(str).values
        
        # Track which samples have all modalities
        valid_mask = np.ones(len(clinical_ids), dtype=bool)
        
        # Add imaging features with alignment
        if self.imaging_dict is not None:
            # Find imaging features array
            imaging_key = None
            for key in ['mri_features', 'imaging_features', 'features', 'X']:
                if key in self.imaging_dict:
                    imaging_key = key
                    break
            
            if imaging_key:
                imaging_features = self.imaging_dict[imaging_key]
                
                # Check for sample IDs in imaging data
                imaging_ids_key = None
                for key in ['sample_ids', 'ids', 'IID', 'subjects']:
                    if key in self.imaging_dict:
                        imaging_ids_key = key
                        break
                
                if imaging_ids_key and imaging_sample_ids is None:
                    imaging_sample_ids = self.imaging_dict[imaging_ids_key].astype(str)
                
                if imaging_sample_ids is not None and len(imaging_sample_ids) == imaging_features.shape[0]:
                    # Align by sample IDs
                    imaging_df = pd.DataFrame(
                        imaging_features,
                        columns=[f'imaging_feature_{i}' for i in range(imaging_features.shape[1])]
                    )
                    imaging_df['_merge_id'] = imaging_sample_ids
                    combined_df['_merge_id'] = clinical_ids
                    
                    combined_df = combined_df.merge(
                        imaging_df, 
                        on='_merge_id', 
                        how='inner'
                    )
                    combined_df.drop('_merge_id', axis=1, inplace=True)
                    print(f"  Aligned imaging data: {imaging_features.shape[1]} features")
                else:
                    # Assume same order if shapes match
                    if imaging_features.shape[0] == len(clinical_ids):
                        for i in range(imaging_features.shape[1]):
                            combined_df[f'imaging_feature_{i}'] = imaging_features[:, i]
                        print(f"  Added imaging data (assumed aligned): {imaging_features.shape[1]} features")
                    else:
                        print(f"  Warning: Imaging shape mismatch ({imaging_features.shape[0]} vs {len(clinical_ids)})")
        
        # Add genetic features with alignment
        if self.genotypes is not None:
            genetic_ids = self.genetic_sample_ids.astype(str) if self.genetic_sample_ids is not None else None
            
            # Get selected SNPs and PRS
            selected_geno, selected_snps = self.select_ad_associated_snps()
            
            if genetic_ids is not None:
                # Create genetic DataFrame for merging
                genetic_df = pd.DataFrame(
                    selected_geno,
                    columns=[f'snp_{snp}' for snp in selected_snps]
                )
                genetic_df['_merge_id'] = genetic_ids
                
                # Add PRS
                if self.config.use_prs:
                    prs = self.calculate_polygenic_risk_score()
                    genetic_df['PRS'] = prs
                
                # Add APOE
                if self.config.use_apoe:
                    apoe = self.extract_apoe_status()
                    if apoe is not None:
                        genetic_df['APOE_e4_count'] = apoe
                
                combined_df['_merge_id'] = combined_df[self.sample_id_column].astype(str)
                combined_df = combined_df.merge(
                    genetic_df,
                    on='_merge_id',
                    how='inner'
                )
                combined_df.drop('_merge_id', axis=1, inplace=True)
                print(f"  Aligned genetic data: {selected_geno.shape[1]} SNPs + PRS + APOE")
            else:
                # Assume same order
                if selected_geno.shape[0] == len(combined_df):
                    for i, snp in enumerate(selected_snps):
                        combined_df[f'snp_{snp}'] = selected_geno[:, i]
                    
                    if self.config.use_prs:
                        combined_df['PRS'] = self.calculate_polygenic_risk_score()
                    
                    if self.config.use_apoe:
                        apoe = self.extract_apoe_status()
                        if apoe is not None:
                            combined_df['APOE_e4_count'] = apoe
                    
                    print(f"  Added genetic data (assumed aligned)")
        
        print(f"✓ Combined dataset: {combined_df.shape[0]} samples, {combined_df.shape[1]} features")
        return combined_df


# ==============================================================================
# PREPROCESSING
# ==============================================================================

class Preprocessor:
    """Handles feature preprocessing and data splits"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.feature_names = None
        self.label_encoder = LabelEncoder()
        
    def preprocess(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        select_k_best: Optional[int] = None
    ) -> Dict:
        """
        Preprocess features and create train/val/test splits
        
        Args:
            df: Combined DataFrame
            target_column: Name of target column
            feature_columns: Specific features to use (None = all numeric)
            select_k_best: Number of features to select (None = use all)
        
        Returns:
            Dictionary with splits and preprocessing objects
        """
        # Identify feature columns
        if feature_columns is None:
            exclude_cols = [target_column, 'IID', 'ID', 'SampleID', 'Subject_ID']
            feature_columns = [
                col for col in df.columns 
                if col not in exclude_cols and df[col].dtype in [np.float64, np.int64, np.float32, np.int32]
            ]
        
        self.feature_names = feature_columns
        X = df[feature_columns].values
        
        # Handle target
        y = df[target_column].values
        if y.dtype == object or y.dtype.name == 'category':
            y = self.label_encoder.fit_transform(y)
        else:
            y = y.astype(int)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Feature selection if requested
        if select_k_best and select_k_best < X_scaled.shape[1]:
            self.feature_selector = SelectKBest(mutual_info_classif, k=select_k_best)
            X_scaled = self.feature_selector.fit_transform(X_scaled, y)
            selected_mask = self.feature_selector.get_support()
            self.feature_names = [f for f, s in zip(feature_columns, selected_mask) if s]
            print(f"  Selected {select_k_best} features via mutual information")
        
        # Stratified splits
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_scaled, y,
            test_size=(self.config.test_size + self.config.val_size),
            random_state=self.config.random_seed,
            stratify=y
        )
        
        val_ratio = self.config.val_size / (self.config.test_size + self.config.val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_ratio),
            random_state=self.config.random_seed,
            stratify=y_temp
        )
        
        # Calculate class weights for imbalanced data
        class_counts = np.bincount(y_train)
        class_weights = len(y_train) / (len(class_counts) * class_counts)
        
        result = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_selector': self.feature_selector,
            'class_weights': class_weights,
            'label_encoder': self.label_encoder
        }
        
        print(f"✓ Preprocessed data:")
        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Val: {X_val.shape[0]} samples")
        print(f"  Test: {X_test.shape[0]} samples")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Class distribution (train): {dict(zip(range(len(class_counts)), class_counts))}")
        
        return result


# ==============================================================================
# PYTORCH DATASET AND MODEL
# ==============================================================================

class AlzheimersDataset(Dataset):
    """PyTorch Dataset for Alzheimer's data"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class NeuralNetClassifier(nn.Module):
    """
    Deep neural network for AD classification
    
    Features:
    - Configurable hidden layers
    - Batch normalization
    - Dropout for regularization
    - Residual connections for deeper networks
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        num_classes: int = 2,
        dropout_rate: float = 0.3,
        use_residual: bool = True
    ):
        super().__init__()
        
        self.use_residual = use_residual and len(hidden_dims) > 2
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dims[-1], num_classes)
        
        # Residual projection if dimensions don't match
        if self.use_residual:
            self.residual_proj = nn.Linear(input_dim, hidden_dims[-1])
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        hidden = self.hidden_layers(x)
        
        if self.use_residual:
            residual = self.residual_proj(x)
            hidden = hidden + residual
        
        output = self.output_layer(hidden)
        return output
    
    def predict_proba(self, x):
        """Get probability predictions"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs


# ==============================================================================
# MODEL TRAINERS
# ==============================================================================

class XGBoostTrainer:
    """XGBoost model trainer with built-in evaluation"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model = None
        self.feature_importance = None
    
    def train(self, data: Dict) -> xgb.XGBClassifier:
        """Train XGBoost model"""
        print("\n" + "="*50)
        print("Training XGBoost Model")
        print("="*50)
        
        # Adjust for class imbalance
        params = self.config.xgb_params.copy()
        class_weights = data['class_weights']
        if len(class_weights) == 2:
            params['scale_pos_weight'] = class_weights[1] / class_weights[0]
        
        self.model = xgb.XGBClassifier(**params)
        
        self.model.fit(
            data['X_train'], data['y_train'],
            eval_set=[(data['X_val'], data['y_val'])],
            verbose=False
        )
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': data['feature_names'],
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Evaluate
        train_pred = self.model.predict(data['X_train'])
        val_pred = self.model.predict(data['X_val'])
        
        print(f"  Train Balanced Accuracy: {balanced_accuracy_score(data['y_train'], train_pred):.4f}")
        print(f"  Val Balanced Accuracy: {balanced_accuracy_score(data['y_val'], val_pred):.4f}")
        
        return self.model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def save(self, path: str):
        self.model.save_model(path)
        print(f"✓ Saved XGBoost model to {path}")
    
    def load(self, path: str):
        self.model = xgb.XGBClassifier()
        self.model.load_model(path)


class NeuralNetTrainer:
    """PyTorch neural network trainer"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    def train(self, data: Dict) -> NeuralNetClassifier:
        """Train neural network model"""
        print("\n" + "="*50)
        print("Training Neural Network Model")
        print("="*50)
        print(f"  Device: {self.device}")
        
        # Create datasets
        train_dataset = AlzheimersDataset(data['X_train'], data['y_train'])
        val_dataset = AlzheimersDataset(data['X_val'], data['y_val'])
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.nn_batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.nn_batch_size,
            shuffle=False
        )
        
        # Initialize model
        input_dim = data['X_train'].shape[1]
        self.model = NeuralNetClassifier(
            input_dim=input_dim,
            hidden_dims=self.config.nn_hidden_dims,
            num_classes=2
        ).to(self.device)
        
        # Loss with class weights
        class_weights = torch.FloatTensor(data['class_weights']).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.nn_learning_rate,
            weight_decay=0.01
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=False
        )
        
        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        patience = 20
        
        for epoch in range(self.config.nn_epochs):
            # Train
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += y_batch.size(0)
                train_correct += predicted.eq(y_batch).sum().item()
            
            # Validate
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += y_batch.size(0)
                    val_correct += predicted.eq(y_batch).sum().item()
            
            # Record history
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        print(f"  Best Val Loss: {best_val_loss:.4f}")
        
        return self.model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = outputs.max(1)
        return predicted.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            probs = self.model.predict_proba(X_tensor)
        return probs.cpu().numpy()
    
    def save(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                'input_dim': self.model.output_layer.in_features,
                'hidden_dims': self.config.nn_hidden_dims
            }
        }, path)
        print(f"✓ Saved Neural Network model to {path}")
    
    def load(self, path: str, input_dim: int):
        checkpoint = torch.load(path, map_location=self.device)
        self.model = NeuralNetClassifier(
            input_dim=input_dim,
            hidden_dims=checkpoint['config']['hidden_dims']
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])


# ==============================================================================
# ENSEMBLE MODEL
# ==============================================================================

class EnsembleClassifier:
    """Ensemble of XGBoost and Neural Network models"""
    
    def __init__(
        self,
        xgb_trainer: XGBoostTrainer,
        nn_trainer: NeuralNetTrainer,
        weights: Dict[str, float] = None
    ):
        self.xgb_trainer = xgb_trainer
        self.nn_trainer = nn_trainer
        self.weights = weights or {'xgboost': 0.5, 'neural_net': 0.5}
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble probability predictions"""
        xgb_probs = self.xgb_trainer.predict_proba(X)
        nn_probs = self.nn_trainer.predict_proba(X)
        
        ensemble_probs = (
            self.weights['xgboost'] * xgb_probs +
            self.weights['neural_net'] * nn_probs
        )
        
        return ensemble_probs
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble predictions"""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
    def optimize_weights(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Find optimal ensemble weights using validation set"""
        best_score = 0
        best_weights = self.weights.copy()
        
        for w in np.arange(0, 1.05, 0.05):
            self.weights = {'xgboost': w, 'neural_net': 1 - w}
            preds = self.predict(X_val)
            score = balanced_accuracy_score(y_val, preds)
            
            if score > best_score:
                best_score = score
                best_weights = self.weights.copy()
        
        self.weights = best_weights
        print(f"✓ Optimized ensemble weights: XGBoost={best_weights['xgboost']:.2f}, NN={best_weights['neural_net']:.2f}")
        print(f"  Best validation balanced accuracy: {best_score:.4f}")
        
        return best_weights


# ==============================================================================
# EVALUATION
# ==============================================================================

class Evaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.results = {}
    
    def evaluate(
        self,
        model,
        data: Dict,
        model_name: str = 'model'
    ) -> Dict:
        """
        Evaluate model on all data splits
        
        Returns dict with metrics for train, val, and test sets
        """
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name}")
        print("="*50)
        
        results = {}
        
        for split in ['train', 'val', 'test']:
            X = data[f'X_{split}']
            y_true = data[f'y_{split}']
            
            # Get predictions
            if hasattr(model, 'predict'):
                y_pred = model.predict(X)
            else:
                y_pred = model(X)
            
            # Get probabilities
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X)
                if len(y_proba.shape) > 1:
                    y_proba = y_proba[:, 1]
            else:
                y_proba = y_pred
            
            # Calculate metrics
            metrics = {
                'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
                'f1_score': f1_score(y_true, y_pred, average='weighted'),
                'roc_auc': roc_auc_score(y_true, y_proba),
                'average_precision': average_precision_score(y_true, y_proba),
                'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
            }
            
            results[split] = metrics
            
            print(f"\n  {split.upper()} Set:")
            print(f"    Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
            print(f"    F1 Score: {metrics['f1_score']:.4f}")
            print(f"    ROC AUC: {metrics['roc_auc']:.4f}")
            print(f"    Average Precision: {metrics['average_precision']:.4f}")
        
        self.results[model_name] = results
        return results
    
    def plot_results(self, save_path: Optional[str] = None):
        """Plot evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Model comparison bar chart
        ax = axes[0, 0]
        models = list(self.results.keys())
        metrics = ['balanced_accuracy', 'f1_score', 'roc_auc']
        x = np.arange(len(models))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [self.results[m]['test'][metric] for m in models]
            ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Model Comparison (Test Set)')
        ax.set_xticks(x + width)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        # 2. Confusion matrices
        ax = axes[0, 1]
        if 'Ensemble' in self.results:
            cm = np.array(self.results['Ensemble']['test']['confusion_matrix'])
        else:
            cm = np.array(list(self.results.values())[0]['test']['confusion_matrix'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix (Test Set)')
        ax.set_xticklabels(['Control', 'AD'])
        ax.set_yticklabels(['Control', 'AD'])
        
        # 3. Train/Val/Test comparison
        ax = axes[1, 0]
        splits = ['train', 'val', 'test']
        x = np.arange(len(splits))
        
        for i, model in enumerate(models):
            values = [self.results[model][s]['balanced_accuracy'] for s in splits]
            ax.plot(x, values, 'o-', label=model, markersize=8)
        
        ax.set_xlabel('Data Split')
        ax.set_ylabel('Balanced Accuracy')
        ax.set_title('Performance Across Data Splits')
        ax.set_xticks(x)
        ax.set_xticklabels(splits)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0.5, 1)
        
        # 4. Summary table
        ax = axes[1, 1]
        ax.axis('off')
        
        table_data = []
        for model in models:
            row = [
                model,
                f"{self.results[model]['test']['balanced_accuracy']:.4f}",
                f"{self.results[model]['test']['f1_score']:.4f}",
                f"{self.results[model]['test']['roc_auc']:.4f}"
            ]
            table_data.append(row)
        
        table = ax.table(
            cellText=table_data,
            colLabels=['Model', 'Bal. Accuracy', 'F1 Score', 'ROC AUC'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax.set_title('Summary Metrics (Test Set)', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved evaluation plot to {save_path}")
        
        plt.show()
        
    def save_results(self, path: str):
        """Save results to JSON"""
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Saved results to {path}")


# ==============================================================================
# SHAP FEATURE IMPORTANCE
# ==============================================================================

class SHAPAnalyzer:
    """SHAP-based feature importance analysis"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
    
    def analyze_xgboost(
        self,
        model: xgb.XGBClassifier,
        X: np.ndarray,
        feature_names: List[str]
    ) -> np.ndarray:
        """Analyze XGBoost model with SHAP"""
        print("\n" + "="*50)
        print("SHAP Analysis for XGBoost")
        print("="*50)
        
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)
        self.shap_values = self.explainer.shap_values(X)
        
        # For binary classification, shap_values might be a list
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]  # Use positive class
        
        print(f"✓ Computed SHAP values for {X.shape[0]} samples")
        
        return self.shap_values
    
    def analyze_neural_net(
        self,
        model: NeuralNetClassifier,
        X: np.ndarray,
        feature_names: List[str],
        background_samples: int = 100
    ) -> np.ndarray:
        """Analyze Neural Network model with SHAP (using DeepExplainer)"""
        print("\n" + "="*50)
        print("SHAP Analysis for Neural Network")
        print("="*50)
        
        self.feature_names = feature_names
        
        # Use a subset as background
        background = X[:min(background_samples, len(X))]
        background_tensor = torch.FloatTensor(background)
        
        model.eval()
        self.explainer = shap.DeepExplainer(model, background_tensor)
        
        X_tensor = torch.FloatTensor(X)
        self.shap_values = self.explainer.shap_values(X_tensor)
        
        # For binary classification
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]
        
        if isinstance(self.shap_values, torch.Tensor):
            self.shap_values = self.shap_values.numpy()
        
        print(f"✓ Computed SHAP values for {X.shape[0]} samples")
        
        return self.shap_values
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance ranking"""
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def plot_summary(self, X: np.ndarray, save_path: Optional[str] = None, max_display: int = 20):
        """Create SHAP summary plot"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Summary plot (beeswarm)
        plt.sca(axes[0])
        shap.summary_plot(
            self.shap_values, X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        axes[0].set_title('SHAP Summary Plot')
        
        # Bar plot
        plt.sca(axes[1])
        shap.summary_plot(
            self.shap_values, X,
            feature_names=self.feature_names,
            plot_type='bar',
            max_display=max_display,
            show=False
        )
        axes[1].set_title('Feature Importance (Mean |SHAP|)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved SHAP plot to {save_path}")
        
        plt.show()
    
    def plot_dependence(
        self,
        X: np.ndarray,
        feature: str,
        interaction_feature: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """Create SHAP dependence plot for a specific feature"""
        feature_idx = self.feature_names.index(feature)
        
        plt.figure(figsize=(10, 6))
        
        if interaction_feature:
            interaction_idx = self.feature_names.index(interaction_feature)
            shap.dependence_plot(
                feature_idx, self.shap_values, X,
                feature_names=self.feature_names,
                interaction_index=interaction_idx,
                show=False
            )
        else:
            shap.dependence_plot(
                feature_idx, self.shap_values, X,
                feature_names=self.feature_names,
                show=False
            )
        
        plt.title(f'SHAP Dependence: {feature}')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved dependence plot to {save_path}")
        
        plt.show()


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

class AlzheimersPipeline:
    """Main pipeline orchestrating all components"""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        setup_directories(self.config)
        
        self.data_loader = MultimodalDataLoader(self.config)
        self.preprocessor = Preprocessor(self.config)
        self.xgb_trainer = XGBoostTrainer(self.config)
        self.nn_trainer = NeuralNetTrainer(self.config)
        self.evaluator = Evaluator(self.config)
        self.shap_analyzer = SHAPAnalyzer(self.config)
        
        self.data = None
        self.ensemble = None
    
    def load_data(
        self,
        clinical_path: str,
        imaging_path: Optional[str] = None,
        genetic_path: Optional[str] = None,
        target_column: str = 'AD_status',
        sample_id_col: str = 'IID'
    ) -> pd.DataFrame:
        """Load all data modalities"""
        print("\n" + "="*50)
        print("Loading Data")
        print("="*50)
        
        # Load clinical data
        self.data_loader.load_clinical_data(clinical_path, sample_id_col)
        
        # Load imaging if provided
        if imaging_path and os.path.exists(imaging_path):
            self.data_loader.load_npz_data(imaging_path)
        
        # Load genetic if provided
        if genetic_path and os.path.exists(genetic_path.replace('.bed', '') + '.bed'):
            self.data_loader.load_bed_genetic_data(genetic_path)
        
        # Combine all modalities
        combined_df = self.data_loader.combine_multimodal_data(target_column=target_column)
        
        return combined_df
    
    def preprocess(
        self,
        df: pd.DataFrame,
        target_column: str = 'AD_status',
        select_k_best: Optional[int] = None
    ) -> Dict:
        """Preprocess data and create splits"""
        print("\n" + "="*50)
        print("Preprocessing Data")
        print("="*50)
        
        self.data = self.preprocessor.preprocess(
            df, target_column, select_k_best=select_k_best
        )
        
        return self.data
    
    def train_models(self) -> Dict:
        """Train all models"""
        if self.data is None:
            raise ValueError("Must preprocess data first")
        
        # Train XGBoost
        self.xgb_trainer.train(self.data)
        
        # Train Neural Network
        self.nn_trainer.train(self.data)
        
        # Create ensemble
        self.ensemble = EnsembleClassifier(
            self.xgb_trainer,
            self.nn_trainer,
            self.config.ensemble_weights
        )
        
        # Optimize ensemble weights
        self.ensemble.optimize_weights(self.data['X_val'], self.data['y_val'])
        
        return {
            'xgboost': self.xgb_trainer.model,
            'neural_net': self.nn_trainer.model,
            'ensemble': self.ensemble
        }
    
    def evaluate_models(self) -> Dict:
        """Evaluate all models"""
        # Evaluate individual models
        self.evaluator.evaluate(self.xgb_trainer, self.data, 'XGBoost')
        self.evaluator.evaluate(self.nn_trainer, self.data, 'Neural Network')
        self.evaluator.evaluate(self.ensemble, self.data, 'Ensemble')
        
        return self.evaluator.results
    
    def analyze_features(self, n_samples: int = 500):
        """Perform SHAP analysis"""
        # Use subset for SHAP (for speed)
        X_sample = self.data['X_test'][:n_samples]
        
        # Analyze XGBoost
        self.shap_analyzer.analyze_xgboost(
            self.xgb_trainer.model,
            X_sample,
            self.data['feature_names']
        )
        
        # Get top features
        importance_df = self.shap_analyzer.get_feature_importance(top_n=20)
        print("\nTop 20 Most Important Features:")
        print(importance_df.to_string())
        
        # Save importance
        importance_df.to_csv(
            os.path.join(self.config.output_dir, 'feature_importance.csv'),
            index=False
        )
        
        return importance_df
    
    def plot_results(self):
        """Generate all plots"""
        # Evaluation plots
        self.evaluator.plot_results(
            os.path.join(self.config.output_dir, 'evaluation_results.png')
        )
        
        # SHAP plots
        if self.shap_analyzer.shap_values is not None:
            self.shap_analyzer.plot_summary(
                self.data['X_test'][:500],
                os.path.join(self.config.output_dir, 'shap_summary.png')
            )
        
        # Training history for neural network
        if self.nn_trainer.history['train_loss']:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            axes[0].plot(self.nn_trainer.history['train_loss'], label='Train')
            axes[0].plot(self.nn_trainer.history['val_loss'], label='Validation')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training History - Loss')
            axes[0].legend()
            axes[0].grid(alpha=0.3)
            
            axes[1].plot(self.nn_trainer.history['train_acc'], label='Train')
            axes[1].plot(self.nn_trainer.history['val_acc'], label='Validation')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Training History - Accuracy')
            axes[1].legend()
            axes[1].grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.config.output_dir, 'training_history.png'),
                dpi=150, bbox_inches='tight'
            )
            plt.show()
    
    def save_models(self):
        """Save all trained models"""
        self.xgb_trainer.save(
            os.path.join(self.config.model_dir, 'xgboost_model.json')
        )
        self.nn_trainer.save(
            os.path.join(self.config.model_dir, 'neural_net_model.pt')
        )
        
        # Save preprocessor
        with open(os.path.join(self.config.model_dir, 'preprocessor.pkl'), 'wb') as f:
            pickle.dump({
                'scaler': self.preprocessor.scaler,
                'feature_names': self.data['feature_names'],
                'feature_selector': self.preprocessor.feature_selector
            }, f)
        print(f"✓ Saved preprocessor")
        
        # Save results
        self.evaluator.save_results(
            os.path.join(self.config.output_dir, 'evaluation_results.json')
        )
    
    def run_full_pipeline(
        self,
        clinical_path: str,
        target_column: str = 'AD_status',
        imaging_path: Optional[str] = None,
        genetic_path: Optional[str] = None,
        sample_id_col: str = 'IID',
        select_k_best: Optional[int] = None
    ) -> Dict:
        """
        Run the complete pipeline end-to-end
        
        Args:
            clinical_path: Path to clinical data file
            target_column: Name of target column
            imaging_path: Optional path to imaging NPZ file
            genetic_path: Optional path to genetic BED file
            sample_id_col: Column name for sample IDs
            select_k_best: Optional number of features to select
        
        Returns:
            Dictionary with all results and models
        """
        print("\n" + "#"*60)
        print("# ALZHEIMER'S DISEASE PREDICTION PIPELINE")
        print("#"*60)
        
        # 1. Load data
        combined_df = self.load_data(
            clinical_path=clinical_path,
            imaging_path=imaging_path,
            genetic_path=genetic_path,
            target_column=target_column,
            sample_id_col=sample_id_col
        )
        
        # 2. Preprocess
        self.preprocess(combined_df, target_column, select_k_best)
        
        # 3. Train models
        models = self.train_models()
        
        # 4. Evaluate
        results = self.evaluate_models()
        
        # 5. Feature analysis
        importance_df = self.analyze_features()
        
        # 6. Plot results
        self.plot_results()
        
        # 7. Save everything
        self.save_models()
        
        print("\n" + "#"*60)
        print("# PIPELINE COMPLETE")
        print("#"*60)
        print(f"\nResults saved to: {self.config.output_dir}")
        print(f"Models saved to: {self.config.model_dir}")
        
        return {
            'models': models,
            'results': results,
            'feature_importance': importance_df,
            'data': self.data
        }


# ==============================================================================
# SYNTHETIC DATA GENERATOR FOR DEMO
# ==============================================================================

def create_synthetic_data(n_samples: int = 1000, save_dir: str = 'alzheimers_hackathon/data'):
    """Create synthetic data for demonstration"""
    print("\nCreating synthetic demonstration data...")
    os.makedirs(save_dir, exist_ok=True)
    
    np.random.seed(42)
    
    # Clinical features
    ages = np.random.normal(72, 8, n_samples)
    education_years = np.random.normal(14, 3, n_samples)
    mmse_scores = np.random.normal(26, 4, n_samples)
    cdr_scores = np.random.choice([0, 0.5, 1, 2], n_samples, p=[0.4, 0.3, 0.2, 0.1])
    
    # APOE status (0, 1, or 2 copies of ε4 allele)
    apoe_e4 = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1])
    
    # Generate AD status based on risk factors
    logit = (
        -5 +
        0.05 * (ages - 65) +
        -0.1 * education_years +
        -0.3 * mmse_scores +
        1.5 * cdr_scores +
        1.2 * apoe_e4 +
        np.random.normal(0, 1, n_samples)
    )
    ad_prob = 1 / (1 + np.exp(-logit))
    ad_status = (np.random.random(n_samples) < ad_prob).astype(int)
    
    # Create clinical DataFrame
    clinical_df = pd.DataFrame({
        'IID': [f'SUBJ_{i:04d}' for i in range(n_samples)],
        'age': ages,
        'education_years': education_years,
        'mmse_score': mmse_scores,
        'cdr_score': cdr_scores,
        'gender': np.random.choice(['M', 'F'], n_samples),
        'AD_status': ad_status
    })
    
    # Add some missing values
    for col in ['mmse_score', 'education_years']:
        mask = np.random.random(n_samples) < 0.05
        clinical_df.loc[mask, col] = np.nan
    
    # Save clinical data
    clinical_path = os.path.join(save_dir, 'clinical_data.tsv')
    clinical_df.to_csv(clinical_path, sep='\t', index=False)
    print(f"  Saved clinical data: {clinical_path}")
    
    # Create imaging features (simulated MRI-derived features)
    n_imaging_features = 50
    imaging_features = np.random.randn(n_samples, n_imaging_features)
    
    # Add AD-related signal to imaging
    for i in range(10):
        imaging_features[:, i] += ad_status * np.random.uniform(0.5, 1.5)
    
    # Save imaging data
    imaging_path = os.path.join(save_dir, 'imaging_features.npz')
    np.savez(
        imaging_path,
        mri_features=imaging_features,
        sample_ids=np.array([f'SUBJ_{i:04d}' for i in range(n_samples)])
    )
    print(f"  Saved imaging data: {imaging_path}")
    
    # Create genetic data (simulated SNPs)
    n_snps = 500
    genotypes = np.random.choice([0, 1, 2], size=(n_samples, n_snps), p=[0.6, 0.3, 0.1])
    
    # Add APOE effect
    genotypes[:, 0] = apoe_e4  # First SNP represents APOE
    
    # Add some missing values (represented as NaN)
    missing_mask = np.random.random(genotypes.shape) < 0.01
    genotypes = genotypes.astype(float)
    genotypes[missing_mask] = np.nan
    
    # Create SNP IDs including known AD SNPs
    snp_ids = ['rs429358'] + [f'rs{np.random.randint(1000000, 9999999)}' for _ in range(n_snps - 1)]
    
    # Save genetic data as NPZ
    genetic_path = os.path.join(save_dir, 'genetic_data.npz')
    np.savez(
        genetic_path,
        genotypes=genotypes,
        sample_ids=np.array([f'SUBJ_{i:04d}' for i in range(n_samples)]),
        snp_ids=np.array(snp_ids)
    )
    print(f"  Saved genetic data: {genetic_path}")
    
    print(f"\n✓ Created synthetic dataset with {n_samples} samples")
    print(f"  AD cases: {ad_status.sum()} ({100*ad_status.mean():.1f}%)")
    print(f"  Controls: {n_samples - ad_status.sum()} ({100*(1-ad_status.mean()):.1f}%)")
    
    return clinical_path, imaging_path, genetic_path


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Create configuration
    config = PipelineConfig(
        n_genetic_features=100,
        use_prs=True,
        use_apoe=True,
        nn_epochs=50,
        nn_batch_size=32
    )
    
    # Create synthetic data for demonstration
    clinical_path, imaging_path, genetic_path = create_synthetic_data(
        n_samples=1000,
        save_dir=config.data_dir
    )
    
    # Initialize pipeline
    pipeline = AlzheimersPipeline(config)
    
    # Load clinical data
    clinical_df = pipeline.data_loader.load_clinical_data(clinical_path)
    
    # Load imaging data
    imaging_dict = pipeline.data_loader.load_npz_data(imaging_path)
    
    # Load genetic data from NPZ (for demo - in real usage, use BED files)
    genetic_data = np.load(genetic_path)
    pipeline.data_loader.genotypes = genetic_data['genotypes']
    pipeline.data_loader.genetic_sample_ids = genetic_data['sample_ids']
    pipeline.data_loader.snp_ids = genetic_data['snp_ids']
    
    # Handle missing values in genetic data
    missing_mask = np.isnan(pipeline.data_loader.genotypes)
    if missing_mask.any():
        print(f"  Imputing {missing_mask.sum()} missing genetic values")
        for col in range(pipeline.data_loader.genotypes.shape[1]):
            col_missing = missing_mask[:, col]
            if col_missing.any():
                valid_vals = pipeline.data_loader.genotypes[~col_missing, col]
                pipeline.data_loader.genotypes[col_missing, col] = np.median(valid_vals)
    
    print(f"✓ Loaded genetic data: {pipeline.data_loader.genotypes.shape}")
    
    # Combine all data
    combined_df = pipeline.data_loader.combine_multimodal_data(target_column='AD_status')
    
    # Preprocess
    data = pipeline.preprocess(combined_df, target_column='AD_status')
    
    # Train models
    models = pipeline.train_models()
    
    # Evaluate
    results = pipeline.evaluate_models()
    
    # Feature analysis
    importance_df = pipeline.analyze_features(n_samples=200)
    
    # Plot results
    pipeline.plot_results()
    
    # Save everything
    pipeline.save_models()
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)

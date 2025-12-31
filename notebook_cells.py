# Alzheimer's Disease Prediction Pipeline - Notebook Cells
# ========================================================
# Copy each cell below into your Jupyter notebook

# ==============================================================================
# CELL 1: Install Dependencies
# ==============================================================================
"""
!pip install xgboost torch pandas scikit-learn numpy matplotlib seaborn shap bed-reader -q
"""

# ==============================================================================
# CELL 2: Imports and Setup
# ==============================================================================
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
    confusion_matrix, classification_report, average_precision_score
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import warnings
import pickle
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

warnings.filterwarnings('ignore')

# Set random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Create directories
for d in ['alzheimers_hackathon/data', 'alzheimers_hackathon/models', 'alzheimers_hackathon/outputs']:
    os.makedirs(d, exist_ok=True)

print("✓ All libraries imported successfully!")
print(f"✓ PyTorch device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
"""

# ==============================================================================
# CELL 3: Configuration
# ==============================================================================
"""
@dataclass
class PipelineConfig:
    '''Configuration for the pipeline'''
    data_dir: str = 'alzheimers_hackathon/data'
    model_dir: str = 'alzheimers_hackathon/models'
    output_dir: str = 'alzheimers_hackathon/outputs'
    test_size: float = 0.15
    val_size: float = 0.15
    random_seed: int = 42
    n_genetic_features: int = 100
    use_prs: bool = True
    use_apoe: bool = True
    nn_epochs: int = 100
    nn_batch_size: int = 32
    nn_learning_rate: float = 0.001
    nn_hidden_dims: List[int] = None
    xgb_params: dict = None
    
    def __post_init__(self):
        if self.nn_hidden_dims is None:
            self.nn_hidden_dims = [256, 128, 64]
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
                'random_state': self.random_seed
            }

config = PipelineConfig()
print("✓ Configuration created")
"""

# ==============================================================================
# CELL 4: Data Loading Class (FIXED)
# ==============================================================================
"""
class MultimodalDataLoader:
    '''Handles loading of all data modalities with proper alignment'''
    
    # Known AD-associated SNPs
    AD_ASSOCIATED_SNPS = [
        'rs429358', 'rs7412', 'rs6656401', 'rs6733839', 'rs35349669',
        'rs190982', 'rs2718058', 'rs10948363', 'rs11771145', 'rs9331896',
        'rs983392', 'rs10792832', 'rs4147929', 'rs3865444', 'rs28834970',
        'rs9271192', 'rs10838725', 'rs17125944', 'rs10498633', 'rs8093731',
    ]
    
    def __init__(self, config):
        self.config = config
        self.sample_id_column = None
        self.clinical_df = None
        self.imaging_dict = None
        self.genotypes = None
        self.genetic_sample_ids = None
        self.snp_ids = None
        
    def load_clinical_data(self, filepath: str, sample_id_col: str = 'IID') -> pd.DataFrame:
        '''Load clinical data with KNN imputation'''
        with open(filepath, 'r') as f:
            first_line = f.readline()
        sep = '\\t' if '\\t' in first_line else ','
        
        df = pd.read_csv(filepath, sep=sep)
        
        # Find sample ID column
        self.sample_id_column = sample_id_col
        if sample_id_col not in df.columns:
            for alt in ['ID', 'SampleID', 'Subject_ID', 'PTID', 'RID']:
                if alt in df.columns:
                    self.sample_id_column = alt
                    break
            else:
                df['IID'] = df.index.astype(str)
                self.sample_id_column = 'IID'
        
        # KNN imputation for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != self.sample_id_column]
        
        if len(numeric_cols) > 0:
            imputer = KNNImputer(n_neighbors=5)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        self.clinical_df = df
        print(f"✓ Loaded clinical data: {df.shape[0]} samples, {df.shape[1]} features")
        return df
    
    def load_npz_data(self, filepath: str) -> Dict[str, np.ndarray]:
        '''Load imaging data from NPZ'''
        data = np.load(filepath, allow_pickle=True)
        arrays_dict = {}
        
        for key in data.files:
            arr = data[key]
            if arr.dtype == object:
                arr = np.array(arr.tolist())
            arrays_dict[key] = arr
            print(f"  Loaded {key}: shape={arr.shape}")
        
        self.imaging_dict = arrays_dict
        print(f"✓ Loaded NPZ data")
        return arrays_dict
    
    def load_bed_genetic_data(self, bedfile_path: str):
        '''Load genetic data from PLINK BED format with missing value handling'''
        from bed_reader import open_bed
        
        if bedfile_path.endswith('.bed'):
            bedfile_path = bedfile_path[:-4]
        
        bed = open_bed(bedfile_path + '.bed')
        genotypes = bed.read()
        sample_ids = np.array(bed.iid)
        snp_ids = np.array(bed.sid)
        
        # FIX: Handle missing genotype calls
        missing_mask = np.isnan(genotypes) | (genotypes < 0) | (genotypes > 2)
        missing_rate = missing_mask.sum() / genotypes.size
        print(f"  Missing genotype rate: {missing_rate:.4%}")
        
        if missing_rate > 0:
            for col_idx in range(genotypes.shape[1]):
                col = genotypes[:, col_idx]
                col_missing = missing_mask[:, col_idx]
                if col_missing.any():
                    valid_vals = col[~col_missing]
                    if len(valid_vals) > 0:
                        mode_val = np.round(np.median(valid_vals))
                        genotypes[col_missing, col_idx] = mode_val
                    else:
                        genotypes[col_missing, col_idx] = 1
        
        genotypes = np.clip(genotypes, 0, 2)
        del bed
        
        self.genotypes = genotypes
        self.genetic_sample_ids = sample_ids
        self.snp_ids = snp_ids
        
        print(f"✓ Loaded genetic data: {genotypes.shape[0]} samples, {genotypes.shape[1]} SNPs")
        return genotypes, sample_ids, snp_ids
    
    def load_npz_genetic_data(self, filepath: str):
        '''Load genetic data from NPZ file (alternative to BED)'''
        data = np.load(filepath, allow_pickle=True)
        
        self.genotypes = data['genotypes']
        self.genetic_sample_ids = data['sample_ids']
        self.snp_ids = data['snp_ids']
        
        # Handle missing values
        missing_mask = np.isnan(self.genotypes)
        if missing_mask.any():
            print(f"  Imputing {missing_mask.sum()} missing genetic values")
            for col in range(self.genotypes.shape[1]):
                col_missing = missing_mask[:, col]
                if col_missing.any():
                    valid_vals = self.genotypes[~col_missing, col]
                    self.genotypes[col_missing, col] = np.median(valid_vals)
        
        print(f"✓ Loaded genetic data: {self.genotypes.shape}")
        return self.genotypes, self.genetic_sample_ids, self.snp_ids
    
    def extract_apoe_status(self) -> Optional[np.ndarray]:
        '''Extract APOE ε4 carrier status'''
        if self.snp_ids is None or self.genotypes is None:
            return None
        
        snp_list = list(self.snp_ids)
        apoe_e4_snp = 'rs429358'
        
        if apoe_e4_snp in snp_list:
            idx = snp_list.index(apoe_e4_snp)
            apoe_status = self.genotypes[:, idx].astype(int)
            print(f"✓ Extracted APOE ε4 status from {apoe_e4_snp}")
            return apoe_status
        
        # Try alternatives
        apoe_snps = [s for s in snp_list if 'APOE' in s.upper() or s in ['rs429358', 'rs7412']]
        if apoe_snps:
            idx = snp_list.index(apoe_snps[0])
            return self.genotypes[:, idx].astype(int)
        
        print("  Warning: APOE SNPs not found")
        return None
    
    def calculate_polygenic_risk_score(self, weights_file: Optional[str] = None) -> np.ndarray:
        '''Calculate Polygenic Risk Score (PRS)'''
        if self.genotypes is None or self.snp_ids is None:
            raise ValueError("Genetic data must be loaded first")
        
        snp_list = list(self.snp_ids)
        
        # Default AD GWAS weights
        snp_weights = {
            'rs429358': 1.2, 'rs7412': -0.5, 'rs6656401': 0.15, 'rs6733839': 0.18,
            'rs35349669': 0.08, 'rs190982': 0.07, 'rs10948363': 0.10, 'rs9331896': 0.14,
            'rs10792832': 0.12, 'rs4147929': 0.15,
        }
        
        if weights_file and os.path.exists(weights_file):
            weights_df = pd.read_csv(weights_file)
            snp_weights = dict(zip(weights_df.iloc[:, 0], weights_df.iloc[:, 1]))
        
        prs = np.zeros(self.genotypes.shape[0])
        snps_used = 0
        
        for snp, weight in snp_weights.items():
            if snp in snp_list:
                idx = snp_list.index(snp)
                prs += weight * self.genotypes[:, idx]
                snps_used += 1
        
        if snps_used == 0:
            # Fallback: use high-variance SNPs
            variances = np.var(self.genotypes, axis=0)
            top_indices = np.argsort(variances)[-20:]
            for idx in top_indices:
                prs += self.genotypes[:, idx] / 20
            snps_used = 20
        
        prs = (prs - prs.mean()) / (prs.std() + 1e-8)
        print(f"✓ Calculated PRS using {snps_used} SNPs")
        return prs
    
    def select_ad_associated_snps(self) -> Tuple[np.ndarray, List[str]]:
        '''Select AD-associated SNPs + high-variance SNPs'''
        snp_list = list(self.snp_ids)
        selected_indices = []
        selected_snps = []
        
        # Get known AD SNPs
        for snp in self.AD_ASSOCIATED_SNPS:
            if snp in snp_list:
                selected_indices.append(snp_list.index(snp))
                selected_snps.append(snp)
        
        print(f"  Found {len(selected_indices)} known AD-associated SNPs")
        
        # Add high-variance SNPs
        n_needed = self.config.n_genetic_features - len(selected_indices)
        if n_needed > 0:
            variances = np.var(self.genotypes, axis=0)
            variances[selected_indices] = -1
            top_var_indices = np.argsort(variances)[-n_needed:]
            
            for idx in top_var_indices:
                if idx not in selected_indices:
                    selected_indices.append(idx)
                    selected_snps.append(self.snp_ids[idx])
        
        selected_genotypes = self.genotypes[:, selected_indices]
        print(f"✓ Selected {len(selected_snps)} genetic features")
        
        return selected_genotypes, selected_snps
    
    def combine_multimodal_data(self, target_column: str = 'AD_status') -> pd.DataFrame:
        '''FIX: Combine all modalities with proper sample alignment'''
        if self.clinical_df is None:
            raise ValueError("Clinical data must be loaded first")
        
        combined_df = self.clinical_df.copy()
        clinical_ids = combined_df[self.sample_id_column].astype(str).values
        
        # Add imaging features with alignment
        if self.imaging_dict is not None:
            imaging_key = None
            for key in ['mri_features', 'imaging_features', 'features', 'X']:
                if key in self.imaging_dict:
                    imaging_key = key
                    break
            
            if imaging_key:
                imaging_features = self.imaging_dict[imaging_key]
                
                # Check for sample IDs
                imaging_ids = None
                for key in ['sample_ids', 'ids', 'IID', 'subjects']:
                    if key in self.imaging_dict:
                        imaging_ids = self.imaging_dict[key].astype(str)
                        break
                
                if imaging_ids is not None and len(imaging_ids) == imaging_features.shape[0]:
                    # Align by sample IDs
                    imaging_df = pd.DataFrame(
                        imaging_features,
                        columns=[f'imaging_feature_{i}' for i in range(imaging_features.shape[1])]
                    )
                    imaging_df['_merge_id'] = imaging_ids
                    combined_df['_merge_id'] = clinical_ids
                    
                    combined_df = combined_df.merge(imaging_df, on='_merge_id', how='inner')
                    combined_df.drop('_merge_id', axis=1, inplace=True)
                    print(f"  Aligned imaging data: {imaging_features.shape[1]} features")
                elif imaging_features.shape[0] == len(clinical_ids):
                    for i in range(imaging_features.shape[1]):
                        combined_df[f'imaging_feature_{i}'] = imaging_features[:, i]
                    print(f"  Added imaging data (assumed aligned): {imaging_features.shape[1]} features")
        
        # Add genetic features with alignment
        if self.genotypes is not None:
            genetic_ids = self.genetic_sample_ids.astype(str) if self.genetic_sample_ids is not None else None
            selected_geno, selected_snps = self.select_ad_associated_snps()
            
            if genetic_ids is not None:
                genetic_df = pd.DataFrame(
                    selected_geno,
                    columns=[f'snp_{snp}' for snp in selected_snps]
                )
                genetic_df['_merge_id'] = genetic_ids
                
                if self.config.use_prs:
                    genetic_df['PRS'] = self.calculate_polygenic_risk_score()
                
                if self.config.use_apoe:
                    apoe = self.extract_apoe_status()
                    if apoe is not None:
                        genetic_df['APOE_e4_count'] = apoe
                
                combined_df['_merge_id'] = combined_df[self.sample_id_column].astype(str)
                combined_df = combined_df.merge(genetic_df, on='_merge_id', how='inner')
                combined_df.drop('_merge_id', axis=1, inplace=True)
                print(f"  Aligned genetic data")
            elif selected_geno.shape[0] == len(combined_df):
                for i, snp in enumerate(selected_snps):
                    combined_df[f'snp_{snp}'] = selected_geno[:, i]
                
                if self.config.use_prs:
                    combined_df['PRS'] = self.calculate_polygenic_risk_score()
                
                if self.config.use_apoe:
                    apoe = self.extract_apoe_status()
                    if apoe is not None:
                        combined_df['APOE_e4_count'] = apoe
        
        print(f"✓ Combined dataset: {combined_df.shape[0]} samples, {combined_df.shape[1]} features")
        return combined_df

print("✓ MultimodalDataLoader class defined")
"""

# ==============================================================================
# CELL 5: Preprocessing Class
# ==============================================================================
"""
class Preprocessor:
    '''Handles feature preprocessing and data splits'''
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.feature_names = None
        self.label_encoder = LabelEncoder()
        
    def preprocess(self, df: pd.DataFrame, target_column: str, 
                   feature_columns: Optional[List[str]] = None,
                   select_k_best: Optional[int] = None) -> Dict:
        '''Preprocess features and create train/val/test splits'''
        
        if feature_columns is None:
            exclude_cols = [target_column, 'IID', 'ID', 'SampleID', 'Subject_ID']
            feature_columns = [
                col for col in df.columns 
                if col not in exclude_cols and df[col].dtype in [np.float64, np.int64, np.float32, np.int32]
            ]
        
        self.feature_names = feature_columns
        X = df[feature_columns].values
        
        y = df[target_column].values
        if y.dtype == object or y.dtype.name == 'category':
            y = self.label_encoder.fit_transform(y)
        else:
            y = y.astype(int)
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Feature selection
        if select_k_best and select_k_best < X_scaled.shape[1]:
            self.feature_selector = SelectKBest(mutual_info_classif, k=select_k_best)
            X_scaled = self.feature_selector.fit_transform(X_scaled, y)
            selected_mask = self.feature_selector.get_support()
            self.feature_names = [f for f, s in zip(feature_columns, selected_mask) if s]
            print(f"  Selected {select_k_best} features via mutual information")
        
        # Stratified splits (70/15/15)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_scaled, y, test_size=0.30, random_state=self.config.random_seed, stratify=y
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=self.config.random_seed, stratify=y_temp
        )
        
        # Class weights for imbalanced data
        class_counts = np.bincount(y_train)
        class_weights = len(y_train) / (len(class_counts) * class_counts)
        
        result = {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'scaler': self.scaler, 'feature_names': self.feature_names,
            'feature_selector': self.feature_selector, 'class_weights': class_weights,
        }
        
        print(f"✓ Preprocessed: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
        print(f"  Features: {X_train.shape[1]}, Class distribution: {dict(zip(range(len(class_counts)), class_counts))}")
        
        return result

print("✓ Preprocessor class defined")
"""

# ==============================================================================
# CELL 6: PyTorch Neural Network Model
# ==============================================================================
"""
class AlzheimersDataset(Dataset):
    '''PyTorch Dataset for Alzheimer's data'''
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class NeuralNetClassifier(nn.Module):
    '''Deep neural network with BatchNorm, Dropout, and Residual connections'''
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], num_classes=2, 
                 dropout_rate=0.3, use_residual=True):
        super().__init__()
        
        self.use_residual = use_residual and len(hidden_dims) > 2
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dims[-1], num_classes)
        
        if self.use_residual:
            self.residual_proj = nn.Linear(input_dim, hidden_dims[-1])
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        hidden = self.hidden_layers(x)
        if self.use_residual:
            hidden = hidden + self.residual_proj(x)
        return self.output_layer(hidden)
    
    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=1)

print("✓ Neural Network model defined")
"""

# ==============================================================================
# CELL 7: XGBoost Trainer
# ==============================================================================
"""
class XGBoostTrainer:
    '''XGBoost model trainer'''
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.feature_importance = None
    
    def train(self, data: Dict) -> xgb.XGBClassifier:
        print("\\n" + "="*50)
        print("Training XGBoost Model")
        print("="*50)
        
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
        
        self.feature_importance = pd.DataFrame({
            'feature': data['feature_names'],
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        train_pred = self.model.predict(data['X_train'])
        val_pred = self.model.predict(data['X_val'])
        
        print(f"  Train Balanced Accuracy: {balanced_accuracy_score(data['y_train'], train_pred):.4f}")
        print(f"  Val Balanced Accuracy: {balanced_accuracy_score(data['y_val'], val_pred):.4f}")
        
        return self.model
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def save(self, path):
        self.model.save_model(path)
        print(f"✓ Saved XGBoost model to {path}")

print("✓ XGBoostTrainer class defined")
"""

# ==============================================================================
# CELL 8: Neural Network Trainer
# ==============================================================================
"""
class NeuralNetTrainer:
    '''PyTorch neural network trainer with early stopping'''
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    def train(self, data: Dict) -> NeuralNetClassifier:
        print("\\n" + "="*50)
        print("Training Neural Network Model")
        print("="*50)
        print(f"  Device: {self.device}")
        
        train_dataset = AlzheimersDataset(data['X_train'], data['y_train'])
        val_dataset = AlzheimersDataset(data['X_val'], data['y_val'])
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.nn_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.nn_batch_size, shuffle=False)
        
        input_dim = data['X_train'].shape[1]
        self.model = NeuralNetClassifier(
            input_dim=input_dim,
            hidden_dims=self.config.nn_hidden_dims,
            num_classes=2
        ).to(self.device)
        
        class_weights = torch.FloatTensor(data['class_weights']).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config.nn_learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        patience = 20
        
        for epoch in range(self.config.nn_epochs):
            # Train
            self.model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            
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
            val_loss, val_correct, val_total = 0, 0, 0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += y_batch.size(0)
                    val_correct += predicted.eq(y_batch).sum().item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            scheduler.step(val_loss)
            
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
        
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        print(f"  Best Val Loss: {best_val_loss:.4f}")
        return self.model
    
    def predict(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            _, predicted = self.model(X_tensor).max(1)
        return predicted.cpu().numpy()
    
    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            probs = self.model.predict_proba(X_tensor)
        return probs.cpu().numpy()
    
    def save(self, path):
        torch.save({'model_state_dict': self.model.state_dict(), 
                    'config': {'hidden_dims': self.config.nn_hidden_dims}}, path)
        print(f"✓ Saved Neural Network model to {path}")

print("✓ NeuralNetTrainer class defined")
"""

# ==============================================================================
# CELL 9: Ensemble Model
# ==============================================================================
"""
class EnsembleClassifier:
    '''Ensemble of XGBoost and Neural Network'''
    
    def __init__(self, xgb_trainer, nn_trainer, weights=None):
        self.xgb_trainer = xgb_trainer
        self.nn_trainer = nn_trainer
        self.weights = weights or {'xgboost': 0.5, 'neural_net': 0.5}
    
    def predict_proba(self, X):
        xgb_probs = self.xgb_trainer.predict_proba(X)
        nn_probs = self.nn_trainer.predict_proba(X)
        return self.weights['xgboost'] * xgb_probs + self.weights['neural_net'] * nn_probs
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
    
    def optimize_weights(self, X_val, y_val):
        best_score, best_weights = 0, self.weights.copy()
        
        for w in np.arange(0, 1.05, 0.05):
            self.weights = {'xgboost': w, 'neural_net': 1 - w}
            score = balanced_accuracy_score(y_val, self.predict(X_val))
            if score > best_score:
                best_score = score
                best_weights = self.weights.copy()
        
        self.weights = best_weights
        print(f"✓ Optimized weights: XGBoost={best_weights['xgboost']:.2f}, NN={best_weights['neural_net']:.2f}")
        print(f"  Best validation balanced accuracy: {best_score:.4f}")
        return best_weights

print("✓ EnsembleClassifier class defined")
"""

# ==============================================================================
# CELL 10: Evaluator and SHAP Analyzer
# ==============================================================================
"""
class Evaluator:
    '''Comprehensive model evaluation'''
    
    def __init__(self, config):
        self.config = config
        self.results = {}
    
    def evaluate(self, model, data, model_name='model'):
        print(f"\\nEvaluating {model_name}")
        print("="*40)
        
        results = {}
        for split in ['train', 'val', 'test']:
            X, y_true = data[f'X_{split}'], data[f'y_{split}']
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)
            y_proba = y_proba[:, 1] if len(y_proba.shape) > 1 else y_proba
            
            results[split] = {
                'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
                'f1_score': f1_score(y_true, y_pred, average='weighted'),
                'roc_auc': roc_auc_score(y_true, y_proba),
                'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
            }
            
            print(f"  {split.upper()}: Bal.Acc={results[split]['balanced_accuracy']:.4f}, "
                  f"F1={results[split]['f1_score']:.4f}, ROC-AUC={results[split]['roc_auc']:.4f}")
        
        self.results[model_name] = results
        return results
    
    def plot_results(self, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Model comparison
        models = list(self.results.keys())
        metrics = ['balanced_accuracy', 'f1_score', 'roc_auc']
        x = np.arange(len(models))
        
        for i, metric in enumerate(metrics):
            values = [self.results[m]['test'][metric] for m in models]
            axes[0, 0].bar(x + i*0.25, values, 0.25, label=metric.replace('_', ' ').title())
        
        axes[0, 0].set_xticks(x + 0.25)
        axes[0, 0].set_xticklabels(models)
        axes[0, 0].legend()
        axes[0, 0].set_title('Model Comparison (Test Set)')
        
        # Confusion matrix
        if 'Ensemble' in self.results:
            cm = np.array(self.results['Ensemble']['test']['confusion_matrix'])
        else:
            cm = np.array(list(self.results.values())[0]['test']['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
        axes[0, 1].set_title('Confusion Matrix')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
        
        # Performance across splits
        for model in models:
            values = [self.results[model][s]['balanced_accuracy'] for s in ['train', 'val', 'test']]
            axes[1, 0].plot(['Train', 'Val', 'Test'], values, 'o-', label=model)
        axes[1, 0].legend()
        axes[1, 0].set_title('Performance Across Splits')
        
        # Summary table
        axes[1, 1].axis('off')
        table_data = [[m, f"{self.results[m]['test']['balanced_accuracy']:.4f}",
                       f"{self.results[m]['test']['f1_score']:.4f}",
                       f"{self.results[m]['test']['roc_auc']:.4f}"] for m in models]
        axes[1, 1].table(cellText=table_data, colLabels=['Model', 'Bal.Acc', 'F1', 'ROC-AUC'],
                         loc='center', cellLoc='center')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


class SHAPAnalyzer:
    '''SHAP-based feature importance'''
    
    def __init__(self, config):
        self.config = config
        self.shap_values = None
        self.feature_names = None
    
    def analyze_xgboost(self, model, X, feature_names):
        print("\\nSHAP Analysis for XGBoost")
        print("="*40)
        
        self.feature_names = feature_names
        explainer = shap.TreeExplainer(model)
        self.shap_values = explainer.shap_values(X)
        
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]
        
        print(f"✓ Computed SHAP values for {X.shape[0]} samples")
        return self.shap_values
    
    def get_feature_importance(self, top_n=20):
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False).head(top_n)
    
    def plot_summary(self, X, save_path=None, max_display=20):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        plt.sca(axes[0])
        shap.summary_plot(self.shap_values, X, feature_names=self.feature_names, 
                          max_display=max_display, show=False)
        
        plt.sca(axes[1])
        shap.summary_plot(self.shap_values, X, feature_names=self.feature_names,
                          plot_type='bar', max_display=max_display, show=False)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

print("✓ Evaluator and SHAPAnalyzer classes defined")
"""

# ==============================================================================
# CELL 11: Synthetic Data Generator (for testing)
# ==============================================================================
"""
def create_synthetic_data(n_samples=1000, save_dir='alzheimers_hackathon/data'):
    '''Create synthetic data for testing'''
    print("Creating synthetic data...")
    os.makedirs(save_dir, exist_ok=True)
    np.random.seed(42)
    
    # Clinical features
    ages = np.random.normal(72, 8, n_samples)
    education = np.random.normal(14, 3, n_samples)
    mmse = np.random.normal(26, 4, n_samples)
    cdr = np.random.choice([0, 0.5, 1, 2], n_samples, p=[0.4, 0.3, 0.2, 0.1])
    apoe_e4 = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1])
    
    # Generate AD status
    logit = -5 + 0.05*(ages-65) - 0.1*education - 0.3*mmse + 1.5*cdr + 1.2*apoe_e4 + np.random.normal(0, 1, n_samples)
    ad_status = (np.random.random(n_samples) < 1/(1+np.exp(-logit))).astype(int)
    
    # Clinical data
    clinical_df = pd.DataFrame({
        'IID': [f'SUBJ_{i:04d}' for i in range(n_samples)],
        'age': ages, 'education_years': education, 'mmse_score': mmse,
        'cdr_score': cdr, 'gender': np.random.choice(['M', 'F'], n_samples),
        'AD_status': ad_status
    })
    
    # Add missing values
    for col in ['mmse_score', 'education_years']:
        clinical_df.loc[np.random.random(n_samples) < 0.05, col] = np.nan
    
    clinical_path = os.path.join(save_dir, 'clinical_data.tsv')
    clinical_df.to_csv(clinical_path, sep='\\t', index=False)
    
    # Imaging features
    imaging_features = np.random.randn(n_samples, 50)
    for i in range(10):
        imaging_features[:, i] += ad_status * np.random.uniform(0.5, 1.5)
    
    imaging_path = os.path.join(save_dir, 'imaging_features.npz')
    np.savez(imaging_path, mri_features=imaging_features,
             sample_ids=np.array([f'SUBJ_{i:04d}' for i in range(n_samples)]))
    
    # Genetic data
    genotypes = np.random.choice([0, 1, 2], size=(n_samples, 500), p=[0.6, 0.3, 0.1]).astype(float)
    genotypes[:, 0] = apoe_e4
    genotypes[np.random.random(genotypes.shape) < 0.01] = np.nan
    
    genetic_path = os.path.join(save_dir, 'genetic_data.npz')
    np.savez(genetic_path, genotypes=genotypes,
             sample_ids=np.array([f'SUBJ_{i:04d}' for i in range(n_samples)]),
             snp_ids=np.array(['rs429358'] + [f'rs{np.random.randint(1000000, 9999999)}' for _ in range(499)]))
    
    print(f"✓ Created synthetic data: {n_samples} samples, AD cases={ad_status.sum()} ({100*ad_status.mean():.1f}%)")
    return clinical_path, imaging_path, genetic_path

# Create synthetic data for demo
clinical_path, imaging_path, genetic_path = create_synthetic_data()
"""

# ==============================================================================
# CELL 12: Run Full Pipeline
# ==============================================================================
"""
# Initialize components
data_loader = MultimodalDataLoader(config)
preprocessor = Preprocessor(config)
xgb_trainer = XGBoostTrainer(config)
nn_trainer = NeuralNetTrainer(config)
evaluator = Evaluator(config)
shap_analyzer = SHAPAnalyzer(config)

# Load data
clinical_df = data_loader.load_clinical_data(clinical_path)
imaging_dict = data_loader.load_npz_data(imaging_path)
data_loader.load_npz_genetic_data(genetic_path)

# Combine all modalities
combined_df = data_loader.combine_multimodal_data(target_column='AD_status')

# Preprocess
data = preprocessor.preprocess(combined_df, target_column='AD_status')

# Train models
xgb_trainer.train(data)
nn_trainer.train(data)

# Create and optimize ensemble
ensemble = EnsembleClassifier(xgb_trainer, nn_trainer)
ensemble.optimize_weights(data['X_val'], data['y_val'])

# Evaluate
evaluator.evaluate(xgb_trainer, data, 'XGBoost')
evaluator.evaluate(nn_trainer, data, 'Neural Network')
evaluator.evaluate(ensemble, data, 'Ensemble')

# Plot results
evaluator.plot_results(os.path.join(config.output_dir, 'evaluation_results.png'))
"""

# ==============================================================================
# CELL 13: SHAP Analysis
# ==============================================================================
"""
# SHAP analysis
X_sample = data['X_test'][:200]
shap_analyzer.analyze_xgboost(xgb_trainer.model, X_sample, data['feature_names'])

# Get feature importance
importance_df = shap_analyzer.get_feature_importance(top_n=20)
print("\\nTop 20 Most Important Features:")
print(importance_df.to_string())

# Plot SHAP summary
shap_analyzer.plot_summary(X_sample, os.path.join(config.output_dir, 'shap_summary.png'))

# Save importance
importance_df.to_csv(os.path.join(config.output_dir, 'feature_importance.csv'), index=False)
"""

# ==============================================================================
# CELL 14: Training History Plot
# ==============================================================================
"""
# Plot neural network training history
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(nn_trainer.history['train_loss'], label='Train')
axes[0].plot(nn_trainer.history['val_loss'], label='Validation')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training History - Loss')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(nn_trainer.history['train_acc'], label='Train')
axes[1].plot(nn_trainer.history['val_acc'], label='Validation')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training History - Accuracy')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(config.output_dir, 'training_history.png'), dpi=150, bbox_inches='tight')
plt.show()
"""

# ==============================================================================
# CELL 15: Save Models
# ==============================================================================
"""
# Save models
xgb_trainer.save(os.path.join(config.model_dir, 'xgboost_model.json'))
nn_trainer.save(os.path.join(config.model_dir, 'neural_net_model.pt'))

# Save preprocessor
with open(os.path.join(config.model_dir, 'preprocessor.pkl'), 'wb') as f:
    pickle.dump({
        'scaler': preprocessor.scaler,
        'feature_names': data['feature_names'],
        'feature_selector': preprocessor.feature_selector
    }, f)

# Save results
with open(os.path.join(config.output_dir, 'evaluation_results.json'), 'w') as f:
    json.dump(evaluator.results, f, indent=2)

print("\\n" + "="*60)
print("PIPELINE COMPLETE!")
print("="*60)
print(f"Results saved to: {config.output_dir}")
print(f"Models saved to: {config.model_dir}")
"""

# ==============================================================================
# CELL 16: Load Your Actual Hackathon Data
# ==============================================================================
"""
# ============================================================
# REPLACE THIS SECTION WITH YOUR ACTUAL DATA PATHS
# ============================================================

# Example for loading your actual hackathon data:

# clinical_path = '/path/to/your/clinical_data.tsv'  # or .csv
# imaging_path = '/path/to/your/imaging_features.npz'
# genetic_path = '/path/to/your/genetic_data'  # .bed/.bim/.fam for PLINK or .npz

# For BED files:
# data_loader.load_bed_genetic_data(genetic_path)

# For NPZ genetic files:
# data_loader.load_npz_genetic_data(genetic_path)

# Then run:
# combined_df = data_loader.combine_multimodal_data(target_column='YOUR_TARGET_COLUMN')
# data = preprocessor.preprocess(combined_df, target_column='YOUR_TARGET_COLUMN')
# ... continue with training
"""

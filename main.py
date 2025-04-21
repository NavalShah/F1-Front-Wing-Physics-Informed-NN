import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import joblib

#############################################
# 1. DATASET DEFINITION
#############################################

class WingDataset(Dataset):
    def __init__(self, forces_csv1, forces_csv2, coeff_csv, common_key=None):
        """
        forces_csv1, forces_csv2: Paths to CSV files containing forces/moments.
        coeff_csv: Path to CSV file containing aerodynamic coefficients.
        common_key: Column name to merge by (e.g., 'time' or 'case_id'). 
                    If None, rows are assumed to correspond by order.
        """
        df1 = pd.read_csv(forces_csv1)
        df2 = pd.read_csv(forces_csv2)
        
        # Merge forces data based on a common key if available or by row order.
        if common_key is not None and common_key in df1.columns and common_key in df2.columns:
            forces_df = pd.merge(df1, df2, on=common_key)
        else:
            forces_df = pd.concat([df1.reset_index(drop=True), df2.reset_index(drop=True)], axis=1)
        
        coeff_df = pd.read_csv(coeff_csv)
        if common_key is not None and common_key in coeff_df.columns:
            self.data = pd.merge(forces_df, coeff_df, on=common_key)
        else:
            self.data = pd.concat([forces_df.reset_index(drop=True), coeff_df.reset_index(drop=True)], axis=1)
        
        # Define columns: 12 inputs (forces & moments) and 2 target coefficients.
        self.feature_cols = [
            'PRESSURE_FORCE_X', 'PRESSURE_FORCE_Y', 'PRESSURE_FORCE_Z',
            'VISCOUS_FORCE_X', 'VISCOUS_FORCE_Y', 'VISCOUS_FORCE_Z',
            'PRESSURE_MOMENT_X', 'PRESSURE_MOMENT_Y', 'PRESSURE_MOMENT_Z',
            'VISCOUS_MOMENT_X', 'VISCOUS_MOMENT_Y', 'VISCOUS_MOMENT_Z'
        ]
        self.target_cols = ['FORCE_COEFFICIENT_CD', 'FORCE_COEFFICIENT_CL']
        
        missing_features = set(self.feature_cols) - set(self.data.columns)
        missing_targets = set(self.target_cols) - set(self.data.columns)
        if missing_features or missing_targets:
            raise ValueError(f"Missing columns. Features missing: {missing_features}, Targets missing: {missing_targets}")
        
        # Standardize features and targets.
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.X = self.scaler_X.fit_transform(self.data[self.feature_cols])
        self.y = self.scaler_y.fit_transform(self.data[self.target_cols])
        
        # For physics-informed loss: use measured forces as provided in CSV.
        # Here we assume that PRESSURE_FORCE_X represents drag force and PRESSURE_FORCE_Y represents lift force.
        self.measured_drag = self.data[['PRESSURE_FORCE_X']].values.astype(np.float32)
        self.measured_lift = self.data[['PRESSURE_FORCE_Y']].values.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.X[idx], dtype=torch.float32)
        targets = torch.tensor(self.y[idx], dtype=torch.float32)
        sample = {
            'features': features,
            'targets': targets,
            'measured_drag': torch.tensor(self.measured_drag[idx][0], dtype=torch.float32),
            'measured_lift': torch.tensor(self.measured_lift[idx][0], dtype=torch.float32)
        }
        return sample

#############################################
# 2. MODEL DEFINITION (Enhanced Architecture)
#############################################

class PINN(nn.Module):
    def __init__(self, input_dim=12, output_dim=2):
        super(PINN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)

#############################################
# 3. LOSS FUNCTIONS
#############################################

def physics_loss(outputs, measured_drag, measured_lift, rho=1.225, v=50.0):
    """
    Computes a physics-informed loss.
    Parameters:
      outputs: predicted coefficients tensor of shape [batch, 2] (Cd, Cl).
      measured_drag, measured_lift: measured forces from CSV.
    Physics relation: F = 0.5 * rho * v² * C.
    """
    Cd_pred = outputs[:, 0]
    Cl_pred = outputs[:, 1]
    
    # Compute predicted forces from aerodynamic relations.
    drag_pred = 0.5 * rho * (v ** 2) * Cd_pred
    lift_pred = 0.5 * rho * (v ** 2) * Cl_pred
    
    loss_drag = torch.mean((measured_drag - drag_pred) ** 2)
    loss_lift = torch.mean((measured_lift - lift_pred) ** 2)
    return loss_drag + loss_lift

#############################################
# 4. TRAINING FUNCTION WITH TRAIN/VALID SPLIT
#############################################

def train_model(forces_csv1, forces_csv2, coeff_csv, common_key=None,
                num_epochs=200, batch_size=32, lr=1e-3, lambda_phys=0.1, val_split=0.2):
    # Load the full dataset.
    full_dataset = WingDataset(forces_csv1, forces_csv2, coeff_csv, common_key=common_key)
    
    # Create train/validation split.
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=val_split, random_state=42)
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = PINN()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    
    print("Starting Training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad()
            features = batch['features']
            targets = batch['targets']
            outputs = model(features)
            
            loss_reg = mse_loss(outputs, targets)
            measured_drag = batch['measured_drag']
            measured_lift = batch['measured_lift']
            loss_phys = physics_loss(outputs, measured_drag, measured_lift)
            
            loss = loss_reg + lambda_phys * loss_phys
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation evaluation.
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features']
                targets = batch['targets']
                outputs = model(features)
                loss_reg = mse_loss(outputs, targets)
                measured_drag = batch['measured_drag']
                measured_lift = batch['measured_lift']
                loss_phys = physics_loss(outputs, measured_drag, measured_lift)
                loss = loss_reg + lambda_phys * loss_phys
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
    
    # Save model and scalers.
    torch.save(model.state_dict(), 'pinn_model_enhanced.pt')
    joblib.dump(full_dataset.scaler_X, 'scaler_X.pkl')
    joblib.dump(full_dataset.scaler_y, 'scaler_y.pkl')
    
    return model, full_dataset, train_dataset, val_dataset

#############################################
# 5. EVALUATION FUNCTION WITH METRIC CALCULATION
#############################################

def evaluate_model(model, dataset, split_name="Full dataset"):
    """
    Evaluates the model on a dataset (training, validation, or full) and computes:
      - Mean Squared Error (MSE)
      - R² Score for Cd and Cl
    """
    model.eval()
    y_true_all = []
    y_pred_all = []
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    scaler_y = joblib.load('scaler_y.pkl')
    
    with torch.no_grad():
        for batch in loader:
            features = batch['features']
            targets = batch['targets']
            outputs = model(features)
            
            # Inverse transform predictions and targets.
            preds = scaler_y.inverse_transform(outputs.cpu().numpy())
            targets_inv = scaler_y.inverse_transform(targets.cpu().numpy())
            
            y_pred_all.append(preds)
            y_true_all.append(targets_inv)
    
    y_pred_all = np.vstack(y_pred_all)
    y_true_all = np.vstack(y_true_all)
    
    mse_cd = mean_squared_error(y_true_all[:, 0], y_pred_all[:, 0])
    mse_cl = mean_squared_error(y_true_all[:, 1], y_pred_all[:, 1])
    r2_cd  = r2_score(y_true_all[:, 0], y_pred_all[:, 0])
    r2_cl  = r2_score(y_true_all[:, 1], y_pred_all[:, 1])
    
    print(f"\nEvaluation Metrics on {split_name}:")
    print(f"Drag Coefficient (Cd): MSE = {mse_cd:.4f}, R² = {r2_cd:.4f}")
    print(f"Lift Coefficient (Cl): MSE = {mse_cl:.4f}, R² = {r2_cl:.4f}")
    
    return y_true_all, y_pred_all

#############################################
# 6. MAIN EXECUTION
#############################################

if __name__ == '__main__':
    # Set paths to your CSV files.
    forces_csv1 = "Forces_and_moments_1 (1).csv"
    forces_csv2 = "Forces_and_moments_1 (2).csv"
    coeff_csv   = "Force_and_moment_coefficients_2.csv"
    
    # Set common_key if available (or use None).
    common_key = None
    
    # Train the model with the updated hyperparameters.
    model, full_dataset, train_dataset, val_dataset = train_model(
        forces_csv1, forces_csv2, coeff_csv, common_key=common_key,
        num_epochs=200, batch_size=32, lr=1e-3, lambda_phys=0.1, val_split=0.2
    )
    
    # Evaluate on Training, Validation, and Full dataset.
    evaluate_model(model, train_dataset, split_name="Training Set")
    evaluate_model(model, val_dataset, split_name="Validation Set")
    evaluate_model(model, full_dataset, split_name="Full Dataset")

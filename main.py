import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# Custom dataset for handling three CSV files
class WingDataset(Dataset):
    def __init__(self, forces_csv1, forces_csv2, coeff_csv, common_key=None):
        """
        forces_csv1, forces_csv2: Paths to CSV files containing forces/moments data.
        coeff_csv: Path to CSV file containing the aerodynamic coefficients.
        common_key: The column name to use for merging (e.g., 'time' or 'case_id'). 
                    If None, we assume rows correspond by order.
        """
        # Load the forces data from both files
        df1 = pd.read_csv(forces_csv1)
        df2 = pd.read_csv(forces_csv2)
        
        # Merge forces datasets
        if common_key is not None and common_key in df1.columns and common_key in df2.columns:
            forces_df = pd.merge(df1, df2, on=common_key)
        else:
            # If no common key, assume row order matches and join side-by-side
            forces_df = pd.concat([df1.reset_index(drop=True), df2.reset_index(drop=True)], axis=1)
        
        # Load the coefficients CSV
        coeff_df = pd.read_csv(coeff_csv)
        
        # Merge forces and coefficients
        if common_key is not None and common_key in coeff_df.columns:
            self.data = pd.merge(forces_df, coeff_df, on=common_key)
        else:
            # If no explicit key, assume row order corresponds
            self.data = pd.concat([forces_df.reset_index(drop=True), coeff_df.reset_index(drop=True)], axis=1)
            
        # Define the feature columns (use only one set for forces and moments)
        self.feature_cols = ['PRESSURE_FORCE_X', 'PRESSURE_FORCE_Y', 'PRESSURE_FORCE_Z',
                             'VISCOUS_FORCE_X', 'VISCOUS_FORCE_Y', 'VISCOUS_FORCE_Z',
                             'PRESSURE_MOMENT_X', 'PRESSURE_MOMENT_Y', 'PRESSURE_MOMENT_Z',
                             'VISCOUS_MOMENT_X', 'VISCOUS_MOMENT_Y', 'VISCOUS_MOMENT_Z']
        # Define the target columns (aerodynamic coefficients)
        self.target_cols = ['FORCE_COEFFICIENT_CD', 'FORCE_COEFFICIENT_CL']
        
        # Ensure the columns exist in the data – adjust these if your CSV column names differ
        missing_features = set(self.feature_cols) - set(self.data.columns)
        missing_targets = set(self.target_cols) - set(self.data.columns)
        if missing_features or missing_targets:
            raise ValueError(f"Missing columns in CSV data. Features missing: {missing_features}, Targets missing: {missing_targets}")
        
        # Normalize inputs and outputs
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.X = self.scaler_X.fit_transform(self.data[self.feature_cols])
        self.y = self.scaler_y.fit_transform(self.data[self.target_cols])

        # (Optional) If you have measured forces to use in the physics loss, add them here.
        # For example, if your CSV includes 'MEASURED_DRAG' and 'MEASURED_LIFT' columns,
        # you can scale and store them similarly. For now, we use dummy values.
        # Here, we assume a placeholder that simply uses one of the forces as a proxy.
        if 'PRESSURE_FORCE_X' in self.data.columns:
            self.measured_drag = self.scaler_y.fit_transform(self.data[['PRESSURE_FORCE_X']])
        else:
            self.measured_drag = None
        
        if 'PRESSURE_FORCE_Y' in self.data.columns:
            self.measured_lift = self.scaler_y.fit_transform(self.data[['PRESSURE_FORCE_Y']])
        else:
            self.measured_lift = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.X[idx], dtype=torch.float32)
        targets = torch.tensor(self.y[idx], dtype=torch.float32)
        sample = {
            'features': features,
            'targets': targets
        }
        # Optionally include measured forces for the physics loss
        if self.measured_drag is not None and self.measured_lift is not None:
            sample['measured_drag'] = torch.tensor(self.measured_drag[idx][0], dtype=torch.float32)
            sample['measured_lift'] = torch.tensor(self.measured_lift[idx][0], dtype=torch.float32)
        return sample

# Simple neural network model (PINN)
class PINN(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=64, output_dim=2):
        super(PINN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)

# Physics-informed loss function
def physics_loss(outputs, measured_drag, measured_lift, rho=1.225, v=50.0):
    """
    outputs: tensor with predictions [Cd, Cl]
    measured_drag, measured_lift: measured or estimated forces (or proxies) corresponding to the simulation.
    """
    # Extract predicted coefficients
    Cd_pred = outputs[:, 0]
    Cl_pred = outputs[:, 1]
    
    # Predicted forces from aerodynamic relation: F = 0.5 * rho * v^2 * C
    drag_pred = 0.5 * rho * v**2 * Cd_pred
    lift_pred = 0.5 * rho * v**2 * Cl_pred
    
    # Compute and return physics loss. If measured forces are available, penalize the difference.
    loss_drag = torch.mean((measured_drag - drag_pred) ** 2)
    loss_lift = torch.mean((measured_lift - lift_pred) ** 2)
    return loss_drag + loss_lift

# Full training procedure
def train_model(forces_csv1, forces_csv2, coeff_csv, common_key=None,
                num_epochs=100, batch_size=32, lr=1e-3, lambda_phys=0.1):
    
    dataset = WingDataset(forces_csv1, forces_csv2, coeff_csv, common_key=common_key)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = PINN()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()  # Loss between predicted coefficients and ground truth

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            features = batch['features']
            targets = batch['targets']
            outputs = model(features)
            
            # Compute standard regression loss
            loss_reg = mse_loss(outputs, targets)
            
            # Check if measured forces are available; if not, physics loss is omitted.
            if 'measured_drag' in batch and 'measured_lift' in batch:
                measured_drag = batch['measured_drag'].unsqueeze(1)  # ensure proper shape
                measured_lift = batch['measured_lift'].unsqueeze(1)
                
                # Concatenate to match the batch shape (if needed)
                measured_drag = measured_drag.squeeze()
                measured_lift = measured_lift.squeeze()
                
                loss_phys = physics_loss(outputs, measured_drag, measured_lift)
            else:
                loss_phys = 0.0

            # Combined loss
            loss = loss_reg + lambda_phys * loss_phys
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}')
    
    return model

# Example usage:
if __name__ == '__main__':
    # Replace these with the paths to your CSV files.
    forces_csv1 = "Forces_and_moments_1 (1).csv"
    forces_csv2 = "Forces_and_moments_1 (2).csv"
    coeff_csv   = "Force_and_moment_coefficients_2.csv"
    
    # If you have a common key/column, e.g., 'time' or 'case_id', set common_key accordingly:
    common_key = None  # For now, None assumes order is aligned.
    
    model = train_model(forces_csv1, forces_csv2, coeff_csv, common_key=common_key,
                        num_epochs=100, batch_size=32, lr=1e-3, lambda_phys=0.1)

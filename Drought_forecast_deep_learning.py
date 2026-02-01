"""
ICEEMDAN-GRU Hybrid Framework for Hydrological Drought Forecasting
Author: Based on research methodology
Purpose: Multi-horizon SSI prediction using signal decomposition + deep learning
"""

# ============================================================================
# 1. IMPORTS AND SETUP
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# For ICEEMDAN - Install: pip install EMD-signal
from PyEMD import ICEEMDAN

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# 2. DATA LOADING AND PREPROCESSING
# ============================================================================

def load_ssi_data(filepath):
    """
    Load SSI (Standardized Streamflow Index) time series data
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file with columns ['Date', 'SSI']
    
    Returns:
    --------
    df : pandas DataFrame
        Preprocessed SSI data
    """
    df = pd.read_csv(filepath, parse_dates=['Date'])
    df = df.set_index('Date')
    df = df.sort_index()
    
    # Handle missing values
    df['SSI'] = df['SSI'].interpolate(method='linear')
    
    print(f"Data loaded: {len(df)} observations")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"SSI range: [{df['SSI'].min():.2f}, {df['SSI'].max():.2f}]")
    
    return df

# ============================================================================
# 3. ICEEMDAN DECOMPOSITION
# ============================================================================

class ICEEMDANDecomposer:
    """
    ICEEMDAN (Improved Complete Ensemble EMD with Adaptive Noise) Decomposer
    """
    
    def __init__(self, num_imfs=None, max_imf=10):
        """
        Parameters:
        -----------
        num_imfs : int, optional
            Number of IMFs to extract (None = automatic)
        max_imf : int
            Maximum number of IMFs to extract
        """
        self.num_imfs = num_imfs
        self.max_imf = max_imf
        self.iceemdan = None
        self.imfs = None
        self.residual = None
        
    def decompose(self, signal):
        """
        Decompose signal using ICEEMDAN
        
        Parameters:
        -----------
        signal : array-like
            Input time series (SSI values)
        
        Returns:
        --------
        imfs : ndarray
            Intrinsic Mode Functions (shape: [n_imfs, n_samples])
        residual : ndarray
            Residual trend component
        """
        print("\n" + "="*60)
        print("ICEEMDAN Decomposition")
        print("="*60)
        
        # Initialize ICEEMDAN
        self.iceemdan = ICEEMDAN(trials=100, max_imf=self.max_imf)
        
        # Perform decomposition
        print("Decomposing signal...")
        imfs_and_residue = self.iceemdan(signal)
        
        # Separate IMFs and residual
        self.imfs = imfs_and_residue[:-1]  # All except last
        self.residual = imfs_and_residue[-1]  # Last component
        
        n_imfs = len(self.imfs)
        print(f"✓ Decomposition complete: {n_imfs} IMFs extracted")
        
        # Print IMF statistics
        print("\nIMF Statistics:")
        print("-" * 60)
        for i, imf in enumerate(self.imfs):
            print(f"IMF-{i+1}: Mean={imf.mean():.4f}, Std={imf.std():.4f}, "
                  f"Range=[{imf.min():.4f}, {imf.max():.4f}]")
        print(f"Residual: Mean={self.residual.mean():.4f}, "
              f"Std={self.residual.std():.4f}")
        
        return self.imfs, self.residual
    
    def visualize_decomposition(self, original_signal, save_path=None):
        """
        Visualize ICEEMDAN decomposition results
        """
        n_imfs = len(self.imfs)
        fig, axes = plt.subplots(n_imfs + 2, 1, figsize=(15, 2*(n_imfs+2)))
        
        # Original signal
        axes[0].plot(original_signal, 'b-', linewidth=1)
        axes[0].set_ylabel('Original\nSSI', fontsize=10, fontweight='bold')
        axes[0].set_title('ICEEMDAN Decomposition of SSI Time Series', 
                          fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # IMFs
        for i, imf in enumerate(self.imfs):
            axes[i+1].plot(imf, 'g-', linewidth=0.8)
            axes[i+1].set_ylabel(f'IMF-{i+1}', fontsize=10, fontweight='bold')
            axes[i+1].grid(True, alpha=0.3)
        
        # Residual
        axes[-1].plot(self.residual, 'r-', linewidth=1)
        axes[-1].set_ylabel('Residual', fontsize=10, fontweight='bold')
        axes[-1].set_xlabel('Time Index', fontsize=11)
        axes[-1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Decomposition plot saved: {save_path}")
        
        plt.show()

# ============================================================================
# 4. DATA PREPARATION FOR GRU
# ============================================================================

def create_sequences(data, n_steps_in, n_steps_out=1):
    """
    Create input-output sequences for time series forecasting
    
    Parameters:
    -----------
    data : array-like
        Time series data
    n_steps_in : int
        Number of time steps for input (lookback window)
    n_steps_out : int
        Number of time steps to forecast (forecast horizon)
    
    Returns:
    --------
    X : ndarray
        Input sequences (shape: [n_samples, n_steps_in, 1])
    y : ndarray
        Target values (shape: [n_samples, n_steps_out])
    """
    X, y = [], []
    
    for i in range(len(data) - n_steps_in - n_steps_out + 1):
        X.append(data[i:(i + n_steps_in)])
        y.append(data[(i + n_steps_in):(i + n_steps_in + n_steps_out)])
    
    X = np.array(X).reshape(-1, n_steps_in, 1)
    y = np.array(y)
    
    if n_steps_out == 1:
        y = y.flatten()
    
    return X, y

def prepare_imf_data(imfs, residual, n_steps_in, n_steps_out, 
                     train_ratio=0.7, val_ratio=0.15):
    """
    Prepare training, validation, and test sets for each IMF component
    
    Parameters:
    -----------
    imfs : list of arrays
        IMF components from ICEEMDAN
    residual : array
        Residual component
    n_steps_in : int
        Lookback window
    n_steps_out : int
        Forecast horizon (1 for t+1, 3 for t+3)
    train_ratio : float
        Proportion of data for training
    val_ratio : float
        Proportion of data for validation
    
    Returns:
    --------
    components_data : list of dicts
        Data dictionaries for each component
    scalers : list
        Fitted scalers for each component
    """
    n_total = len(imfs[0])
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    components_data = []
    scalers = []
    
    # Combine IMFs and residual
    all_components = list(imfs) + [residual]
    
    for idx, component in enumerate(all_components):
        # Scale data
        scaler = MinMaxScaler(feature_range=(-1, 1))
        component_scaled = scaler.fit_transform(component.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = create_sequences(component_scaled, n_steps_in, n_steps_out)
        
        # Split into train/val/test
        X_train = X[:n_train]
        y_train = y[:n_train]
        
        X_val = X[n_train:n_train+n_val]
        y_val = y[n_train:n_train+n_val]
        
        X_test = X[n_train+n_val:]
        y_test = y[n_train+n_val:]
        
        component_data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'name': f'IMF-{idx+1}' if idx < len(imfs) else 'Residual'
        }
        
        components_data.append(component_data)
        scalers.append(scaler)
    
    print(f"\n✓ Data prepared for {len(components_data)} components")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")
    
    return components_data, scalers

# ============================================================================
# 5. GRU MODEL ARCHITECTURE
# ============================================================================

def build_gru_model(n_steps_in, n_steps_out=1, units=64, dropout=0.3, 
                    learning_rate=0.001):
    """
    Build GRU model for IMF/residual forecasting
    
    Parameters:
    -----------
    n_steps_in : int
        Input sequence length
    n_steps_out : int
        Output sequence length
    units : int
        Number of GRU units
    dropout : float
        Dropout rate
    learning_rate : float
        Learning rate for Adam optimizer
    
    Returns:
    --------
    model : keras Model
        Compiled GRU model
    """
    model = Sequential([
        GRU(units=units, 
            activation='tanh',
            return_sequences=False,
            input_shape=(n_steps_in, 1)),
        Dropout(dropout),
        Dense(32, activation='relu'),
        Dropout(dropout/2),
        Dense(n_steps_out)
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

def train_component_models(components_data, n_steps_in, n_steps_out, 
                           epochs=100, batch_size=32):
    """
    Train GRU models for each IMF and residual component
    
    Parameters:
    -----------
    components_data : list of dicts
        Prepared data for each component
    n_steps_in : int
        Input sequence length
    n_steps_out : int
        Output sequence length
    epochs : int
        Maximum training epochs
    batch_size : int
        Batch size for training
    
    Returns:
    --------
    models : list
        Trained GRU models
    histories : list
        Training histories
    """
    print("\n" + "="*60)
    print("Training GRU Models for Each Component")
    print("="*60)
    
    models = []
    histories = []
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=15, 
                                restore_best_weights=True, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                   patience=10, min_lr=1e-7, verbose=0)
    
    for idx, comp_data in enumerate(components_data):
        print(f"\n{comp_data['name']}:")
        print("-" * 40)
        
        # Build model
        model = build_gru_model(n_steps_in, n_steps_out)
        
        # Train model
        history = model.fit(
            comp_data['X_train'], 
            comp_data['y_train'],
            validation_data=(comp_data['X_val'], comp_data['y_val']),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        # Evaluate on validation set
        val_loss, val_mae = model.evaluate(comp_data['X_val'], 
                                            comp_data['y_val'], 
                                            verbose=0)
        
        print(f"✓ Training complete")
        print(f"  Epochs trained: {len(history.history['loss'])}")
        print(f"  Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
        
        models.append(model)
        histories.append(history)
    
    return models, histories

# ============================================================================
# 6. PREDICTION AND RECONSTRUCTION
# ============================================================================

def predict_and_reconstruct(models, components_data, scalers):
    """
    Generate predictions for each component and reconstruct final SSI forecast
    
    Parameters:
    -----------
    models : list
        Trained GRU models
    components_data : list
        Component datasets
    scalers : list
        Fitted scalers
    
    Returns:
    --------
    predictions : dict
        Predictions for train, val, and test sets
    """
    predictions = {
        'train': [],
        'val': [],
        'test': []
    }
    
    for model, comp_data, scaler in zip(models, components_data, scalers):
        # Predict for each split
        for split in ['train', 'val', 'test']:
            X = comp_data[f'X_{split}']
            
            # Predict (scaled)
            pred_scaled = model.predict(X, verbose=0)
            
            # Inverse transform
            pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
            
            predictions[split].append(pred)
    
    # Reconstruct by summing all components
    reconstructed = {}
    for split in ['train', 'val', 'test']:
        reconstructed[split] = np.sum(predictions[split], axis=0)
    
    return reconstructed

# ============================================================================
# 7. EVALUATION METRICS
# ============================================================================

def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive evaluation metrics
    
    Returns:
    --------
    metrics : dict
        Dictionary of evaluation metrics
    """
    # Basic metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Nash-Sutcliffe Efficiency (NSE)
    nse = 1 - (np.sum((y_true - y_pred)**2) / 
               np.sum((y_true - np.mean(y_true))**2))
    
    # Percent Bias (PBIAS)
    pbias = 100 * np.sum(y_true - y_pred) / np.sum(y_true)
    
    # Index of Agreement (IoA)
    ioa = 1 - (np.sum((y_true - y_pred)**2) / 
               np.sum((np.abs(y_pred - np.mean(y_true)) + 
                       np.abs(y_true - np.mean(y_true)))**2))
    
    # Kling-Gupta Efficiency (KGE)
    r = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = np.std(y_pred) / np.std(y_true)
    beta = np.mean(y_pred) / np.mean(y_true)
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'NSE': nse,
        'PBIAS': pbias,
        'IoA': ioa,
        'KGE': kge
    }
    
    return metrics

def print_evaluation_results(y_test, predictions_test, horizon):
    """
    Print formatted evaluation results
    """
    metrics = calculate_metrics(y_test, predictions_test)
    
    print("\n" + "="*60)
    print(f"Test Set Performance (t+{horizon})")
    print("="*60)
    print(f"RMSE:  {metrics['RMSE']:.4f}")
    print(f"MAE:   {metrics['MAE']:.4f}")
    print(f"R²:    {metrics['R²']:.4f}")
    print(f"NSE:   {metrics['NSE']:.4f}")
    print(f"PBIAS: {metrics['PBIAS']:.2f}%")
    print(f"IoA:   {metrics['IoA']:.4f}")
    print(f"KGE:   {metrics['KGE']:.4f}")
    print("="*60)
    
    return metrics

# ============================================================================
# 8. VISUALIZATION
# ============================================================================

def plot_predictions(y_true, y_pred, title, save_path=None):
    """
    Plot actual vs predicted values
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Time series plot
    axes[0].plot(y_true, 'b-', label='Actual SSI', linewidth=1.5, alpha=0.7)
    axes[0].plot(y_pred, 'r-', label='Predicted SSI', linewidth=1.5, alpha=0.7)
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0].fill_between(range(len(y_true)), -2, 0, alpha=0.1, color='red', 
                          label='Drought Zone')
    axes[0].set_xlabel('Time Index', fontsize=12)
    axes[0].set_ylabel('SSI', fontsize=12)
    axes[0].set_title(title, fontsize=14, fontweight='bold')
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Scatter plot
    axes[1].scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
    
    # Perfect prediction line
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, 
                  label='Perfect Prediction')
    
    # R² annotation
    r2 = r2_score(y_true, y_pred)
    axes[1].text(0.05, 0.95, f'R² = {r2:.4f}', 
                  transform=axes[1].transAxes,
                  fontsize=12, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axes[1].set_xlabel('Actual SSI', fontsize=12)
    axes[1].set_ylabel('Predicted SSI', fontsize=12)
    axes[1].set_title('Scatter Plot: Actual vs Predicted', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Prediction plot saved: {save_path}")
    
    plt.show()

def plot_training_history(histories, save_path=None):
    """
    Plot training history for all component models
    """
    n_models = len(histories)
    fig, axes = plt.subplots(n_models, 2, figsize=(15, 3*n_models))
    
    if n_models == 1:
        axes = axes.reshape(1, -1)
    
    for idx, history in enumerate(histories):
        # Loss plot
        axes[idx, 0].plot(history.history['loss'], label='Training Loss')
        axes[idx, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[idx, 0].set_xlabel('Epoch')
        axes[idx, 0].set_ylabel('Loss (MSE)')
        axes[idx, 0].set_title(f'Component {idx+1} - Loss', fontweight='bold')
        axes[idx, 0].legend()
        axes[idx, 0].grid(True, alpha=0.3)
        
        # MAE plot
        axes[idx, 1].plot(history.history['mae'], label='Training MAE')
        axes[idx, 1].plot(history.history['val_mae'], label='Validation MAE')
        axes[idx, 1].set_xlabel('Epoch')
        axes[idx, 1].set_ylabel('MAE')
        axes[idx, 1].set_title(f'Component {idx+1} - MAE', fontweight='bold')
        axes[idx, 1].legend()
        axes[idx, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training history plot saved: {save_path}")
    
    plt.show()

# ============================================================================
# 9. MAIN PIPELINE
# ============================================================================

def iceemdan_gru_pipeline(ssi_data, n_steps_in=24, n_steps_out=1, 
                          train_ratio=0.7, val_ratio=0.15,
                          epochs=100, batch_size=32,
                          save_dir='outputs'):
    """
    Complete ICEEMDAN-GRU drought forecasting pipeline
    
    Parameters:
    -----------
    ssi_data : array-like
        SSI time series data
    n_steps_in : int
        Lookback window (e.g., 24 months)
    n_steps_out : int
        Forecast horizon (1 for t+1, 3 for t+3)
    train_ratio : float
        Training data proportion
    val_ratio : float
        Validation data proportion
    epochs : int
        Maximum training epochs
    batch_size : int
        Training batch size
    save_dir : str
        Directory to save outputs
    
    Returns:
    --------
    results : dict
        Complete results dictionary
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("ICEEMDAN-GRU DROUGHT FORECASTING PIPELINE")
    print("="*60)
    print(f"Configuration:")
    print(f"  Lookback window: {n_steps_in} months")
    print(f"  Forecast horizon: t+{n_steps_out}")
    print(f"  Train/Val/Test split: {train_ratio:.0%}/{val_ratio:.0%}/"
          f"{1-train_ratio-val_ratio:.0%}")
    
    # Step 1: ICEEMDAN Decomposition
    decomposer = ICEEMDANDecomposer()
    imfs, residual = decomposer.decompose(ssi_data)
    decomposer.visualize_decomposition(ssi_data, 
                                        f'{save_dir}/iceemdan_decomposition.png')
    
    # Step 2: Prepare Data
    components_data, scalers = prepare_imf_data(
        imfs, residual, n_steps_in, n_steps_out, 
        train_ratio, val_ratio
    )
    
    # Get actual test values
    n_total = len(ssi_data)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    test_start = n_train + n_val + n_steps_in
    y_test_actual = ssi_data[test_start + n_steps_out - 1:]
    
    # Step 3: Train Models
    models, histories = train_component_models(
        components_data, n_steps_in, n_steps_out,
        epochs, batch_size
    )
    
    # Plot training history
    plot_training_history(histories, f'{save_dir}/training_history.png')
    
    # Step 4: Predict and Reconstruct
    print("\n" + "="*60)
    print("Generating Predictions")
    print("="*60)
    predictions = predict_and_reconstruct(models, components_data, scalers)
    
    # Step 5: Evaluate
    metrics = print_evaluation_results(y_test_actual, 
                                        predictions['test'], 
                                        n_steps_out)
    
    # Step 6: Visualize Results
    plot_predictions(y_test_actual, predictions['test'],
                      f'ICEEMDAN-GRU Drought Forecast (t+{n_steps_out})',
                      f'{save_dir}/predictions_test.png')
    
    # Compile results
    results = {
        'decomposer': decomposer,
        'models': models,
        'scalers': scalers,
        'predictions': predictions,
        'metrics': metrics,
        'y_test': y_test_actual,
        'histories': histories
    }
    
    print("\n✓ Pipeline completed successfully!")
    print(f"✓ Outputs saved to: {save_dir}/")
    
    return results

# ============================================================================
# 10. EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Generate synthetic SSI data (replace with your actual data)
    print("Generating example SSI data...")
    
    # Simulate SSI time series (408 months ≈ 34 years)
    np.random.seed(42)
    t = np.arange(408)
    
    # Combine multiple components to simulate realistic SSI
    seasonal = 0.5 * np.sin(2 * np.pi * t / 12)  # Annual cycle
    trend = -0.0005 * t  # Slight declining trend
    enso = 0.3 * np.sin(2 * np.pi * t / 36)  # ENSO-like 3-year cycle
    noise = np.random.normal(0, 0.3, len(t))
    
    ssi_synthetic = seasonal + trend + enso + noise
    
    # Create DataFrame
    dates = pd.date_range('1982-01-01', periods=len(ssi_synthetic), freq='MS')
    df_ssi = pd.DataFrame({'Date': dates, 'SSI': ssi_synthetic})
    
    print(f"✓ Generated {len(df_ssi)} months of synthetic SSI data")
    
    # ========================================================================
    # RUN PIPELINE FOR t+1 (One-month ahead forecast)
    # ========================================================================
    print("\n" + "#"*60)
    print("# ONE-MONTH AHEAD FORECASTING (t+1)")
    print("#"*60)
    
    results_t1 = iceemdan_gru_pipeline(
        ssi_data=df_ssi['SSI'].values,
        n_steps_in=24,  # 24-month lookback
        n_steps_out=1,  # 1-month ahead
        train_ratio=0.7,
        val_ratio=0.15,
        epochs=100,
        batch_size=32,
        save_dir='outputs/t1_forecast'
    )
    
    # ========================================================================
    # RUN PIPELINE FOR t+3 (Three-month ahead forecast)
    # ========================================================================
    print("\n" + "#"*60)
    print("# THREE-MONTH AHEAD FORECASTING (t+3)")
    print("#"*60)
    
    results_t3 = iceemdan_gru_pipeline(
        ssi_data=df_ssi['SSI'].values,
        n_steps_in=24,  # 24-month lookback
        n_steps_out=3,  # 3-months ahead
        train_ratio=0.7,
        val_ratio=0.15,
        epochs=100,
        batch_size=32,
        save_dir='outputs/t3_forecast'
    )
    
    # ========================================================================
    # COMPARISON TABLE
    # ========================================================================
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON: t+1 vs t+3")
    print("="*60)
    
    comparison_df = pd.DataFrame({
        'Metric': ['RMSE', 'MAE', 'R²', 'NSE', 'PBIAS', 'IoA', 'KGE'],
        't+1': [results_t1['metrics'][m] for m in 
                ['RMSE', 'MAE', 'R²', 'NSE', 'PBIAS', 'IoA', 'KGE']],
        't+3': [results_t3['metrics'][m] for m in 
                ['RMSE', 'MAE', 'R²', 'NSE', 'PBIAS', 'IoA', 'KGE']]
    })
    
    print(comparison_df.to_string(index=False))
    print("="*60)
    
    # Save comparison
    comparison_df.to_csv('outputs/performance_comparison.csv', index=False)
    print("\n✓ Comparison table saved: outputs/performance_comparison.csv")
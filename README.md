# Drought-forecasting-Deep-Learning
Drought forecasting based on deep learning and ICEEMDAN based preprocessing
ICEEMDAN = Improved Complete Ensemble Empirical Mode Decomposition with Adaptive NoiseIt decomposes the SSI time series into:
6-7 IMFs (Intrinsic Mode Functions)

IMF-1: High-frequency noise (sub-seasonal variability, ~days to weeks)
IMF-2-3: Seasonal patterns (3-12 months, monsoon cycles)
IMF-4-5: Interannual oscillations (2-4 years, ENSO-like cycles)
IMF-6-7: Multi-year trends (4-6 years, decadal variability)

SSI_predicted = IMF-1_pred + IMF-2_pred + IMF-3_pred + IMF-4_pred 
                + IMF-5_pred + IMF-6_pred + IMF-7_pred + Residual_pred

1 Residual Component

Long-term trend (climate change signal, anthropogenic impacts)


How It Works (Simple Analogy):Imagine SSI as an audio recording with multiple voices talking simultaneously:

ICEEMDAN is like having perfect "audio filters" that isolate each voice
Each voice (IMF) speaks at a different pitch (frequency)
Once isolated, each voice is much easier to understand (predict)


What is GRU?GRU = Gated Recurrent UnitA type of Recurrent Neural Network (RNN) designed for sequential data:
Specialized for time series forecasting
Has "memory" - remembers patterns from past timesteps
Simpler than LSTM but often equally effective
Good at learning complex temporal dependencies
GRU Architecture (Simplified):Input Sequence → GRU Layer → Dense Layer → Output Prediction
     ↓              ↓              ↓              ↓
  [24 months]   [64 units]    [32 neurons]   [forecast]


For EACH component:
  1. Build GRU model (64 units, dropout layers)
  2. Train on component-specific data
  3. Use early stopping to prevent overfitting
  4. Validate and optimize

Compare predicted vs actual SSI using:
  • RMSE (error magnitude)
  • MAE (average error)
  • R² (variance explained)
  • NSE (Nash-Sutcliffe Efficiency)
  • PBIAS (bias)
  • IoA (agreement)
  • KGE (Kling-Gupta Efficiency)  

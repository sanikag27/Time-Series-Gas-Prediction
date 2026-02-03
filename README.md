# Gas Price Prediction using Deep Learning

Time series forecasting project comparing LSTM, Transformer, and ARIMA models for predicting U.S. weekly gasoline prices using over 30 years of historical data.

## Project Overview

This project explores the use of deep learning models to forecast weekly U.S. gasoline prices using data from April 1993 to April 2025 (1,672 observations). Motivated by the increasing need for accurate energy price forecasting, especially amid economic shocks like the 2008 financial crisis and COVID-19, we implemented two modern deep learning models—Long Short-Term Memory (LSTM) and Transformer-based neural networks—alongside a classical ARIMA baseline.

## Problem Statement

Consumers face challenges with:
- Confusing vehicle comparisons and hidden long-term costs
- Lack of transparency in fuel price trends
- Uncertainty about future gas prices when making vehicle purchase decisions
- Need for trustworthy forecasting tools

## Dataset

**Source:** U.S. Energy Information Administration (EIA)  
**Period:** April 1993 - April 2025  
**Size:** 1,672 weekly observations  
**Features:** National average price (USD per gallon) of regular gasoline

### Data Characteristics
- Clear upward trend over three decades
- Disruptions from real-world events (2008 crisis, COVID-19)
- No strong seasonal patterns (confirmed by ACF/PACF analysis)
- First-order differencing applied for stationarity

## Models Implemented

### 1. ARIMA (Baseline)
**Configuration:** ARIMA(1,1,4) selected via auto ARIMA

**Approach:**
- Traditional fixed-horizon forecast (trained on first 90%, predicted last 10%)
- Sliding window strategy with point-level predictions for fair comparison

**Strengths:**
- Interpretable and computationally efficient
- Solid performance under stable conditions

**Limitations:**
- Limited flexibility in capturing volatility
- Multi-step predictions converge toward mean

### 2. LSTM (Long Short-Term Memory)
**Architecture:** 
- Bidirectional, two-layer LSTM
- 64-128 hidden units with dropout (0.2)
- PyTorch implementation

**Approach:**
- Sliding window of 52 weeks (one year)
- Window-based prediction to minimize error accumulation
- Multi-step forecasting (up to 4 weeks ahead)

**Strengths:**
- Captures complex non-linear patterns
- Handles long-term dependencies
- Addresses vanishing gradient problem

### 3. Transformer (Time-Series)
**Architecture:**
- Attention-based architecture adapted from NLP
- 4 attention blocks
- Sliding window of 12 sequences
- 50 training epochs

**Approach:**
- Self-attention mechanism instead of recurrence
- Positional encoding for temporal information
- Multi-step predictions (4 weeks at a time)

**Strengths:**
- Parallel processing enables faster training
- Captures long-range dependencies efficiently
- Scalable and lightweight architecture

## Results

### One-Step Forecasting Metrics

| Model | MAE | RMSE | MSE |
|-------|-----|------|-----|
| **ARIMA (Baseline)** | **0.0321** | **0.0513** | **0.0026** |
| LSTM | 0.0830 | 0.1153 | 0.0133 |
| Transformer | 0.1242 | 0.1399 | 0.0196 |

### Key Findings

**Best Performance:** ARIMA achieved the lowest error metrics for one-step-ahead forecasting

**Why ARIMA Outperformed:**
- Small dataset size (~1,600 observations)
- Lack of strong seasonality
- Simpler data structure favors traditional methods
- Potential overfitting in neural network models

**Deep Learning Advantages:**
- Better multi-step forecasting capability
- More robust to dynamic patterns and noise
- Adaptable to changing market conditions
- Superior long-range dependency modeling

## Technologies Used

**Languages & Frameworks:**
- Python
- PyTorch (deep learning)
- statsmodels (ARIMA)

**Libraries:**
- pandas, numpy (data processing)
- matplotlib, seaborn (visualization)
- sklearn (evaluation metrics)

**Development Environment:**
- Jupyter Notebook
- Google Colab / Local GPU

## Key Insights

1. **Dataset Size Matters:** Small datasets (~1,600 observations) favor traditional statistical methods
2. **Model Complexity Trade-off:** LSTM and Transformer showed higher volatility, possibly due to overfitting
3. **Rolling Predictions:** Neural networks demonstrated better capability for multi-step forecasting
4. **Real-World Events:** Models struggled with sudden shocks (financial crisis, pandemic)

## Future Improvements

- **Larger Dataset:** Incorporate county-specific or daily data for more observations
- **Hyperparameter Tuning:** Extensive grid search and cross-validation
- **Modern Architectures:** Explore long-context learning models (Informer, Autoformer)
- **Feature Engineering:** Include economic indicators (oil prices, inflation, GDP)
- **Ensemble Methods:** Combine predictions from multiple models


## Course Information

**Institution:** Carnegie Mellon University  
**Course:** Neural Networks & Deep Learning
**Semester:** Spring 2025  
**Project Type:** Final Group Project

## Conclusion

While traditional ARIMA outperformed deep learning models on this specific small-scale dataset, neural network approaches (LSTM and Transformer) offer significant potential for handling complex patterns and dynamic conditions in larger, more feature-rich datasets. The project demonstrates the importance of matching model complexity to dataset characteristics and the continued relevance of classical statistical methods for certain forecasting tasks.

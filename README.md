# GPR_composite
data_analysisDentalcomposites
# **Dental Composite Degradation Analysis: Experimental Data Modeling with Gaussian Process Regression (GPR)**

## **Overview**
This project combines materials science research with machine learning to analyze and predict the degradation of dental composites in different chemical environments. The study focuses on Diametral Tensile Strength (DTS) changes over time for three composites in four storage media, using Gaussian Process Regression for predictive modeling.

## **Key Features**
- **Experimental Data Analysis**: DTS measurements for 3 composites (CU, BL, SP) across 4 environments (DW, AS, EW, MW) at 5 time points
- **Degradation Rate Calculation**: Linear regression to compute degradation slopes for each environment-composite combination
- **Machine Learning Modeling**: Gaussian Process Regression (GPR) to predict degradation rates from categorical features
- **Uncertainty Quantification**: GPR provides both predictions and uncertainty estimates
- **Cross-Validation**: 5-fold cross-validation for robust model evaluation
- **Visualization**: Comparative plots of experimental vs. predicted degradation rates

## **Scientific Background**
Dental composites degrade in oral environments through:
1. **Hydrolytic degradation** (water absorption → matrix/filler interface breakdown)
2. **Resin plasticization** (ethanol penetration → softening of polymer matrix)

This project quantifies these effects and builds predictive models for material performance.

## **Data Structure**
The dataset contains DTS measurements (MPa) for:
- **Composites**: Competence Universal (CU), Bright Light (BL), Spectrum (SP)
- **Environments**: Distilled Water (DW), Artificial Saliva (AS), Ethanol/Water (EW), Chlorhexidine Mouthwash (MW)
- **Time Points**: 1, 15, 30, 60, 90 days

## **Methodology**
### **1. Data Processing**
- Compute linear degradation slopes: `slope = d(DTS)/dt` for each environment-composite pair
- One-Hot Encoding of categorical variables (Media, Composite)

### **2. Machine Learning Model**
- **Algorithm**: Gaussian Process Regression (GPR)
- **Kernel**: RBF + WhiteKernel (for noise modeling)
- **Evaluation**: 5-fold cross-validation
- **Metrics**: R² score and RMSE (Root Mean Squared Error)

### **3. Key Advantages of GPR**
- Works well with small datasets (12 samples in this case)
- Provides uncertainty estimates with predictions
- Non-parametric and Bayesian approach
- Excellent for scientific data with inherent noise

## **Results Summary**
- **Best Performance**: Ethanol/Water environment (most degradation: -0.268 MPa/day for CU)
- **Most Stable Composite**: Spectrum (lowest degradation across all environments)
- **Model Performance**: R² = 0.983, RMSE = 0.0062 MPa/day (5-fold CV)
- **Key Finding**: Environment type has stronger effect than composite type on degradation rate

## **Installation & Requirements**

### **Python Environment Setup**
```bash
# Create new conda environment
conda create -n dental_ml python=3.10 scikit-learn=1.3 numpy pandas matplotlib

# Activate environment
conda activate dental_ml

# Or install using pip
pip install -r requirements.txt
```

### **Requirements**
- Python 3.10 (recommended)
- scikit-learn ≥ 1.3.0
- NumPy ≥ 1.21.0
- Pandas ≥ 1.3.0
- Matplotlib ≥ 3.4.0

## **Usage**

### **Run Complete Analysis**
```bash
python dental_composite_analysis.py
```

### **Code Structure**
```
dental_composite_analysis.py
├── 1. Data Loading & Slope Calculation
├── 2. Feature Encoding (One-Hot)
├── 3. GPR Model Training (5-fold CV)
├── 4. Model Evaluation & Metrics
├── 5. Visualization
└── 6. Results Export
```

### **Key Functions**
- `calculate_degradation_slopes()`: Computes linear regression slopes
- `train_gpr_model()`: Trains Gaussian Process Regressor
- `cross_validate_model()`: Performs 5-fold cross-validation
- `plot_results()`: Generates publication-quality figures

## **Output Files**
1. **`GPR_Prediction_Plot.png`**: Predicted vs. experimental degradation rates
2. **`Slope_Comparison_Plot.png`**: Bar chart comparison with uncertainty bars
3. **Console output**: Performance metrics and prediction tables

## **Scientific Applications**
This approach can be extended to:
- **Material Screening**: Predict performance of new composite formulations
- **Lifetime Prediction**: Estimate clinical service life under different conditions
- **Optimization**: Identify optimal material properties for specific environments
- **Risk Assessment**: Quantify uncertainty in material performance predictions

## **Interpretation of Results**
- **High R² (~0.98)**: Strong predictive capability of the GPR model
- **Small RMSE**: Accurate prediction of degradation rates
- **Small uncertainty bars**: High confidence in predictions, indicating reliable experimental data
- **Ethanol/Water effect**: Confirms plasticization as dominant degradation mechanism

## **Future Directions**
1. **Non-linear Modeling**: Extend to non-linear degradation kinetics
2. **Additional Features**: Incorporate material properties (filler content, degree of conversion)
3. **Multi-output GPR**: Predict multiple mechanical properties simultaneously
4. **Bayesian Optimization**: For material formulation optimization

## **Publication Reference**
This methodology is suitable for materials science journals focusing on:
- Dental materials characterization
- Machine learning in materials science
- Predictive modeling of material degradation
- Experimental data analysis with uncertainty quantification

## **Citation**
If using this approach in your research, please cite:
```
Predictive Modeling of Dental Composite Degradation Using Gaussian Process Regression. 
Journal of Dental Materials Research, 
```

## **License**
This project is available for academic and research use. For commercial applications, please contact the authors.

## **Contact**
For questions, collaborations, or technical issues:
- Email: armenarmen25@gmail.com





**Note**: This README corresponds to the complete analysis described in the accompanying paper. The code implements the methodology section of the publication and produces the figures shown in the results section.

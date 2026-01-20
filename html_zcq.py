"""
HTML交互式可视化统一入口

【功能】
混淆矩阵热力图、分类准确率、ROC曲线、箱线图等

【运行】
python html_zcq.py

【输出】
docs/index.html - 交互式可视化页面
"""

import os
import json
import base64
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 配置 - 使用相对于脚本位置的路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # 作图目录
MAC_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'mac')  # mac目录（数据和模型所在）
DATA_DIR = os.path.join(MAC_DIR, 'data')
MODEL_PATH = os.path.join(MAC_DIR, 'checkpoints', 'phase2_stage5classes.keras')
RESULTS_PATH = os.path.join(MAC_DIR, 'checkpoints', 'phase2_results.json')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'docs')  # 输出到作图/docs文件夹

# 颜色配置
COLORS = {
    'Normal': '#2E86AB', 'Ball': '#E94F37', 'Cage': '#F39C12',
    'Inner': '#9B59B6', 'Outer': '#2ECC71',
}



# ============================================================
# 混淆矩阵 HTML
# ============================================================

def compute_confusion_matrix(y_true, y_pred, n_classes=5):
    """计算混淆矩阵"""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t][p] += 1
    return cm


def create_cm_heatmap(cm, class_names, show_percentage=False):
    """创建混淆矩阵热力图（y轴翻转）"""
    cm_flipped = np.flipud(cm)
    class_names_flipped = class_names[::-1]
    
    if show_percentage:
        row_sums = cm_flipped.sum(axis=1)[:, np.newaxis]
        row_sums[row_sums == 0] = 1
        cm_display = cm_flipped.astype('float') / row_sums * 100
        text = [[f'{v:.1f}%' for v in row] for row in cm_display]
    else:
        cm_display = cm_flipped
        text = [[str(v) for v in row] for row in cm_display]
    
    return go.Heatmap(z=cm_display, x=class_names, y=class_names_flipped,
                      text=text, texttemplate='%{text}', textfont={'size': 12},
                      colorscale='Blues', showscale=False)


def load_image_as_base64(filename):
    """将图片转为base64 data URI"""
    filepath = os.path.join(SCRIPT_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            data = base64.b64encode(f.read()).decode('utf-8')
            ext = filename.split('.')[-1].lower()
            mime = 'image/jpeg' if ext in ['jpg', 'jpeg'] else f'image/{ext}'
            return f'data:{mime};base64,{data}'
    return filename  # 文件不存在则返回原路径

def generate_confusion_html():
    """生成混淆矩阵HTML（带颜色选择、数值编辑功能）"""
    print("=" * 60)
    print("混淆矩阵 HTML (增强版)")
    print("=" * 60)
    
    # 加载图片为base64
    title_bg_base64 = load_image_as_base64('title_bg.jpg')
    ai_bg_base64 = load_image_as_base64('ai_bg.jpg')
    ai_btn_base64 = load_image_as_base64('ai_btn.jpg')
    
    with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    class_names = ['Normal', 'Ball', 'Cage', 'Inner', 'Outer']
    
    # 总体混淆矩阵 - 使用自定义初始值（每类280样本，准确率>85%）
    # 生成随机误分类的混淆矩阵
    import random
    n_classes = len(class_names)
    samples_per_class = 280
    min_accuracy = 0.85
    cm_total = []
    for i in range(n_classes):
        row = [0] * n_classes
        correct = int(samples_per_class * (min_accuracy + random.uniform(0, 0.10)))  # 85%-95%正确
        row[i] = correct
        remaining = samples_per_class - correct
        # 随机分配误分类到其他类
        for _ in range(remaining):
            j = random.choice([x for x in range(n_classes) if x != i])
            row[j] += 1
        cm_total.append(row)
    
    # 分转速混淆矩阵
    confusion_by_rpm = data.get('confusion_by_rpm', {})
    cm_by_rpm = {}
    for rpm in ['400', '600', '1000']:
        if rpm in confusion_by_rpm:
            cm_by_rpm[rpm] = compute_confusion_matrix(
                confusion_by_rpm[rpm]['y_true'], confusion_by_rpm[rpm]['y_pred']).tolist()
    
    # 构建数据结构 - 图1混淆矩阵，图2柱状图，图3箱线图，图4 ROC
    import random
    
    # 默认X轴类别标签
    default_labels = ['A', 'B', 'C', 'D', 'E']
    
    # 图2: 分类准确率柱状图数据
    bar_data = {
        'labels': default_labels,
        'values': [round(random.uniform(85, 99), 1) for _ in range(5)]
    }
    
    # 图3: 箱线图数据 (每类5个统计值: min, Q1, median, Q3, max)
    boxplot_data = {
        'labels': default_labels,
        'values': [[round(random.uniform(70, 80), 1), round(random.uniform(80, 85), 1), 
                    round(random.uniform(85, 90), 1), round(random.uniform(90, 95), 1), 
                    round(random.uniform(95, 100), 1)] for _ in range(5)]
    }
    
    # 图4: ROC曲线数据
    roc_data = {
        'labels': default_labels,
        'curves': [{'fpr': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    'tpr': sorted([0] + [round(random.uniform(0.6, 1.0), 2) for _ in range(9)] + [1.0]),
                    'auc': round(random.uniform(0.9, 0.99), 3)} for _ in range(5)]
    }
    
    # 图5: 准确率箱线图数据 (每系列包含: 平均值, 最小值, 最大值) - 五分类
    errorbar_data = {
        'labels': ['A', 'B', 'C', 'D', 'E'],
        'series': [
            {'name': 'Ours', 'avg': [85.5, 88.3, 91.1, 93.8, 95.2], 'min': [80.2, 83.1, 86.5, 88.3, 90.1], 'max': [90.8, 93.5, 96.7, 98.1, 99.3], 'color': '#27ae60'},
            {'name': 'Method A', 'avg': [82.2, 85.5, 88.2, 90.0, 92.5], 'min': [77.1, 80.3, 83.5, 85.7, 87.2], 'max': [87.3, 90.7, 92.9, 94.3, 97.8], 'color': '#3498db'},
        ]
    }
    
    # 图6: 折线图数据
    line_data = {
        'xAxis': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        'series': [
            {'name': 'Series A', 'data': [150, 230, 224, 218, 135, 147, 260], 'color': '#5470c6'},
            {'name': 'Series B', 'data': [80, 122, 201, 154, 190, 130, 110], 'color': '#91cc75'}
        ]
    }
    
    # 图7: 散点图数据
    scatter_data = {
        'series': [
            {'name': 'Group A', 'data': [[10.0, 8.04], [8.07, 6.95], [13.0, 7.58], [9.05, 8.81], [11.0, 8.33], [14.0, 7.66], [13.4, 6.81], [10.0, 6.33], [14.0, 8.96], [12.5, 6.82]], 'color': '#5470c6'},
            {'name': 'Group B', 'data': [[8.0, 6.58], [8.0, 5.76], [8.0, 7.71], [8.0, 8.84], [8.0, 8.47], [8.0, 7.04], [8.0, 5.25], [19.0, 12.5], [8.0, 5.56], [8.0, 7.91]], 'color': '#ee6666'}
        ]
    }
    
    # 图8: 蜘蛛图(雷达图)数据
    radar_data = {
        'indicator': [{'name': 'Sales', 'max': 100}, {'name': 'Admin', 'max': 100}, {'name': 'IT', 'max': 100}, {'name': 'Support', 'max': 100}, {'name': 'Dev', 'max': 100}, {'name': 'Marketing', 'max': 100}],
        'series': [
            {'name': 'Budget', 'data': [80, 90, 70, 85, 95, 75], 'color': '#5470c6'},
            {'name': 'Actual', 'data': [70, 85, 65, 90, 80, 85], 'color': '#91cc75'}
        ]
    }
    
    # 图9: 双轴图数据
    dualaxis_data = {
        'xAxis': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        'series': [
            {'name': 'Revenue', 'data': [2.0, 4.9, 7.0, 23.2, 25.6, 76.7], 'type': 'bar', 'yAxisIndex': 0, 'color': '#5470c6'},
            {'name': 'Growth Rate', 'data': [2.6, 5.9, 9.0, 26.4, 28.7, 70.7], 'type': 'line', 'yAxisIndex': 1, 'color': '#ee6666'}
        ],
        'yAxis': [{'name': 'Revenue ($M)', 'position': 'left'}, {'name': 'Growth (%)', 'position': 'right'}]
    }
    
    # 图10: 面积图数据
    area_data = {
        'xAxis': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        'series': [
            {'name': 'Email', 'data': [120, 132, 101, 134, 90, 230, 210], 'color': '#5470c6'},
            {'name': 'Union', 'data': [220, 182, 191, 234, 290, 330, 310], 'color': '#91cc75'}
        ]
    }
    
    # 图11: 带状图数据
    band_data = {
        'xAxis': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        'series': [
            {'name': 'Lower', 'data': [100, 120, 110, 130, 125, 140, 150], 'color': '#5470c6'},
            {'name': 'Upper', 'data': [180, 200, 190, 210, 205, 220, 230], 'color': '#91cc75'}
        ]
    }
    
    # 图12: 等高线图数据
    contour_data = {
        'data': [[i, j, np.sin(i/5)*np.cos(j/5)*10] for i in range(20) for j in range(20)]
    }
    
    # 图13: 极坐标图数据
    polar_data = {
        'series': [
            {'name': 'Polar A', 'data': [[0, 5], [45, 8], [90, 12], [135, 7], [180, 10], [225, 6], [270, 9], [315, 11]], 'color': '#5470c6'},
        ]
    }
    
    # 图14: 3D曲面图数据(简化为热力图展示)
    surface3d_data = {
        'data': [[i, j, np.sin(np.sqrt(i*i+j*j))*10] for i in range(-10, 11) for j in range(-10, 11)]
    }
    
    # 图15: 3D散点图数据
    scatter3d_data = {
        'series': [
            {'name': '3D Points', 'data': [[np.random.randn()*5, np.random.randn()*5, np.random.randn()*5] for _ in range(50)], 'color': '#5470c6'}
        ]
    }
    
    # 图16: 3D条形图数据
    bar3d_data = {
        'data': [[i, j, (i+1)*(j+1)] for i in range(5) for j in range(5)]
    }
    
    # 图17: 直方图数据
    histogram_data = {
        'data': [np.random.randn()*15+50 for _ in range(100)],
        'bins': 15
    }
    
    # 图18: 小提琴图数据(用箱线图+密度表示)
    violin_data = {
        'categories': ['Group A', 'Group B', 'Group C', 'Group D'],
        'data': [[np.random.randn()*10+50 for _ in range(30)] for _ in range(4)]
    }
    
    # 图19: 成对关系图数据(用散点矩阵表示)
    pairplot_data = {
        'variables': ['Var1', 'Var2', 'Var3'],
        'data': [[np.random.randn()*10+50 for _ in range(3)] for _ in range(50)]
    }
    
    # 图20: Facet Grid图数据
    facet_data = {
        'categories': ['Time A', 'Time B'],
        'groups': ['Group 1', 'Group 2'],
        'data': {
            'Time A': {'Group 1': [[1,2],[2,3],[3,5],[4,4]], 'Group 2': [[1,3],[2,4],[3,3],[4,6]]},
            'Time B': {'Group 1': [[1,4],[2,5],[3,6],[4,7]], 'Group 2': [[1,2],[2,3],[3,4],[4,5]]}
        }
    }
    
    # 图21: 热力图数据(额外)
    heatmap_data = {
        'xAxis': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
        'yAxis': ['Morning', 'Noon', 'Afternoon', 'Evening'],
        'data': [[i, j, np.random.randint(0, 100)] for i in range(5) for j in range(4)]
    }
    
    # 图22: 饼图数据
    pie_data = {
        'series': [
            {'name': 'Category A', 'value': 335},
            {'name': 'Category B', 'value': 310},
            {'name': 'Category C', 'value': 234},
            {'name': 'Category D', 'value': 135},
            {'name': 'Category E', 'value': 148}
        ]
    }
    
    # 图23: 瀑布图数据
    waterfall_data = {
        'xAxis': ['Start', 'Q1', 'Q2', 'Q3', 'Q4', 'End'],
        'data': [1000, 200, -150, 300, -100, 1250]
    }
    
    matrices_data = {
        'fig1': {'name': '图1', 'subtitle': '混淆矩阵', 'type': 'confusion', 'data': cm_total},
        'fig2': {'name': '图2', 'subtitle': '分类准确率', 'type': 'bar', 'data': bar_data},
        'fig3': {'name': '图3', 'subtitle': '对比图', 'type': 'multibar', 'data': boxplot_data},
        'fig4': {'name': '图4', 'subtitle': 'ROC曲线', 'type': 'roc', 'data': roc_data},
        'fig5': {'name': '图5', 'subtitle': '误差准确率图', 'type': 'boxplot', 'data': errorbar_data},
        'fig6': {'name': '图6', 'subtitle': '折线图', 'type': 'line', 'data': line_data},
        'fig7': {'name': '图7', 'subtitle': '箱线图', 'type': 'realboxplot', 'data': violin_data},
        'fig8': {'name': '图8', 'subtitle': '蜘蛛图', 'type': 'radar', 'data': radar_data},
        'fig9': {'name': '图9', 'subtitle': '双轴图', 'type': 'dualaxis', 'data': dualaxis_data},
        'fig10': {'name': '图10', 'subtitle': '饼图', 'type': 'pie', 'data': pie_data},
    }
    
    html_content = _generate_enhanced_cm_html(matrices_data, class_names, title_bg_base64, ai_bg_base64, ai_btn_base64)
    
    # 直接输出到docs目录（GitHub Pages发布目录）
    output_path = os.path.join(OUTPUT_DIR, 'index.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"已保存: {output_path}")
    print(f"  → 公开地址: https://zcq991029.github.io/zcq-visualization/")
    print("功能: 颜色切换、数值编辑、样式控制")
    
    return output_path


def _generate_enhanced_cm_html(matrices_data, class_names, title_bg_base64='title_bg.jpg', ai_bg_base64='ai_bg.jpg', ai_btn_base64='ai_btn.jpg'):
    """生成增强版混淆矩阵HTML"""
    import json as json_lib
    
    import time
    version = str(int(time.time()))  # 使用时间戳作为版本号
    
    # 加载本地JS库（内嵌到HTML，离线可用）
    libs_dir = os.path.join(SCRIPT_DIR, 'libs')
    xlsx_js = echarts_js = plotly_js = ''
    try:
        with open(os.path.join(libs_dir, 'xlsx.min.js'), 'r', encoding='utf-8') as f:
            xlsx_js = f.read()
        with open(os.path.join(libs_dir, 'echarts.min.js'), 'r', encoding='utf-8') as f:
            echarts_js = f.read()
        with open(os.path.join(libs_dir, 'plotly.min.js'), 'r', encoding='utf-8') as f:
            plotly_js = f.read()
        print("  已加载本地JS库（离线模式）")
    except FileNotFoundError:
        print("  警告: 本地JS库不存在，使用CDN")
    
    html = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="0">
    <title>可视化仓库 v''' + version + '''</title>
''' + (f'    <script>{xlsx_js}</script>\n    <script>{echarts_js}</script>\n    <script>{plotly_js}</script>' if xlsx_js else f'''    <script src="https://cdn.jsdelivr.net/npm/xlsx@0.18.5/dist/xlsx.full.min.js?v={version}"></script>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js?v={version}"></script>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js?v={version}"></script>
    <script src="https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js?v={version}"></script>''') + '''
    <style>
        :root { --bg: #f5f5f7; --card: #fff; --text: #1d1d1f; --sub: #86868b; --accent: #0071e3; }
        * { box-sizing: border-box; }
        body { font-family: "Times New Roman", Times, serif;
               background: var(--bg); color: var(--text); margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 15px; padding: 100px 40px; background: url("''' + title_bg_base64 + '''") center/cover; color: #fff; text-shadow: 2px 2px 8px rgba(0,0,0,0.8); border-radius: 12px; min-height: 320px; display:flex; align-items:center; justify-content:center; }
        .subtitle { text-align: center; color: var(--sub); margin-bottom: 20px; }
        
        /* 全局颜色选择器 */
        .global-controls { display: flex; justify-content: center; gap: 10px; margin-bottom: 25px; flex-wrap: wrap; }
        .theme-btn { border: 1px solid #ddd; background: #fff; padding: 8px 16px; border-radius: 20px;
                     cursor: pointer; font-size: 13px; display: flex; align-items: center; gap: 6px; transition: all 0.2s; }
        .theme-btn:hover { background: #f0f0f0; }
        .theme-btn.active { border-color: var(--accent); background: #eefbff; color: var(--accent); font-weight: bold; }
        .theme-dot { width: 14px; height: 14px; border-radius: 50%; }
        
        /* 矩阵网格布局 - 单列 */
        .matrices-grid { display: flex; flex-direction: column; gap: 25px; margin: 0 auto; padding: 0 20px; align-items: center; width: 100%; overflow-x: auto; }
        
        /* 单个矩阵卡片 */
        .matrix-card { background: var(--card); border-radius: 16px; padding: 20px;
                       box-shadow: 0 2px 12px rgba(0,0,0,0.08); overflow: visible; width: fit-content; min-width: auto; max-width: none; margin: 0 auto; }
        .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }
        .card-title { font-size: 16px; font-weight: 600; }
        .card-controls { display: flex; gap: 8px; }
        .ctrl-btn { background: #f0f0f0; border: none; padding: 6px 12px; border-radius: 8px;
                    cursor: pointer; font-size: 12px; transition: all 0.2s; }
        .ctrl-btn:hover { background: #e0e0e0; }
        .ctrl-btn.active { background: var(--accent); color: #fff; }
        
        /* 样式面板 - 可拖动浮窗 */
        .style-panel { 
            display: none; 
            position: fixed; 
            top: 100px; 
            right: 20px; 
            width: 320px;
            max-height: 70vh;
            overflow-y: auto;
            background: #fff; 
            border-radius: 10px; 
            padding: 12px; 
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            z-index: 1000;
            border: 1px solid #ddd;
        }
        .style-panel.show { display: block; }
        .style-panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            margin-bottom: 8px;
            border-bottom: 1px solid #eee;
            cursor: move;
            user-select: none;
        }
        .style-panel-header span { font-weight: bold; font-size: 13px; }
        .style-panel-close { 
            background: #e74c3c; 
            color: #fff; 
            border: none; 
            border-radius: 50%; 
            width: 22px; 
            height: 22px; 
            cursor: pointer; 
            font-size: 12px;
            line-height: 22px;
            text-align: center;
        }
        .style-panel-close:hover { background: #c0392b; }
        .style-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; font-size: 12px; flex-wrap: wrap; }
        .style-row label { min-width: 70px; }
        .style-row select { padding: 4px 8px; border-radius: 4px; border: 1px solid #ddd; }
        
        /* 矩阵表格 */
        .matrix-wrapper { position: relative; display: flex; justify-content: center; }
        .axis-label { font-size: 12px; font-weight: bold; color: var(--sub); position: absolute; }
        .y-axis { transform: rotate(-90deg); left: -30px; top: 50%; }
        .x-axis { bottom: -22px; left: 50%; transform: translateX(-50%); }
        
        .matrix-table { border-collapse: collapse; }
        .matrix-table th, .matrix-table td { width: 50px; height: 50px; text-align: center;
                                              font-size: 13px; transition: background 0.3s; }
        .matrix-table td { border: 1px solid #eee; }
        .matrix-table th { background: #fff; font-weight: 600; font-size: 11px; border: none; }
        .matrix-table td { cursor: pointer; }
        .matrix-table td:hover { outline: 2px solid var(--accent); outline-offset: -2px; }
        .matrix-table td.dark { color: #fff; }
        
        /* 编辑弹窗 */
        .modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                 background: rgba(0,0,0,0.4); z-index: 1000; justify-content: center; align-items: center; }
        .modal.show { display: flex; }
        .modal-content { background: #fff; border-radius: 16px; padding: 25px; min-width: 300px;
                         box-shadow: 0 10px 40px rgba(0,0,0,0.2); }
        .modal-title { font-size: 18px; font-weight: 600; margin-bottom: 15px; }
        .modal-input { width: 100%; padding: 12px; font-size: 16px; border: 1px solid #ddd;
                       border-radius: 8px; margin-bottom: 15px; }
        .modal-btns { display: flex; gap: 10px; }
        .modal-btn { flex: 1; padding: 10px; border: none; border-radius: 8px; cursor: pointer; font-size: 14px; }
        .modal-btn.primary { background: var(--accent); color: #fff; }
        .modal-btn.secondary { background: #f0f0f0; }
        
        /* 统计信息 */
        .stats { margin-top: 12px; font-size: 12px; color: var(--sub); text-align: center; }
        
        /* 下载按钮 */
        .download-btns { display: flex; gap: 8px; justify-content: center; margin-top: 10px; }
        .dl-btn { background: #f0f0f0; border: 1px solid #ddd; padding: 5px 12px; border-radius: 6px;
                  cursor: pointer; font-size: 11px; transition: all 0.2s; }
        .dl-btn:hover { background: #e0e0e0; }
        
        /* Logo和头部 */
        .header-bar { display: flex; justify-content: space-between; align-items: center; padding: 10px 20px; background: #fff; 
                      box-shadow: 0 2px 8px rgba(0,0,0,0.1); position: fixed; top: 0; left: 0; right: 0; z-index: 100; }
        .logo-area { display: flex; align-items: center; gap: 10px; }
        .logo { height: 40px; width: auto; }
        .logo-text { font-size: 16px; font-weight: bold; color: #1a5c1a; }
        .help-btn { background: var(--accent); color: #fff; border: none; padding: 8px 16px; border-radius: 20px; 
                    cursor: pointer; font-size: 13px; transition: all 0.2s; }
        .help-btn:hover { background: #005bb5; }
        body { padding-top: 70px; }
        
        /* 水印 */
        .watermark { position: fixed; bottom: 20px; right: 20px; font-size: 48px; font-weight: bold; color: rgba(0,0,0,0.05);
                     pointer-events: none; z-index: 99; font-family: 'Times New Roman', serif; }
        
        /* AI聊天窗口 */
        .ai-toggle { position: fixed; right: 20px; bottom: 80px; width: 50px; height: 50px; border-radius: 50%;
                     background: url("''' + ai_btn_base64 + '''") center/cover; border: 2px solid #fff;
                     cursor: pointer; box-shadow: 0 4px 15px rgba(0,0,0,0.3); z-index: 150; transition: all 0.3s ease; overflow: hidden; }
        .ai-toggle.expanded { width: 300px; height: 200px; border-radius: 12px; }
        .ai-panel { position: fixed; right: 20px; bottom: 140px; width: 380px; max-height: 500px; background: #fff;
                    border-radius: 16px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); z-index: 150; display: none; flex-direction: column; }
        .ai-panel.show { display: flex; }
        .ai-header { padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #fff;
                     border-radius: 16px 16px 0 0; display: flex; justify-content: space-between; align-items: center; }
        .ai-header h3 { margin: 0; font-size: 16px; }
        .ai-close { background: none; border: none; color: #fff; font-size: 20px; cursor: pointer; }
        .ai-config { padding: 10px 15px; background: #f8f9fa; border-bottom: 1px solid #eee; font-size: 12px; }
        .ai-config input, .ai-config select { padding: 5px 8px; border: 1px solid #ddd; border-radius: 4px; font-size: 12px; }
        .ai-config input[type="password"] { width: 180px; }
        .ai-messages { flex: 1; overflow-y: auto; padding: 15px; max-height: 280px; }
        .ai-msg { margin-bottom: 12px; }
        .ai-msg.user { text-align: right; }
        .ai-msg .bubble { display: inline-block; padding: 10px 14px; border-radius: 12px; max-width: 85%; word-wrap: break-word; font-size: 13px; line-height: 1.5; }
        .ai-msg.user .bubble { background: #667eea; color: #fff; }
        .ai-msg.ai .bubble { background: #f0f0f0; color: #333; }
        .ai-msg.ai .bubble pre { margin: 8px 0 0; padding: 8px; background: #282c34; color: #abb2bf; border-radius: 6px; overflow-x: auto; font-size: 11px; }
        .ai-input-area { padding: 10px 15px; border-top: 1px solid #eee; display: flex; gap: 8px; }
        .ai-input { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 8px; font-size: 13px; resize: none; }
        .ai-send { padding: 10px 16px; background: #667eea; color: #fff; border: none; border-radius: 8px; cursor: pointer; font-size: 13px; }
        .ai-send:disabled { background: #ccc; cursor: not-allowed; }
        
        /* README弹窗 */
        .readme-modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5);
                        z-index: 200; justify-content: center; align-items: center; }
        .readme-modal.show { display: flex; }
        .readme-content { background: #fff; border-radius: 16px; padding: 30px; max-width: 700px; max-height: 80vh; 
                          overflow-y: auto; box-shadow: 0 10px 40px rgba(0,0,0,0.3); }
        .readme-content h2 { margin-top: 0; color: var(--accent); }
        .readme-content h3 { margin: 15px 0 8px; color: #333; border-bottom: 1px solid #eee; padding-bottom: 5px; }
        .readme-content ul { margin: 5px 0; padding-left: 20px; }
        .readme-content li { margin: 4px 0; }
        .readme-close { position: absolute; top: 15px; right: 20px; font-size: 24px; cursor: pointer; color: #666; }
    </style>
</head>
<body>
    <!-- Logo和水印 -->
    <div class="header-bar">
        <div class="logo-area">
            <img src="logo.png" alt="DGUT" class="logo" style="height:45px" onerror="this.style.display='none';this.nextElementSibling.style.display='block'">
            <span class="logo-text" style="display:none;font-size:14px;color:#1a5c1a;font-weight:bold">东莞理工学院</span>
        </div>
        <div style="display:flex;gap:8px;align-items:center">
            <button class="help-btn" onclick="resetToProgram()" style="background:#e74c3c">⚠️ 全局重置</button>
            <button class="help-btn" onclick="hardRefresh()" style="background:#95a5a6">🔄 刷新</button>
            <button class="help-btn" onclick="showReadme()">📖 说明</button>
        </div>
    </div>
    <div class="watermark">zcq</div>
    
    <div class="container">
        <h1>可视化仓库 - 无限扩展版</h1>
        
        <!-- 矩阵网格 -->
        <div class="matrices-grid" id="matricesGrid"></div>
    </div>
    
    <!-- 编辑弹窗 -->
    <div class="modal" id="editModal">
        <div class="modal-content">
            <div class="modal-title" id="modalTitle">编辑单元格</div>
            <input type="number" class="modal-input" id="modalInput" min="0">
            <div class="modal-btns">
                <button class="modal-btn secondary" onclick="closeModal()">取消</button>
                <button class="modal-btn primary" onclick="saveEdit()">保存</button>
            </div>
        </div>
    </div>
    
    <!-- 保存节点弹窗 -->
    <div class="modal" id="saveModal">
        <div class="modal-content">
            <div class="modal-title">💾 保存新节点</div>
            <input type="text" class="modal-input" id="checkpointName" placeholder="输入节点名称（如：调试版v1）" style="font-size:14px">
            <div class="modal-btns">
                <button class="modal-btn secondary" onclick="closeSaveModal()">取消</button>
                <button class="modal-btn primary" onclick="saveNamedCheckpoint()">保存</button>
            </div>
        </div>
    </div>
    
    <!-- 节点列表弹窗 -->
    <div class="modal" id="checkpointModal">
        <div class="modal-content" style="max-width:500px">
            <div class="modal-title">🔄 选择要恢复的节点</div>
            <div id="checkpointList" style="max-height:300px;overflow-y:auto;margin:15px 0"></div>
            <div class="modal-btns">
                <button class="modal-btn secondary" onclick="closeCheckpointList()">关闭</button>
            </div>
        </div>
    </div>
    
    <!-- AI导入弹窗 -->
    <div class="modal" id="importModal">
        <div class="modal-content" style="max-width:600px">
            <div class="modal-title" id="importModalTitle">📥 AI智能导入</div>
            <div id="importTargetRow" style="margin:15px 0;display:none">
                <label style="display:block;margin-bottom:8px;font-weight:600">1. 选择目标图:</label>
                <select id="importTargetMatrix" style="width:100%;padding:8px;border:1px solid #ddd;border-radius:6px"></select>
            </div>
            <div style="margin:15px 0">
                <label style="display:block;margin-bottom:8px;font-weight:600">粘贴数据或截图 (Ctrl+V):</label>
                <div id="pasteArea" style="width:100%;min-height:120px;padding:10px;border:2px dashed #ddd;border-radius:6px;background:#fafafa;cursor:pointer;text-align:center;line-height:100px;color:#999" onclick="document.getElementById('importDataText').focus()">
                    点击此处或按Ctrl+V粘贴截图/文本
                </div>
                <textarea id="importDataText" rows="4" placeholder="或在此输入文本数据..." style="width:100%;padding:10px;border:1px solid #ddd;border-radius:6px;font-family:monospace;font-size:12px;resize:vertical;margin-top:8px"></textarea>
            </div>
            <div style="margin:15px 0">
                <label style="display:block;margin-bottom:8px;font-weight:600">或上传文件:</label>
                <input type="file" id="importFile" accept=".txt,.csv,.json,.xlsx,.xls,.png,.jpg,.jpeg,.tif,.tiff" onchange="handleImportFile(this)" style="font-size:12px">
                <div id="imagePreview" style="display:none;margin-top:10px;max-width:300px"><img id="previewImg" style="max-width:100%;border:1px solid #ddd;border-radius:4px"></div>
            </div>
            <div id="importStatus" style="display:none;padding:10px;border-radius:6px;margin:10px 0;font-size:12px"></div>
            <div class="modal-btns">
                <button class="modal-btn secondary" onclick="closeImportModal()">取消</button>
                <button class="modal-btn primary" onclick="processAIImport()" id="importBtn">🤖 AI识别并填充</button>
            </div>
        </div>
    </div>
    
    <!-- AI聊天窗口 -->
    <button class="ai-toggle" id="aiToggleBtn" title="刚起床吗"></button>
    <div class="ai-panel" id="aiPanel">
        <div class="ai-header" id="aiHeader" style="cursor:move;background:url(''' + "'" + ai_bg_base64 + "'" + ''') center/cover;min-height:80px;padding:15px;border-radius:12px 12px 0 0">
            <h3 style="color:#fff;text-shadow:1px 1px 4px rgba(0,0,0,0.8);margin:0">🤖 刚起床吗</h3>
            <button class="ai-close" onclick="toggleAI()" style="color:#fff;text-shadow:1px 1px 2px rgba(0,0,0,0.8)">×</button>
        </div>
        <div class="ai-config">
            <div style="margin-bottom:8px;display:flex;align-items:center;gap:8px">
                <label style="min-width:55px">API Key:</label>
                <input type="password" id="aiApiKey" placeholder="sk-..." oninput="saveAIConfig()" style="flex:1">
                <button onclick="toggleApiKeyVisibility()" style="padding:4px 8px;border:1px solid #ddd;border-radius:4px;cursor:pointer;font-size:11px" id="toggleKeyBtn">👁️</button>
            </div>
            <div style="display:flex;align-items:center;gap:8px">
                <label style="min-width:55px">平台:</label>
                <select id="aiPlatform" onchange="saveAIConfig();updateModelOptions()" style="flex:1">
                    <option value="deepseek">DeepSeek</option>
                    <option value="siliconflow">硅基流动 (支持图片)</option>
                    <option value="openai">OpenAI (支持图片)</option>
                </select>
            </div>
            <div style="display:flex;align-items:center;gap:8px;margin-top:5px">
                <label style="min-width:55px">模型:</label>
                <select id="aiModel" onchange="saveAIConfig()" style="flex:1">
                    <option value="deepseek-chat">DeepSeek V3 (快速)</option>
                    <option value="deepseek-reasoner">DeepSeek R1 (推理)</option>
                </select>
            </div>
            <div id="apiKeyWarning" style="margin-top:8px;padding:8px;background:#fff3cd;border-radius:6px;font-size:11px;color:#856404;display:none">
                ⚠️ 请先输入您的 API Key 才能使用AI功能<br>
                <a href="https://platform.deepseek.com/api_keys" target="_blank" id="apiKeyLink" style="color:#0071e3">获取API Key →</a>
            </div>
        </div>
        <div class="ai-messages" id="aiMessages">
            <div class="ai-msg ai" style="text-align:center">
                <div class="bubble" style="background:#f0f7ff;color:#0071e3">
                    👋 您好！我是数据分析助手。<br>请先配置API Key，然后可以询问关于混淆矩阵的问题。
                </div>
            </div>
        </div>
        <div class="ai-input-area">
            <textarea class="ai-input" id="aiInput" rows="2" placeholder="输入问题（需先配置API Key）..." onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendAI()}"></textarea>
            <button class="ai-send" id="aiSend" onclick="sendAI()">发送</button>
        </div>
    </div>
    
    <!-- README弹窗 -->
    <div class="readme-modal" id="readmeModal" onclick="if(event.target===this)closeReadme()">
        <div class="readme-content">
            <span class="readme-close" onclick="closeReadme()">&times;</span>
            <h2>📖 SCI可视化仓库 使用说明</h2>
            
            <h3>🚀 快速入门</h3>
            <ul>
                <li><strong>点击数据</strong>：直接编辑图表数值（所有图表支持）</li>
                <li><strong>点击轴标签</strong>：直接编辑X/Y轴刻度标签</li>
                <li><strong>点击轴名称</strong>：直接编辑X/Y轴名称</li>
                <li><strong>🎨样式面板</strong>：调整字体、颜色、尺寸等样式</li>
                <li><strong>📝编辑数据</strong>：表格形式批量编辑数据</li>
                <li><strong>📥AI导入</strong>：粘贴数据或截图智能导入</li>
                <li><strong>💾节点管理</strong>：保存/恢复当前状态</li>
            </ul>
            
            <h3>📊 图表类型（共10种）</h3>
            <ul>
                <li><strong>图1 混淆矩阵</strong>：点击单元格/标签编辑，支持5种颜色主题</li>
                <li><strong>图2 柱状图</strong>：点击柱子/轴标签编辑，柱宽/颜色可调</li>
                <li><strong>图3 对比图</strong>：多组数据对比，系列数/颜色可调</li>
                <li><strong>图4 ROC曲线</strong>：点击轴名编辑，支持多条曲线</li>
                <li><strong>图5 误差条图</strong>：点击柱子/误差线编辑</li>
                <li><strong>图6 折线图</strong>：点击刻度/数据点编辑，矩形框边框</li>
                <li><strong>图7 箱线图</strong>：点击箱体编辑数据，支持Y轴范围设置</li>
                <li><strong>图8 蜘蛛图</strong>：点击数据点编辑，雷达图形式</li>
                <li><strong>图9 双轴图</strong>：点击柱/线/轴编辑，左右双Y轴</li>
                <li><strong>图10 饼图</strong>：点击扇区编辑数值，点击图例编辑名称</li>
            </ul>
            
            <h3>🎨 样式配置（点击🎨）</h3>
            <ul>
                <li><strong>图表尺寸</strong>：宽×高可调</li>
                <li><strong>坐标轴</strong>：名称/刻度标签/颜色可编辑</li>
                <li><strong>矩形框</strong>：黑色封闭边框（SCI标准）</li>
                <li><strong>字体</strong>：Times/Arial可选，黑色刻度</li>
            </ul>
            
            <h3>💾 节点管理</h3>
            <ul>
                <li><strong>保存节点</strong>：命名保存当前完整状态</li>
                <li><strong>恢复节点</strong>：列表选择精准恢复</li>
                <li><strong>完全重置</strong>：恢复程序初始值</li>
            </ul>
            
            <h3>📷 导出图片</h3>
            <ul>
                <li><strong>PNG/JPG/TIF</strong>：4倍分辨率高清导出</li>
            </ul>
            
            <h3>🔄 刷新说明</h3>
            <ul>
                <li><strong>强制刷新</strong>：Cmd+Shift+R (Mac) / Ctrl+Shift+R (Win)</li>
                <li><strong>清除缓存</strong>：点击右上角"🔄 刷新"按钮</li>
            </ul>
            
            <p style="margin-top:20px;color:#666;font-size:12px;text-align:center">
                东莞理工学院 · zcq · SCI可视化仓库 v2.0
            </p>
        </div>
    </div>
    
    <script>
        // 强制硬刷新（清除缓存）
        function hardRefresh() {
            if ('caches' in window) {
                caches.keys().then(names => { names.forEach(name => caches.delete(name)); });
            }
            localStorage.clear();
            sessionStorage.clear();
            window.location.href = window.location.href.split('?')[0] + '?t=' + Date.now();
        }
        
        // README弹窗
        function showReadme() { document.getElementById('readmeModal').classList.add('show'); }
        function closeReadme() { document.getElementById('readmeModal').classList.remove('show'); }
        // 原始数据（程序生成的初始值）
        const PROGRAM_DATA = JSON.parse(JSON.stringify(''' + json_lib.dumps(matrices_data, ensure_ascii=False) + '''));
        const STORAGE_KEY = 'confusion_matrix_data';
        const SETTINGS_KEY = 'confusion_matrix_settings';
        const CHECKPOINT_KEY = 'confusion_matrix_checkpoint';
        
        // 加载数据：优先 localStorage
        let matricesData;
        const savedData = localStorage.getItem(STORAGE_KEY);
        if (savedData) {
            try { matricesData = JSON.parse(savedData); }
            catch (e) { matricesData = JSON.parse(JSON.stringify(PROGRAM_DATA)); }
        } else {
            matricesData = JSON.parse(JSON.stringify(PROGRAM_DATA));
        }
        
        // 加载检查点列表（多节点保存）
        const CHECKPOINTS_KEY = 'confusion_matrix_checkpoints';
        let checkpointsList = [];
        const savedCheckpoints = localStorage.getItem(CHECKPOINTS_KEY);
        if (savedCheckpoints) {
            try { checkpointsList = JSON.parse(savedCheckpoints); } catch (e) { checkpointsList = []; }
        }
        
        const classNames = ''' + json_lib.dumps(class_names) + ''';
        
        // 保存全部状态到 localStorage
        function saveAllToStorage() {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(matricesData));
            localStorage.setItem(SETTINGS_KEY, JSON.stringify({ matrixThemes, matrixModes, colorModes, globalStyles, customLabels, customTitles, customSubtitles, chartStyles, boxplotSeries }));
        }
        
        // 加载设置
        function loadSettings() {
            const saved = localStorage.getItem(SETTINGS_KEY);
            if (saved) {
                try {
                    const s = JSON.parse(saved);
                    if (s.matrixThemes) Object.assign(matrixThemes, s.matrixThemes);
                    if (s.matrixModes) Object.assign(matrixModes, s.matrixModes);
                    if (s.colorModes) Object.assign(colorModes, s.colorModes);
                    if (s.globalStyles) Object.assign(globalStyles, s.globalStyles);
                    if (s.customLabels) customLabels = [...s.customLabels];
                    if (s.customTitles) Object.assign(customTitles, s.customTitles);
                    if (s.customSubtitles) Object.assign(customSubtitles, s.customSubtitles);
                    if (s.chartStyles) Object.assign(chartStyles, s.chartStyles);
                    if (s.boxplotSeries) Object.assign(boxplotSeries, s.boxplotSeries);
                } catch (e) {}
            }
            // 确保所有矩阵的colorModes都有值，默认为row
            Object.keys(matricesData).forEach(key => {
                if (!colorModes[key]) colorModes[key] = 'row';
            });
        }
        
        // 自定义矩阵标题和副标题
        let customTitles = {};
        let customSubtitles = {
            'fig1': '400rpm',
            'fig2': '',
            'fig3': '',
            'fig4': ''
        };
        
        // 图表样式配置（各图独立）
        let chartStyles = {
            'fig2': { 
                axisLabelFont: 'Times New Roman', axisLabelSize: 12, axisTickFont: 'Times New Roman', axisTickSize: 10, 
                legendFont: 'Times New Roman', legendSize: 11, chartWidth: 10, chartHeight: 5,
                xAxisName: '类别', yAxisName: '准确率(%)', barWidth: 40, barColors: ['#3498db','#e67e22','#27ae60','#e74c3c','#9b59b6','#1abc9c']
            },
            'fig3': { 
                axisLabelFont: 'Times New Roman', axisLabelSize: 12, axisTickFont: 'Times New Roman', axisTickSize: 10, 
                legendFont: 'Times New Roman', legendSize: 11, chartWidth: 10, chartHeight: 5,
                xAxisName: 'Evaluation budget', yAxisName: 'Terminal HV', barWidth: 25
            },
            'fig4': { 
                axisLabelFont: 'Times New Roman', axisLabelSize: 12, axisTickFont: 'Times New Roman', axisTickSize: 10, 
                legendFont: 'Times New Roman', legendSize: 11, chartWidth: 10, chartHeight: 5,
                xAxisName: 'FPR', yAxisName: 'TPR'
            },
            'fig5': { 
                axisLabelFont: 'Times New Roman', axisLabelSize: 12, axisTickFont: 'Times New Roman', axisTickSize: 10, 
                legendFont: 'Times New Roman', legendSize: 11, chartWidth: 10, chartHeight: 5,
                xAxisName: 'Evaluation budget', yAxisName: '准确率(%)', barWidth: 30
            }
        };
        
        // ============ 多节点保存功能 ============
        // 每个图独立的节点存储
        const CHART_CHECKPOINTS_KEY = 'chart_checkpoints';
        let chartCheckpoints = JSON.parse(localStorage.getItem(CHART_CHECKPOINTS_KEY) || '{}');
        
        function openChartSaveModal(key) {
            const chartName = matricesData[key]?.name || key;
            const name = prompt('💾 保存 ' + chartName + ' 节点\\n\\n请输入节点名称:', chartName + ' - 节点' + ((chartCheckpoints[key]?.length || 0) + 1));
            if (!name) return;
            
            if (!chartCheckpoints[key]) chartCheckpoints[key] = [];
            const checkpoint = {
                id: Date.now(),
                name: name,
                time: new Date().toLocaleString('zh-CN'),
                data: JSON.parse(JSON.stringify(matricesData[key])),
                styles: JSON.parse(JSON.stringify(chartStyles[key] || {})),
                theme: matrixThemes[key],
                mode: matrixModes[key],
                colorMode: colorModes[key]
            };
            chartCheckpoints[key].push(checkpoint);
            localStorage.setItem(CHART_CHECKPOINTS_KEY, JSON.stringify(chartCheckpoints));
            alert('✅ 节点 "' + name + '" 已保存！');
        }
        
        function openChartRestoreModal(key) {
            const chartName = matricesData[key]?.name || key;
            const list = chartCheckpoints[key] || [];
            if (list.length === 0) {
                alert('📭 ' + chartName + ' 暂无保存的节点');
                return;
            }
            
            let html = '🔄 恢复 ' + chartName + ' 节点\\n\\n';
            list.forEach((cp, i) => { html += (i+1) + '. ' + cp.name + ' (' + cp.time + ')\\n'; });
            html += '\\n请输入序号 (1-' + list.length + '):';
            
            const choice = prompt(html);
            if (!choice) return;
            const idx = parseInt(choice) - 1;
            if (isNaN(idx) || idx < 0 || idx >= list.length) {
                alert('❌ 无效的序号');
                return;
            }
            
            const cp = list[idx];
            matricesData[key] = JSON.parse(JSON.stringify(cp.data));
            chartStyles[key] = JSON.parse(JSON.stringify(cp.styles || {}));
            if (cp.theme) matrixThemes[key] = cp.theme;
            if (cp.mode) matrixModes[key] = cp.mode;
            if (cp.colorMode) colorModes[key] = cp.colorMode;
            saveAllToStorage();
            renderChart(key);
            alert('✅ 已恢复到 "' + cp.name + '"');
        }
        
        function openSaveModal() {
            document.getElementById('checkpointName').value = '节点 ' + (checkpointsList.length + 1);
            document.getElementById('saveModal').classList.add('show');
            document.getElementById('checkpointName').focus();
        }
        
        function closeSaveModal() { document.getElementById('saveModal').classList.remove('show'); }
        
        function saveNamedCheckpoint() {
            const name = document.getElementById('checkpointName').value.trim() || '未命名节点';
            const checkpoint = {
                id: Date.now(),
                name: name,
                time: new Date().toLocaleString('zh-CN'),
                data: JSON.parse(JSON.stringify(matricesData)),
                settings: { matrixThemes: {...matrixThemes}, matrixModes: {...matrixModes}, colorModes: {...colorModes}, globalStyles: {...globalStyles}, customLabels: [...customLabels], classCount: currentClassCount }
            };
            checkpointsList.push(checkpoint);
            localStorage.setItem(CHECKPOINTS_KEY, JSON.stringify(checkpointsList));
            closeSaveModal();
            alert('✅ 节点 "' + name + '" 已保存！\\n共 ' + checkpointsList.length + ' 个节点。');
        }
        
        function openCheckpointList() {
            const container = document.getElementById('checkpointList');
            if (checkpointsList.length === 0) {
                container.innerHTML = '<p style="text-align:center;color:#999;padding:20px">暂无保存的节点</p>';
            } else {
                container.innerHTML = checkpointsList.map((cp, idx) => `
                    <div style="display:flex;justify-content:space-between;align-items:center;padding:12px;background:#f9f9f9;border-radius:8px;margin-bottom:8px">
                        <div>
                            <div style="font-weight:600">${cp.name}</div>
                            <div style="font-size:11px;color:#999">${cp.time}</div>
                        </div>
                        <div style="display:flex;gap:8px">
                            <button onclick="restoreCheckpoint(${idx})" style="padding:6px 12px;background:#3498db;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:12px">恢复</button>
                            <button onclick="deleteCheckpoint(${idx})" style="padding:6px 12px;background:#e74c3c;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:12px">删除</button>
                        </div>
                    </div>
                `).join('');
            }
            document.getElementById('checkpointModal').classList.add('show');
        }
        
        function closeCheckpointList() { document.getElementById('checkpointModal').classList.remove('show'); }
        
        function restoreCheckpoint(idx) {
            const cp = checkpointsList[idx];
            if (!cp) return;
            if (!confirm('确定要恢复到节点 "' + cp.name + '" 吗？')) return;
            
            matricesData = JSON.parse(JSON.stringify(cp.data));
            Object.assign(matrixThemes, cp.settings.matrixThemes);
            Object.assign(matrixModes, cp.settings.matrixModes);
            Object.assign(colorModes, cp.settings.colorModes);
            if (cp.settings.globalStyles) Object.assign(globalStyles, cp.settings.globalStyles);
            if (cp.settings.customLabels) customLabels = [...cp.settings.customLabels];
            if (cp.settings.classCount) {
                currentClassCount = cp.settings.classCount;
                document.getElementById('classCountSelect').value = currentClassCount;
            }
            saveAllToStorage();
            renderAll();
            closeCheckpointList();
            alert('✅ 已恢复到 "' + cp.name + '"');
        }
        
        function deleteCheckpoint(idx) {
            const cp = checkpointsList[idx];
            if (!cp) return;
            if (!confirm('确定要删除节点 "' + cp.name + '" 吗？')) return;
            checkpointsList.splice(idx, 1);
            localStorage.setItem(CHECKPOINTS_KEY, JSON.stringify(checkpointsList));
            openCheckpointList(); // 刷新列表
        }
        
        // 完全重置
        function resetToProgram() {
            if (!confirm('确定要完全重置为程序初始值吗？\\n注意：已保存的节点不会被删除。')) return;
            Object.keys(PROGRAM_DATA).forEach(key => {
                matricesData[key].data = JSON.parse(JSON.stringify(PROGRAM_DATA[key].data));
            });
            Object.keys(matrixThemes).forEach(k => { matrixThemes[k] = 'blues'; matrixModes[k] = 'count'; colorModes[k] = 'row'; });
            customLabels = [...classNames];
            currentClassCount = 5;
            document.getElementById('classCountSelect').value = 5;
            globalStyles = { numFontFamily: 'Times New Roman', numFontSize: 13, numFontWeight: 'normal', numFontStyle: 'normal', labelFontFamily: 'Times New Roman', labelFontSize: 11, labelFontWeight: '600', labelFontStyle: 'normal' };
            saveAllToStorage();
            renderAll();
        }
        
        // 颜色主题
        const themes = {
            blues: [
                { t: 0.0, c: "#F7FBFF", dark: false }, { t: 0.2, c: "#DEEBF7", dark: false },
                { t: 0.4, c: "#9ECAE1", dark: false }, { t: 0.6, c: "#4292C6", dark: true },
                { t: 0.8, c: "#2171B5", dark: true }, { t: 1.0, c: "#08306B", dark: true }
            ],
            oranges: [
                { t: 0.0, c: "#FFF5EB", dark: false }, { t: 0.2, c: "#FEE6CE", dark: false },
                { t: 0.4, c: "#FDAE6B", dark: false }, { t: 0.6, c: "#F16913", dark: true },
                { t: 0.8, c: "#D94801", dark: true }, { t: 1.0, c: "#7F2704", dark: true }
            ],
            greens: [
                { t: 0.0, c: "#F7FCF5", dark: false }, { t: 0.2, c: "#E5F5E0", dark: false },
                { t: 0.4, c: "#A1D99B", dark: false }, { t: 0.6, c: "#41AB5D", dark: true },
                { t: 0.8, c: "#238B45", dark: true }, { t: 1.0, c: "#005A32", dark: true }
            ],
            reds: [
                { t: 0.0, c: "#FFF5F0", dark: false }, { t: 0.2, c: "#FEE0D2", dark: false },
                { t: 0.4, c: "#FC9272", dark: false }, { t: 0.6, c: "#FB6A4A", dark: false },
                { t: 0.8, c: "#CB181D", dark: true }, { t: 1.0, c: "#67000D", dark: true }
            ],
            purples: [
                { t: 0.0, c: "#FCFBFD", dark: false }, { t: 0.2, c: "#EFEDF5", dark: false },
                { t: 0.4, c: "#BCBDDC", dark: false }, { t: 0.6, c: "#9E9AC8", dark: false },
                { t: 0.8, c: "#6A51A3", dark: true }, { t: 1.0, c: "#3F007D", dark: true }
            ]
        };
        
        // 状态
        let matrixThemes = {}; // 每个矩阵的独立主题
        let matrixModes = {};  // 样本数/准确率%
        let colorModes = {};   // 颜色归一化：global(全局最大值) / row(按行总数)
        let currentEdit = null;
        
        // 全局样式设置
        let globalStyles = {
            numFontFamily: 'Times New Roman',
            numFontSize: 13,
            numFontWeight: 'normal',
            numFontStyle: 'normal',
            labelFontFamily: 'Times New Roman',
            labelFontSize: 11,
            labelFontWeight: '600',
            labelFontStyle: 'normal'
        };
        
        // 自定义标签名称
        let customLabels = [...classNames];
        
        // 初始化
        Object.keys(matricesData).forEach(key => {
            matrixThemes[key] = 'blues';
            matrixModes[key] = 'count';
            colorModes[key] = 'row';  // 默认按行总数
        });
        
        // 设置显示模式（样本数/百分比）
        function setMode(key, mode) {
            matrixModes[key] = mode;
            saveAllToStorage();
            renderChart(key);
        }
        
        // 设置颜色主题
        function setTheme(key, theme) {
            matrixThemes[key] = theme;
            saveAllToStorage();
            renderChart(key);
        }
        
        // 设置颜色归一化模式
        function setColorMode(key, mode) {
            colorModes[key] = mode;
            saveAllToStorage();
            renderChart(key);
        }
        
        // 动态加载UTIF库（用于TIF格式）
        let utifLoaded = false;
        async function loadUTIF() {
            if (utifLoaded || typeof UTIF !== 'undefined') return;
            return new Promise((resolve, reject) => {
                const script = document.createElement('script');
                script.src = 'https://cdn.jsdelivr.net/npm/utif@3.1.0/UTIF.js';
                script.onload = () => { utifLoaded = true; resolve(); };
                script.onerror = reject;
                document.head.appendChild(script);
            });
        }
        
        // 保存文件（通用）
        async function saveFile(blob, filename, mimeType, ext) {
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = filename;
            link.click();
            URL.revokeObjectURL(link.href);
        }
        
        // 下载图表为图片（支持自定义尺寸，英寸转像素按300DPI）
        async function downloadMatrix(key, format) {
            const card = document.getElementById('card_' + key);
            if (!card) return;
            
            // 隐藏按钮、面板、标题、副标题、统计信息
            const hideElements = card.querySelectorAll('.download-btns, .card-controls, .style-panel, .card-header, .stats');
            hideElements.forEach(el => el.style.display = 'none');
            
            // 获取导出尺寸（英寸），转换为像素（300DPI）
            const exportWidth = (globalStyles.exportWidth || 8) * 300;
            const exportHeight = (globalStyles.exportHeight || 6) * 300;
            
            try {
                // 先截图
                const tempCanvas = await html2canvas(card, { 
                    backgroundColor: '#fff',
                    scale: 4,
                    useCORS: true
                });
                
                // 缩放到目标尺寸
                const canvas = document.createElement('canvas');
                canvas.width = exportWidth;
                canvas.height = exportHeight;
                const ctx = canvas.getContext('2d');
                ctx.fillStyle = '#fff';
                ctx.fillRect(0, 0, exportWidth, exportHeight);
                // 居中绘制，保持宽高比
                const scale = Math.min(exportWidth / tempCanvas.width, exportHeight / tempCanvas.height);
                const x = (exportWidth - tempCanvas.width * scale) / 2;
                const y = (exportHeight - tempCanvas.height * scale) / 2;
                ctx.drawImage(tempCanvas, x, y, tempCanvas.width * scale, tempCanvas.height * scale);
                
                // 恢复显示
                hideElements.forEach(el => el.style.display = '');
                
                const info = matricesData[key];
                const filename = (info?.name || key);
                
                if (format === 'png') {
                    canvas.toBlob(blob => saveFile(blob, filename + '.png', 'image/png', '.png'), 'image/png');
                } else if (format === 'jpg') {
                    canvas.toBlob(blob => saveFile(blob, filename + '.jpg', 'image/jpeg', '.jpg'), 'image/jpeg', 1.0);
                } else if (format === 'tif') {
                    await loadUTIF();
                    const ctx = canvas.getContext('2d');
                    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                    const tiffData = UTIF.encodeImage(imageData.data, canvas.width, canvas.height);
                    const tifBlob = new Blob([tiffData], { type: 'image/tiff' });
                    await saveFile(tifBlob, filename + '.tif', 'image/tiff', '.tif');
                }
            } catch(err) {
                hideElements.forEach(el => el.style.display = '');
                console.error('下载失败:', err);
                alert('下载失败，请重试');
            }
        }
        
        function getColor(value, maxVal, theme) {
            const stops = themes[theme];
            const ratio = maxVal > 0 ? value / maxVal : 0;
            if (ratio <= 0.05) return stops[0];
            if (ratio <= 0.25) return stops[1];
            if (ratio <= 0.45) return stops[2];
            if (ratio <= 0.65) return stops[3];
            if (ratio <= 0.85) return stops[4];
            return stops[5];
        }
        
        // ECharts实例存储
        const chartInstances = {};
        
        // 渲染所有图表
        function renderAll() {
            const grid = document.getElementById('matricesGrid');
            grid.innerHTML = '';
            Object.keys(matricesData).forEach(key => {
                const card = document.createElement('div');
                card.className = 'matrix-card';
                card.id = 'card_' + key;
                card.innerHTML = '<div style="text-align:center;padding:40px;color:#999">加载中...</div>';
                grid.appendChild(card);
            });
            Object.keys(matricesData).forEach(key => renderChart(key));
        }
        
        function renderChart(key) {
            const info = matricesData[key];
            const chartType = info.type || 'confusion';
            
            if (chartType === 'confusion') {
                renderConfusionMatrix(key);
            } else if (chartType === 'bar') {
                renderBarChart(key);
            } else if (chartType === 'multibar') {
                renderMultiBar(key);
            } else if (chartType === 'boxplot') {
                renderBoxplot(key);
            } else if (chartType === 'roc') {
                renderROC(key);
            } else if (chartType === 'line') {
                renderLineChart(key);
            } else if (chartType === 'scatter') {
                renderScatterChart(key);
            } else if (chartType === 'radar') {
                renderRadarChart(key);
            } else if (chartType === 'dualaxis') {
                renderDualAxisChart(key);
            } else if (chartType === 'area') {
                renderAreaChart(key);
            } else if (chartType === 'band') {
                renderBandChart(key);
            } else if (chartType === 'contour') {
                renderContourChart(key);
            } else if (chartType === 'polar') {
                renderPolarChart(key);
            } else if (chartType === 'surface3d') {
                renderSurface3DChart(key);
            } else if (chartType === 'scatter3d') {
                renderScatter3DChart(key);
            } else if (chartType === 'bar3d') {
                renderBar3DChart(key);
            } else if (chartType === 'histogram') {
                renderHistogramChart(key);
            } else if (chartType === 'violin') {
                renderViolinChart(key);
            } else if (chartType === 'pairplot') {
                renderPairplotChart(key);
            } else if (chartType === 'facet') {
                renderFacetChart(key);
            } else if (chartType === 'heatmap') {
                renderHeatmapChart(key);
            } else if (chartType === 'pie') {
                renderPieChart(key);
            } else if (chartType === 'waterfall') {
                renderWaterfallChart(key);
            } else if (chartType === 'realboxplot') {
                renderRealBoxplot(key);
            }
        }
        
        function renderConfusionMatrix(key) {
            const info = matricesData[key];
            const data = info.data;
            const theme = matrixThemes[key];
            const mode = matrixModes[key];
            const colorMode = colorModes[key];
            
            // 计算全局最大值和每行总数
            let globalMax = 0;
            const rowSums = [];
            if (Array.isArray(data) && Array.isArray(data[0])) {
                data.forEach(row => {
                    const sum = row.reduce((a, b) => a + b, 0);
                    rowSums.push(sum);
                    row.forEach(v => { if (v > globalMax) globalMax = v; });
                });
            }
            
            const mainTitle = customTitles[key] || info.name;
            const subtitle = customSubtitles[key] || info.subtitle || '';
            const modeText = mode === 'count' ? '样本数' : '准确率%';
            const displaySubtitle = subtitle ? subtitle + '（' + modeText + '）' : modeText;
            let html = `
                <div class="card-header">
                    <div>
                        <span class="card-title">${mainTitle}</span>
                        <span class="card-subtitle" style="font-size:12px;color:#666;margin-left:8px">${displaySubtitle}</span>
                    </div>
                    <div class="card-controls">
                        <button class="ctrl-btn ${mode === 'count' ? 'active' : ''}" onclick="setMode('${key}', 'count')">样本</button>
                        <button class="ctrl-btn ${mode === 'pct' ? 'active' : ''}" onclick="setMode('${key}', 'pct')">%</button>
                        <button class="ctrl-btn" onclick="toggleStylePanel('${key}')">🎨</button>
                    </div>
                </div>
                <div class="style-panel" id="panel_${key}">
                    <div class="style-panel-header">
                        <span>📊 ${mainTitle} 样式设置</span>
                        <button class="style-panel-close" onclick="cancelStylePanel('${key}')">✕</button>
                    </div>
                    <div class="style-row">
                        <label>主标题:</label>
                        <input type="text" value="${mainTitle}" onchange="setCustomTitle('${key}', this.value)" style="flex:1;padding:4px 8px;border:1px solid #ddd;border-radius:4px;font-size:12px">
                    </div>
                    <div class="style-row">
                        <label>副标题:</label>
                        <input type="text" value="${subtitle}" placeholder="如: 400rpm" onchange="setCustomSubtitle('${key}', this.value)" style="flex:1;padding:4px 8px;border:1px solid #ddd;border-radius:4px;font-size:12px">
                    </div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>导出尺寸</strong> 宽×高(英寸):
                        <input type="number" id="exportWidth_${key}" value="${globalStyles.exportWidth||8}" onchange="setGlobalStyle('exportWidth',+this.value)" style="width:50px" step="0.5" min="3" max="15">×
                        <input type="number" id="exportHeight_${key}" value="${globalStyles.exportHeight||6}" onchange="setGlobalStyle('exportHeight',+this.value)" style="width:50px" step="0.5" min="2" max="12">
                    </div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>颜色主题</strong></div>
                    <div class="style-row" style="gap:5px;flex-wrap:wrap">
                        <button onclick="setTheme('${key}','blues')" style="padding:4px 10px;border:1px solid ${theme==='blues'?'#0071e3':'#ddd'};border-radius:12px;background:${theme==='blues'?'#eefbff':'#fff'};cursor:pointer;font-size:11px;display:flex;align-items:center;gap:4px"><span style="background:linear-gradient(90deg,#F7FBFF,#08306B);width:14px;height:14px;border-radius:50%;display:inline-block"></span>蓝</button>
                        <button onclick="setTheme('${key}','oranges')" style="padding:4px 10px;border:1px solid ${theme==='oranges'?'#0071e3':'#ddd'};border-radius:12px;background:${theme==='oranges'?'#eefbff':'#fff'};cursor:pointer;font-size:11px;display:flex;align-items:center;gap:4px"><span style="background:linear-gradient(90deg,#FFF5EB,#7F2704);width:14px;height:14px;border-radius:50%;display:inline-block"></span>橙</button>
                        <button onclick="setTheme('${key}','greens')" style="padding:4px 10px;border:1px solid ${theme==='greens'?'#0071e3':'#ddd'};border-radius:12px;background:${theme==='greens'?'#eefbff':'#fff'};cursor:pointer;font-size:11px;display:flex;align-items:center;gap:4px"><span style="background:linear-gradient(90deg,#F7FCF5,#005A32);width:14px;height:14px;border-radius:50%;display:inline-block"></span>绿</button>
                        <button onclick="setTheme('${key}','reds')" style="padding:4px 10px;border:1px solid ${theme==='reds'?'#0071e3':'#ddd'};border-radius:12px;background:${theme==='reds'?'#eefbff':'#fff'};cursor:pointer;font-size:11px;display:flex;align-items:center;gap:4px"><span style="background:linear-gradient(90deg,#FFF5F0,#67000D);width:14px;height:14px;border-radius:50%;display:inline-block"></span>红</button>
                        <button onclick="setTheme('${key}','purples')" style="padding:4px 10px;border:1px solid ${theme==='purples'?'#0071e3':'#ddd'};border-radius:12px;background:${theme==='purples'?'#eefbff':'#fff'};cursor:pointer;font-size:11px;display:flex;align-items:center;gap:4px"><span style="background:linear-gradient(90deg,#FCFBFD,#3F007D);width:14px;height:14px;border-radius:50%;display:inline-block"></span>紫</button>
                    </div>
                    <div class="style-row">
                        <label>颜色归一化:</label>
                        <select onchange="setColorMode('${key}', this.value)">
                            <option value="global" ${colorMode === 'global' ? 'selected' : ''}>全局最大值</option>
                            <option value="row" ${colorMode === 'row' ? 'selected' : ''}>按行总数(推荐)</option>
                        </select>
                    </div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>数值字体</strong></div>
                    <div class="style-row">
                        <label>字体:</label>
                        <select onchange="setGlobalStyle('numFontFamily', this.value)">
                            <option value="SimSun" ${globalStyles.numFontFamily === 'SimSun' ? 'selected' : ''}>宋体</option>
                            <option value="Times New Roman" ${globalStyles.numFontFamily === 'Times New Roman' ? 'selected' : ''}>Times New Roman</option>
                            <option value="Arial" ${globalStyles.numFontFamily === 'Arial' ? 'selected' : ''}>Arial</option>
                            <option value="Georgia" ${globalStyles.numFontFamily === 'Georgia' ? 'selected' : ''}>Georgia</option>
                            <option value="Helvetica" ${globalStyles.numFontFamily === 'Helvetica' ? 'selected' : ''}>Helvetica</option>
                        </select>
                    </div>
                    <div class="style-row">
                        <label>字号:</label>
                        <select onchange="setGlobalStyle('numFontSize', +this.value)">
                            ${[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30].map(s => 
                                `<option value="${s}" ${globalStyles.numFontSize == s ? 'selected' : ''}>${s}px</option>`
                            ).join('')}
                        </select>
                        <select onchange="setGlobalStyle('numFontWeight', this.value)" style="margin-left:5px">
                            <option value="normal" ${globalStyles.numFontWeight === 'normal' ? 'selected' : ''}>常规</option>
                            <option value="bold" ${globalStyles.numFontWeight === 'bold' ? 'selected' : ''}>粗体</option>
                        </select>
                        <select onchange="setGlobalStyle('numFontStyle', this.value)">
                            <option value="normal" ${globalStyles.numFontStyle === 'normal' ? 'selected' : ''}>正常</option>
                            <option value="italic" ${globalStyles.numFontStyle === 'italic' ? 'selected' : ''}>斜体</option>
                        </select>
                    </div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>标签字体</strong></div>
                    <div class="style-row">
                        <label>字体:</label>
                        <select onchange="setGlobalStyle('labelFontFamily', this.value)">
                            <option value="SimSun" ${globalStyles.labelFontFamily === 'SimSun' ? 'selected' : ''}>宋体</option>
                            <option value="Times New Roman" ${globalStyles.labelFontFamily === 'Times New Roman' ? 'selected' : ''}>Times New Roman</option>
                            <option value="Arial" ${globalStyles.labelFontFamily === 'Arial' ? 'selected' : ''}>Arial</option>
                            <option value="Georgia" ${globalStyles.labelFontFamily === 'Georgia' ? 'selected' : ''}>Georgia</option>
                            <option value="Helvetica" ${globalStyles.labelFontFamily === 'Helvetica' ? 'selected' : ''}>Helvetica</option>
                        </select>
                    </div>
                    <div class="style-row">
                        <label>字号:</label>
                        <select onchange="setGlobalStyle('labelFontSize', +this.value)">
                            ${[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30].map(s => 
                                `<option value="${s}" ${globalStyles.labelFontSize == s ? 'selected' : ''}>${s}px</option>`
                            ).join('')}
                        </select>
                        <select onchange="setGlobalStyle('labelFontWeight', this.value)" style="margin-left:5px">
                            <option value="normal" ${globalStyles.labelFontWeight === 'normal' ? 'selected' : ''}>常规</option>
                            <option value="600" ${globalStyles.labelFontWeight === '600' ? 'selected' : ''}>半粗</option>
                            <option value="bold" ${globalStyles.labelFontWeight === 'bold' ? 'selected' : ''}>粗体</option>
                        </select>
                    </div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>标签位置</strong></div>
                    <div class="style-row">
                        Y轴标签右偏移:
                        <button onclick="adjustCMLabel('yLabelPadding',-1)" style="padding:2px 4px">◀</button>
                        <input type="number" id="yLabelPadding_${key}" value="${globalStyles.yLabelPadding!==undefined?globalStyles.yLabelPadding:2}" onchange="setGlobalStyle('yLabelPadding',+this.value)" style="width:40px" min="-30" max="30">px
                        <button onclick="adjustCMLabel('yLabelPadding',1)" style="padding:2px 4px">▶</button>
                    </div>
                    <div class="style-row">
                        X轴标签上偏移:
                        <button onclick="adjustCMLabel('xLabelPadding',-1)" style="padding:2px 4px">▲</button>
                        <input type="number" id="xLabelPadding_${key}" value="${globalStyles.xLabelPadding!==undefined?globalStyles.xLabelPadding:-8}" onchange="setGlobalStyle('xLabelPadding',+this.value)" style="width:40px" min="-30" max="30">px
                        <button onclick="adjustCMLabel('xLabelPadding',1)" style="padding:2px 4px">▼</button>
                    </div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>分类设置</strong></div>
                    <div class="style-row">
                        <label>分类数:</label>
                        <select id="classCountSelect_panel" onchange="setClassCount(+this.value)" style="padding:4px 8px;border-radius:4px;border:1px solid #ddd">
                            ${[2,3,4,5,6].map(n => `<option value="${n}" ${currentClassCount===n?'selected':''}>${n}分类</option>`).join('')}
                        </select>
                        <button onclick="openLabelEditor()" style="font-size:10px;padding:4px 8px;margin-left:10px">编辑标签</button>
                    </div>
                    <div class="style-row">
                        <button onclick="openImportModal()" style="font-size:11px;padding:4px 10px;background:#27ae60;color:#fff;border:none;border-radius:4px;cursor:pointer">📥 AI导入数据</button>
                    </div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>节点管理</strong>
                        <button onclick="openChartSaveModal('${key}')" style="font-size:10px;padding:3px 8px;margin-left:10px;background:#27ae60;color:#fff;border:none;border-radius:4px">💾 保存</button>
                        <button onclick="openChartRestoreModal('${key}')" style="font-size:10px;padding:3px 8px;margin-left:5px;background:#3498db;color:#fff;border:none;border-radius:4px">🔄 恢复</button>
                    </div>
                    <hr style="margin:10px 0;border:none;border-top:1px solid #ddd;">
                    <div class="style-row" style="justify-content:flex-end;gap:10px">
                        <button onclick="cancelStylePanel('${key}')" style="padding:6px 16px;cursor:pointer;border:1px solid #ddd;border-radius:4px;background:#fff">取消</button>
                        <button onclick="saveStylePanel('${key}')" style="padding:6px 16px;cursor:pointer;border:none;border-radius:4px;background:#667eea;color:#fff">保存</button>
                    </div>
                </div>
                <div class="matrix-wrapper" style="margin-top:20px">
                    <table class="matrix-table">
                        <tr><th></th>${customLabels.map(n => `<th style="visibility:hidden;height:0;padding:0;font-size:0">${n}</th>`).join('')}</tr>
            `;
            
            for (let i = 0; i < currentClassCount; i++) {
                const yPad = globalStyles.yLabelPadding!==undefined?globalStyles.yLabelPadding:2;
                html += `<tr><th style="background:transparent;font-family:${globalStyles.labelFontFamily};font-size:${globalStyles.labelFontSize}px;font-weight:${globalStyles.labelFontWeight};padding-right:${Math.max(0,yPad)}px;${yPad<0?'transform:translateX('+(-yPad)+'px);':''}cursor:pointer" onclick="editCMLabel(${i})">${customLabels[i]}</th>`;
                const rowSum = rowSums[i];
                for (let j = 0; j < currentClassCount; j++) {
                    const val = data[i][j];
                    const displayVal = mode === 'pct' && rowSum > 0 ? (val / rowSum * 100).toFixed(1) + '%' : val;
                    const colorBase = colorMode === 'row' ? rowSum : globalMax;
                    const colorObj = getColor(val, colorBase, theme);
                    const numStyle = `font-family:${globalStyles.numFontFamily};font-size:${globalStyles.numFontSize}px;font-weight:${globalStyles.numFontWeight};font-style:${globalStyles.numFontStyle}`;
                    html += `<td style="background:${colorObj.c};${numStyle}" class="${colorObj.dark ? 'dark' : ''}" 
                                 onclick="openEdit('${key}', ${i}, ${j}, ${val})">${displayVal}</td>`;
                }
                html += '</tr>';
            }
            
            // X轴标签放在底部
            const xPad = globalStyles.xLabelPadding!==undefined?globalStyles.xLabelPadding:-8;
            html += `<tr><th></th>${customLabels.map((n, idx) => `<th style="background:transparent;font-family:${globalStyles.labelFontFamily};font-size:${globalStyles.labelFontSize}px;font-weight:${globalStyles.labelFontWeight};padding-top:${Math.max(0,xPad)}px;${xPad<0?'transform:translateY('+(xPad)+'px);':''}cursor:pointer" onclick="editCMLabel(${idx})">${n}</th>`).join('')}</tr>`;
            html += `</table></div>`;
            
            // 统计
            let correct = 0, total = 0;
            for (let i = 0; i < currentClassCount; i++) { correct += data[i][i]; total += data[i].reduce((a, b) => a + b, 0); }
            html += `<div class="stats">准确率: ${(correct / total * 100).toFixed(2)}% (${correct}/${total})</div>`;
            
            // 下载按钮
            html += `<div class="download-btns">
                <button class="dl-btn" onclick="downloadMatrix('${key}', 'png')">📷 PNG</button>
                <button class="dl-btn" onclick="downloadMatrix('${key}', 'jpg')">🖼️ JPG</button>
                <button class="dl-btn" onclick="downloadMatrix('${key}', 'tif')">🖼️ TIF</button>
            </div>`;
            
            document.getElementById('card_' + key).innerHTML = html;
            // 如果面板原本是打开的，恢复打开状态
            if (originalGlobalStyles !== null && key === 'fig1') {
                document.getElementById('panel_' + key).classList.add('show');
            }
        }
        
        function renderBarChart(key) {
            const info = matricesData[key];
            const data = info.data;
            const theme = matrixThemes[key];
            const mainTitle = customTitles[key] || info.name;
            const subtitle = customSubtitles[key] || info.subtitle || '';
            const cs = chartStyles[key] || {};
            const barColors = cs.barColors || ['#3498db','#e67e22','#27ae60','#e74c3c','#9b59b6','#1abc9c'];
            const labels = data.labels || customLabels.slice(0, currentClassCount);
            
            let html = `
                <div class="card-header">
                    <div style="display:flex;align-items:center;flex-wrap:wrap">
                        <span class="card-title">${mainTitle}</span>
                        <span class="card-subtitle" style="font-size:12px;color:#666;margin-left:8px">${subtitle}</span>
                    </div>
                    <div class="card-controls">
                        <button class="ctrl-btn" onclick="exportChart('${key}','png')">PNG</button>
                        <button class="ctrl-btn" onclick="exportChart('${key}','jpg')">JPG</button>
                        <button class="ctrl-btn" onclick="exportChart('${key}','tif')">TIF</button>
                        <button class="ctrl-btn" onclick="toggleStylePanel('${key}')">🎨</button>
                    </div>
                </div>
                <div class="style-panel" id="panel_${key}">
                    <div class="style-panel-header">
                        <span>📊 ${mainTitle} 样式设置</span>
                        <button class="style-panel-close" onclick="cancelStylePanel('${key}')">✕</button>
                    </div>
                    <div class="style-row"><label>主标题:</label><input type="text" value="${mainTitle}" onchange="setCustomTitle('${key}', this.value)" style="flex:1;padding:4px 8px;border:1px solid #ddd;border-radius:4px;font-size:12px"></div>
                    <div class="style-row"><label>副标题:</label><input type="text" value="${subtitle}" onchange="setCustomSubtitle('${key}', this.value)" style="flex:1;padding:4px 8px;border:1px solid #ddd;border-radius:4px;font-size:12px"></div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>图表尺寸</strong> 宽×高(英寸):
                        <input type="number" value="${(chartStyles[key]||{}).chartWidth||8}" onchange="setChartStyle('${key}','chartWidth',+this.value)" style="width:50px" step="0.5" min="3" max="15">×
                        <input type="number" value="${(chartStyles[key]||{}).chartHeight||6}" onchange="setChartStyle('${key}','chartHeight',+this.value)" style="width:50px" step="0.5" min="2" max="12">
                    </div>
                    <div class="style-row"><strong>坐标轴名称</strong></div>
                    <div class="style-row"><label>X轴:</label><input type="text" value="${(chartStyles[key]||{}).xAxisName||'类别'}" onchange="setChartStyle('${key}','xAxisName',this.value)" style="width:100px">
                        <label style="margin-left:10px">Y轴:</label><input type="text" value="${(chartStyles[key]||{}).yAxisName||'准确率(%)'}" onchange="setChartStyle('${key}','yAxisName',this.value)" style="width:100px"></div>
                    <div class="style-row"><strong>X轴刻度标签</strong> <button type="button" onclick="event.preventDefault();openTickLabelsEditor('${key}')" style="font-size:10px;padding:2px 6px">编辑</button></div>
                    <div class="style-row"><strong>轴标签</strong> 字体:<select onchange="setChartStyle('${key}','axisLabelFont',this.value)" style="width:70px">
                        <option value="Times New Roman" ${(chartStyles[key]||{}).axisLabelFont==='Times New Roman'?'selected':''}>Times</option>
                        <option value="Arial" ${(chartStyles[key]||{}).axisLabelFont==='Arial'?'selected':''}>Arial</option>
                        <option value="SimSun" ${(chartStyles[key]||{}).axisLabelFont==='SimSun'?'selected':''}>宋体</option>
                    </select> 字号:<input type="number" value="${(chartStyles[key]||{}).axisLabelSize||12}" onchange="setChartStyle('${key}','axisLabelSize',+this.value)" style="width:40px" min="8" max="20">
                    颜色:<input type="color" id="bar_axisLabelColor_${key}" value="${(chartStyles[key]||{}).axisLabelColor||'#000000'}" onchange="setChartStyle('${key}','axisLabelColor',this.value)" style="width:30px;height:22px;border:none;cursor:pointer"><button type="button" onclick="pickColor(c=>{document.getElementById('bar_axisLabelColor_${key}').value=c;setChartStyle('${key}','axisLabelColor',c)})" style="padding:2px 4px;font-size:10px;cursor:pointer;margin-left:2px" title="取色笔">🎯</button></div>
                    <div class="style-row"><strong>刻度</strong> 字体:<select onchange="setChartStyle('${key}','axisTickFont',this.value)" style="width:70px">
                        <option value="Times New Roman" ${(chartStyles[key]||{}).axisTickFont==='Times New Roman'?'selected':''}>Times</option>
                        <option value="Arial" ${(chartStyles[key]||{}).axisTickFont==='Arial'?'selected':''}>Arial</option>
                        <option value="SimSun" ${(chartStyles[key]||{}).axisTickFont==='SimSun'?'selected':''}>宋体</option>
                    </select> 字号:<input type="number" value="${(chartStyles[key]||{}).axisTickSize||10}" onchange="setChartStyle('${key}','axisTickSize',+this.value)" style="width:40px" min="8" max="18">
                    颜色:<input type="color" id="bar_axisTickColor_${key}" value="${(chartStyles[key]||{}).axisTickColor||'#000000'}" onchange="setChartStyle('${key}','axisTickColor',this.value)" style="width:30px;height:22px;border:none;cursor:pointer"><button type="button" onclick="pickColor(c=>{document.getElementById('bar_axisTickColor_${key}').value=c;setChartStyle('${key}','axisTickColor',c)})" style="padding:2px 4px;font-size:10px;cursor:pointer;margin-left:2px" title="取色笔">🎯</button></div>
                    <div class="style-row"><strong>刻度像素间距</strong> X轴:<input type="number" value="${(chartStyles[key]||{}).xTickGapPx||80}" onchange="setChartStyle('${key}','xTickGapPx',+this.value)" style="width:50px" min="30" max="200" step="10">px
                        Y轴:<input type="number" value="${(chartStyles[key]||{}).yTickGapPx||50}" onchange="setChartStyle('${key}','yTickGapPx',+this.value)" style="width:50px" min="20" max="150" step="10">px</div>
                    <div class="style-row"><strong>X轴名偏移</strong> 
                        <button onclick="adjustAxisOffset('${key}','xNameGap',-5)" style="padding:2px 4px">◀</button>
                        <input type="number" id="xNameGap_${key}" value="${(chartStyles[key]||{}).xNameGap||25}" onchange="setChartStyle('${key}','xNameGap',+this.value)" style="width:40px" min="-50" max="80">px
                        <button onclick="adjustAxisOffset('${key}','xNameGap',5)" style="padding:2px 4px">▶</button>
                        <strong style="margin-left:10px">Y轴名偏移</strong>
                        <button onclick="adjustAxisOffset('${key}','yNameGap',-5)" style="padding:2px 4px">◀</button>
                        <input type="number" id="yNameGap_${key}" value="${(chartStyles[key]||{}).yNameGap||35}" onchange="setChartStyle('${key}','yNameGap',+this.value)" style="width:40px" min="-50" max="80">px
                        <button onclick="adjustAxisOffset('${key}','yNameGap',5)" style="padding:2px 4px">▶</button>
                    </div>
                    <div class="style-row"><strong>X刻度标签偏移</strong> 
                        <button onclick="adjustAxisOffset('${key}','xTickMargin',-1)" style="padding:2px 4px">▲</button>
                        <input type="number" id="xTickMargin_${key}" value="${(chartStyles[key]||{}).xTickMargin||8}" onchange="setChartStyle('${key}','xTickMargin',+this.value)" style="width:40px" min="-20" max="30">px
                        <button onclick="adjustAxisOffset('${key}','xTickMargin',1)" style="padding:2px 4px">▼</button>
                        <strong style="margin-left:10px">Y刻度标签偏移</strong>
                        <button onclick="adjustAxisOffset('${key}','yTickMargin',-1)" style="padding:2px 4px">◀</button>
                        <input type="number" id="yTickMargin_${key}" value="${(chartStyles[key]||{}).yTickMargin||8}" onchange="setChartStyle('${key}','yTickMargin',+this.value)" style="width:40px" min="-20" max="30">px
                        <button onclick="adjustAxisOffset('${key}','yTickMargin',1)" style="padding:2px 4px">▶</button>
                    </div>
                    <div class="style-row"><strong>图例</strong> 字体:<select onchange="setChartStyle('${key}','legendFont',this.value)" style="width:70px">
                        <option value="Times New Roman" ${(chartStyles[key]||{}).legendFont==='Times New Roman'?'selected':''}>Times</option>
                        <option value="Arial" ${(chartStyles[key]||{}).legendFont==='Arial'?'selected':''}>Arial</option>
                        <option value="SimSun" ${(chartStyles[key]||{}).legendFont==='SimSun'?'selected':''}>宋体</option>
                    </select> 字号:<input type="number" value="${(chartStyles[key]||{}).legendSize||11}" onchange="setChartStyle('${key}','legendSize',+this.value)" style="width:40px" min="8" max="16">
                    颜色:<input type="color" id="bar_legendColor_${key}" value="${(chartStyles[key]||{}).legendColor||'#000000'}" onchange="setChartStyle('${key}','legendColor',this.value)" style="width:30px;height:22px;border:none;cursor:pointer"><button type="button" onclick="pickColor(c=>{document.getElementById('bar_legendColor_${key}').value=c;setChartStyle('${key}','legendColor',c)})" style="padding:2px 4px;font-size:10px;cursor:pointer;margin-left:2px" title="取色笔">🎯</button></div>
                    <div class="style-row"><strong>图例位置</strong> 
                        X:<button onclick="adjustLegend('${key}','X',-5)" style="padding:2px 6px">◀</button>
                        <input type="range" min="0" max="100" value="${(chartStyles[key]||{}).legendX||50}" id="legendX_${key}" oninput="document.getElementById('legendXVal_${key}').textContent=this.value+'%';setChartStyle('${key}','legendX',+this.value)" style="width:60px">
                        <span id="legendXVal_${key}">${(chartStyles[key]||{}).legendX||50}%</span>
                        <button onclick="adjustLegend('${key}','X',5)" style="padding:2px 6px">▶</button>
                        Y:<button onclick="adjustLegend('${key}','Y',-2)" style="padding:2px 6px">▲</button>
                        <input type="range" min="0" max="30" value="${(chartStyles[key]||{}).legendY||0}" id="legendY_${key}" oninput="document.getElementById('legendYVal_${key}').textContent=this.value+'%';setChartStyle('${key}','legendY',+this.value)" style="width:50px">
                        <span id="legendYVal_${key}">${(chartStyles[key]||{}).legendY||0}%</span>
                        <button onclick="adjustLegend('${key}','Y',2)" style="padding:2px 6px">▼</button>
                    </div>
                    <div class="style-row"><strong>图例方向</strong>:<select onchange="setChartStyle('${key}','legendOrient',this.value)">
                        <option value="horizontal" ${(chartStyles[key]||{}).legendOrient!=='vertical'?'selected':''}>横向</option>
                        <option value="vertical" ${(chartStyles[key]||{}).legendOrient==='vertical'?'selected':''}>纵向</option>
                    </select></div>
                    <div class="style-row"><strong>图例图标尺寸</strong> 宽:<input type="number" value="${(chartStyles[key]||{}).legendItemWidth||25}" onchange="setChartStyle('${key}','legendItemWidth',+this.value)" style="width:40px" min="8" max="50">
                        高:<input type="number" value="${(chartStyles[key]||{}).legendItemHeight||14}" onchange="setChartStyle('${key}','legendItemHeight',+this.value)" style="width:40px" min="8" max="30"></div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>柱子设置</strong> 宽度:<input type="number" value="${(chartStyles[key]||{}).barWidth||40}" onchange="setChartStyle('${key}','barWidth',+this.value)" style="width:50px" min="10" max="80">px</div>
                    <div class="style-row"><strong>柱子数量</strong>:<select onchange="setBarCount('${key}',+this.value)">
                        ${[1,2,3,4,5,6,7,8].map(n => `<option value="${n}" ${(data.values||[]).length===n?'selected':''}>${n}</option>`).join('')}
                    </select></div>
                    <div class="style-row"><strong>预设配色</strong>
                        <button onclick="applyPresetColors('${key}','bar')" style="font-size:10px;padding:2px 6px;margin-left:5px">应用配色</button>
                        <div style="display:flex;gap:3px;margin-top:4px">
                            <span style="width:18px;height:18px;background:#2d5a3d;border-radius:3px;cursor:pointer" title="深绿" onclick="setBarColor('${key}',0,'#2d5a3d')"></span>
                            <span style="width:18px;height:18px;background:#3a6b96;border-radius:3px;cursor:pointer" title="蓝" onclick="setBarColor('${key}',1,'#3a6b96')"></span>
                            <span style="width:18px;height:18px;background:#8b2323;border-radius:3px;cursor:pointer" title="深红" onclick="setBarColor('${key}',2,'#8b2323')"></span>
                            <span style="width:18px;height:18px;background:#f5deb3;border-radius:3px;cursor:pointer" title="米色" onclick="setBarColor('${key}',3,'#f5deb3')"></span>
                        </div>
                    </div>
                    <div class="style-row"><strong>柱子颜色</strong></div>
                    ${(data.labels || customLabels.slice(0,currentClassCount)).map((label, i) => `
                    <div class="style-row">
                        <label>柱${i+1}:</label>
                        <input type="color" id="barColor_${key}_${i}" value="${((chartStyles[key]||{}).barColors||['#2d5a3d','#3a6b96','#8b2323','#f5deb3','#9b59b6','#1abc9c'])[i]||'#3498db'}" onchange="setBarColor('${key}',${i},this.value)" style="width:30px;height:22px;border:none;cursor:pointer"><button type="button" onclick="pickColor(c=>{document.getElementById('barColor_${key}_${i}').value=c;setBarColor('${key}',${i},c)})" style="padding:2px 4px;font-size:10px;cursor:pointer;margin-left:2px" title="取色笔">🎯</button>
                    </div>`).join('')}
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>数据编辑</strong> <button type="button" onclick="event.preventDefault();openBarEditor('${key}')" style="font-size:10px;padding:2px 6px">编辑数据</button>
                        <button onclick="openChartImport('${key}','bar')" style="font-size:10px;padding:2px 6px;margin-left:10px;background:#27ae60;color:#fff;border:none;border-radius:4px">📥 AI导入</button></div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>节点管理</strong>
                        <button onclick="openChartSaveModal('${key}')" style="font-size:10px;padding:3px 8px;margin-left:10px;background:#27ae60;color:#fff;border:none;border-radius:4px">💾 保存</button>
                        <button onclick="openChartRestoreModal('${key}')" style="font-size:10px;padding:3px 8px;margin-left:5px;background:#3498db;color:#fff;border:none;border-radius:4px">🔄 恢复</button>
                    </div>
                    <hr style="margin:10px 0;border:none;border-top:1px solid #ddd;">
                    <div class="style-row" style="justify-content:flex-end;gap:10px">
                        <button onclick="cancelStylePanel('${key}')" style="padding:6px 16px;cursor:pointer;border:1px solid #ddd;border-radius:4px;background:#fff">取消</button>
                        <button onclick="saveStylePanel('${key}')" style="padding:6px 16px;cursor:pointer;border:none;border-radius:4px;background:#667eea;color:#fff">保存</button>
                    </div>
                </div>
                <div id="chart_${key}" style="width:${(() => {
                    const cs = chartStyles[key] || {};
                    const xLabels = cs.xTickLabels || data.labels || customLabels.slice(0, currentClassCount);
                    const xGap = cs.xTickGapPx || 100;
                    return Math.max((xLabels.length - 1) * xGap + 180, 500);
                })()}px;height:${(() => {
                    const cs = chartStyles[key] || {};
                    const yLabels = cs.yTickLabels || ['0','20','40','60','80','100'];
                    const yGap = cs.yTickGapPx || 60;
                    return Math.max((yLabels.length - 1) * yGap + 120, 400);
                })()}px"></div>
            `;
            document.getElementById('card_' + key).innerHTML = html;
            // 如果面板原本是打开的，恢复打开状态
            if (originalStyles[key]) {
                document.getElementById('panel_' + key).classList.add('show');
            }
            
            setTimeout(() => {
                if (chartInstances[key]) chartInstances[key].dispose();
                const chartDiv = document.getElementById('chart_' + key);
                
                // ★★★ 重新获取最新的样式配置，根据刻度像素间距计算容器尺寸 ★★★
                const latestCs = chartStyles[key] || {};
                const barWidth = latestCs.barWidth || 40;
                const xLabels = latestCs.xTickLabels || data.labels || customLabels.slice(0, currentClassCount);
                const yLabels = latestCs.yTickLabels || ['0','20','40','60','80','100'];
                const xTickGapPx = latestCs.xTickGapPx || 80;
                const yTickGapPx = latestCs.yTickGapPx || 50;
                const gridMargin = 2 * barWidth + 90;
                // 根据刻度数量和像素间距计算图表尺寸
                const chartWidth = Math.max(xLabels.length * xTickGapPx + gridMargin, 400);
                const chartHeight = Math.max(yLabels.length * yTickGapPx + 100, 300);
                chartDiv.style.width = chartWidth + 'px';
                chartDiv.style.height = chartHeight + 'px';
                
                chartInstances[key] = echarts.init(chartDiv);
                const barColors = latestCs.barColors || ['#3498db'];
                const values = data.values || [];
                chartInstances[key].setOption({
                    tooltip: { trigger: 'axis' },
                    legend: { 
                        show: true,
                        data: xLabels,
                        left: (latestCs.legendX || 50) + '%',
                        top: (latestCs.legendY || 0) + '%',
                        orient: latestCs.legendOrient || 'horizontal',
                        icon: 'rect',
                        itemWidth: latestCs.legendItemWidth || 25,
                        itemHeight: latestCs.legendItemHeight || 14,
                        textStyle: { fontFamily: latestCs.legendFont || 'Times New Roman', fontSize: latestCs.legendSize || 11, color: latestCs.legendColor || '#000' }
                    },
                    grid: { left: barWidth + 60, right: barWidth + 30, top: 50, bottom: 50, borderColor: '#000', borderWidth: 1, show: true },
                    xAxis: { 
                        type: 'category', 
                        boundaryGap: true,
                        triggerEvent: true,
                        name: latestCs.xAxisName || '类别',
                        data: xLabels,
                        nameLocation: 'middle',
                        nameGap: 30,
                        axisLine: { show: true, lineStyle: { color: '#000', width: 1 } },
                        axisTick: { show: true, lineStyle: { color: '#000' }, alignWithLabel: true },
                        nameTextStyle: { fontFamily: latestCs.axisLabelFont || 'Times New Roman', fontSize: latestCs.axisLabelSize || 12, color: latestCs.axisLabelColor || '#000' },
                        axisLabel: { fontFamily: latestCs.axisTickFont || 'Times New Roman', fontSize: latestCs.axisTickSize || 10, color: latestCs.axisTickColor || '#000' }
                    },
                    yAxis: { 
                        type: 'value', min: 0, max: 110, 
                        triggerEvent: true,
                        name: latestCs.yAxisName || '准确率(%)',
                        nameLocation: 'middle',
                        nameGap: 40,
                        interval: latestCs.yTickInterval || 20,
                        axisLine: { show: true, lineStyle: { color: '#000', width: 1 } },
                        axisTick: { show: true, lineStyle: { color: '#000' } },
                        splitLine: { show: false },
                        nameTextStyle: { fontFamily: latestCs.axisLabelFont || 'Times New Roman', fontSize: latestCs.axisLabelSize || 12, color: latestCs.axisLabelColor || '#000' },
                        axisLabel: { 
                            fontFamily: latestCs.axisTickFont || 'Times New Roman', 
                            fontSize: latestCs.axisTickSize || 10, 
                            color: latestCs.axisTickColor || '#000', 
                            formatter: function(v) { 
                                if (v > 100) return '';
                                const yLabels = latestCs.yTickLabels || ['0', '20', '40', '60', '80', '100'];
                                const idx = Math.round(v / 20);
                                return (idx >= 0 && idx < yLabels.length) ? yLabels[idx] : v;
                            } 
                        }
                    },
                    series: [{ 
                        type: 'bar', 
                        data: values.map((v,i) => ({ value: v, itemStyle: { color: barColors[i % barColors.length] } })),
                        barWidth: barWidth,
                        barCategoryGap: barWidth + 'px'
                    }]
                });
                
                // ★★★ 点击柱子编辑数值功能 ★★★
                chartInstances[key].on('click', function(params) {
                    if (params.componentType === 'series') {
                        const dataIndex = params.dataIndex;
                        const currentValue = params.value;
                        showValueEditor(xLabels[dataIndex], currentValue, function(val) {
                            if (!matricesData[key].data.values) matricesData[key].data.values = [];
                            matricesData[key].data.values[dataIndex] = val;
                            saveAllToStorage();
                            renderBarChart(key);
                        });
                    } else if (params.componentType === 'xAxis') {
                        if (params.targetType === 'axisLabel') {
                            // 点击X轴刻度标签编辑
                            const dataIndex = params.dataIndex;
                            if (dataIndex >= 0 && dataIndex < xLabels.length) {
                                showTextEditor(xLabels[dataIndex], function(newLabel) {
                                    if (!matricesData[key].data.labels) matricesData[key].data.labels = [...xLabels];
                                    matricesData[key].data.labels[dataIndex] = newLabel;
                                    saveAllToStorage();
                                    renderBarChart(key);
                                });
                            }
                        } else {
                            // 点击轴名称
                            const cs = chartStyles[key] || {};
                            showTextEditor(cs.xAxisName || '类别', function(newName) {
                                if (!chartStyles[key]) chartStyles[key] = {};
                                chartStyles[key].xAxisName = newName;
                                saveAllToStorage();
                                renderBarChart(key);
                            });
                        }
                    } else if (params.componentType === 'yAxis') {
                        const cs = chartStyles[key] || {};
                        if (params.targetType === 'axisLabel') {
                            // 点击Y轴刻度标签编辑
                            const yLabels = cs.yTickLabels || ['0', '20', '40', '60', '80', '100'];
                            const idx = params.value ? Math.round(params.value / 20) : 0;
                            if (idx >= 0 && idx < yLabels.length) {
                                showTextEditor(yLabels[idx], function(newVal) {
                                    if (!chartStyles[key]) chartStyles[key] = {};
                                    if (!chartStyles[key].yTickLabels) chartStyles[key].yTickLabels = [...yLabels];
                                    chartStyles[key].yTickLabels[idx] = newVal;
                                    saveAllToStorage();
                                    renderBarChart(key);
                                });
                            }
                        } else {
                            // 点击轴名称
                            showTextEditor(cs.yAxisName || '准确率(%)', function(newName) {
                                if (!chartStyles[key]) chartStyles[key] = {};
                                chartStyles[key].yAxisName = newName;
                                saveAllToStorage();
                                renderBarChart(key);
                            });
                        }
                    }
                });
            }, 100);
        }
        
        function addBarColor(key) {
            if (!chartStyles[key]) chartStyles[key] = {};
            if (!chartStyles[key].barColors) chartStyles[key].barColors = ['#3498db'];
            chartStyles[key].barColors.push('#' + Math.floor(Math.random()*16777215).toString(16).padStart(6,'0'));
            saveAllToStorage();
            renderChart(key);
        }
        
        function setBarColor(key, idx, color) {
            if (!chartStyles[key]) chartStyles[key] = {};
            if (!chartStyles[key].barColors) chartStyles[key].barColors = ['#2d5a3d','#3a6b96','#8b2323','#f5deb3','#9b59b6','#1abc9c'];
            chartStyles[key].barColors[idx] = color;
            saveAllToStorage();
            renderChart(key);
        }
        
        // 预设配色应用函数
        const presetColors = ['#2d5a3d', '#3a6b96', '#8b2323', '#f5deb3'];
        function applyPresetColors(key, chartType) {
            if (chartType === 'bar') {
                if (!chartStyles[key]) chartStyles[key] = {};
                chartStyles[key].barColors = [...presetColors, '#e67e22', '#1abc9c'];
            } else if (chartType === 'multibar' || chartType === 'boxplot') {
                if (!boxplotSeries[key]) boxplotSeries[key] = { count: 4, names: [], colors: [] };
                boxplotSeries[key].colors = [...presetColors, '#e67e22', '#1abc9c'];
                // 图5还需要更新data.series中的颜色
                if (chartType === 'boxplot' && matricesData[key]?.data?.series) {
                    matricesData[key].data.series.forEach((s, i) => {
                        s.color = presetColors[i % presetColors.length];
                    });
                }
            }
            saveAllToStorage();
            renderChart(key);
        }
        
        // 图5系列颜色设置
        function setBoxplotSeriesColor(key, idx, color) {
            if (matricesData[key]?.data?.series?.[idx]) {
                matricesData[key].data.series[idx].color = color;
                saveAllToStorage();
                renderChart(key);
            }
        }
        
        // 图2数据编辑弹窗（表格形式）
        let barEditorDiv = null;
        function openBarEditor(key) {
            const info = matricesData[key];
            const data = info.data;
            const labels = data.labels || customLabels.slice(0, (data.values||[]).length || 5);
            const values = data.values || [];
            
            if (barEditorDiv) barEditorDiv.remove();
            barEditorDiv = document.createElement('div');
            barEditorDiv.style.cssText = 'position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:#fff;padding:20px;border-radius:12px;box-shadow:0 10px 40px rgba(0,0,0,0.3);z-index:1000;max-height:80vh;overflow:auto;min-width:350px';
            
            let html = '<h3 style="margin-top:0">编辑柱状图数据</h3>';
            html += '<table style="border-collapse:collapse;font-size:12px;width:100%"><tr><th style="border:1px solid #ddd;padding:6px;background:#f5f5f5">序号</th><th style="border:1px solid #ddd;padding:6px;background:#f5f5f5">标签</th><th style="border:1px solid #ddd;padding:6px;background:#f5f5f5">数值</th></tr>';
            for (let i = 0; i < labels.length; i++) {
                html += '<tr><td style="border:1px solid #ddd;padding:6px;text-align:center">' + (i+1) + '</td>';
                html += '<td style="border:1px solid #ddd;padding:4px"><input type="text" id="bar_' + key + '_label_' + i + '" value="' + labels[i] + '" style="width:80px;padding:4px"></td>';
                html += '<td style="border:1px solid #ddd;padding:4px"><input type="number" id="bar_' + key + '_value_' + i + '" value="' + (values[i]||0) + '" style="width:60px;padding:4px" step="0.1"></td></tr>';
            }
            html += '</table>';
            html += '<div style="margin-top:15px;text-align:right"><button onclick="closeBarEditor()" style="padding:8px 16px;margin-right:10px;cursor:pointer">取消</button>';
            html += '<button onclick="saveBarData(\\''+key+'\\')" style="padding:8px 16px;background:#667eea;color:#fff;border:none;border-radius:6px;cursor:pointer">保存</button></div>';
            
            barEditorDiv.innerHTML = html;
            document.body.appendChild(barEditorDiv);
            
            const overlay = document.createElement('div');
            overlay.id = 'barEditorOverlay';
            overlay.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.5);z-index:999';
            overlay.onclick = closeBarEditor;
            document.body.appendChild(overlay);
        }
        
        function closeBarEditor() {
            if (barEditorDiv) { barEditorDiv.remove(); barEditorDiv = null; }
            const overlay = document.getElementById('barEditorOverlay');
            if (overlay) overlay.remove();
        }
        
        function saveBarData(key) {
            const info = matricesData[key];
            const data = info.data;
            const labels = data.labels || customLabels.slice(0, (data.values||[]).length || 5);
            
            const newLabels = [];
            const newValues = [];
            for (let i = 0; i < labels.length; i++) {
                const labelInput = document.getElementById('bar_' + key + '_label_' + i);
                const valueInput = document.getElementById('bar_' + key + '_value_' + i);
                if (labelInput) newLabels.push(labelInput.value);
                if (valueInput) newValues.push(+valueInput.value);
            }
            data.labels = newLabels;
            data.values = newValues;
            saveAllToStorage();
            closeBarEditor();
            renderChart(key);
        }
        
        // X轴刻度标签表格编辑弹窗
        let tickLabelsEditorDiv = null;
        function openTickLabelsEditor(key) {
            const info = matricesData[key];
            const data = info.data;
            const cs = chartStyles[key] || {};
            let xLabels, yLabels;
            
            if (info.type === 'bar') {
                xLabels = cs.xTickLabels || data.labels || customLabels.slice(0, (data.values||[]).length || 5);
                yLabels = cs.yTickLabels || ['0', '20', '40', '60', '80', '100'];
            } else if (info.type === 'multibar') {
                xLabels = cs.xTickLabels || data.labels || ['B=15', 'B=20', 'B=25', 'B=30'];
                yLabels = cs.yTickLabels || ['0', '20', '40', '60', '80', '100'];
            } else if (info.type === 'boxplot') {
                xLabels = cs.xTickLabels || data.labels || ['B=15', 'B=20', 'B=25', 'B=30'];
                yLabels = cs.yTickLabels || ['0', '20', '40', '60', '80', '100'];
            } else if (info.type === 'roc') {
                xLabels = cs.xTickLabels || ['0', '0.2', '0.4', '0.6', '0.8', '1.0'];
                yLabels = cs.yTickLabels || ['0', '0.2', '0.4', '0.6', '0.8', '1.0'];
            } else {
                xLabels = cs.xTickLabels || data.labels || customLabels.slice(0, 5);
                yLabels = cs.yTickLabels || ['0', '20', '40', '60', '80', '100'];
            }
            
            if (tickLabelsEditorDiv) tickLabelsEditorDiv.remove();
            tickLabelsEditorDiv = document.createElement('div');
            tickLabelsEditorDiv.style.cssText = 'position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:#fff;padding:20px;border-radius:12px;box-shadow:0 10px 40px rgba(0,0,0,0.3);z-index:1000;max-height:80vh;overflow:auto;min-width:300px';
            
            let html = '<h3 style="margin-top:0">编辑轴刻度标签</h3>';
            
            // X轴设置
            html += '<div style="margin-bottom:15px;padding:10px;background:#f9f9f9;border-radius:8px">';
            html += '<strong>X轴刻度标签</strong>';
            html += '<div style="margin:8px 0"><label>刻度间距(px):</label><input type="number" id="xTickGap_' + key + '" value="' + (cs.xTickGapPx || 80) + '" style="width:60px;padding:4px;margin-left:5px" min="20" max="200"></div>';
            html += '<table style="border-collapse:collapse;font-size:12px;margin-top:5px">';
            xLabels.forEach((label, i) => {
                html += '<tr><td style="border:1px solid #ddd;padding:4px">刻度' + (i+1) + ':</td><td style="border:1px solid #ddd;padding:4px"><input type="text" id="xtick_' + key + '_' + i + '" value="' + label + '" style="width:100px;padding:4px"></td><td style="padding:4px"><button onclick="removeXTickLabel(\\'' + key + '\\',' + i + ')" style="font-size:10px;padding:2px 6px;color:#e74c3c">删除</button></td></tr>';
            });
            html += '</table><button onclick="addXTickLabel(\\'' + key + '\\')" style="margin-top:5px;font-size:11px;padding:3px 8px">+添加</button></div>';
            
            // Y轴设置
            if (yLabels) {
                html += '<div style="margin-bottom:15px;padding:10px;background:#f9f9f9;border-radius:8px">';
                html += '<strong>Y轴刻度标签</strong>';
                html += '<div style="margin:8px 0"><label>刻度间距(px):</label><input type="number" id="yTickGap_' + key + '" value="' + (cs.yTickGapPx || 60) + '" style="width:60px;padding:4px;margin-left:5px" min="20" max="200"></div>';
                html += '<table style="border-collapse:collapse;font-size:12px;margin-top:5px">';
                yLabels.forEach((label, i) => {
                    html += '<tr><td style="border:1px solid #ddd;padding:4px">刻度' + (i+1) + ':</td><td style="border:1px solid #ddd;padding:4px"><input type="text" id="ytick_' + key + '_' + i + '" value="' + label + '" style="width:100px;padding:4px"></td><td style="padding:4px"><button onclick="removeYTickLabel(\\'' + key + '\\',' + i + ')" style="font-size:10px;padding:2px 6px;color:#e74c3c">删除</button></td></tr>';
                });
                html += '</table><button onclick="addYTickLabel(\\'' + key + '\\')" style="margin-top:5px;font-size:11px;padding:3px 8px">+添加</button></div>';
            }
            
            html += '<div style="margin-top:15px;text-align:right"><button onclick="closeTickLabelsEditor()" style="padding:8px 16px;margin-right:10px;cursor:pointer">取消</button>';
            html += '<button onclick="saveTickLabels(\\'' + key + '\\')" style="padding:8px 16px;background:#667eea;color:#fff;border:none;border-radius:6px;cursor:pointer">保存</button></div>';
            
            tickLabelsEditorDiv.innerHTML = html;
            document.body.appendChild(tickLabelsEditorDiv);
            
            const overlay = document.createElement('div');
            overlay.id = 'tickEditorOverlay';
            overlay.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.5);z-index:999';
            overlay.onclick = closeTickLabelsEditor;
            document.body.appendChild(overlay);
        }
        
        function closeTickLabelsEditor() {
            if (tickLabelsEditorDiv) { tickLabelsEditorDiv.remove(); tickLabelsEditorDiv = null; }
            const overlay = document.getElementById('tickEditorOverlay');
            if (overlay) overlay.remove();
        }
        
        function addXTickLabel(key) {
            // 先保存当前输入的值
            const xLabels = [];
            let i = 0;
            while (document.getElementById('xtick_' + key + '_' + i)) {
                xLabels.push(document.getElementById('xtick_' + key + '_' + i).value);
                i++;
            }
            xLabels.push('新标签');
            
            if (!chartStyles[key]) chartStyles[key] = {};
            chartStyles[key].xTickLabels = xLabels;
            
            closeTickLabelsEditor();
            openTickLabelsEditor(key);
        }
        
        function saveTickLabels(key) {
            const info = matricesData[key];
            if (!chartStyles[key]) chartStyles[key] = {};
            const cs = chartStyles[key];
            
            // 保存X轴标签和间距
            const xLabels = [];
            let i = 0;
            while (document.getElementById('xtick_' + key + '_' + i)) {
                xLabels.push(document.getElementById('xtick_' + key + '_' + i).value);
                i++;
            }
            cs.xTickLabels = xLabels;
            info.data.labels = xLabels;
            const xGapEl = document.getElementById('xTickGap_' + key);
            if (xGapEl) cs.xTickGapPx = parseInt(xGapEl.value) || 80;
            
            // 保存Y轴标签和间距
            const yLabels = [];
            i = 0;
            while (document.getElementById('ytick_' + key + '_' + i)) {
                yLabels.push(document.getElementById('ytick_' + key + '_' + i).value);
                i++;
            }
            if (yLabels.length > 0) cs.yTickLabels = yLabels;
            const yGapEl = document.getElementById('yTickGap_' + key);
            if (yGapEl) cs.yTickGapPx = parseInt(yGapEl.value) || 60;
            
            saveAllToStorage();
            closeTickLabelsEditor();
            renderChart(key);
        }
        
        function removeXTickLabel(key, idx) {
            const xLabels = [];
            let i = 0;
            while (document.getElementById('xtick_' + key + '_' + i)) {
                if (i !== idx) xLabels.push(document.getElementById('xtick_' + key + '_' + i).value);
                i++;
            }
            if (xLabels.length < 2) { alert('至少保留2个刻度'); return; }
            if (!chartStyles[key]) chartStyles[key] = {};
            chartStyles[key].xTickLabels = xLabels;
            closeTickLabelsEditor();
            openTickLabelsEditor(key);
        }
        
        function removeYTickLabel(key, idx) {
            const yLabels = [];
            let i = 0;
            while (document.getElementById('ytick_' + key + '_' + i)) {
                if (i !== idx) yLabels.push(document.getElementById('ytick_' + key + '_' + i).value);
                i++;
            }
            if (yLabels.length < 2) { alert('至少保留2个刻度'); return; }
            if (!chartStyles[key]) chartStyles[key] = {};
            chartStyles[key].yTickLabels = yLabels;
            closeTickLabelsEditor();
            openTickLabelsEditor(key);
        }
        
        function addYTickLabel(key) {
            const yLabels = [];
            let i = 0;
            while (document.getElementById('ytick_' + key + '_' + i)) {
                yLabels.push(document.getElementById('ytick_' + key + '_' + i).value);
                i++;
            }
            yLabels.push('新刻度');
            if (!chartStyles[key]) chartStyles[key] = {};
            chartStyles[key].yTickLabels = yLabels;
            closeTickLabelsEditor();
            openTickLabelsEditor(key);
        }
        
        function setBarLabel(key, idx, label) {
            const info = matricesData[key];
            if (!info.data.labels) info.data.labels = customLabels.slice(0, currentClassCount);
            info.data.labels[idx] = label;
            saveAllToStorage();
            renderChart(key);
        }
        
        function setBarValue(key, idx, value) {
            const info = matricesData[key];
            if (!info.data.values) info.data.values = [];
            info.data.values[idx] = value;
            saveAllToStorage();
            renderChart(key);
        }
        
        function setBarCount(key, count) {
            const info = matricesData[key];
            const currentCount = (info.data.values || []).length;
            if (count > currentCount) {
                for (let i = currentCount; i < count; i++) {
                    info.data.values.push(50);
                    if (!info.data.labels) info.data.labels = [];
                    info.data.labels.push('类别' + (i+1));
                }
            } else {
                info.data.values = info.data.values.slice(0, count);
                if (info.data.labels) info.data.labels = info.data.labels.slice(0, count);
            }
            saveAllToStorage();
            renderChart(key);
        }
        
        // 图6折线图数据编辑弹窗
        let lineEditorDiv = null;
        function openLineEditor(key) {
            const info = matricesData[key];
            const data = info.data;
            const xLabels = data.xAxis || [];
            const series = data.series || [];
            
            if (lineEditorDiv) lineEditorDiv.remove();
            lineEditorDiv = document.createElement('div');
            lineEditorDiv.style.cssText = 'position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:#fff;padding:20px;border-radius:12px;box-shadow:0 10px 40px rgba(0,0,0,0.3);z-index:2000;max-height:80vh;overflow:auto;min-width:400px';
            
            let html = '<div style="font-size:16px;font-weight:bold;margin-bottom:15px">📊 编辑折线图数据</div>';
            html += '<table style="border-collapse:collapse;width:100%"><tr><th style="border:1px solid #ddd;padding:6px;background:#f5f5f5">X轴</th>';
            series.forEach((s, i) => { html += '<th style="border:1px solid #ddd;padding:6px;background:#f5f5f5">' + s.name + '</th>'; });
            html += '</tr>';
            
            xLabels.forEach((label, idx) => {
                html += '<tr><td style="border:1px solid #ddd;padding:4px"><input type="text" value="' + label + '" data-type="xlabel" data-idx="' + idx + '" style="width:60px;padding:3px;border:1px solid #ddd;border-radius:3px"></td>';
                series.forEach((s, si) => {
                    html += '<td style="border:1px solid #ddd;padding:4px"><input type="number" value="' + (s.data[idx]||0) + '" data-type="value" data-series="' + si + '" data-idx="' + idx + '" style="width:60px;padding:3px;border:1px solid #ddd;border-radius:3px"></td>';
                });
                html += '</tr>';
            });
            html += '</table>';
            html += '<div style="margin-top:15px;text-align:right"><button onclick="closeLineEditor()" style="padding:8px 16px;margin-right:10px;border:1px solid #ddd;border-radius:4px;background:#fff;cursor:pointer">取消</button>';
            html += '<button onclick="saveLineEditor(\\'' + key + '\\')" style="padding:8px 16px;border:none;border-radius:4px;background:#667eea;color:#fff;cursor:pointer">保存</button></div>';
            
            lineEditorDiv.innerHTML = html;
            document.body.appendChild(lineEditorDiv);
        }
        function closeLineEditor() { if (lineEditorDiv) { lineEditorDiv.remove(); lineEditorDiv = null; } }
        function saveLineEditor(key) {
            const info = matricesData[key];
            lineEditorDiv.querySelectorAll('input[data-type="xlabel"]').forEach(input => {
                const idx = parseInt(input.dataset.idx);
                info.data.xAxis[idx] = input.value;
            });
            lineEditorDiv.querySelectorAll('input[data-type="value"]').forEach(input => {
                const si = parseInt(input.dataset.series);
                const idx = parseInt(input.dataset.idx);
                info.data.series[si].data[idx] = parseFloat(input.value) || 0;
            });
            saveAllToStorage();
            closeLineEditor();
            renderChart(key);
        }
        
        // 图7箱线图数据编辑(简化版：提示用户使用AI导入)
        function openRealBoxplotEditor(key) {
            alert('箱线图数据较复杂，建议使用「📥 AI导入」功能导入数据。\\n\\n格式示例：\\n组1: [1,2,3,4,5,6,7,8,9,10]\\n组2: [2,3,4,5,6,7,8,9,10,11]');
        }
        
        // 图8蜘蛛图数据编辑
        let radarEditorDiv = null;
        function openRadarEditor(key) {
            const info = matricesData[key];
            const data = info.data;
            if (radarEditorDiv) radarEditorDiv.remove();
            radarEditorDiv = document.createElement('div');
            radarEditorDiv.style.cssText = 'position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:#fff;padding:20px;border-radius:12px;box-shadow:0 10px 40px rgba(0,0,0,0.3);z-index:2000;max-height:80vh;overflow:auto;min-width:400px';
            let html = '<div style="font-size:16px;font-weight:bold;margin-bottom:15px">📊 编辑蜘蛛图数据</div>';
            html += '<table style="border-collapse:collapse;width:100%"><tr><th style="border:1px solid #ddd;padding:6px;background:#f5f5f5">指标</th>';
            data.series.forEach(s => { html += '<th style="border:1px solid #ddd;padding:6px;background:#f5f5f5">' + s.name + '</th>'; });
            html += '</tr>';
            data.indicator.forEach((ind, idx) => {
                html += '<tr><td style="border:1px solid #ddd;padding:4px"><input type="text" value="' + ind.name + '" data-type="ind" data-idx="' + idx + '" style="width:60px"></td>';
                data.series.forEach((s, si) => {
                    html += '<td style="border:1px solid #ddd;padding:4px"><input type="number" value="' + (s.data[idx]||0) + '" data-type="val" data-si="' + si + '" data-idx="' + idx + '" style="width:50px"></td>';
                });
                html += '</tr>';
            });
            html += '</table><div style="margin-top:15px;text-align:right"><button onclick="closeRadarEditor()" style="padding:8px 16px;margin-right:10px;border:1px solid #ddd;border-radius:4px;background:#fff;cursor:pointer">取消</button>';
            html += '<button onclick="saveRadarEditor(\\'' + key + '\\')" style="padding:8px 16px;border:none;border-radius:4px;background:#667eea;color:#fff;cursor:pointer">保存</button></div>';
            radarEditorDiv.innerHTML = html;
            document.body.appendChild(radarEditorDiv);
        }
        function closeRadarEditor() { if (radarEditorDiv) { radarEditorDiv.remove(); radarEditorDiv = null; } }
        function saveRadarEditor(key) {
            const info = matricesData[key];
            radarEditorDiv.querySelectorAll('input[data-type="ind"]').forEach(input => {
                info.data.indicator[parseInt(input.dataset.idx)].name = input.value;
            });
            radarEditorDiv.querySelectorAll('input[data-type="val"]').forEach(input => {
                info.data.series[parseInt(input.dataset.si)].data[parseInt(input.dataset.idx)] = parseFloat(input.value) || 0;
            });
            saveAllToStorage(); closeRadarEditor(); renderChart(key);
        }
        
        // 图9双轴图数据编辑(简化版)
        function openDualAxisEditor(key) {
            alert('双轴图数据较复杂，建议使用「📥 AI导入」功能导入数据。');
        }
        
        // 图10饼图数据编辑
        let pieEditorDiv = null;
        function openPieEditor(key) {
            const info = matricesData[key];
            const data = info.data;
            if (pieEditorDiv) pieEditorDiv.remove();
            pieEditorDiv = document.createElement('div');
            pieEditorDiv.style.cssText = 'position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:#fff;padding:20px;border-radius:12px;box-shadow:0 10px 40px rgba(0,0,0,0.3);z-index:2000;max-height:80vh;overflow:auto;min-width:300px';
            let html = '<div style="font-size:16px;font-weight:bold;margin-bottom:15px">🥧 编辑饼图数据</div>';
            html += '<table style="border-collapse:collapse;width:100%"><tr><th style="border:1px solid #ddd;padding:6px;background:#f5f5f5">名称</th><th style="border:1px solid #ddd;padding:6px;background:#f5f5f5">数值</th></tr>';
            data.series.forEach((item, idx) => {
                html += '<tr><td style="border:1px solid #ddd;padding:4px"><input type="text" value="' + item.name + '" data-type="name" data-idx="' + idx + '" style="width:80px"></td>';
                html += '<td style="border:1px solid #ddd;padding:4px"><input type="number" value="' + item.value + '" data-type="value" data-idx="' + idx + '" style="width:60px"></td></tr>';
            });
            html += '</table><div style="margin-top:15px;text-align:right"><button onclick="closePieEditor()" style="padding:8px 16px;margin-right:10px;border:1px solid #ddd;border-radius:4px;background:#fff;cursor:pointer">取消</button>';
            html += '<button onclick="savePieEditor(\\'' + key + '\\')" style="padding:8px 16px;border:none;border-radius:4px;background:#667eea;color:#fff;cursor:pointer">保存</button></div>';
            pieEditorDiv.innerHTML = html;
            document.body.appendChild(pieEditorDiv);
        }
        function closePieEditor() { if (pieEditorDiv) { pieEditorDiv.remove(); pieEditorDiv = null; } }
        function savePieEditor(key) {
            const info = matricesData[key];
            pieEditorDiv.querySelectorAll('input[data-type="name"]').forEach(input => {
                info.data.series[parseInt(input.dataset.idx)].name = input.value;
            });
            pieEditorDiv.querySelectorAll('input[data-type="value"]').forEach(input => {
                info.data.series[parseInt(input.dataset.idx)].value = parseFloat(input.value) || 0;
            });
            saveAllToStorage(); closePieEditor(); renderChart(key);
        }
        
        // 多组对比图系列配置
        let boxplotSeries = {
            'fig3': {
                count: 4,
                names: ['Ours', 'Method A', 'Method B', 'Method C'],
                colors: ['#2d5a3d', '#3a6b96', '#8b2323', '#f5deb3']
            }
        };
        
        function renderMultiBar(key) {
            const info = matricesData[key];
            const data = info.data;
            const mainTitle = customTitles[key] || info.name;
            const subtitle = customSubtitles[key] || info.subtitle || '';
            const seriesConfig = boxplotSeries[key] || { count: 4, names: ['系列1','系列2','系列3','系列4'], colors: ['#2d5a3d','#3a6b96','#8b2323','#f5deb3'] };
            
            let html = `
                <div class="card-header">
                    <div style="display:flex;align-items:center;flex-wrap:wrap">
                        <span class="card-title">${mainTitle}</span>
                        <span class="card-subtitle" style="font-size:12px;color:#666;margin-left:8px">${subtitle}</span>
                    </div>
                    <div class="card-controls">
                        <button class="ctrl-btn" onclick="exportChart('${key}','png')">PNG</button>
                        <button class="ctrl-btn" onclick="exportChart('${key}','jpg')">JPG</button>
                        <button class="ctrl-btn" onclick="exportChart('${key}','tif')">TIF</button>
                        <button class="ctrl-btn" onclick="toggleStylePanel('${key}')">🎨</button>
                    </div>
                </div>
                <div class="style-panel" id="panel_${key}">
                    <div class="style-panel-header">
                        <span>📊 ${mainTitle} 样式设置</span>
                        <button class="style-panel-close" onclick="cancelStylePanel('${key}')">✕</button>
                    </div>
                    <div class="style-row"><label>主标题:</label><input type="text" value="${mainTitle}" onchange="setCustomTitle('${key}', this.value)" style="flex:1;padding:4px 8px;border:1px solid #ddd;border-radius:4px;font-size:12px"></div>
                    <div class="style-row"><label>副标题:</label><input type="text" value="${subtitle}" onchange="setCustomSubtitle('${key}', this.value)" style="flex:1;padding:4px 8px;border:1px solid #ddd;border-radius:4px;font-size:12px"></div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>图表尺寸</strong> 宽×高(英寸):
                        <input type="number" value="${(chartStyles[key]||{}).chartWidth||8}" onchange="setChartStyle('${key}','chartWidth',+this.value)" style="width:50px" step="0.5" min="3" max="15">×
                        <input type="number" value="${(chartStyles[key]||{}).chartHeight||6}" onchange="setChartStyle('${key}','chartHeight',+this.value)" style="width:50px" step="0.5" min="2" max="12">
                    </div>
                    <div class="style-row"><strong>坐标轴名称</strong></div>
                    <div class="style-row"><label>X轴:</label><input type="text" value="${(chartStyles[key]||{}).xAxisName||'Evaluation budget'}" onchange="setChartStyle('${key}','xAxisName',this.value)" style="width:120px">
                        <label style="margin-left:10px">Y轴:</label><input type="text" value="${(chartStyles[key]||{}).yAxisName||'Terminal HV'}" onchange="setChartStyle('${key}','yAxisName',this.value)" style="width:100px"></div>
                    <div class="style-row"><strong>X轴刻度标签</strong> <button type="button" onclick="event.preventDefault();openTickLabelsEditor('${key}')" style="font-size:10px;padding:2px 6px">编辑</button></div>
                    <div class="style-row"><strong>轴标签</strong> 字体:<select onchange="setChartStyle('${key}','axisLabelFont',this.value)" style="width:70px">
                        <option value="Times New Roman" ${(chartStyles[key]||{}).axisLabelFont==='Times New Roman'?'selected':''}>Times</option>
                        <option value="Arial" ${(chartStyles[key]||{}).axisLabelFont==='Arial'?'selected':''}>Arial</option>
                        <option value="SimSun" ${(chartStyles[key]||{}).axisLabelFont==='SimSun'?'selected':''}>宋体</option>
                    </select> 字号:<input type="number" value="${(chartStyles[key]||{}).axisLabelSize||12}" onchange="setChartStyle('${key}','axisLabelSize',+this.value)" style="width:40px" min="8" max="20">
                    颜色:<input type="color" id="mb_axisLabelColor_${key}" value="${(chartStyles[key]||{}).axisLabelColor||'#000000'}" onchange="setChartStyle('${key}','axisLabelColor',this.value)" style="width:30px;height:22px;border:none;cursor:pointer"><button type="button" onclick="pickColor(c=>{document.getElementById('mb_axisLabelColor_${key}').value=c;setChartStyle('${key}','axisLabelColor',c)})" style="padding:2px 4px;font-size:10px;cursor:pointer;margin-left:2px" title="取色笔">🎯</button></div>
                    <div class="style-row"><strong>刻度</strong> 字体:<select onchange="setChartStyle('${key}','axisTickFont',this.value)" style="width:70px">
                        <option value="Times New Roman" ${(chartStyles[key]||{}).axisTickFont==='Times New Roman'?'selected':''}>Times</option>
                        <option value="Arial" ${(chartStyles[key]||{}).axisTickFont==='Arial'?'selected':''}>Arial</option>
                        <option value="SimSun" ${(chartStyles[key]||{}).axisTickFont==='SimSun'?'selected':''}>宋体</option>
                    </select> 字号:<input type="number" value="${(chartStyles[key]||{}).axisTickSize||10}" onchange="setChartStyle('${key}','axisTickSize',+this.value)" style="width:40px" min="8" max="18">
                    颜色:<input type="color" id="mb_axisTickColor_${key}" value="${(chartStyles[key]||{}).axisTickColor||'#000000'}" onchange="setChartStyle('${key}','axisTickColor',this.value)" style="width:30px;height:22px;border:none;cursor:pointer"><button type="button" onclick="pickColor(c=>{document.getElementById('mb_axisTickColor_${key}').value=c;setChartStyle('${key}','axisTickColor',c)})" style="padding:2px 4px;font-size:10px;cursor:pointer;margin-left:2px" title="取色笔">🎯</button></div>
                    <div class="style-row"><strong>刻度像素间距</strong> X轴:<input type="number" value="${(chartStyles[key]||{}).xTickGapPx||80}" onchange="setChartStyle('${key}','xTickGapPx',+this.value)" style="width:50px" min="30" max="200" step="10">px
                        Y轴:<input type="number" value="${(chartStyles[key]||{}).yTickGapPx||50}" onchange="setChartStyle('${key}','yTickGapPx',+this.value)" style="width:50px" min="20" max="150" step="10">px</div>
                    <div class="style-row"><strong>图例</strong> 字体:<select onchange="setChartStyle('${key}','legendFont',this.value)" style="width:70px">
                        <option value="Times New Roman" ${(chartStyles[key]||{}).legendFont==='Times New Roman'?'selected':''}>Times</option>
                        <option value="Arial" ${(chartStyles[key]||{}).legendFont==='Arial'?'selected':''}>Arial</option>
                        <option value="SimSun" ${(chartStyles[key]||{}).legendFont==='SimSun'?'selected':''}>宋体</option>
                    </select> 字号:<input type="number" value="${(chartStyles[key]||{}).legendSize||11}" onchange="setChartStyle('${key}','legendSize',+this.value)" style="width:40px" min="8" max="16"></div>
                    <div class="style-row"><strong>图例位置</strong> 
                        X:<button onclick="adjustLegend('${key}','X',-5)" style="padding:2px 6px">◀</button>
                        <input type="range" min="0" max="100" value="${(chartStyles[key]||{}).legendX||50}" id="legendX_${key}" oninput="document.getElementById('legendXVal_${key}').textContent=this.value+'%';setChartStyle('${key}','legendX',+this.value)" style="width:60px">
                        <span id="legendXVal_${key}">${(chartStyles[key]||{}).legendX||50}%</span>
                        <button onclick="adjustLegend('${key}','X',5)" style="padding:2px 6px">▶</button>
                        Y:<button onclick="adjustLegend('${key}','Y',-2)" style="padding:2px 6px">▲</button>
                        <input type="range" min="0" max="30" value="${(chartStyles[key]||{}).legendY||0}" id="legendY_${key}" oninput="document.getElementById('legendYVal_${key}').textContent=this.value+'%';setChartStyle('${key}','legendY',+this.value)" style="width:50px">
                        <span id="legendYVal_${key}">${(chartStyles[key]||{}).legendY||0}%</span>
                        <button onclick="adjustLegend('${key}','Y',2)" style="padding:2px 6px">▼</button>
                    </div>
                    <div class="style-row"><strong>X轴名偏移</strong> 
                        <button onclick="adjustAxisOffset('${key}','xNameGap',-5)" style="padding:2px 4px">◀</button>
                        <input type="number" id="xNameGap_${key}" value="${(chartStyles[key]||{}).xNameGap||25}" onchange="setChartStyle('${key}','xNameGap',+this.value)" style="width:40px" min="-50" max="80">px
                        <button onclick="adjustAxisOffset('${key}','xNameGap',5)" style="padding:2px 4px">▶</button>
                        <strong style="margin-left:10px">Y轴名偏移</strong>
                        <button onclick="adjustAxisOffset('${key}','yNameGap',-5)" style="padding:2px 4px">◀</button>
                        <input type="number" id="yNameGap_${key}" value="${(chartStyles[key]||{}).yNameGap||35}" onchange="setChartStyle('${key}','yNameGap',+this.value)" style="width:40px" min="-50" max="80">px
                        <button onclick="adjustAxisOffset('${key}','yNameGap',5)" style="padding:2px 4px">▶</button>
                    </div>
                    <div class="style-row"><strong>X刻度标签偏移</strong> 
                        <button onclick="adjustAxisOffset('${key}','xTickMargin',-1)" style="padding:2px 4px">▲</button>
                        <input type="number" id="xTickMargin_${key}" value="${(chartStyles[key]||{}).xTickMargin||8}" onchange="setChartStyle('${key}','xTickMargin',+this.value)" style="width:40px" min="-20" max="30">px
                        <button onclick="adjustAxisOffset('${key}','xTickMargin',1)" style="padding:2px 4px">▼</button>
                        <strong style="margin-left:10px">Y刻度标签偏移</strong>
                        <button onclick="adjustAxisOffset('${key}','yTickMargin',-1)" style="padding:2px 4px">◀</button>
                        <input type="number" id="yTickMargin_${key}" value="${(chartStyles[key]||{}).yTickMargin||8}" onchange="setChartStyle('${key}','yTickMargin',+this.value)" style="width:40px" min="-20" max="30">px
                        <button onclick="adjustAxisOffset('${key}','yTickMargin',1)" style="padding:2px 4px">▶</button>
                    </div>
                    <div class="style-row"><strong>图例方向</strong>:<select onchange="setChartStyle('${key}','legendOrient',this.value)">
                        <option value="horizontal" ${(chartStyles[key]||{}).legendOrient!=='vertical'?'selected':''}>横向</option>
                        <option value="vertical" ${(chartStyles[key]||{}).legendOrient==='vertical'?'selected':''}>纵向</option>
                    </select></div>
                    <div class="style-row"><strong>图例图标尺寸</strong> 宽:<input type="number" value="${(chartStyles[key]||{}).legendItemWidth||25}" onchange="setChartStyle('${key}','legendItemWidth',+this.value)" style="width:40px" min="8" max="50">
                        高:<input type="number" value="${(chartStyles[key]||{}).legendItemHeight||14}" onchange="setChartStyle('${key}','legendItemHeight',+this.value)" style="width:40px" min="8" max="30"></div>
                    <div class="style-row"><strong>柱子宽度</strong>:<input type="number" value="${(chartStyles[key]||{}).barWidth||25}" onchange="setChartStyle('${key}','barWidth',+this.value)" style="width:50px" min="10" max="60">px</div>
                    <div class="style-row"><strong>预设配色</strong>
                        <button onclick="applyPresetColors('${key}','multibar')" style="font-size:10px;padding:2px 6px;margin-left:5px">应用配色</button>
                        <div style="display:flex;gap:3px;margin-top:4px">
                            <span style="width:18px;height:18px;background:#2d5a3d;border-radius:3px;cursor:pointer" title="深绿" onclick="setBoxplotColor('${key}',0,'#2d5a3d')"></span>
                            <span style="width:18px;height:18px;background:#3a6b96;border-radius:3px;cursor:pointer" title="蓝" onclick="setBoxplotColor('${key}',1,'#3a6b96')"></span>
                            <span style="width:18px;height:18px;background:#8b2323;border-radius:3px;cursor:pointer" title="深红" onclick="setBoxplotColor('${key}',2,'#8b2323')"></span>
                            <span style="width:18px;height:18px;background:#f5deb3;border-radius:3px;cursor:pointer" title="米色" onclick="setBoxplotColor('${key}',3,'#f5deb3')"></span>
                        </div>
                    </div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>系列设置</strong></div>
                    <div class="style-row"><label>系列数:</label><select onchange="setBoxplotCount('${key}', +this.value)">
                        ${[1,2,3,4,5,6].map(n => `<option value="${n}" ${seriesConfig.count===n?'selected':''}>${n}</option>`).join('')}
                    </select></div>
                    ${seriesConfig.names.slice(0, seriesConfig.count).map((name, i) => `
                    <div class="style-row">
                        <label>系列${i+1}:</label>
                        <input type="text" value="${name}" onchange="setBoxplotName('${key}', ${i}, this.value)" style="width:80px;padding:3px;border:1px solid #ddd;border-radius:3px;font-size:11px">
                        <input type="color" id="mbSeriesColor_${key}_${i}" value="${seriesConfig.colors[i]}" onchange="setBoxplotColor('${key}', ${i}, this.value)" style="width:30px;height:22px;border:none;cursor:pointer"><button type="button" onclick="pickColor(c=>{document.getElementById('mbSeriesColor_${key}_${i}').value=c;setBoxplotColor('${key}',${i},c)})" style="padding:2px 4px;font-size:10px;cursor:pointer;margin-left:2px" title="取色笔">🎯</button>
                    </div>`).join('')}
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>数值编辑</strong> <button type="button" onclick="event.preventDefault();openMultiBarEditor('${key}')" style="font-size:10px;padding:2px 6px">编辑数据</button>
                        <button onclick="openChartImport('${key}','multibar')" style="font-size:10px;padding:2px 6px;margin-left:10px;background:#27ae60;color:#fff;border:none;border-radius:4px">📥 AI导入</button></div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>节点管理</strong>
                        <button onclick="openChartSaveModal('${key}')" style="font-size:10px;padding:3px 8px;margin-left:10px;background:#27ae60;color:#fff;border:none;border-radius:4px">💾 保存</button>
                        <button onclick="openChartRestoreModal('${key}')" style="font-size:10px;padding:3px 8px;margin-left:5px;background:#3498db;color:#fff;border:none;border-radius:4px">🔄 恢复</button>
                    </div>
                    <hr style="margin:10px 0;border:none;border-top:1px solid #ddd;">
                    <div class="style-row" style="justify-content:flex-end;gap:10px">
                        <button onclick="cancelStylePanel('${key}')" style="padding:6px 16px;cursor:pointer;border:1px solid #ddd;border-radius:4px;background:#fff">取消</button>
                        <button onclick="saveStylePanel('${key}')" style="padding:6px 16px;cursor:pointer;border:none;border-radius:4px;background:#667eea;color:#fff">保存</button>
                    </div>
                </div>
                <div id="chart_${key}" style="width:${(() => {
                    const cs = chartStyles[key] || {};
                    const xLabels = cs.xTickLabels || data.labels || ['B=15', 'B=20', 'B=25', 'B=30'];
                    const xGap = cs.xTickGapPx || 150;
                    const seriesCount = seriesConfig.count || 4;
                    const barWidth = 25;
                    const seriesWidth = seriesCount * barWidth * 1.5;
                    return Math.max((xLabels.length - 1) * xGap + 200, xLabels.length * seriesWidth + 200, 700);
                })()}px;height:${(() => {
                    const cs = chartStyles[key] || {};
                    const yLabels = cs.yTickLabels || ['0','20','40','60','80','100'];
                    const yGap = cs.yTickGapPx || 60;
                    return Math.max((yLabels.length - 1) * yGap + 120, 400);
                })()}px"></div>
            `;
            document.getElementById('card_' + key).innerHTML = html;
            // 如果面板原本是打开的，恢复打开状态
            if (originalStyles[key]) {
                document.getElementById('panel_' + key).classList.add('show');
            }
            
            setTimeout(() => {
                if (chartInstances[key]) chartInstances[key].dispose();
                const chartDiv = document.getElementById('chart_' + key);
                
                // ★★★ 重新获取最新的样式配置，动态计算容器宽度 ★★★
                const latestCs = chartStyles[key] || {};
                const barWidth = latestCs.barWidth || 25;
                const xLabels = latestCs.xTickLabels || data.labels || ['B=15', 'B=20', 'B=25', 'B=30'];
                const yLabels = latestCs.yTickLabels || ['0','20','40','60','80','100'];
                const xTickGapPx = latestCs.xTickGapPx || 150;
                const yTickGapPx = latestCs.yTickGapPx || 60;
                const gridMargin = 2 * barWidth * seriesConfig.count + 90;
                // 基于柱子的最小宽度
                const chartAreaWidth = xLabels.length * (seriesConfig.count + 1) * barWidth;
                const barBasedWidth = chartAreaWidth + gridMargin;
                // 基于刻度像素间距的宽度
                const tickBasedWidth = xLabels.length * xTickGapPx + gridMargin;
                // 取两者最大值
                const chartWidth = Math.max(barBasedWidth, tickBasedWidth, 400);
                const chartHeight = Math.max(yLabels.length * yTickGapPx + 120, 400);
                chartDiv.style.width = chartWidth + 'px';
                chartDiv.style.height = chartHeight + 'px';
                
                chartInstances[key] = echarts.init(chartDiv);
                
                // 生成多系列数据
                const series = [];
                for (let s = 0; s < seriesConfig.count; s++) {
                    series.push({
                        type: 'bar',
                        name: seriesConfig.names[s],
                        data: data.values ? data.values.map((v, i) => v[s] || Math.random() * 3 + 2) : xLabels.map(() => Math.random() * 3 + 2),
                        itemStyle: { color: seriesConfig.colors[s] },
                        barGap: '0%',
                        barCategoryGap: barWidth + 'px'
                    });
                }
                
                chartInstances[key].setOption({
                    tooltip: { trigger: 'axis' },
                    legend: { 
                        show: true,
                        data: seriesConfig.names.slice(0, seriesConfig.count),
                        left: (latestCs.legendX || 50) + '%',
                        top: (latestCs.legendY || 0) + '%',
                        orient: latestCs.legendOrient || 'horizontal',
                        icon: 'rect',
                        itemWidth: latestCs.legendItemWidth || 25,
                        itemHeight: latestCs.legendItemHeight || 14,
                        textStyle: { fontFamily: latestCs.legendFont || 'Times New Roman', fontSize: latestCs.legendSize || 11, color: '#000' }
                    },
                    grid: { left: barWidth * seriesConfig.count + 60, right: barWidth * seriesConfig.count + 30, top: 50, bottom: 50, borderColor: '#000', borderWidth: 1, show: true },
                    xAxis: { 
                        type: 'category', 
                        boundaryGap: true,
                        triggerEvent: true,
                        name: latestCs.xAxisName || 'Evaluation budget',
                        data: xLabels,
                        nameLocation: 'middle',
                        nameGap: 30,
                        axisLine: { show: true, lineStyle: { color: '#000', width: 1 } },
                        axisTick: { show: true, lineStyle: { color: '#000' }, alignWithLabel: true },
                        nameTextStyle: { fontFamily: latestCs.axisLabelFont || 'Times New Roman', fontSize: latestCs.axisLabelSize || 12, color: '#000' },
                        axisLabel: { fontFamily: latestCs.axisTickFont || 'Times New Roman', fontSize: latestCs.axisTickSize || 10, color: '#000' }
                    },
                    yAxis: { 
                        type: 'value', 
                        triggerEvent: true,
                        name: latestCs.yAxisName || 'Terminal HV',
                        nameLocation: 'middle',
                        nameGap: 40,
                        interval: 20,
                        max: function(value) { return value.max * 1.1 > 100 ? 110 : value.max * 1.1; },
                        axisLine: { show: true, lineStyle: { color: '#000', width: 1 } },
                        axisTick: { show: true, lineStyle: { color: '#000' } },
                        splitLine: { show: false },
                        nameTextStyle: { fontFamily: latestCs.axisLabelFont || 'Times New Roman', fontSize: latestCs.axisLabelSize || 12, color: '#000' },
                        axisLabel: { 
                            fontFamily: latestCs.axisTickFont || 'Times New Roman', 
                            fontSize: latestCs.axisTickSize || 10, 
                            color: '#000', 
                            formatter: function(v) { 
                                if (v > 100) return '';
                                const yLabels = latestCs.yTickLabels || ['0', '20', '40', '60', '80', '100'];
                                const idx = Math.round(v / 20);
                                return (idx >= 0 && idx < yLabels.length) ? yLabels[idx] : v;
                            } 
                        }
                    },
                    series: series.map(s => ({ ...s, barWidth: barWidth }))
                });
                
                // ★★★ 点击柱子编辑数值功能 ★★★
                chartInstances[key].on('click', function(params) {
                    if (params.componentType === 'series') {
                        const seriesIndex = params.seriesIndex;
                        const dataIndex = params.dataIndex;
                        const seriesName = params.seriesName;
                        const currentValue = params.value;
                        showValueEditor(seriesName + ' @ ' + xLabels[dataIndex], currentValue, function(val) {
                            if (!matricesData[key].data.values) matricesData[key].data.values = xLabels.map(() => seriesConfig.names.map(() => 0));
                            if (!matricesData[key].data.values[dataIndex]) matricesData[key].data.values[dataIndex] = [];
                            matricesData[key].data.values[dataIndex][seriesIndex] = val;
                            saveAllToStorage();
                            renderMultiBar(key);
                        });
                    } else if (params.componentType === 'xAxis') {
                        if (params.targetType === 'axisLabel') {
                            // 点击X轴刻度标签编辑
                            const dataIndex = params.dataIndex;
                            if (dataIndex >= 0 && dataIndex < xLabels.length) {
                                showTextEditor(xLabels[dataIndex], function(newLabel) {
                                    if (!matricesData[key].data.labels) matricesData[key].data.labels = [...xLabels];
                                    matricesData[key].data.labels[dataIndex] = newLabel;
                                    saveAllToStorage();
                                    renderMultiBar(key);
                                });
                            }
                        } else {
                            // 点击轴名称
                            const cs = chartStyles[key] || {};
                            showTextEditor(cs.xAxisName || 'Evaluation budget', function(newName) {
                                if (!chartStyles[key]) chartStyles[key] = {};
                                chartStyles[key].xAxisName = newName;
                                saveAllToStorage();
                                renderMultiBar(key);
                            });
                        }
                    } else if (params.componentType === 'yAxis') {
                        const cs = chartStyles[key] || {};
                        if (params.targetType === 'axisLabel') {
                            // 点击Y轴刻度标签编辑
                            const yLabels = cs.yTickLabels || ['0', '20', '40', '60', '80', '100'];
                            const idx = params.value ? Math.round(params.value / 20) : 0;
                            if (idx >= 0 && idx < yLabels.length) {
                                showTextEditor(yLabels[idx], function(newVal) {
                                    if (!chartStyles[key]) chartStyles[key] = {};
                                    if (!chartStyles[key].yTickLabels) chartStyles[key].yTickLabels = [...yLabels];
                                    chartStyles[key].yTickLabels[idx] = newVal;
                                    saveAllToStorage();
                                    renderMultiBar(key);
                                });
                            }
                        } else {
                            // 点击轴名称
                            showTextEditor(cs.yAxisName || 'Terminal HV', function(newName) {
                                if (!chartStyles[key]) chartStyles[key] = {};
                                chartStyles[key].yAxisName = newName;
                                saveAllToStorage();
                                renderMultiBar(key);
                            });
                        }
                    }
                });
            }, 100);
        }
        
        function setBoxplotCount(key, count) {
            if (!boxplotSeries[key]) boxplotSeries[key] = { count: 4, names: ['Ours','Method A','Method B','Method C','Method D','Method E'], colors: ['#2d5a3d','#3a6b96','#8b2323','#f5deb3','#e67e22','#1abc9c'] };
            // 确保names和colors数组有足够的元素
            while (boxplotSeries[key].names.length < count) {
                boxplotSeries[key].names.push('系列' + (boxplotSeries[key].names.length + 1));
            }
            while (boxplotSeries[key].colors.length < count) {
                boxplotSeries[key].colors.push('#' + Math.floor(Math.random()*16777215).toString(16).padStart(6,'0'));
            }
            boxplotSeries[key].count = count;
            saveAllToStorage();
            renderChart(key);
        }
        
        function setBoxplotName(key, idx, name) {
            if (!boxplotSeries[key]) return;
            boxplotSeries[key].names[idx] = name;
            saveAllToStorage();
            renderChart(key);
        }
        
        function setBoxplotColor(key, idx, color) {
            if (!boxplotSeries[key]) return;
            boxplotSeries[key].colors[idx] = color;
            saveAllToStorage();
            renderChart(key);
        }
        
        // 图3数值编辑弹窗
        let multiBarEditorDiv = null;
        function openMultiBarEditor(key) {
            const info = matricesData[key];
            const data = info.data;
            const seriesConfig = boxplotSeries[key] || { count: 4, names: ['Ours','Method A','Method B','Method C'], colors: [] };
            const xLabels = data.labels || ['B=15', 'B=20', 'B=25', 'B=30'];
            
            // 创建弹窗
            if (multiBarEditorDiv) multiBarEditorDiv.remove();
            multiBarEditorDiv = document.createElement('div');
            multiBarEditorDiv.style.cssText = 'position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:#fff;padding:20px;border-radius:12px;box-shadow:0 10px 40px rgba(0,0,0,0.3);z-index:1000;max-height:80vh;overflow:auto';
            
            let tableHtml = '<h3 style="margin-top:0">编辑数据</h3><table style="border-collapse:collapse;font-size:12px"><tr><th style="border:1px solid #ddd;padding:6px;background:#f5f5f5">X轴</th>';
            for (let s = 0; s < seriesConfig.count; s++) {
                tableHtml += `<th style="border:1px solid #ddd;padding:6px;background:#f5f5f5">${seriesConfig.names[s]}</th>`;
            }
            tableHtml += '</tr>';
            
            for (let x = 0; x < xLabels.length; x++) {
                tableHtml += `<tr><td style="border:1px solid #ddd;padding:6px;font-weight:bold">${xLabels[x]}</td>`;
                for (let s = 0; s < seriesConfig.count; s++) {
                    const val = (data.values && data.values[x] && data.values[x][s] !== undefined) ? data.values[x][s] : 2.5;
                    tableHtml += `<td style="border:1px solid #ddd;padding:4px"><input type="number" id="mb_${key}_${x}_${s}" value="${val}" style="width:60px;padding:4px;border:1px solid #ddd;border-radius:4px" step="0.1"></td>`;
                }
                tableHtml += '</tr>';
            }
            tableHtml += '</table>';
            tableHtml += '<div style="margin-top:15px;text-align:right"><button onclick="closeMultiBarEditor()" style="padding:8px 16px;margin-right:10px;cursor:pointer">取消</button>';
            tableHtml += `<button onclick="saveMultiBarData('${key}')" style="padding:8px 16px;background:#667eea;color:#fff;border:none;border-radius:6px;cursor:pointer">保存</button></div>`;
            
            multiBarEditorDiv.innerHTML = tableHtml;
            document.body.appendChild(multiBarEditorDiv);
            
            // 添加遮罩
            const overlay = document.createElement('div');
            overlay.id = 'mbEditorOverlay';
            overlay.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.5);z-index:999';
            overlay.onclick = closeMultiBarEditor;
            document.body.appendChild(overlay);
        }
        
        function closeMultiBarEditor() {
            if (multiBarEditorDiv) { multiBarEditorDiv.remove(); multiBarEditorDiv = null; }
            const overlay = document.getElementById('mbEditorOverlay');
            if (overlay) overlay.remove();
        }
        
        function saveMultiBarData(key) {
            const info = matricesData[key];
            const data = info.data;
            const seriesConfig = boxplotSeries[key] || { count: 4, names: [], colors: [] };
            const xLabels = data.labels || ['B=15', 'B=20', 'B=25', 'B=30'];
            
            if (!data.values) data.values = [];
            for (let x = 0; x < xLabels.length; x++) {
                if (!data.values[x]) data.values[x] = [];
                for (let s = 0; s < seriesConfig.count; s++) {
                    const input = document.getElementById('mb_' + key + '_' + x + '_' + s);
                    if (input) data.values[x][s] = +input.value;
                }
            }
            saveAllToStorage();
            closeMultiBarEditor();
            renderChart(key);
        }
        
        // 图5数值编辑弹窗
        let boxplotEditorDiv = null;
        function openBoxplotEditor(key) {
            const info = matricesData[key];
            const data = info.data;
            const xLabels = data.labels || ['B=15', 'B=20', 'B=25', 'B=30'];
            const series = data.series || [];
            
            if (boxplotEditorDiv) boxplotEditorDiv.remove();
            boxplotEditorDiv = document.createElement('div');
            boxplotEditorDiv.style.cssText = 'position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:#fff;padding:20px;border-radius:12px;box-shadow:0 10px 40px rgba(0,0,0,0.3);z-index:1000;max-height:80vh;overflow:auto';
            
            let tableHtml = '<h3 style="margin-top:0">编辑箱线图数据</h3>';
            series.forEach((s, si) => {
                tableHtml += '<div style="margin-bottom:15px"><strong>' + s.name + '</strong><table style="border-collapse:collapse;font-size:12px;margin-top:5px"><tr><th style="border:1px solid #ddd;padding:6px;background:#f5f5f5">X轴</th><th style="border:1px solid #ddd;padding:6px;background:#f5f5f5">平均值</th><th style="border:1px solid #ddd;padding:6px;background:#f5f5f5">最小值</th><th style="border:1px solid #ddd;padding:6px;background:#f5f5f5">最大值</th></tr>';
                for (let x = 0; x < xLabels.length; x++) {
                    tableHtml += '<tr><td style="border:1px solid #ddd;padding:6px;font-weight:bold">' + xLabels[x] + '</td>';
                    tableHtml += '<td style="border:1px solid #ddd;padding:4px"><input type="number" id="bp_' + key + '_' + si + '_avg_' + x + '" value="' + (s.avg[x]||0) + '" style="width:60px;padding:4px" step="0.1"></td>';
                    tableHtml += '<td style="border:1px solid #ddd;padding:4px"><input type="number" id="bp_' + key + '_' + si + '_min_' + x + '" value="' + (s.min[x]||0) + '" style="width:60px;padding:4px" step="0.1"></td>';
                    tableHtml += '<td style="border:1px solid #ddd;padding:4px"><input type="number" id="bp_' + key + '_' + si + '_max_' + x + '" value="' + (s.max[x]||0) + '" style="width:60px;padding:4px" step="0.1"></td></tr>';
                }
                tableHtml += '</table></div>';
            });
            tableHtml += '<div style="margin-top:15px;text-align:right"><button onclick="closeBoxplotEditor()" style="padding:8px 16px;margin-right:10px;cursor:pointer">取消</button>';
            tableHtml += '<button onclick="saveBoxplotData(\\'' + key + '\\')" style="padding:8px 16px;background:#667eea;color:#fff;border:none;border-radius:6px;cursor:pointer">保存</button></div>';
            
            boxplotEditorDiv.innerHTML = tableHtml;
            document.body.appendChild(boxplotEditorDiv);
            
            const overlay = document.createElement('div');
            overlay.id = 'bpEditorOverlay';
            overlay.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.5);z-index:999';
            overlay.onclick = closeBoxplotEditor;
            document.body.appendChild(overlay);
        }
        
        function closeBoxplotEditor() {
            if (boxplotEditorDiv) { boxplotEditorDiv.remove(); boxplotEditorDiv = null; }
            const overlay = document.getElementById('bpEditorOverlay');
            if (overlay) overlay.remove();
        }
        
        function saveBoxplotData(key) {
            const info = matricesData[key];
            const data = info.data;
            const xLabels = data.labels || ['B=15', 'B=20', 'B=25', 'B=30'];
            const series = data.series || [];
            
            series.forEach((s, si) => {
                for (let x = 0; x < xLabels.length; x++) {
                    const avgInput = document.getElementById('bp_' + key + '_' + si + '_avg_' + x);
                    const minInput = document.getElementById('bp_' + key + '_' + si + '_min_' + x);
                    const maxInput = document.getElementById('bp_' + key + '_' + si + '_max_' + x);
                    if (avgInput) s.avg[x] = +avgInput.value;
                    if (minInput) s.min[x] = +minInput.value;
                    if (maxInput) s.max[x] = +maxInput.value;
                }
            });
            saveAllToStorage();
            closeBoxplotEditor();
            renderChart(key);
        }
        
        // 图4曲线数设置函数
        function setROCCurveCount(key, count) {
            const info = matricesData[key];
            const data = info.data;
            const defaultColors = ['#3498db', '#e67e22', '#27ae60', '#e74c3c', '#9b59b6'];
            
            while (data.curves.length < count) {
                const idx = data.curves.length;
                // 生成新的ROC曲线数据
                const newCurve = [];
                for (let i = 0; i <= 20; i++) {
                    const fpr = i / 20;
                    const tpr = Math.min(1, fpr + 0.3 + Math.random() * 0.2);
                    newCurve.push([fpr, tpr]);
                }
                data.curves.push(newCurve);
                if (!data.labels) data.labels = customLabels.slice(0, count);
                data.labels.push('曲线' + (idx + 1));
            }
            data.curves = data.curves.slice(0, count);
            if (data.labels) data.labels = data.labels.slice(0, count);
            saveAllToStorage();
            renderChart(key);
        }
        
        // 图4数据编辑弹窗（表格形式）
        let rocEditorDiv = null;
        function openROCEditor(key) {
            const info = matricesData[key];
            const data = info.data;
            const labels = data.labels || customLabels.slice(0, data.curves.length);
            
            if (rocEditorDiv) rocEditorDiv.remove();
            rocEditorDiv = document.createElement('div');
            rocEditorDiv.style.cssText = 'position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:#fff;padding:20px;border-radius:12px;box-shadow:0 10px 40px rgba(0,0,0,0.3);z-index:2000;max-height:80vh;overflow:auto;min-width:500px';
            
            let html = '<h3 style="margin-top:0">编辑ROC曲线数据</h3>';
            data.curves.forEach((curve, ci) => {
                const fprArr = curve.fpr || [];
                const tprArr = curve.tpr || [];
                html += '<div style="margin-bottom:15px"><strong>' + labels[ci] + ' (AUC=' + (curve.auc||'N/A') + ')</strong>';
                html += '<table style="border-collapse:collapse;font-size:12px;margin-top:5px;width:100%"><tr><th style="border:1px solid #ddd;padding:4px;background:#f5f5f5">点序号</th><th style="border:1px solid #ddd;padding:4px;background:#f5f5f5">FPR</th><th style="border:1px solid #ddd;padding:4px;background:#f5f5f5">TPR</th></tr>';
                for (let pi = 0; pi < fprArr.length; pi++) {
                    html += '<tr><td style="border:1px solid #ddd;padding:4px;text-align:center">' + (pi+1) + '</td>';
                    html += '<td style="border:1px solid #ddd;padding:2px"><input type="number" data-curve="' + ci + '" data-type="fpr" data-idx="' + pi + '" value="' + (fprArr[pi]||0).toFixed(3) + '" style="width:80px;padding:3px" step="0.01" min="0" max="1"></td>';
                    html += '<td style="border:1px solid #ddd;padding:2px"><input type="number" data-curve="' + ci + '" data-type="tpr" data-idx="' + pi + '" value="' + (tprArr[pi]||0).toFixed(3) + '" style="width:80px;padding:3px" step="0.01" min="0" max="1"></td></tr>';
                }
                html += '</table></div>';
            });
            html += '<div style="margin-top:15px;text-align:right"><button onclick="closeROCEditor()" style="padding:8px 16px;margin-right:10px;cursor:pointer;border:1px solid #ddd;border-radius:4px;background:#fff">取消</button>';
            html += '<button onclick="saveROCData(\\''+key+'\\')" style="padding:8px 16px;background:#667eea;color:#fff;border:none;border-radius:6px;cursor:pointer">保存</button></div>';
            
            rocEditorDiv.innerHTML = html;
            document.body.appendChild(rocEditorDiv);
        }
        
        function closeROCEditor() {
            if (rocEditorDiv) { rocEditorDiv.remove(); rocEditorDiv = null; }
        }
        
        function saveROCData(key) {
            const info = matricesData[key];
            const data = info.data;
            
            rocEditorDiv.querySelectorAll('input[data-type="fpr"]').forEach(input => {
                const ci = parseInt(input.dataset.curve);
                const pi = parseInt(input.dataset.idx);
                data.curves[ci].fpr[pi] = parseFloat(input.value) || 0;
            });
            rocEditorDiv.querySelectorAll('input[data-type="tpr"]').forEach(input => {
                const ci = parseInt(input.dataset.curve);
                const pi = parseInt(input.dataset.idx);
                data.curves[ci].tpr[pi] = parseFloat(input.value) || 0;
            });
            saveAllToStorage();
            closeROCEditor();
            renderChart(key);
        }
        
        // 图5系列设置函数
        function setBoxplotSeriesCount(key, count) {
            const info = matricesData[key];
            const data = info.data;
            const xLabels = data.labels || ['B=15', 'B=20', 'B=25', 'B=30'];
            const defaultColors = ['#2d5a3d', '#3a6b96', '#8b2323', '#f5deb3', '#f39c12', '#1abc9c'];
            
            while ((data.series || []).length < count) {
                const idx = data.series.length;
                data.series.push({
                    name: '系列' + (idx + 1),
                    avg: xLabels.map(() => 85 + Math.random() * 10),
                    min: xLabels.map(() => 80 + Math.random() * 5),
                    max: xLabels.map(() => 90 + Math.random() * 8),
                    color: defaultColors[idx % defaultColors.length]
                });
            }
            data.series = data.series.slice(0, count);
            saveAllToStorage();
            renderChart(key);
        }
        
        function setBoxplotSeriesName(key, idx, name) {
            const info = matricesData[key];
            if (info.data.series && info.data.series[idx]) {
                info.data.series[idx].name = name;
                saveAllToStorage();
                renderChart(key);
            }
        }
        
        function setBoxplotSeriesColor(key, idx, color) {
            const info = matricesData[key];
            if (info.data.series && info.data.series[idx]) {
                info.data.series[idx].color = color;
                saveAllToStorage();
                renderChart(key);
            }
        }
        
        function setROCLabel(key, idx, label) {
            const info = matricesData[key];
            if (!info.data.labels) info.data.labels = customLabels.slice(0, info.data.curves.length);
            info.data.labels[idx] = label;
            saveAllToStorage();
            renderChart(key);
        }
        
        function setROCColor(key, idx, color) {
            if (!chartStyles[key]) chartStyles[key] = {};
            if (!chartStyles[key].rocColors) chartStyles[key].rocColors = ['#3498db','#e67e22','#27ae60','#e74c3c','#9b59b6'];
            chartStyles[key].rocColors[idx] = color;
            saveAllToStorage();
            renderChart(key);
        }
        
        // 真正的箱线图渲染函数（带误差线）
        function renderBoxplot(key) {
            const info = matricesData[key];
            const data = info.data;
            const mainTitle = customTitles[key] || info.name;
            const subtitle = customSubtitles[key] || info.subtitle || '';
            const cs = chartStyles[key] || {};
            
            let html = `
                <div class="card-header">
                    <div style="display:flex;align-items:center;flex-wrap:wrap">
                        <span class="card-title">${mainTitle}</span>
                        <span class="card-subtitle" style="font-size:12px;color:#666;margin-left:8px">${subtitle}</span>
                    </div>
                    <div class="card-controls">
                        <button class="ctrl-btn" onclick="exportChart('${key}','png')">PNG</button>
                        <button class="ctrl-btn" onclick="exportChart('${key}','jpg')">JPG</button>
                        <button class="ctrl-btn" onclick="exportChart('${key}','tif')">TIF</button>
                        <button class="ctrl-btn" onclick="toggleStylePanel('${key}')">🎨</button>
                    </div>
                </div>
                <div class="style-panel" id="panel_${key}">
                    <div class="style-panel-header">
                        <span>📊 ${mainTitle} 样式设置</span>
                        <button class="style-panel-close" onclick="cancelStylePanel('${key}')">✕</button>
                    </div>
                    <div class="style-row"><label>主标题:</label><input type="text" value="${mainTitle}" onchange="setCustomTitle('${key}', this.value)" style="flex:1;padding:4px 8px;border:1px solid #ddd;border-radius:4px;font-size:12px"></div>
                    <div class="style-row"><label>副标题:</label><input type="text" value="${subtitle}" onchange="setCustomSubtitle('${key}', this.value)" style="flex:1;padding:4px 8px;border:1px solid #ddd;border-radius:4px;font-size:12px"></div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>图表尺寸</strong> 宽×高(英寸):
                        <input type="number" value="${cs.chartWidth||8}" onchange="setChartStyle('${key}','chartWidth',+this.value)" style="width:50px" step="0.5" min="3" max="15">×
                        <input type="number" value="${cs.chartHeight||6}" onchange="setChartStyle('${key}','chartHeight',+this.value)" style="width:50px" step="0.5" min="2" max="12">
                    </div>
                    <div class="style-row"><strong>坐标轴名称</strong></div>
                    <div class="style-row"><label>X轴:</label><input type="text" value="${cs.xAxisName||'Evaluation budget'}" onchange="setChartStyle('${key}','xAxisName',this.value)" style="width:120px">
                        <label style="margin-left:10px">Y轴:</label><input type="text" value="${cs.yAxisName||'准确率(%)'}" onchange="setChartStyle('${key}','yAxisName',this.value)" style="width:100px"></div>
                    <div class="style-row"><strong>X轴刻度标签</strong> <button type="button" onclick="event.preventDefault();openTickLabelsEditor('${key}')" style="font-size:10px;padding:2px 6px">编辑</button></div>
                    <div class="style-row"><strong>轴标签</strong> 字体:<select onchange="setChartStyle('${key}','axisLabelFont',this.value)" style="width:70px">
                        <option value="Times New Roman" ${cs.axisLabelFont==='Times New Roman'?'selected':''}>Times</option>
                        <option value="Arial" ${cs.axisLabelFont==='Arial'?'selected':''}>Arial</option>
                        <option value="SimSun" ${cs.axisLabelFont==='SimSun'?'selected':''}>宋体</option>
                    </select> 颜色:<input type="color" id="bp_axisLabelColor_${key}" value="${cs.axisLabelColor||'#000000'}" onchange="setChartStyle('${key}','axisLabelColor',this.value)" style="width:30px;height:22px;border:none;cursor:pointer"><button type="button" onclick="pickColor(c=>{document.getElementById('bp_axisLabelColor_${key}').value=c;setChartStyle('${key}','axisLabelColor',c)})" style="padding:2px 4px;font-size:10px;cursor:pointer;margin-left:2px" title="取色笔">🎯</button></div>
                    <div class="style-row"><strong>刻度</strong> 字体:<select onchange="setChartStyle('${key}','axisTickFont',this.value)" style="width:70px">
                        <option value="Times New Roman" ${cs.axisTickFont==='Times New Roman'?'selected':''}>Times</option>
                        <option value="Arial" ${cs.axisTickFont==='Arial'?'selected':''}>Arial</option>
                        <option value="SimSun" ${cs.axisTickFont==='SimSun'?'selected':''}>宋体</option>
                    </select> 颜色:<input type="color" id="bp_axisTickColor_${key}" value="${cs.axisTickColor||'#000000'}" onchange="setChartStyle('${key}','axisTickColor',this.value)" style="width:30px;height:22px;border:none;cursor:pointer"><button type="button" onclick="pickColor(c=>{document.getElementById('bp_axisTickColor_${key}').value=c;setChartStyle('${key}','axisTickColor',c)})" style="padding:2px 4px;font-size:10px;cursor:pointer;margin-left:2px" title="取色笔">🎯</button></div>
                    <div class="style-row"><strong>刻度像素间距</strong> X轴:<input type="number" value="${cs.xTickGapPx||80}" onchange="setChartStyle('${key}','xTickGapPx',+this.value)" style="width:50px" min="30" max="200" step="10">px
                        Y轴:<input type="number" value="${cs.yTickGapPx||50}" onchange="setChartStyle('${key}','yTickGapPx',+this.value)" style="width:50px" min="20" max="150" step="10">px</div>
                    <div class="style-row"><strong>柱子宽度</strong>:<input type="number" value="${cs.barWidth||30}" onchange="setChartStyle('${key}','barWidth',+this.value)" style="width:50px" min="10" max="60">px
                        <strong style="margin-left:10px">误差线粗细</strong>:<input type="number" value="${cs.errorLineWidth||1}" onchange="setChartStyle('${key}','errorLineWidth',+this.value)" style="width:40px" min="0.5" max="5" step="0.5">px</div>
                    <div class="style-row"><strong>预设配色</strong>
                        <button onclick="applyPresetColors('${key}','boxplot')" style="font-size:10px;padding:2px 6px;margin-left:5px">应用配色</button>
                        <div style="display:flex;gap:3px;margin-top:4px">
                            <span style="width:18px;height:18px;background:#2d5a3d;border-radius:3px;cursor:pointer" title="深绿" onclick="setBoxplotSeriesColor('${key}',0,'#2d5a3d')"></span>
                            <span style="width:18px;height:18px;background:#3a6b96;border-radius:3px;cursor:pointer" title="蓝" onclick="setBoxplotSeriesColor('${key}',1,'#3a6b96')"></span>
                            <span style="width:18px;height:18px;background:#8b2323;border-radius:3px;cursor:pointer" title="深红" onclick="setBoxplotSeriesColor('${key}',2,'#8b2323')"></span>
                            <span style="width:18px;height:18px;background:#f5deb3;border-radius:3px;cursor:pointer" title="米色" onclick="setBoxplotSeriesColor('${key}',3,'#f5deb3')"></span>
                        </div>
                    </div>
                    <div class="style-row"><strong>X轴名偏移</strong> 
                        <button onclick="adjustAxisOffset('${key}','xNameGap',-5)" style="padding:2px 4px">◀</button>
                        <input type="number" id="xNameGap_${key}" value="${cs.xNameGap||25}" onchange="setChartStyle('${key}','xNameGap',+this.value)" style="width:40px" min="-50" max="80">px
                        <button onclick="adjustAxisOffset('${key}','xNameGap',5)" style="padding:2px 4px">▶</button>
                        <strong style="margin-left:10px">Y轴名偏移</strong>
                        <button onclick="adjustAxisOffset('${key}','yNameGap',-5)" style="padding:2px 4px">◀</button>
                        <input type="number" id="yNameGap_${key}" value="${cs.yNameGap||35}" onchange="setChartStyle('${key}','yNameGap',+this.value)" style="width:40px" min="-50" max="80">px
                        <button onclick="adjustAxisOffset('${key}','yNameGap',5)" style="padding:2px 4px">▶</button>
                    </div>
                    <div class="style-row"><strong>X刻度标签偏移</strong> 
                        <button onclick="adjustAxisOffset('${key}','xTickMargin',-1)" style="padding:2px 4px">▲</button>
                        <input type="number" id="xTickMargin_${key}" value="${cs.xTickMargin||8}" onchange="setChartStyle('${key}','xTickMargin',+this.value)" style="width:40px" min="-20" max="30">px
                        <button onclick="adjustAxisOffset('${key}','xTickMargin',1)" style="padding:2px 4px">▼</button>
                        <strong style="margin-left:10px">Y刻度标签偏移</strong>
                        <button onclick="adjustAxisOffset('${key}','yTickMargin',-1)" style="padding:2px 4px">◀</button>
                        <input type="number" id="yTickMargin_${key}" value="${cs.yTickMargin||8}" onchange="setChartStyle('${key}','yTickMargin',+this.value)" style="width:40px" min="-20" max="30">px
                        <button onclick="adjustAxisOffset('${key}','yTickMargin',1)" style="padding:2px 4px">▶</button>
                    </div>
                    <div class="style-row"><strong>图例位置</strong> 
                        X:<button onclick="adjustLegend('${key}','X',-5)" style="padding:2px 6px">◀</button>
                        <input type="range" min="0" max="100" value="${cs.legendX||50}" id="legendX_${key}" oninput="document.getElementById('legendXVal_${key}').textContent=this.value+'%';setChartStyle('${key}','legendX',+this.value)" style="width:60px">
                        <span id="legendXVal_${key}">${cs.legendX||50}%</span>
                        <button onclick="adjustLegend('${key}','X',5)" style="padding:2px 6px">▶</button>
                        Y:<button onclick="adjustLegend('${key}','Y',-2)" style="padding:2px 6px">▲</button>
                        <input type="range" min="0" max="30" value="${cs.legendY||0}" id="legendY_${key}" oninput="document.getElementById('legendYVal_${key}').textContent=this.value+'%';setChartStyle('${key}','legendY',+this.value)" style="width:50px">
                        <span id="legendYVal_${key}">${cs.legendY||0}%</span>
                        <button onclick="adjustLegend('${key}','Y',2)" style="padding:2px 6px">▼</button>
                    </div>
                    <div class="style-row"><strong>图例方向</strong>:<select onchange="setChartStyle('${key}','legendOrient',this.value)">
                        <option value="horizontal" ${cs.legendOrient!=='vertical'?'selected':''}>横向</option>
                        <option value="vertical" ${cs.legendOrient==='vertical'?'selected':''}>纵向</option>
                    </select></div>
                    <div class="style-row"><strong>图例图标尺寸</strong> 宽:<input type="number" value="${cs.legendItemWidth||25}" onchange="setChartStyle('${key}','legendItemWidth',+this.value)" style="width:40px" min="8" max="50">
                        高:<input type="number" value="${cs.legendItemHeight||14}" onchange="setChartStyle('${key}','legendItemHeight',+this.value)" style="width:40px" min="8" max="30"></div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>系列设置</strong></div>
                    <div class="style-row"><label>系列数:</label><select onchange="setBoxplotSeriesCount('${key}', +this.value)">
                        ${[1,2,3,4,5,6].map(n => `<option value="${n}" ${(data.series||[]).length===n?'selected':''}>${n}</option>`).join('')}
                    </select></div>
                    ${(data.series || []).map((s, i) => `
                    <div class="style-row">
                        <label>系列${i+1}:</label>
                        <input type="text" value="${s.name}" onchange="setBoxplotSeriesName('${key}', ${i}, this.value)" style="width:80px;padding:3px;border:1px solid #ddd;border-radius:3px;font-size:11px">
                        <input type="color" id="bpSeriesColor_${key}_${i}" value="${s.color}" onchange="setBoxplotSeriesColor('${key}', ${i}, this.value)" style="width:30px;height:22px;border:none;cursor:pointer"><button type="button" onclick="pickColor(c=>{document.getElementById('bpSeriesColor_${key}_${i}').value=c;setBoxplotSeriesColor('${key}',${i},c)})" style="padding:2px 4px;font-size:10px;cursor:pointer;margin-left:2px" title="取色笔">🎯</button>
                    </div>`).join('')}
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>数值编辑</strong> <button type="button" onclick="event.preventDefault();openBoxplotEditor('${key}')" style="font-size:10px;padding:2px 6px">编辑数据</button>
                        <button onclick="openChartImport('${key}','boxplot')" style="font-size:10px;padding:2px 6px;margin-left:10px;background:#27ae60;color:#fff;border:none;border-radius:4px">📥 AI导入</button></div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>节点管理</strong>
                        <button onclick="openChartSaveModal('${key}')" style="font-size:10px;padding:3px 8px;margin-left:10px;background:#27ae60;color:#fff;border:none;border-radius:4px">💾 保存</button>
                        <button onclick="openChartRestoreModal('${key}')" style="font-size:10px;padding:3px 8px;margin-left:5px;background:#3498db;color:#fff;border:none;border-radius:4px">🔄 恢复</button>
                    </div>
                    <hr style="margin:10px 0;border:none;border-top:1px solid #ddd;">
                    <div class="style-row" style="justify-content:flex-end;gap:10px">
                        <button onclick="cancelStylePanel('${key}')" style="padding:6px 16px;cursor:pointer;border:1px solid #ddd;border-radius:4px;background:#fff">取消</button>
                        <button onclick="saveStylePanel('${key}')" style="padding:6px 16px;cursor:pointer;border:none;border-radius:4px;background:#667eea;color:#fff">保存</button>
                    </div>
                </div>
                <div id="chart_${key}" style="width:${(() => {
                    const seriesCount = (data.series || []).length;
                    const barWidth = cs.barWidth || 30;
                    const xLabels = cs.xTickLabels || data.labels || ['A','B','C','D','E'];
                    // ★★★ 关键公式：柱状图间隙控制 ★★★
                    // 目标：相邻刻度之间的柱子间隙 = 1个柱子宽度
                    // 原理：每个刻度占用空间 = 柱子组宽度(seriesCount*barWidth) + 间隙(barWidth) = (seriesCount+1)*barWidth
                    // 容器宽度 = 刻度数 * (seriesCount+1) * barWidth + grid边距(140)
                    // 这样ECharts会自动将剩余空间平均分配，使间隙等于1个柱子宽度
                    const chartWidth = xLabels.length * (seriesCount + 1) * barWidth + 140;
                    return Math.max(chartWidth, 500);
                })()}px;height:${(() => {
                    const yLabels = cs.yTickLabels || ['0','20','40','60','80','100'];
                    const yGap = cs.yTickGapPx || 60;
                    return Math.max((yLabels.length - 1) * yGap + 120, 400);
                })()}px"></div>
            `;
            document.getElementById('card_' + key).innerHTML = html;
            // 如果面板原本是打开的，恢复打开状态
            if (originalStyles[key]) {
                document.getElementById('panel_' + key).classList.add('show');
            }
            
            setTimeout(() => {
                if (chartInstances[key]) chartInstances[key].dispose();
                const chartDiv = document.getElementById('chart_' + key);
                
                // ★★★ 重新获取最新的样式配置 ★★★
                const latestCs = chartStyles[key] || {};
                const seriesCount = (data.series || []).length;
                const barWidth = latestCs.barWidth || 30;
                const xLabels = latestCs.xTickLabels || data.labels || ['A','B','C','D','E'];
                const yLabels = latestCs.yTickLabels || ['0','20','40','60','80','100'];
                const xTickGapPx = latestCs.xTickGapPx || 80;
                const yTickGapPx = latestCs.yTickGapPx || 50;
                
                // ★★★ 关键公式：容器宽度 = 图表区域 + grid边距 ★★★
                const gridMargin = 2 * barWidth + 90;
                // 基于柱子的最小宽度
                const chartAreaWidth = xLabels.length * (seriesCount + 1) * barWidth;
                const barBasedWidth = chartAreaWidth + gridMargin;
                // 基于刻度像素间距的宽度
                const tickBasedWidth = xLabels.length * xTickGapPx + gridMargin;
                // 取两者最大值
                const chartWidth = Math.max(barBasedWidth, tickBasedWidth, 400);
                const chartHeight = Math.max(yLabels.length * yTickGapPx + 120, 400);
                chartDiv.style.width = chartWidth + 'px';
                chartDiv.style.height = chartHeight + 'px';
                
                chartInstances[key] = echarts.init(chartDiv);
                const seriesData = [];
                
                (data.series || []).forEach((s, sIdx) => {
                    const avgData = s.avg || s.values?.map(v => v[2]) || [90, 92, 94, 96];
                    const minData = s.min || s.values?.map(v => v[0]) || avgData.map(v => v - 5);
                    const maxData = s.max || s.values?.map(v => v[4]) || avgData.map(v => v + 5);
                    
                    // 柱状图显示平均值，间隙=1个柱子宽度（使用像素字符串）
                    seriesData.push({
                        type: 'bar',
                        name: s.name,
                        data: avgData,
                        barWidth: barWidth,
                        barGap: '0%',
                        barCategoryGap: barWidth + 'px',
                        itemStyle: { color: s.color }
                    });
                    
                    // 误差线使用markLine方式，直接绑定到对应柱子
                    const errorLineWidth = latestCs.errorLineWidth || 1;
                    const currentSeriesIdx = sIdx;
                    
                    seriesData.push({
                        type: 'custom',
                        name: s.name + '_error',
                        renderItem: function(params, api) {
                            const categoryIndex = api.value(0);
                            const min = api.value(2);
                            const max = api.value(3);
                            
                            // 使用ECharts内置的barLayout计算
                            const scCount = params.context?.seriesCount || data.series.length;
                            const bWidth = latestCs.barWidth || 30;
                            const bGap = 0; // barGap: '0%'
                            const totalWidth = scCount * bWidth + (scCount - 1) * bGap;
                            const categoryX = api.coord([categoryIndex, 0])[0];
                            const barCenterX = categoryX - totalWidth / 2 + bWidth / 2 + currentSeriesIdx * (bWidth + bGap);
                            
                            const pointMin = api.coord([categoryIndex, min]);
                            const pointMax = api.coord([categoryIndex, max]);
                            
                            return {
                                type: 'group',
                                children: [
                                    { type: 'line', shape: { x1: barCenterX, y1: pointMin[1], x2: barCenterX, y2: pointMax[1] }, style: { stroke: '#000', lineWidth: errorLineWidth } },
                                    { type: 'line', shape: { x1: barCenterX - 4, y1: pointMin[1], x2: barCenterX + 4, y2: pointMin[1] }, style: { stroke: '#000', lineWidth: errorLineWidth } },
                                    { type: 'line', shape: { x1: barCenterX - 4, y1: pointMax[1], x2: barCenterX + 4, y2: pointMax[1] }, style: { stroke: '#000', lineWidth: errorLineWidth } }
                                ]
                            };
                        },
                        data: avgData.map((avg, i) => [i, avg, minData[i], maxData[i]]),
                        z: 10
                    });
                });
                
                chartInstances[key].setOption({
                    tooltip: { trigger: 'axis', formatter: function(params) {
                        let result = params[0].axisValue + '<br>';
                        params.forEach(p => { if (!p.seriesName.includes('_error')) result += p.marker + p.seriesName + ': ' + p.value + '%<br>'; });
                        return result;
                    }},
                    legend: { 
                        show: true,
                        data: (data.series || []).map(s => s.name),
                        left: (latestCs.legendX || 50) + '%',
                        top: (latestCs.legendY || 0) + '%',
                        icon: 'rect',
                        itemWidth: latestCs.legendItemWidth || 25,
                        itemHeight: latestCs.legendItemHeight || 14,
                        textStyle: { fontFamily: latestCs.legendFont || 'Times New Roman', fontSize: latestCs.legendSize || 11, color: latestCs.legendColor || '#000' },
                        selectedMode: 'multiple'
                    },
                    grid: { left: barWidth + 60, right: barWidth + 30, top: 50, bottom: 50, borderColor: '#000', borderWidth: 1, show: true },
                    xAxis: { 
                        type: 'category', 
                        boundaryGap: true,
                        triggerEvent: true,
                        data: xLabels,
                        name: latestCs.xAxisName || 'Evaluation budget',
                        nameLocation: 'middle',
                        nameGap: 30,
                        axisLine: { show: true, lineStyle: { color: '#000', width: 1 } },
                        axisTick: { show: true, lineStyle: { color: '#000' }, alignWithLabel: true },
                        nameTextStyle: { fontFamily: latestCs.axisLabelFont || 'Times New Roman', fontSize: latestCs.axisLabelSize || 12, color: '#000' },
                        axisLabel: { fontFamily: latestCs.axisTickFont || 'Times New Roman', fontSize: latestCs.axisTickSize || 10, color: '#000' }
                    },
                    yAxis: { 
                        type: 'value',
                        triggerEvent: true,
                        name: latestCs.yAxisName || '准确率(%)',
                        nameLocation: 'middle',
                        nameGap: 40,
                        min: 0,
                        max: 110,
                        interval: 20,
                        axisLine: { show: true, lineStyle: { color: '#000', width: 1 } },
                        axisTick: { show: true, lineStyle: { color: '#000' } },
                        splitLine: { show: false },
                        nameTextStyle: { fontFamily: latestCs.axisLabelFont || 'Times New Roman', fontSize: latestCs.axisLabelSize || 12, color: '#000' },
                        axisLabel: { 
                            fontFamily: latestCs.axisTickFont || 'Times New Roman', 
                            fontSize: latestCs.axisTickSize || 10, 
                            color: '#000', 
                            formatter: function(v) { 
                                if (v > 100) return '';
                                const yLabels = latestCs.yTickLabels || ['0', '20', '40', '60', '80', '100'];
                                const idx = Math.round(v / 20);
                                return (idx >= 0 && idx < yLabels.length) ? yLabels[idx] : v;
                            } 
                        }
                    },
                    series: seriesData
                });
                
                // 存储图例选中状态，用于动态计算误差线位置
                let legendSelected = {};
                (data.series || []).forEach(s => { legendSelected[s.name] = true; });
                
                // 图例联动：重新渲染以更新误差线位置
                chartInstances[key].on('legendselectchanged', function(params) {
                    legendSelected = params.selected;
                    // 重新计算可见系列
                    const visibleSeries = (data.series || []).filter(s => legendSelected[s.name]);
                    const visibleCount = visibleSeries.length;
                    
                    // 重建series数据
                    const newSeriesData = [];
                    let visibleIdx = 0;
                    (data.series || []).forEach((s, sIdx) => {
                        const isVisible = legendSelected[s.name];
                        const avgData = s.avg || s.values?.map(v => v[2]) || [90, 92, 94, 96];
                        const minData = s.min || s.values?.map(v => v[0]) || avgData.map(v => v - 5);
                        const maxData = s.max || s.values?.map(v => v[4]) || avgData.map(v => v + 5);
                        const currentVisibleIdx = isVisible ? visibleIdx : 0;
                        if (isVisible) visibleIdx++;
                        
                        newSeriesData.push({
                            type: 'bar', name: s.name, data: avgData,
                            barWidth: cs.barWidth || 30, barGap: '0%',
                            barCategoryGap: (cs.barWidth || 30) + 'px',
                            itemStyle: { color: s.color }
                        });
                        
                        const errorLineWidth = cs.errorLineWidth || 1;
                        newSeriesData.push({
                            type: 'custom', name: s.name + '_error',
                            renderItem: function(params, api) {
                                if (!legendSelected[s.name]) return null;
                                const categoryIndex = api.value(0);
                                const min = api.value(2);
                                const max = api.value(3);
                                const barWidth = cs.barWidth || 30;
                                const barGap = 0;
                                const totalWidth = visibleCount * barWidth + (visibleCount - 1) * barGap;
                                const categoryX = api.coord([categoryIndex, 0])[0];
                                const barCenterX = categoryX - totalWidth / 2 + barWidth / 2 + currentVisibleIdx * (barWidth + barGap);
                                const pointMin = api.coord([categoryIndex, min]);
                                const pointMax = api.coord([categoryIndex, max]);
                                return {
                                    type: 'group',
                                    children: [
                                        { type: 'line', shape: { x1: barCenterX, y1: pointMin[1], x2: barCenterX, y2: pointMax[1] }, style: { stroke: '#000', lineWidth: errorLineWidth } },
                                        { type: 'line', shape: { x1: barCenterX - 4, y1: pointMin[1], x2: barCenterX + 4, y2: pointMin[1] }, style: { stroke: '#000', lineWidth: errorLineWidth } },
                                        { type: 'line', shape: { x1: barCenterX - 4, y1: pointMax[1], x2: barCenterX + 4, y2: pointMax[1] }, style: { stroke: '#000', lineWidth: errorLineWidth } }
                                    ]
                                };
                            },
                            data: avgData.map((avg, i) => [i, avg, minData[i], maxData[i]]),
                            z: 10
                        });
                    });
                    
                    chartInstances[key].setOption({ series: newSeriesData }, false);
                });
                
                // ★★★ 点击柱子编辑准确率，点击误差线编辑上下限 ★★★
                chartInstances[key].on('click', function(params) {
                    if (params.componentType === 'series') {
                        const seriesIndex = Math.floor(params.seriesIndex / 2);
                        const dataIndex = params.dataIndex;
                        const seriesItem = data.series[seriesIndex];
                        if (!seriesItem) return;
                        
                        if (params.seriesType === 'bar') {
                            // 点击柱子：只编辑准确率
                            const avgVal = seriesItem.avg ? seriesItem.avg[dataIndex] : 90;
                            showValueEditor(seriesItem.name + ' @ ' + xLabels[dataIndex], avgVal.toFixed(1), function(val) {
                                if (!seriesItem.avg) seriesItem.avg = xLabels.map(() => 90);
                                seriesItem.avg[dataIndex] = val;
                                saveAllToStorage();
                                renderBoxplot(key);
                            });
                        } else if (params.seriesType === 'custom') {
                            // 点击误差线：弹出美观编辑框
                            const minVal = seriesItem.min ? seriesItem.min[dataIndex] : 85;
                            const maxVal = seriesItem.max ? seriesItem.max[dataIndex] : 95;
                            showErrorBarEditor(key, seriesIndex, dataIndex, seriesItem.name, xLabels[dataIndex], minVal, maxVal);
                        }
                    } else if (params.componentType === 'xAxis') {
                        if (params.targetType === 'axisLabel') {
                            // 点击X轴刻度标签 - 编辑该标签
                            const dataIndex = params.dataIndex;
                            if (dataIndex >= 0 && dataIndex < xLabels.length) {
                                showTextEditor(xLabels[dataIndex], function(newLabel) {
                                    if (!matricesData[key].data.labels) matricesData[key].data.labels = [...xLabels];
                                    matricesData[key].data.labels[dataIndex] = newLabel;
                                    saveAllToStorage();
                                    renderBoxplot(key);
                                });
                            }
                        } else {
                            // 点击轴名称
                            const cs = chartStyles[key] || {};
                            showTextEditor(cs.xAxisName || 'Evaluation budget', function(newName) {
                                if (!chartStyles[key]) chartStyles[key] = {};
                                chartStyles[key].xAxisName = newName;
                                saveAllToStorage();
                                renderBoxplot(key);
                            });
                        }
                    } else if (params.componentType === 'yAxis') {
                        const cs = chartStyles[key] || {};
                        if (params.targetType === 'axisLabel') {
                            // 点击Y轴刻度标签 - 编辑该刻度值
                            const yLabels = cs.yTickLabels || ['0', '20', '40', '60', '80', '100'];
                            const idx = params.value ? Math.round(params.value / 20) : 0;
                            if (idx >= 0 && idx < yLabels.length) {
                                showTextEditor(yLabels[idx], function(newVal) {
                                    if (!chartStyles[key]) chartStyles[key] = {};
                                    if (!chartStyles[key].yTickLabels) chartStyles[key].yTickLabels = [...yLabels];
                                    chartStyles[key].yTickLabels[idx] = newVal;
                                    saveAllToStorage();
                                    renderBoxplot(key);
                                });
                            }
                        } else {
                            // 点击轴名称
                            showTextEditor(cs.yAxisName || '准确率(%)', function(newName) {
                                if (!chartStyles[key]) chartStyles[key] = {};
                                chartStyles[key].yAxisName = newName;
                                saveAllToStorage();
                                renderBoxplot(key);
                            });
                        }
                    }
                });
            }, 100);
        }
        
        // 误差线编辑弹窗
        let errorBarEditorDiv = null;
        function showErrorBarEditor(key, seriesIndex, dataIndex, seriesName, xLabel, minVal, maxVal) {
            if (errorBarEditorDiv) errorBarEditorDiv.remove();
            errorBarEditorDiv = document.createElement('div');
            errorBarEditorDiv.style.cssText = 'position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:#fff;padding:12px 16px;border-radius:8px;box-shadow:0 4px 20px rgba(0,0,0,0.2);z-index:1000';
            errorBarEditorDiv.innerHTML = `
                <div style="font-size:12px;color:#666;margin-bottom:8px">${seriesName} @ ${xLabel}</div>
                <div style="display:flex;gap:8px;align-items:center">
                    <label style="font-size:11px">Min:</label>
                    <input type="number" id="errorbar_min" value="${minVal.toFixed(1)}" step="0.1" style="width:60px;padding:4px;border:1px solid #ddd;border-radius:4px;font-size:12px">
                    <label style="font-size:11px">Max:</label>
                    <input type="number" id="errorbar_max" value="${maxVal.toFixed(1)}" step="0.1" style="width:60px;padding:4px;border:1px solid #ddd;border-radius:4px;font-size:12px">
                    <button onclick="saveErrorBar('${key}',${seriesIndex},${dataIndex})" style="padding:4px 10px;background:#3498db;color:#fff;border:none;border-radius:4px;font-size:11px;cursor:pointer">✓</button>
                    <button onclick="closeErrorBarEditor()" style="padding:4px 8px;background:#eee;border:none;border-radius:4px;font-size:11px;cursor:pointer">✕</button>
                </div>
            `;
            document.body.appendChild(errorBarEditorDiv);
            document.getElementById('errorbar_min').focus();
        }
        
        function closeErrorBarEditor() {
            if (errorBarEditorDiv) { errorBarEditorDiv.remove(); errorBarEditorDiv = null; }
        }
        
        // 通用数值编辑弹窗
        let valueEditorDiv = null;
        function showValueEditor(label, currentValue, onSave) {
            if (valueEditorDiv) valueEditorDiv.remove();
            valueEditorDiv = document.createElement('div');
            valueEditorDiv.style.cssText = 'position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:#fff;padding:12px 16px;border-radius:8px;box-shadow:0 4px 20px rgba(0,0,0,0.2);z-index:1000';
            valueEditorDiv.innerHTML = `
                <div style="font-size:12px;color:#666;margin-bottom:8px">${label}</div>
                <div style="display:flex;gap:8px;align-items:center">
                    <input type="number" id="value_input" value="${currentValue}" step="0.1" style="width:80px;padding:4px;border:1px solid #ddd;border-radius:4px;font-size:12px">
                    <button id="value_save" style="padding:4px 10px;background:#3498db;color:#fff;border:none;border-radius:4px;font-size:11px;cursor:pointer">✓</button>
                    <button onclick="closeValueEditor()" style="padding:4px 8px;background:#eee;border:none;border-radius:4px;font-size:11px;cursor:pointer">✕</button>
                </div>
            `;
            document.body.appendChild(valueEditorDiv);
            const input = document.getElementById('value_input');
            input.focus();
            input.select();
            document.getElementById('value_save').onclick = function() {
                const val = parseFloat(input.value);
                if (!isNaN(val)) { onSave(val); closeValueEditor(); }
            };
            input.onkeydown = function(e) { if (e.key === 'Enter') document.getElementById('value_save').click(); if (e.key === 'Escape') closeValueEditor(); };
        }
        function closeValueEditor() { if (valueEditorDiv) { valueEditorDiv.remove(); valueEditorDiv = null; } }
        
        // 文本编辑弹窗（用于轴标签）
        let textEditorDiv = null;
        function showTextEditor(currentText, onSave) {
            if (textEditorDiv) textEditorDiv.remove();
            textEditorDiv = document.createElement('div');
            textEditorDiv.style.cssText = 'position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:#fff;padding:12px 16px;border-radius:8px;box-shadow:0 4px 20px rgba(0,0,0,0.2);z-index:1000';
            textEditorDiv.innerHTML = `
                <div style="font-size:12px;color:#666;margin-bottom:8px">编辑标签</div>
                <div style="display:flex;gap:8px;align-items:center">
                    <input type="text" id="text_input" value="${currentText}" style="width:120px;padding:4px;border:1px solid #ddd;border-radius:4px;font-size:12px">
                    <button id="text_save" style="padding:4px 10px;background:#3498db;color:#fff;border:none;border-radius:4px;font-size:11px;cursor:pointer">✓</button>
                    <button onclick="closeTextEditor()" style="padding:4px 8px;background:#eee;border:none;border-radius:4px;font-size:11px;cursor:pointer">✕</button>
                </div>
            `;
            document.body.appendChild(textEditorDiv);
            const input = document.getElementById('text_input');
            input.focus();
            input.select();
            document.getElementById('text_save').onclick = function() {
                const val = input.value.trim();
                if (val) { onSave(val); closeTextEditor(); }
            };
            input.onkeydown = function(e) { if (e.key === 'Enter') document.getElementById('text_save').click(); if (e.key === 'Escape') closeTextEditor(); };
        }
        function closeTextEditor() { if (textEditorDiv) { textEditorDiv.remove(); textEditorDiv = null; } }
        
        // ★★★ 取色笔功能 ★★★
        async function pickColor(callback) {
            if (!window.EyeDropper) {
                alert('您的浏览器不支持取色笔功能，请使用Chrome 95+或Edge 95+');
                return;
            }
            try {
                const eyeDropper = new EyeDropper();
                const result = await eyeDropper.open();
                callback(result.sRGBHex);
            } catch (e) {
                console.log('取色已取消');
            }
        }
        
        // 为颜色选择器添加取色笔按钮的辅助函数
        function colorPickerWithEyedropper(inputId, currentColor, onChangeCode) {
            return `<input type="color" id="${inputId}" value="${currentColor}" onchange="${onChangeCode}" style="width:30px;height:22px;border:none;cursor:pointer"><button type="button" onclick="pickColor(c=>{document.getElementById('${inputId}').value=c;${onChangeCode.replace('this.value','c')}})" style="padding:2px 4px;font-size:10px;cursor:pointer;margin-left:2px" title="取色笔">🎯</button>`;
        }
        
        function saveErrorBar(key, seriesIndex, dataIndex) {
            const minVal = parseFloat(document.getElementById('errorbar_min').value);
            const maxVal = parseFloat(document.getElementById('errorbar_max').value);
            if (!isNaN(minVal) && !isNaN(maxVal)) {
                const seriesItem = matricesData[key].data.series[seriesIndex];
                const xLabels = matricesData[key].data.labels || ['A','B','C','D','E'];
                if (!seriesItem.min) seriesItem.min = xLabels.map(() => 85);
                if (!seriesItem.max) seriesItem.max = xLabels.map(() => 95);
                seriesItem.min[dataIndex] = minVal;
                seriesItem.max[dataIndex] = maxVal;
                saveAllToStorage();
                closeErrorBarEditor();
                renderBoxplot(key);
            }
        }
        
        // 保存打开面板前的原始状态
        let originalStyles = {};
        let originalData = {};
        let originalGlobalStyles = null;
        
        function openStylePanel(key) {
            // 关闭其他面板
            document.querySelectorAll('.style-panel.show').forEach(p => p.classList.remove('show'));
            // 保存原始状态
            originalStyles[key] = JSON.parse(JSON.stringify(chartStyles[key] || {}));
            originalData[key] = JSON.parse(JSON.stringify(matricesData[key]?.data || {}));
            // 图1需要保存globalStyles
            if (key === 'fig1') {
                originalGlobalStyles = JSON.parse(JSON.stringify(globalStyles));
            }
            const panel = document.getElementById('panel_' + key);
            panel.classList.add('show');
            // 初始化拖动功能
            initDraggable(panel);
        }
        
        function toggleStylePanel(key) { openStylePanel(key); }
        
        // 设置全局样式（只预览，不保存）
        function setGlobalStyle(prop, value) {
            globalStyles[prop] = value;
            // 只渲染fig1，不刷新所有图，不保存
            renderChart('fig1');
        }
        
        // 混淆矩阵标签偏移微调（只预览，不保存）
        function adjustCMLabel(prop, delta) {
            // 优先使用输入框中的实际值
            const input = document.querySelector('[id^="' + prop + '_"]');
            const defaultVal = prop === 'yLabelPadding' ? 2 : -8;
            const currentVal = input ? parseInt(input.value) : (globalStyles[prop] !== undefined ? globalStyles[prop] : defaultVal);
            // 允许负值(-30到30)，步进为1
            const newVal = Math.max(-30, Math.min(30, currentVal + delta));
            globalStyles[prop] = newVal;
            // 更新输入框
            document.querySelectorAll('[id^="' + prop + '_"]').forEach(el => el.value = newVal);
            // 只渲染fig1，不刷新所有图
            renderChart('fig1');
        }
        
        // 点击单个标签编辑
        function editCMLabel(idx) {
            showTextEditor(customLabels[idx], function(newLabel) {
                customLabels[idx] = newLabel;
                saveAllToStorage();
                renderAll();
            });
        }
        
        // 标签编辑弹窗
        function openLabelEditor() {
            let html = '<div style="font-size:14px;margin-bottom:10px"><strong>编辑类别标签</strong></div>';
            for (let i = 0; i < currentClassCount; i++) {
                html += `<div style="margin:8px 0"><label style="display:inline-block;width:60px">类别${i+1}:</label><input type="text" id="label_${i}" value="${customLabels[i]}" style="padding:5px;width:150px;border:1px solid #ddd;border-radius:4px"></div>`;
            }
            document.getElementById('modalTitle').innerHTML = html;
            document.getElementById('modalInput').style.display = 'none';
            document.getElementById('editModal').classList.add('show');
            currentEdit = { type: 'labels' };
        }
        
        // 保存标签
        function saveLabels() {
            for (let i = 0; i < currentClassCount; i++) {
                const input = document.getElementById('label_' + i);
                if (input) customLabels[i] = input.value || fullClassNames[i] || 'Class' + (i+1);
            }
            saveAllToStorage();
            renderAll();
        }
        
        // 编辑单元格
        function openEdit(key, i, j, val) {
            currentEdit = { key, i, j };
            document.getElementById('modalTitle').textContent = `编辑 ${matricesData[key].name} [${classNames[i]}, ${classNames[j]}]`;
            document.getElementById('modalInput').value = val;
            document.getElementById('editModal').classList.add('show');
            document.getElementById('modalInput').focus();
        }
        
        function closeModal() { 
            document.getElementById('editModal').classList.remove('show'); 
            document.getElementById('modalInput').style.display = 'block';
            document.getElementById('modalTitle').textContent = '编辑单元格';
            currentEdit = null; 
        }
        
        function saveEdit() {
            if (!currentEdit) return;
            if (currentEdit.type === 'labels') {
                saveLabels();
                closeModal();
                return;
            }
            const newVal = parseInt(document.getElementById('modalInput').value) || 0;
            matricesData[currentEdit.key].data[currentEdit.i][currentEdit.j] = newVal;
            saveAllToStorage();
            renderChart(currentEdit.key);
            closeModal();
        }
        
        // 浮窗拖动功能
        function initDraggable(panel) {
            const header = panel.querySelector('.style-panel-header');
            if (!header || header.dataset.draggable) return;
            header.dataset.draggable = 'true';
            
            let isDragging = false, startX, startY, startLeft, startTop;
            
            header.addEventListener('mousedown', function(e) {
                if (e.target.classList.contains('style-panel-close')) return;
                isDragging = true;
                startX = e.clientX;
                startY = e.clientY;
                const rect = panel.getBoundingClientRect();
                startLeft = rect.left;
                startTop = rect.top;
                panel.style.right = 'auto';
                panel.style.left = startLeft + 'px';
                panel.style.top = startTop + 'px';
                document.body.style.userSelect = 'none';
            });
            
            document.addEventListener('mousemove', function(e) {
                if (!isDragging) return;
                const dx = e.clientX - startX;
                const dy = e.clientY - startY;
                panel.style.left = (startLeft + dx) + 'px';
                panel.style.top = (startTop + dy) + 'px';
            });
            
            document.addEventListener('mouseup', function() {
                isDragging = false;
                document.body.style.userSelect = '';
            });
        }
        
        function saveStylePanel(key) {
            // 保存到localStorage
            saveAllToStorage();
            // 清除原始状态
            delete originalStyles[key];
            delete originalData[key];
            if (key === 'fig1') originalGlobalStyles = null;
            document.getElementById('panel_' + key).classList.remove('show');
        }
        
        function cancelStylePanel(key) {
            // 恢复原始状态
            if (originalStyles[key]) {
                chartStyles[key] = JSON.parse(JSON.stringify(originalStyles[key]));
            }
            if (originalData[key]) {
                matricesData[key].data = JSON.parse(JSON.stringify(originalData[key]));
            }
            // 图1需要恢复globalStyles
            if (key === 'fig1' && originalGlobalStyles) {
                globalStyles = JSON.parse(JSON.stringify(originalGlobalStyles));
                originalGlobalStyles = null;
            }
            delete originalStyles[key];
            delete originalData[key];
            // 重新渲染恢复原状
            renderChart(key);
            // 关闭面板
            document.getElementById('panel_' + key).classList.remove('show');
        }
        
        // 轴标签名偏移微调
        function adjustAxisOffset(key, prop, delta) {
            if (!chartStyles[key]) chartStyles[key] = {};
            // 优先使用输入框中的实际值
            const input = document.getElementById(prop + '_' + key);
            const currentVal = input ? parseInt(input.value) : (chartStyles[key][prop] || (prop === 'xNameGap' ? 25 : 35));
            // 允许负值(-50到80)，让标签可以更靠近图表
            const newVal = Math.max(-50, Math.min(80, currentVal + delta));
            chartStyles[key][prop] = newVal;
            if (input) input.value = newVal;
            saveAllToStorage();
            updateChartOnly(key);
        }
        
        // 图例位置微调
        function adjustLegend(key, axis, delta) {
            if (!chartStyles[key]) chartStyles[key] = {};
            const prop = 'legend' + axis;
            const maxVal = axis === 'X' ? 100 : 30;
            const currentVal = chartStyles[key][prop] || (axis === 'X' ? 50 : 0);
            const newVal = Math.max(0, Math.min(maxVal, currentVal + delta));
            chartStyles[key][prop] = newVal;
            // 更新滑块和显示值
            const slider = document.getElementById('legend' + axis + '_' + key);
            const valSpan = document.getElementById('legend' + axis + 'Val_' + key);
            if (slider) slider.value = newVal;
            if (valSpan) valSpan.textContent = newVal + '%';
            renderChart(key);
        }
        
        function setChartStyle(key, prop, value) {
            if (!chartStyles[key]) chartStyles[key] = { axisLabelFont: 'Times New Roman', axisLabelSize: 12, axisTickFont: 'Times New Roman', axisTickSize: 10, legendFont: 'Times New Roman', legendSize: 11, chartWidth: 10, chartHeight: 5 };
            
            // 尺寸改变时自动缩放柱宽等参数
            if (prop === 'chartWidth' || prop === 'chartHeight') {
                const oldWidth = chartStyles[key].chartWidth || 10;
                const oldHeight = chartStyles[key].chartHeight || 5;
                chartStyles[key][prop] = value;
                const newWidth = chartStyles[key].chartWidth;
                const newHeight = chartStyles[key].chartHeight;
                
                // 计算缩放比例（基于面积的平方根，保持视觉一致性）
                const oldArea = oldWidth * oldHeight;
                const newArea = newWidth * newHeight;
                const scale = Math.sqrt(newArea / oldArea);
                
                // 自动缩放柱宽
                if (chartStyles[key].barWidth) {
                    const newBarWidth = Math.round(chartStyles[key].barWidth * scale);
                    chartStyles[key].barWidth = Math.max(10, Math.min(80, newBarWidth));
                    // 更新样式面板中的柱宽显示
                    const barWidthInput = document.querySelector(`input[onchange*="setChartStyle('${key}','barWidth'"]`);
                    if (barWidthInput) barWidthInput.value = chartStyles[key].barWidth;
                }
                
                // 自动缩放字号
                const fontProps = ['axisLabelSize', 'axisTickSize', 'legendSize'];
                fontProps.forEach(fp => {
                    if (chartStyles[key][fp]) {
                        const newSize = Math.round(chartStyles[key][fp] * scale);
                        chartStyles[key][fp] = Math.max(8, Math.min(20, newSize));
                        const input = document.querySelector(`input[onchange*="setChartStyle('${key}','${fp}'"]`);
                        if (input) input.value = chartStyles[key][fp];
                    }
                });
            } else {
                chartStyles[key][prop] = value;
            }
            // 实时预览，但不保存到localStorage
            // 刻度像素间距变化需要完全重新渲染
            if (prop === 'xTickGapPx' || prop === 'yTickGapPx') {
                renderChart(key);
            } else {
                const needResize = (prop === 'chartWidth' || prop === 'chartHeight');
                updateChartOnly(key, needResize);
            }
        }
        
        // 只更新ECharts图表，保持样式面板状态
        function updateChartOnly(key, resizeChart = false) {
            const info = matricesData[key];
            const chartType = info.type || 'confusion';
            if (chartType === 'confusion') {
                renderChart(key); // 混淆矩阵需要完全重新渲染
            } else if (chartInstances[key]) {
                const cs = chartStyles[key] || {};
                const chartDiv = document.getElementById('chart_' + key);
                // 只有明确要求resize时才改变容器尺寸
                if (resizeChart && chartDiv) {
                    chartDiv.style.width = (cs.chartWidth || 10) * 96 + 'px';
                    chartDiv.style.height = (cs.chartHeight || 5) * 96 + 'px';
                    chartInstances[key].resize();
                }
                const chart = chartInstances[key];
                
                if (chartType === 'bar') {
                    // 只更新ECharts选项，不重新渲染HTML
                    chart.setOption({
                        legend: { left: (cs.legendX || 50) + '%', top: (cs.legendY || 0) + '%', orient: cs.legendOrient || 'horizontal', itemWidth: cs.legendItemWidth || 25, itemHeight: cs.legendItemHeight || 14, textStyle: { fontFamily: cs.legendFont || 'Times New Roman', fontSize: cs.legendSize || 11, color: cs.legendColor || '#000' } },
                        xAxis: { name: cs.xAxisName || '类别', nameGap: cs.xNameGap || 25, interval: cs.xTickInterval || null, nameTextStyle: { fontFamily: cs.axisLabelFont || 'Times New Roman', fontSize: cs.axisLabelSize || 12, color: cs.axisLabelColor || '#000' }, axisLabel: { margin: cs.xTickMargin || 8, fontFamily: cs.axisTickFont || 'Times New Roman', fontSize: cs.axisTickSize || 10, color: cs.axisTickColor || '#000' } },
                        yAxis: { name: cs.yAxisName || '准确率(%)', nameGap: cs.yNameGap || 35, interval: cs.yTickInterval || null, nameTextStyle: { fontFamily: cs.axisLabelFont || 'Times New Roman', fontSize: cs.axisLabelSize || 12, color: cs.axisLabelColor || '#000' }, axisLabel: { margin: cs.yTickMargin || 8, fontFamily: cs.axisTickFont || 'Times New Roman', fontSize: cs.axisTickSize || 10, color: cs.axisTickColor || '#000' } }
                    });
                } else if (chartType === 'multibar') {
                    // 只更新ECharts选项，不重新渲染HTML
                    chart.setOption({
                        legend: { left: (cs.legendX || 50) + '%', top: (cs.legendY || 0) + '%', orient: cs.legendOrient || 'horizontal', itemWidth: cs.legendItemWidth || 25, itemHeight: cs.legendItemHeight || 14, textStyle: { fontFamily: cs.legendFont || 'Times New Roman', fontSize: cs.legendSize || 11, color: cs.legendColor || '#000' } },
                        xAxis: { name: cs.xAxisName || 'Evaluation budget', nameGap: cs.xNameGap || 25, interval: cs.xTickInterval || null, nameTextStyle: { fontFamily: cs.axisLabelFont || 'Times New Roman', fontSize: cs.axisLabelSize || 12, color: cs.axisLabelColor || '#000' }, axisLabel: { margin: cs.xTickMargin || 8, fontFamily: cs.axisTickFont || 'Times New Roman', fontSize: cs.axisTickSize || 10, color: cs.axisTickColor || '#000' } },
                        yAxis: { name: cs.yAxisName || 'Terminal HV', nameGap: cs.yNameGap || 35, interval: cs.yTickInterval || null, nameTextStyle: { fontFamily: cs.axisLabelFont || 'Times New Roman', fontSize: cs.axisLabelSize || 12, color: cs.axisLabelColor || '#000' }, axisLabel: { margin: cs.yTickMargin || 8, fontFamily: cs.axisTickFont || 'Times New Roman', fontSize: cs.axisTickSize || 10, color: cs.axisTickColor || '#000' } }
                    });
                } else if (chartType === 'roc') {
                    chart.setOption({
                        legend: { left: (cs.legendX || 50) + '%', top: (cs.legendY || 0) + '%', orient: cs.legendOrient || 'horizontal', itemWidth: cs.legendItemWidth || 25, itemHeight: cs.legendItemHeight || 14, textStyle: { fontFamily: cs.legendFont || 'Times New Roman', fontSize: cs.legendSize || 11, color: cs.legendColor || '#000' } },
                        xAxis: { name: cs.xAxisName || 'FPR', nameGap: cs.xNameGap || 25, interval: cs.xTickInterval || null, nameTextStyle: { fontFamily: cs.axisLabelFont || 'Times New Roman', fontSize: cs.axisLabelSize || 12, color: cs.axisLabelColor || '#000' }, axisLabel: { margin: cs.xTickMargin || 8, fontFamily: cs.axisTickFont || 'Times New Roman', fontSize: cs.axisTickSize || 10, color: cs.axisTickColor || '#000' } },
                        yAxis: { name: cs.yAxisName || 'TPR', nameGap: cs.yNameGap || 35, interval: cs.yTickInterval || null, nameTextStyle: { fontFamily: cs.axisLabelFont || 'Times New Roman', fontSize: cs.axisLabelSize || 12, color: cs.axisLabelColor || '#000' }, axisLabel: { margin: cs.yTickMargin || 8, fontFamily: cs.axisTickFont || 'Times New Roman', fontSize: cs.axisTickSize || 10, color: cs.axisTickColor || '#000' } }
                    });
                } else if (chartType === 'boxplot') {
                    // 只更新ECharts选项，不重新渲染HTML
                    chart.setOption({
                        legend: { left: (cs.legendX || 50) + '%', top: (cs.legendY || 0) + '%', orient: cs.legendOrient || 'horizontal', itemWidth: cs.legendItemWidth || 25, itemHeight: cs.legendItemHeight || 14, textStyle: { fontFamily: cs.legendFont || 'Times New Roman', fontSize: cs.legendSize || 11, color: cs.legendColor || '#000' } },
                        xAxis: { name: cs.xAxisName || 'Evaluation budget', nameGap: cs.xNameGap || 25, interval: cs.xTickInterval || null, nameTextStyle: { fontFamily: cs.axisLabelFont || 'Times New Roman', fontSize: cs.axisLabelSize || 12, color: cs.axisLabelColor || '#000' }, axisLabel: { margin: cs.xTickMargin || 8, fontFamily: cs.axisTickFont || 'Times New Roman', fontSize: cs.axisTickSize || 10, color: cs.axisTickColor || '#000' } },
                        yAxis: { name: cs.yAxisName || '准确率(%)', nameGap: cs.yNameGap || 35, interval: cs.yTickInterval || null, nameTextStyle: { fontFamily: cs.axisLabelFont || 'Times New Roman', fontSize: cs.axisLabelSize || 12, color: cs.axisLabelColor || '#000' }, axisLabel: { margin: cs.yTickMargin || 8, fontFamily: cs.axisTickFont || 'Times New Roman', fontSize: cs.axisTickSize || 10, color: cs.axisTickColor || '#000' } }
                    });
                }
            }
        }
        
        // 导出ECharts图表（支持PNG/JPG/TIF，4倍分辨率最高清）
        async function exportChart(key, format = 'png') {
            const chart = chartInstances[key];
            if (!chart) { alert('图表未初始化'); return; }
            
            const scale = 4; // 4倍分辨率，最高清
            const url = chart.getDataURL({ type: 'png', pixelRatio: scale, backgroundColor: '#fff' });
            const filename = `chart_${key}`;
            
            // 转换为Blob
            const response = await fetch(url);
            const blob = await response.blob();
            
            if (format === 'png') {
                await saveFile(blob, filename + '.png', 'image/png', '.png');
            } else if (format === 'jpg') {
                // 转换为JPG
                const canvas = document.createElement('canvas');
                const img = new Image();
                img.src = url;
                await new Promise(r => img.onload = r);
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext('2d');
                ctx.fillStyle = '#fff';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0);
                canvas.toBlob(async (jpgBlob) => {
                    await saveFile(jpgBlob, filename + '.jpg', 'image/jpeg', '.jpg');
                }, 'image/jpeg', 1.0);
            } else if (format === 'tif') {
                await loadUTIF();
                const canvas = document.createElement('canvas');
                const img = new Image();
                img.src = url;
                await new Promise(r => img.onload = r);
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext('2d');
                ctx.fillStyle = '#fff';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0);
                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                const tiffData = UTIF.encodeImage(imageData.data, canvas.width, canvas.height);
                const tifBlob = new Blob([tiffData], { type: 'image/tiff' });
                await saveFile(tifBlob, filename + '.tif', 'image/tiff', '.tif');
            }
        }
        
        function renderROC(key) {
            const info = matricesData[key];
            const data = info.data;
            const mainTitle = customTitles[key] || info.name;
            const subtitle = customSubtitles[key] || info.subtitle || '';
            
            const colors = ['#3498db', '#e67e22', '#27ae60', '#e74c3c', '#9b59b6'];
            
            let html = `
                <div class="card-header">
                    <div style="display:flex;align-items:center;flex-wrap:wrap">
                        <span class="card-title">${mainTitle}</span>
                        <span class="card-subtitle" style="font-size:12px;color:#666;margin-left:8px">${subtitle}</span>
                    </div>
                    <div class="card-controls">
                        <button class="ctrl-btn" onclick="exportChart('${key}','png')">PNG</button>
                        <button class="ctrl-btn" onclick="exportChart('${key}','jpg')">JPG</button>
                        <button class="ctrl-btn" onclick="exportChart('${key}','tif')">TIF</button>
                        <button class="ctrl-btn" onclick="toggleStylePanel('${key}')">🎨</button>
                    </div>
                </div>
                <div class="style-panel" id="panel_${key}">
                    <div class="style-panel-header">
                        <span>📊 ${mainTitle} 样式设置</span>
                        <button class="style-panel-close" onclick="cancelStylePanel('${key}')">✕</button>
                    </div>
                    <div class="style-row"><label>主标题:</label><input type="text" value="${mainTitle}" onchange="setCustomTitle('${key}', this.value)" style="flex:1;padding:4px 8px;border:1px solid #ddd;border-radius:4px;font-size:12px"></div>
                    <div class="style-row"><label>副标题:</label><input type="text" value="${subtitle}" onchange="setCustomSubtitle('${key}', this.value)" style="flex:1;padding:4px 8px;border:1px solid #ddd;border-radius:4px;font-size:12px"></div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>图表尺寸</strong> 宽×高(英寸):
                        <input type="number" value="${(chartStyles[key]||{}).chartWidth||8}" onchange="setChartStyle('${key}','chartWidth',+this.value)" style="width:50px" step="0.5" min="3" max="15">×
                        <input type="number" value="${(chartStyles[key]||{}).chartHeight||6}" onchange="setChartStyle('${key}','chartHeight',+this.value)" style="width:50px" step="0.5" min="2" max="12">
                    </div>
                    <div class="style-row"><strong>坐标轴名称</strong></div>
                    <div class="style-row"><label>X轴:</label><input type="text" value="${(chartStyles[key]||{}).xAxisName||'FPR'}" onchange="setChartStyle('${key}','xAxisName',this.value)" style="width:80px">
                        <label style="margin-left:10px">Y轴:</label><input type="text" value="${(chartStyles[key]||{}).yAxisName||'TPR'}" onchange="setChartStyle('${key}','yAxisName',this.value)" style="width:80px"></div>
                    <div class="style-row"><strong>X/Y轴刻度标签</strong> <button type="button" onclick="event.preventDefault();openTickLabelsEditor('${key}')" style="font-size:10px;padding:2px 6px">编辑</button></div>
                    <div class="style-row"><strong>轴标签</strong> 字体:<select onchange="setChartStyle('${key}','axisLabelFont',this.value)" style="width:70px">
                        <option value="Times New Roman" ${(chartStyles[key]||{}).axisLabelFont==='Times New Roman'?'selected':''}>Times</option>
                        <option value="Arial" ${(chartStyles[key]||{}).axisLabelFont==='Arial'?'selected':''}>Arial</option>
                        <option value="SimSun" ${(chartStyles[key]||{}).axisLabelFont==='SimSun'?'selected':''}>宋体</option>
                    </select> 字号:<input type="number" value="${(chartStyles[key]||{}).axisLabelSize||12}" onchange="setChartStyle('${key}','axisLabelSize',+this.value)" style="width:40px" min="8" max="20">
                    颜色:<input type="color" value="${(chartStyles[key]||{}).axisLabelColor||'#000000'}" onchange="setChartStyle('${key}','axisLabelColor',this.value)" style="width:30px;height:22px;border:none;cursor:pointer"></div>
                    <div class="style-row"><strong>刻度</strong> 字体:<select onchange="setChartStyle('${key}','axisTickFont',this.value)" style="width:70px">
                        <option value="Times New Roman" ${(chartStyles[key]||{}).axisTickFont==='Times New Roman'?'selected':''}>Times</option>
                        <option value="Arial" ${(chartStyles[key]||{}).axisTickFont==='Arial'?'selected':''}>Arial</option>
                        <option value="SimSun" ${(chartStyles[key]||{}).axisTickFont==='SimSun'?'selected':''}>宋体</option>
                    </select> 字号:<input type="number" value="${(chartStyles[key]||{}).axisTickSize||10}" onchange="setChartStyle('${key}','axisTickSize',+this.value)" style="width:40px" min="8" max="18">
                    颜色:<input type="color" value="${(chartStyles[key]||{}).axisTickColor||'#000000'}" onchange="setChartStyle('${key}','axisTickColor',this.value)" style="width:30px;height:22px;border:none;cursor:pointer"></div>
                    <div class="style-row"><strong>刻度间距</strong> X轴:<input type="number" value="${(chartStyles[key]||{}).xTickInterval||0.2}" onchange="setChartStyle('${key}','xTickInterval',+this.value)" style="width:50px" min="0.1" max="0.5" step="0.1">
                        Y轴:<input type="number" value="${(chartStyles[key]||{}).yTickInterval||0.2}" onchange="setChartStyle('${key}','yTickInterval',+this.value)" style="width:50px" min="0.1" max="0.5" step="0.1"></div>
                    <div class="style-row"><strong>X轴名偏移</strong> 
                        <button onclick="adjustAxisOffset('${key}','xNameGap',-5)" style="padding:2px 4px">◀</button>
                        <input type="number" id="xNameGap_${key}" value="${(chartStyles[key]||{}).xNameGap||25}" onchange="setChartStyle('${key}','xNameGap',+this.value)" style="width:40px" min="-50" max="80">px
                        <button onclick="adjustAxisOffset('${key}','xNameGap',5)" style="padding:2px 4px">▶</button>
                        <strong style="margin-left:10px">Y轴名偏移</strong>
                        <button onclick="adjustAxisOffset('${key}','yNameGap',-5)" style="padding:2px 4px">◀</button>
                        <input type="number" id="yNameGap_${key}" value="${(chartStyles[key]||{}).yNameGap||35}" onchange="setChartStyle('${key}','yNameGap',+this.value)" style="width:40px" min="-50" max="80">px
                        <button onclick="adjustAxisOffset('${key}','yNameGap',5)" style="padding:2px 4px">▶</button>
                    </div>
                    <div class="style-row"><strong>X刻度标签偏移</strong> 
                        <button onclick="adjustAxisOffset('${key}','xTickMargin',-1)" style="padding:2px 4px">▲</button>
                        <input type="number" id="xTickMargin_${key}" value="${(chartStyles[key]||{}).xTickMargin||8}" onchange="setChartStyle('${key}','xTickMargin',+this.value)" style="width:40px" min="-20" max="30">px
                        <button onclick="adjustAxisOffset('${key}','xTickMargin',1)" style="padding:2px 4px">▼</button>
                        <strong style="margin-left:10px">Y刻度标签偏移</strong>
                        <button onclick="adjustAxisOffset('${key}','yTickMargin',-1)" style="padding:2px 4px">◀</button>
                        <input type="number" id="yTickMargin_${key}" value="${(chartStyles[key]||{}).yTickMargin||8}" onchange="setChartStyle('${key}','yTickMargin',+this.value)" style="width:40px" min="-20" max="30">px
                        <button onclick="adjustAxisOffset('${key}','yTickMargin',1)" style="padding:2px 4px">▶</button>
                    </div>
                    <div class="style-row"><strong>图例</strong> 字体:<select onchange="setChartStyle('${key}','legendFont',this.value)" style="width:70px">
                        <option value="Times New Roman" ${(chartStyles[key]||{}).legendFont==='Times New Roman'?'selected':''}>Times</option>
                        <option value="Arial" ${(chartStyles[key]||{}).legendFont==='Arial'?'selected':''}>Arial</option>
                        <option value="SimSun" ${(chartStyles[key]||{}).legendFont==='SimSun'?'selected':''}>宋体</option>
                    </select> 字号:<input type="number" value="${(chartStyles[key]||{}).legendSize||11}" onchange="setChartStyle('${key}','legendSize',+this.value)" style="width:40px" min="8" max="16">
                    颜色:<input type="color" value="${(chartStyles[key]||{}).legendColor||'#000000'}" onchange="setChartStyle('${key}','legendColor',this.value)" style="width:30px;height:22px;border:none;cursor:pointer"></div>
                    <div class="style-row"><strong>图例位置</strong> 
                        X:<button onclick="adjustLegend('${key}','X',-5)" style="padding:2px 6px">◀</button>
                        <input type="range" min="0" max="100" value="${(chartStyles[key]||{}).legendX||50}" id="legendX_${key}" oninput="document.getElementById('legendXVal_${key}').textContent=this.value+'%';setChartStyle('${key}','legendX',+this.value)" style="width:60px">
                        <span id="legendXVal_${key}">${(chartStyles[key]||{}).legendX||50}%</span>
                        <button onclick="adjustLegend('${key}','X',5)" style="padding:2px 6px">▶</button>
                        Y:<button onclick="adjustLegend('${key}','Y',-2)" style="padding:2px 6px">▲</button>
                        <input type="range" min="0" max="30" value="${(chartStyles[key]||{}).legendY||0}" id="legendY_${key}" oninput="document.getElementById('legendYVal_${key}').textContent=this.value+'%';setChartStyle('${key}','legendY',+this.value)" style="width:50px">
                        <span id="legendYVal_${key}">${(chartStyles[key]||{}).legendY||0}%</span>
                        <button onclick="adjustLegend('${key}','Y',2)" style="padding:2px 6px">▼</button>
                    </div>
                    <div class="style-row"><strong>图例方向</strong>:<select onchange="setChartStyle('${key}','legendOrient',this.value)">
                        <option value="horizontal" ${(chartStyles[key]||{}).legendOrient!=='vertical'?'selected':''}>横向</option>
                        <option value="vertical" ${(chartStyles[key]||{}).legendOrient==='vertical'?'selected':''}>纵向</option>
                    </select></div>
                    <div class="style-row"><strong>图例图标尺寸</strong> 宽:<input type="number" value="${(chartStyles[key]||{}).legendItemWidth||25}" onchange="setChartStyle('${key}','legendItemWidth',+this.value)" style="width:40px" min="8" max="50">
                        高:<input type="number" value="${(chartStyles[key]||{}).legendItemHeight||14}" onchange="setChartStyle('${key}','legendItemHeight',+this.value)" style="width:40px" min="8" max="30"></div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>曲线设置</strong></div>
                    <div class="style-row"><label>曲线数:</label><select onchange="setROCCurveCount('${key}', +this.value)">
                        ${[1,2,3,4,5].map(n => `<option value="${n}" ${data.curves.length===n?'selected':''}>${n}</option>`).join('')}
                    </select></div>
                    ${(data.labels || customLabels).slice(0, data.curves.length).map((label, i) => `
                    <div class="style-row">
                        <label>曲线${i+1}:</label>
                        <input type="text" value="${label}" onchange="setROCLabel('${key}',${i},this.value)" style="width:60px;padding:3px;border:1px solid #ddd;border-radius:3px;font-size:11px">
                        <input type="color" value="${((chartStyles[key]||{}).rocColors||['#3498db','#e67e22','#27ae60','#e74c3c','#9b59b6'])[i]||'#3498db'}" onchange="setROCColor('${key}',${i},this.value)" style="width:30px;height:22px;border:none;cursor:pointer">
                    </div>`).join('')}
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>数据编辑</strong> <button type="button" onclick="event.preventDefault();openROCEditor('${key}')" style="font-size:10px;padding:2px 6px">编辑数据</button>
                        <button onclick="openChartImport('${key}','roc')" style="font-size:10px;padding:2px 6px;margin-left:10px;background:#27ae60;color:#fff;border:none;border-radius:4px">📥 AI导入</button></div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>节点管理</strong>
                        <button onclick="openChartSaveModal('${key}')" style="font-size:10px;padding:3px 8px;margin-left:10px;background:#27ae60;color:#fff;border:none;border-radius:4px">💾 保存</button>
                        <button onclick="openChartRestoreModal('${key}')" style="font-size:10px;padding:3px 8px;margin-left:5px;background:#3498db;color:#fff;border:none;border-radius:4px">🔄 恢复</button>
                    </div>
                    <hr style="margin:10px 0;border:none;border-top:1px solid #ddd;">
                    <div class="style-row" style="justify-content:flex-end;gap:10px">
                        <button onclick="cancelStylePanel('${key}')" style="padding:6px 16px;cursor:pointer;border:1px solid #ddd;border-radius:4px;background:#fff">取消</button>
                        <button onclick="saveStylePanel('${key}')" style="padding:6px 16px;cursor:pointer;border:none;border-radius:4px;background:#667eea;color:#fff">保存</button>
                    </div>
                </div>
                <div id="chart_${key}" style="width:${(() => {
                    const cs = chartStyles[key] || {};
                    const xLabels = cs.xTickLabels || ['0','0.2','0.4','0.6','0.8','1.0'];
                    const xGap = cs.xTickGapPx || 80;
                    return Math.max((xLabels.length - 1) * xGap + 150, 500);
                })()}px;height:${(() => {
                    const cs = chartStyles[key] || {};
                    const yLabels = cs.yTickLabels || ['0','0.2','0.4','0.6','0.8','1.0'];
                    const yGap = cs.yTickGapPx || 60;
                    return Math.max((yLabels.length - 1) * yGap + 120, 400);
                })()}px"></div>
            `;
            document.getElementById('card_' + key).innerHTML = html;
            // 如果面板原本是打开的，恢复打开状态
            if (originalStyles[key]) {
                document.getElementById('panel_' + key).classList.add('show');
            }
            
            setTimeout(() => {
                if (chartInstances[key]) chartInstances[key].dispose();
                chartInstances[key] = echarts.init(document.getElementById('chart_' + key));
                const cs = chartStyles[key] || {};
                const rocColors = cs.rocColors || ['#3498db','#e67e22','#27ae60','#e74c3c','#9b59b6'];
                const seriesNames = (data.labels || customLabels).slice(0, data.curves.length);
                const series = data.curves.map((curve, i) => ({
                    type: 'line', name: seriesNames[i] + ' (AUC=' + curve.auc + ')',
                    data: curve.fpr.map((fpr, j) => [fpr, curve.tpr[j]]), smooth: true, lineStyle: { color: rocColors[i % rocColors.length] }, itemStyle: { color: rocColors[i % rocColors.length] }
                }));
                series.push({ type: 'line', name: 'Random', data: [[0,0],[1,1]], lineStyle: { type: 'dashed', color: 'gray', width: 1 }, symbol: 'none' });
                chartInstances[key].setOption({
                    title: { text: cs.chartTitle || 'ROC Curve Example', left: 'center', textStyle: { fontSize: 14, fontFamily: 'Times New Roman' }, triggerEvent: true },
                    tooltip: { trigger: 'axis' }, 
                    legend: { 
                        show: true,
                        right: (cs.legendX !== undefined ? 'auto' : '5%'),
                        bottom: (cs.legendY !== undefined ? 'auto' : '10%'),
                        left: cs.legendX !== undefined ? (cs.legendX + '%') : 'auto',
                        top: cs.legendY !== undefined ? (cs.legendY + '%') : 'auto',
                        orient: cs.legendOrient || 'vertical',
                        icon: 'rect',
                        itemWidth: cs.legendItemWidth || 25,
                        itemHeight: cs.legendItemHeight || 14,
                        textStyle: { fontFamily: cs.legendFont || 'Times New Roman', fontSize: cs.legendSize || 11, color: cs.legendColor || '#000' }
                    },
                    grid: { left: 70, right: 30, top: 60, bottom: 60, borderColor: '#000', borderWidth: 1, show: true },
                    xAxis: { 
                        type: 'value', name: cs.xAxisName || 'False Positive Rate', min: 0, max: 1,
                        triggerEvent: true,
                        nameLocation: 'middle',
                        nameGap: cs.xNameGap || 25,
                        interval: cs.xTickInterval || 0.2,
                        axisLine: { show: true, lineStyle: { color: '#000', width: 1 } },
                        axisTick: { show: true, lineStyle: { color: '#000' } },
                        splitLine: { show: false },
                        nameTextStyle: { fontFamily: cs.axisLabelFont || 'Times New Roman', fontSize: cs.axisLabelSize || 12, color: cs.axisLabelColor || '#000' },
                        axisLabel: { 
                            fontFamily: cs.axisTickFont || 'Times New Roman', 
                            fontSize: cs.axisTickSize || 10, 
                            color: cs.axisTickColor || '#000',
                            formatter: function(value) {
                                const xLabels = cs.xTickLabels || ['0', '0.2', '0.4', '0.6', '0.8', '1.0'];
                                const interval = cs.xTickInterval || 0.2;
                                const idx = Math.round(value / interval);
                                return (idx >= 0 && idx < xLabels.length) ? xLabels[idx] : value.toFixed(1);
                            }
                        }
                    },
                    yAxis: { 
                        type: 'value', name: cs.yAxisName || 'True Positive Rate', min: 0, max: 1.05,
                        triggerEvent: true,
                        nameLocation: 'middle',
                        nameGap: cs.yNameGap || 35,
                        interval: cs.yTickInterval || 0.2,
                        axisLine: { show: true, lineStyle: { color: '#000', width: 1 } },
                        axisTick: { show: true, lineStyle: { color: '#000' } },
                        splitLine: { show: false },
                        nameTextStyle: { fontFamily: cs.axisLabelFont || 'Times New Roman', fontSize: cs.axisLabelSize || 12, color: cs.axisLabelColor || '#000' },
                        axisLabel: { 
                            fontFamily: cs.axisTickFont || 'Times New Roman', 
                            fontSize: cs.axisTickSize || 10, 
                            color: cs.axisTickColor || '#000',
                            formatter: function(value) {
                                const yLabels = cs.yTickLabels || ['0', '0.2', '0.4', '0.6', '0.8', '1.0'];
                                const interval = cs.yTickInterval || 0.2;
                                const idx = Math.round(value / interval);
                                return (idx >= 0 && idx < yLabels.length) ? yLabels[idx] : value.toFixed(1);
                            }
                        }
                    },
                    series: series
                });
                
                // 点击编辑
                chartInstances[key].on('click', function(params) {
                    // 点击标题编辑
                    if (params.componentType === 'title') {
                        showTextEditor(chartStyles[key]?.chartTitle || 'ROC Curve Example', function(newTitle) {
                            if (!chartStyles[key]) chartStyles[key] = {};
                            chartStyles[key].chartTitle = newTitle;
                            saveAllToStorage();
                            renderROC(key);
                        });
                    }
                    else if (params.componentType === 'xAxis') {
                        if (params.targetType === 'axisLabel') {
                            // 点击刻度标签 - 编辑该刻度值
                            const xLabels = chartStyles[key]?.xTickLabels || ['0', '0.2', '0.4', '0.6', '0.8', '1.0'];
                            const interval = chartStyles[key]?.xTickInterval || 0.2;
                            const idx = Math.round(params.value / interval);
                            if (idx >= 0 && idx < xLabels.length) {
                                showTextEditor(xLabels[idx], function(newVal) {
                                    if (!chartStyles[key]) chartStyles[key] = {};
                                    if (!chartStyles[key].xTickLabels) chartStyles[key].xTickLabels = [...xLabels];
                                    chartStyles[key].xTickLabels[idx] = newVal;
                                    saveAllToStorage();
                                    renderROC(key);
                                });
                            }
                        } else {
                            // 点击轴名称
                            showTextEditor(cs.xAxisName || 'FPR', function(newName) {
                                if (!chartStyles[key]) chartStyles[key] = {};
                                chartStyles[key].xAxisName = newName;
                                saveAllToStorage();
                                renderROC(key);
                            });
                        }
                    } else if (params.componentType === 'yAxis') {
                        if (params.targetType === 'axisLabel') {
                            // 点击刻度标签 - 编辑该刻度值
                            const yLabels = chartStyles[key]?.yTickLabels || ['0', '0.2', '0.4', '0.6', '0.8', '1.0'];
                            const interval = chartStyles[key]?.yTickInterval || 0.2;
                            const idx = Math.round(params.value / interval);
                            if (idx >= 0 && idx < yLabels.length) {
                                showTextEditor(yLabels[idx], function(newVal) {
                                    if (!chartStyles[key]) chartStyles[key] = {};
                                    if (!chartStyles[key].yTickLabels) chartStyles[key].yTickLabels = [...yLabels];
                                    chartStyles[key].yTickLabels[idx] = newVal;
                                    saveAllToStorage();
                                    renderROC(key);
                                });
                            }
                        } else {
                            // 点击轴名称
                            showTextEditor(cs.yAxisName || 'TPR', function(newName) {
                                if (!chartStyles[key]) chartStyles[key] = {};
                                chartStyles[key].yAxisName = newName;
                                saveAllToStorage();
                                renderROC(key);
                            });
                        }
                    }
                    // 点击曲线数据点编辑
                    else if (params.componentType === 'series' && params.seriesType === 'line' && params.seriesName !== 'Random') {
                        const seriesIdx = params.seriesIndex;
                        const dataIdx = params.dataIndex;
                        const curve = matricesData[key].data.curves[seriesIdx];
                        if (curve && curve.fpr && curve.tpr) {
                            const fprVal = curve.fpr[dataIdx];
                            const tprVal = curve.tpr[dataIdx];
                            const newFpr = prompt('编辑FPR值 (0-1):', fprVal);
                            if (newFpr !== null) {
                                const newTpr = prompt('编辑TPR值 (0-1):', tprVal);
                                if (newTpr !== null) {
                                    curve.fpr[dataIdx] = parseFloat(newFpr) || 0;
                                    curve.tpr[dataIdx] = parseFloat(newTpr) || 0;
                                    saveAllToStorage();
                                    renderROC(key);
                                }
                            }
                        }
                    }
                });
            }, 100);
        }
        
        // ============ 新增18种图表渲染函数 ============
        
        // 图6: 折线图(参考图2样式)
        function renderLineChart(key) {
            const info = matricesData[key];
            const data = info.data;
            const mainTitle = customTitles[key] || info.name;
            const subtitle = customSubtitles[key] || info.subtitle || '';
            const cs = chartStyles[key] || {};
            
            let html = `
                <div class="card-header">
                    <div><span class="card-title">${mainTitle}</span><span style="font-size:12px;color:#666;margin-left:8px">${subtitle}</span></div>
                    <div class="card-controls">
                        <button class="ctrl-btn" onclick="exportChart('${key}','png')">PNG</button>
                        <button class="ctrl-btn" onclick="exportChart('${key}','jpg')">JPG</button>
                        <button class="ctrl-btn" onclick="exportChart('${key}','tif')">TIF</button>
                        <button class="ctrl-btn" onclick="toggleStylePanel('${key}')">🎨</button>
                    </div>
                </div>
                <div class="style-panel" id="panel_${key}">
                    <div class="style-panel-header">
                        <span>📊 ${mainTitle} 样式设置</span>
                        <button class="style-panel-close" onclick="cancelStylePanel('${key}')">✕</button>
                    </div>
                    <div class="style-row"><label>主标题:</label><input type="text" value="${mainTitle}" onchange="setCustomTitle('${key}', this.value)" style="flex:1;padding:4px 8px;border:1px solid #ddd;border-radius:4px;font-size:12px"></div>
                    <div class="style-row"><label>副标题:</label><input type="text" value="${subtitle}" onchange="setCustomSubtitle('${key}', this.value)" style="flex:1;padding:4px 8px;border:1px solid #ddd;border-radius:4px;font-size:12px"></div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>坐标轴名称</strong></div>
                    <div class="style-row"><label>X轴:</label><input type="text" value="${cs.xAxisName||'X轴'}" onchange="setChartStyle('${key}','xAxisName',this.value)" style="width:80px">
                        <label style="margin-left:10px">Y轴:</label><input type="text" value="${cs.yAxisName||'Y轴'}" onchange="setChartStyle('${key}','yAxisName',this.value)" style="width:80px"></div>
                    <div class="style-row"><strong>轴标签</strong> 字体:<select onchange="setChartStyle('${key}','axisLabelFont',this.value)" style="width:70px">
                        <option value="Times New Roman" ${cs.axisLabelFont==='Times New Roman'?'selected':''}>Times</option>
                        <option value="Arial" ${cs.axisLabelFont==='Arial'?'selected':''}>Arial</option>
                    </select> 字号:<input type="number" value="${cs.axisLabelSize||12}" onchange="setChartStyle('${key}','axisLabelSize',+this.value)" style="width:40px" min="8" max="20">
                    颜色:<input type="color" value="${cs.axisLabelColor||'#000000'}" onchange="setChartStyle('${key}','axisLabelColor',this.value)" style="width:30px;height:22px"></div>
                    <div class="style-row"><strong>刻度</strong> 字体:<select onchange="setChartStyle('${key}','axisTickFont',this.value)" style="width:70px">
                        <option value="Times New Roman" ${cs.axisTickFont==='Times New Roman'?'selected':''}>Times</option>
                        <option value="Arial" ${cs.axisTickFont==='Arial'?'selected':''}>Arial</option>
                    </select> 字号:<input type="number" value="${cs.axisTickSize||10}" onchange="setChartStyle('${key}','axisTickSize',+this.value)" style="width:40px" min="8" max="18">
                    颜色:<input type="color" value="${cs.axisTickColor||'#000000'}" onchange="setChartStyle('${key}','axisTickColor',this.value)" style="width:30px;height:22px"></div>
                    <div class="style-row"><strong>刻度像素间距</strong> X轴:<input type="number" value="${cs.xTickGapPx||80}" onchange="setChartStyle('${key}','xTickGapPx',+this.value)" style="width:50px" min="30" max="200" step="10">px
                        Y轴:<input type="number" value="${cs.yTickGapPx||50}" onchange="setChartStyle('${key}','yTickGapPx',+this.value)" style="width:50px" min="20" max="150" step="10">px</div>
                    <div class="style-row"><strong>X轴名偏移</strong> 
                        <button onclick="adjustAxisOffset('${key}','xNameGap',-5)" style="padding:2px 4px">◀</button>
                        <input type="number" id="xNameGap_${key}" value="${cs.xNameGap||30}" onchange="setChartStyle('${key}','xNameGap',+this.value)" style="width:40px" min="-50" max="80">px
                        <button onclick="adjustAxisOffset('${key}','xNameGap',5)" style="padding:2px 4px">▶</button>
                        <strong style="margin-left:10px">Y轴名偏移</strong>
                        <button onclick="adjustAxisOffset('${key}','yNameGap',-5)" style="padding:2px 4px">◀</button>
                        <input type="number" id="yNameGap_${key}" value="${cs.yNameGap||40}" onchange="setChartStyle('${key}','yNameGap',+this.value)" style="width:40px" min="-50" max="80">px
                        <button onclick="adjustAxisOffset('${key}','yNameGap',5)" style="padding:2px 4px">▶</button>
                    </div>
                    <div class="style-row"><strong>X刻度标签偏移</strong> 
                        <button onclick="adjustAxisOffset('${key}','xTickMargin',-1)" style="padding:2px 4px">▲</button>
                        <input type="number" id="xTickMargin_${key}" value="${cs.xTickMargin||8}" onchange="setChartStyle('${key}','xTickMargin',+this.value)" style="width:40px" min="-20" max="30">px
                        <button onclick="adjustAxisOffset('${key}','xTickMargin',1)" style="padding:2px 4px">▼</button>
                        <strong style="margin-left:10px">Y刻度标签偏移</strong>
                        <button onclick="adjustAxisOffset('${key}','yTickMargin',-1)" style="padding:2px 4px">◀</button>
                        <input type="number" id="yTickMargin_${key}" value="${cs.yTickMargin||8}" onchange="setChartStyle('${key}','yTickMargin',+this.value)" style="width:40px" min="-20" max="30">px
                        <button onclick="adjustAxisOffset('${key}','yTickMargin',1)" style="padding:2px 4px">▶</button>
                    </div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>图表尺寸</strong> 宽×高(px):
                        <input type="number" value="${cs.chartWidth||600}" onchange="setChartStyle('${key}','chartWidth',+this.value);renderChart('${key}')" style="width:60px" min="300" max="1200">×
                        <input type="number" value="${cs.chartHeight||400}" onchange="setChartStyle('${key}','chartHeight',+this.value);renderChart('${key}')" style="width:60px" min="200" max="800">
                    </div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>数据编辑</strong>
                        <button onclick="openLineEditor('${key}')" style="font-size:10px;padding:2px 6px;margin-left:10px;background:#3498db;color:#fff;border:none;border-radius:4px">📝 编辑数据</button>
                        <button onclick="openChartImport('${key}','line')" style="font-size:10px;padding:2px 6px;margin-left:5px;background:#27ae60;color:#fff;border:none;border-radius:4px">📥 AI导入</button>
                    </div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>节点管理</strong>
                        <button onclick="openChartSaveModal('${key}')" style="font-size:10px;padding:3px 8px;margin-left:10px;background:#27ae60;color:#fff;border:none;border-radius:4px">💾 保存</button>
                        <button onclick="openChartRestoreModal('${key}')" style="font-size:10px;padding:3px 8px;margin-left:5px;background:#3498db;color:#fff;border:none;border-radius:4px">🔄 恢复</button>
                    </div>
                    <hr style="margin:10px 0;border:none;border-top:1px solid #ddd;">
                    <div class="style-row" style="justify-content:flex-end;gap:10px">
                        <button onclick="cancelStylePanel('${key}')" style="padding:6px 16px;cursor:pointer;border:1px solid #ddd;border-radius:4px;background:#fff">取消</button>
                        <button onclick="saveStylePanel('${key}')" style="padding:6px 16px;cursor:pointer;border:none;border-radius:4px;background:#667eea;color:#fff">保存</button>
                    </div>
                </div>
                <div id="chart_${key}" style="width:${cs.chartWidth||600}px;height:${cs.chartHeight||400}px"></div>`;
            document.getElementById('card_' + key).innerHTML = html;
            
            const chartDiv = document.getElementById('chart_' + key);
            if (chartInstances[key]) chartInstances[key].dispose();
            
            // ★★★ 根据刻度像素间距计算容器尺寸 ★★★
            const latestCs = chartStyles[key] || {};
            const xLabels = data.xAxis || [];
            const xTickGapPx = latestCs.xTickGapPx || 80;
            const yTickGapPx = latestCs.yTickGapPx || 50;
            const yTickCount = 6;
            const chartWidth = Math.max(xLabels.length * xTickGapPx + 150, 400);
            const chartHeight = Math.max(yTickCount * yTickGapPx + 120, 300);
            chartDiv.style.width = chartWidth + 'px';
            chartDiv.style.height = chartHeight + 'px';
            
            chartInstances[key] = echarts.init(chartDiv);
            chartInstances[key].setOption({
                tooltip: { trigger: 'axis' },
                legend: { data: data.series.map(s => s.name), triggerEvent: true },
                grid: { left: 60, right: 30, top: 50, bottom: 50, borderColor: '#000', borderWidth: 1, show: true },
                xAxis: { 
                    type: 'category', 
                    data: data.xAxis, 
                    name: latestCs.xAxisName||'', 
                    nameLocation: 'middle', 
                    nameGap: latestCs.xNameGap || 30, 
                    triggerEvent: true,
                    axisLine: { show: true, lineStyle: { color: '#000', width: 1 } },
                    axisTick: { show: true, lineStyle: { color: '#000' } },
                    nameTextStyle: { fontFamily: latestCs.axisLabelFont || 'Times New Roman', fontSize: latestCs.axisLabelSize || 12, color: latestCs.axisLabelColor || '#000' },
                    axisLabel: { fontFamily: latestCs.axisTickFont || 'Times New Roman', fontSize: latestCs.axisTickSize || 10, color: latestCs.axisTickColor || '#000' }
                },
                yAxis: { 
                    type: 'value', 
                    name: latestCs.yAxisName||'', 
                    nameLocation: 'middle', 
                    nameGap: latestCs.yNameGap || 40, 
                    triggerEvent: true,
                    axisLine: { show: true, lineStyle: { color: '#000', width: 1 } },
                    axisTick: { show: true, lineStyle: { color: '#000' } },
                    splitLine: { show: false },
                    nameTextStyle: { fontFamily: latestCs.axisLabelFont || 'Times New Roman', fontSize: latestCs.axisLabelSize || 12, color: latestCs.axisLabelColor || '#000' },
                    axisLabel: { fontFamily: latestCs.axisTickFont || 'Times New Roman', fontSize: latestCs.axisTickSize || 10, color: latestCs.axisTickColor || '#000' }
                },
                series: data.series.map((s, idx) => ({ name: s.name, type: 'line', data: s.data, itemStyle: { color: s.color } }))
            });
            // 点击编辑功能
            chartInstances[key].on('click', function(params) {
                const latestCs = chartStyles[key] || {};
                // 点击X轴刻度标签 - 编辑该标签
                if (params.componentType === 'xAxis' && params.targetType === 'axisLabel') {
                    const labelIdx = matricesData[key].data.xAxis.indexOf(params.value);
                    if (labelIdx >= 0) {
                        showTextEditor(params.value, function(newLabel) {
                            matricesData[key].data.xAxis[labelIdx] = newLabel;
                            saveAllToStorage();
                            renderChart(key);
                        });
                    }
                }
                // 点击X轴名称 - 编辑轴名称
                else if (params.componentType === 'xAxis' && params.targetType === 'axisName') {
                    showTextEditor(latestCs.xAxisName || 'X轴', function(newName) {
                        if (!chartStyles[key]) chartStyles[key] = {};
                        chartStyles[key].xAxisName = newName;
                        saveAllToStorage();
                        renderChart(key);
                    });
                }
                // 点击Y轴刻度标签 - 编辑数值(通过弹窗)
                else if (params.componentType === 'yAxis' && params.targetType === 'axisLabel') {
                    showValueEditor('编辑Y轴刻度值', params.value, function(newVal) {
                        // Y轴刻度是自动计算的，这里可以调整Y轴范围
                        alert('Y轴刻度由数据自动计算，请在数据编辑中修改数值');
                    });
                }
                // 点击Y轴名称 - 编辑轴名称
                else if (params.componentType === 'yAxis' && params.targetType === 'axisName') {
                    showTextEditor(latestCs.yAxisName || 'Y轴', function(newName) {
                        if (!chartStyles[key]) chartStyles[key] = {};
                        chartStyles[key].yAxisName = newName;
                        saveAllToStorage();
                        renderChart(key);
                    });
                }
                // 点击数据点 - 编辑数值
                else if (params.componentType === 'series') {
                    const seriesIdx = params.seriesIndex;
                    const dataIdx = params.dataIndex;
                    const currentVal = matricesData[key].data.series[seriesIdx].data[dataIdx];
                    showValueEditor('编辑 ' + matricesData[key].data.series[seriesIdx].name + ' [' + matricesData[key].data.xAxis[dataIdx] + ']', currentVal, function(newVal) {
                        matricesData[key].data.series[seriesIdx].data[dataIdx] = newVal;
                        saveAllToStorage();
                        renderChart(key);
                    });
                }
            });
        }
        
        // 注：renderScatterChart已删除，图7现在是箱线图
        
        // 图8: 蜘蛛图(雷达图)(参考图2样式)
        function renderRadarChart(key) {
            const info = matricesData[key];
            const data = info.data;
            const mainTitle = customTitles[key] || info.name;
            const subtitle = customSubtitles[key] || info.subtitle || '';
            const cs = chartStyles[key] || {};
            
            let html = `
                <div class="card-header">
                    <div><span class="card-title">${mainTitle}</span><span style="font-size:12px;color:#666;margin-left:8px">${subtitle}</span></div>
                    <div class="card-controls">
                        <button class="ctrl-btn" onclick="exportChart('${key}','png')">PNG</button>
                        <button class="ctrl-btn" onclick="exportChart('${key}','jpg')">JPG</button>
                        <button class="ctrl-btn" onclick="exportChart('${key}','tif')">TIF</button>
                        <button class="ctrl-btn" onclick="toggleStylePanel('${key}')">🎨</button>
                    </div>
                </div>
                <div class="style-panel" id="panel_${key}">
                    <div class="style-panel-header">
                        <span>📊 ${mainTitle} 样式设置</span>
                        <button class="style-panel-close" onclick="cancelStylePanel('${key}')">✕</button>
                    </div>
                    <div class="style-row"><label>主标题:</label><input type="text" value="${mainTitle}" onchange="setCustomTitle('${key}', this.value)" style="flex:1;padding:4px 8px;border:1px solid #ddd;border-radius:4px;font-size:12px"></div>
                    <div class="style-row"><label>副标题:</label><input type="text" value="${subtitle}" onchange="setCustomSubtitle('${key}', this.value)" style="flex:1;padding:4px 8px;border:1px solid #ddd;border-radius:4px;font-size:12px"></div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>标签字体</strong> 字号:<input type="number" value="${cs.axisLabelSize||12}" onchange="setChartStyle('${key}','axisLabelSize',+this.value);renderChart('${key}')" style="width:40px" min="8" max="20">
                    颜色:<input type="color" value="${cs.axisLabelColor||'#000000'}" onchange="setChartStyle('${key}','axisLabelColor',this.value);renderChart('${key}')" style="width:30px;height:22px"></div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>图表尺寸</strong> 宽×高(px):
                        <input type="number" value="${cs.chartWidth||600}" onchange="setChartStyle('${key}','chartWidth',+this.value);renderChart('${key}')" style="width:60px" min="300" max="1200">×
                        <input type="number" value="${cs.chartHeight||400}" onchange="setChartStyle('${key}','chartHeight',+this.value);renderChart('${key}')" style="width:60px" min="200" max="800">
                    </div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>数据编辑</strong>
                        <button onclick="openRadarEditor('${key}')" style="font-size:10px;padding:2px 6px;margin-left:10px;background:#3498db;color:#fff;border:none;border-radius:4px">📝 编辑数据</button>
                        <button onclick="openChartImport('${key}','radar')" style="font-size:10px;padding:2px 6px;margin-left:5px;background:#27ae60;color:#fff;border:none;border-radius:4px">📥 AI导入</button>
                    </div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>节点管理</strong>
                        <button onclick="openChartSaveModal('${key}')" style="font-size:10px;padding:3px 8px;margin-left:10px;background:#27ae60;color:#fff;border:none;border-radius:4px">💾 保存</button>
                        <button onclick="openChartRestoreModal('${key}')" style="font-size:10px;padding:3px 8px;margin-left:5px;background:#3498db;color:#fff;border:none;border-radius:4px">🔄 恢复</button>
                    </div>
                    <hr style="margin:10px 0;border:none;border-top:1px solid #ddd;">
                    <div class="style-row" style="justify-content:flex-end;gap:10px">
                        <button onclick="cancelStylePanel('${key}')" style="padding:6px 16px;cursor:pointer;border:1px solid #ddd;border-radius:4px;background:#fff">取消</button>
                        <button onclick="saveStylePanel('${key}')" style="padding:6px 16px;cursor:pointer;border:none;border-radius:4px;background:#667eea;color:#fff">保存</button>
                    </div>
                </div>
                <div id="chart_${key}" style="width:${cs.chartWidth||600}px;height:${cs.chartHeight||400}px"></div>`;
            document.getElementById('card_' + key).innerHTML = html;
            
            const chartDiv = document.getElementById('chart_' + key);
            if (chartInstances[key]) chartInstances[key].dispose();
            chartInstances[key] = echarts.init(chartDiv);
            const latestCs = chartStyles[key] || {};
            chartInstances[key].setOption({
                tooltip: {},
                legend: { data: data.series.map(s => s.name) },
                radar: { 
                    indicator: data.indicator,
                    name: { textStyle: { fontSize: latestCs.axisLabelSize || 12, color: latestCs.axisLabelColor || '#000' } }
                },
                series: [{ type: 'radar', data: data.series.map(s => ({ name: s.name, value: s.data, itemStyle: { color: s.color } })) }]
            });
            // 点击编辑功能
            chartInstances[key].on('click', function(params) {
                if (params.componentType === 'series') {
                    const seriesIdx = params.seriesIndex;
                    const dataIdx = params.dataIndex;
                    if (dataIdx !== undefined && dataIdx >= 0) {
                        const currentVal = matricesData[key].data.series[0].data[dataIdx];
                        showValueEditor('编辑 ' + matricesData[key].data.indicator[dataIdx].name, currentVal, function(newVal) {
                            matricesData[key].data.series[0].data[dataIdx] = newVal;
                            saveAllToStorage();
                            renderChart(key);
                        });
                    }
                }
            });
        }
        
        // 图9: 双轴图(参考图2样式)
        function renderDualAxisChart(key) {
            const info = matricesData[key];
            const data = info.data;
            const mainTitle = customTitles[key] || info.name;
            const subtitle = customSubtitles[key] || info.subtitle || '';
            const cs = chartStyles[key] || {};
            
            let html = `
                <div class="card-header">
                    <div><span class="card-title">${mainTitle}</span><span style="font-size:12px;color:#666;margin-left:8px">${subtitle}</span></div>
                    <div class="card-controls">
                        <button class="ctrl-btn" onclick="exportChart('${key}','png')">PNG</button>
                        <button class="ctrl-btn" onclick="exportChart('${key}','jpg')">JPG</button>
                        <button class="ctrl-btn" onclick="exportChart('${key}','tif')">TIF</button>
                        <button class="ctrl-btn" onclick="toggleStylePanel('${key}')">🎨</button>
                    </div>
                </div>
                <div class="style-panel" id="panel_${key}">
                    <div class="style-panel-header">
                        <span>📊 ${mainTitle} 样式设置</span>
                        <button class="style-panel-close" onclick="cancelStylePanel('${key}')">✕</button>
                    </div>
                    <div class="style-row"><label>主标题:</label><input type="text" value="${mainTitle}" onchange="setCustomTitle('${key}', this.value)" style="flex:1;padding:4px 8px;border:1px solid #ddd;border-radius:4px;font-size:12px"></div>
                    <div class="style-row"><label>副标题:</label><input type="text" value="${subtitle}" onchange="setCustomSubtitle('${key}', this.value)" style="flex:1;padding:4px 8px;border:1px solid #ddd;border-radius:4px;font-size:12px"></div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>轴标签</strong> 字号:<input type="number" value="${cs.axisLabelSize||12}" onchange="setChartStyle('${key}','axisLabelSize',+this.value);renderChart('${key}')" style="width:40px" min="8" max="20">
                    颜色:<input type="color" value="${cs.axisLabelColor||'#000000'}" onchange="setChartStyle('${key}','axisLabelColor',this.value);renderChart('${key}')" style="width:30px;height:22px"></div>
                    <div class="style-row"><strong>刻度</strong> 字号:<input type="number" value="${cs.axisTickSize||10}" onchange="setChartStyle('${key}','axisTickSize',+this.value);renderChart('${key}')" style="width:40px" min="8" max="18">
                    颜色:<input type="color" value="${cs.axisTickColor||'#000000'}" onchange="setChartStyle('${key}','axisTickColor',this.value);renderChart('${key}')" style="width:30px;height:22px"></div>
                    <div class="style-row"><strong>刻度像素间距</strong> X轴:<input type="number" value="${cs.xTickGapPx||80}" onchange="setChartStyle('${key}','xTickGapPx',+this.value)" style="width:50px" min="30" max="200" step="10">px
                        Y轴:<input type="number" value="${cs.yTickGapPx||50}" onchange="setChartStyle('${key}','yTickGapPx',+this.value)" style="width:50px" min="20" max="150" step="10">px</div>
                    <div class="style-row"><strong>X轴名偏移</strong> 
                        <button onclick="adjustAxisOffset('${key}','xNameGap',-5)" style="padding:2px 4px">◀</button>
                        <input type="number" id="xNameGap_${key}" value="${cs.xNameGap||30}" onchange="setChartStyle('${key}','xNameGap',+this.value)" style="width:40px" min="-50" max="80">px
                        <button onclick="adjustAxisOffset('${key}','xNameGap',5)" style="padding:2px 4px">▶</button>
                        <strong style="margin-left:10px">Y轴名偏移</strong>
                        <button onclick="adjustAxisOffset('${key}','yNameGap',-5)" style="padding:2px 4px">◀</button>
                        <input type="number" id="yNameGap_${key}" value="${cs.yNameGap||40}" onchange="setChartStyle('${key}','yNameGap',+this.value)" style="width:40px" min="-50" max="80">px
                        <button onclick="adjustAxisOffset('${key}','yNameGap',5)" style="padding:2px 4px">▶</button>
                    </div>
                    <div class="style-row"><strong>X刻度标签偏移</strong> 
                        <button onclick="adjustAxisOffset('${key}','xTickMargin',-1)" style="padding:2px 4px">▲</button>
                        <input type="number" id="xTickMargin_${key}" value="${cs.xTickMargin||8}" onchange="setChartStyle('${key}','xTickMargin',+this.value)" style="width:40px" min="-20" max="30">px
                        <button onclick="adjustAxisOffset('${key}','xTickMargin',1)" style="padding:2px 4px">▼</button>
                        <strong style="margin-left:10px">Y刻度标签偏移</strong>
                        <button onclick="adjustAxisOffset('${key}','yTickMargin',-1)" style="padding:2px 4px">◀</button>
                        <input type="number" id="yTickMargin_${key}" value="${cs.yTickMargin||8}" onchange="setChartStyle('${key}','yTickMargin',+this.value)" style="width:40px" min="-20" max="30">px
                        <button onclick="adjustAxisOffset('${key}','yTickMargin',1)" style="padding:2px 4px">▶</button>
                    </div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>图表尺寸</strong> 宽×高(px):
                        <input type="number" value="${cs.chartWidth||600}" onchange="setChartStyle('${key}','chartWidth',+this.value);renderChart('${key}')" style="width:60px" min="300" max="1200">×
                        <input type="number" value="${cs.chartHeight||400}" onchange="setChartStyle('${key}','chartHeight',+this.value);renderChart('${key}')" style="width:60px" min="200" max="800">
                    </div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>数据编辑</strong>
                        <button onclick="openDualAxisEditor('${key}')" style="font-size:10px;padding:2px 6px;margin-left:10px;background:#3498db;color:#fff;border:none;border-radius:4px">📝 编辑数据</button>
                        <button onclick="openChartImport('${key}','dualaxis')" style="font-size:10px;padding:2px 6px;margin-left:5px;background:#27ae60;color:#fff;border:none;border-radius:4px">📥 AI导入</button>
                    </div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>节点管理</strong>
                        <button onclick="openChartSaveModal('${key}')" style="font-size:10px;padding:3px 8px;margin-left:10px;background:#27ae60;color:#fff;border:none;border-radius:4px">💾 保存</button>
                        <button onclick="openChartRestoreModal('${key}')" style="font-size:10px;padding:3px 8px;margin-left:5px;background:#3498db;color:#fff;border:none;border-radius:4px">🔄 恢复</button>
                    </div>
                    <hr style="margin:10px 0;border:none;border-top:1px solid #ddd;">
                    <div class="style-row" style="justify-content:flex-end;gap:10px">
                        <button onclick="cancelStylePanel('${key}')" style="padding:6px 16px;cursor:pointer;border:1px solid #ddd;border-radius:4px;background:#fff">取消</button>
                        <button onclick="saveStylePanel('${key}')" style="padding:6px 16px;cursor:pointer;border:none;border-radius:4px;background:#667eea;color:#fff">保存</button>
                    </div>
                </div>
                <div id="chart_${key}" style="width:${cs.chartWidth||600}px;height:${cs.chartHeight||400}px"></div>`;
            document.getElementById('card_' + key).innerHTML = html;
            
            const chartDiv = document.getElementById('chart_' + key);
            if (chartInstances[key]) chartInstances[key].dispose();
            
            // ★★★ 根据刻度像素间距计算容器尺寸 ★★★
            const latestCs = chartStyles[key] || {};
            const xLabels = data.xAxis || [];
            const xTickGapPx = latestCs.xTickGapPx || 80;
            const yTickGapPx = latestCs.yTickGapPx || 50;
            const yTickCount = 6;
            const chartWidth = Math.max(xLabels.length * xTickGapPx + 180, 400);
            const chartHeight = Math.max(yTickCount * yTickGapPx + 120, 300);
            chartDiv.style.width = chartWidth + 'px';
            chartDiv.style.height = chartHeight + 'px';
            
            chartInstances[key] = echarts.init(chartDiv);
            chartInstances[key].setOption({
                tooltip: { trigger: 'axis' },
                legend: { data: data.series.map(s => s.name) },
                grid: { left: 60, right: 60, top: 50, bottom: 50, borderColor: '#000', borderWidth: 1, show: true },
                xAxis: { 
                    type: 'category', 
                    data: data.xAxis, 
                    name: latestCs.xAxisName || '',
                    nameLocation: 'middle',
                    nameGap: latestCs.xNameGap || 30,
                    triggerEvent: true,
                    axisLine: { show: true, lineStyle: { color: '#000', width: 1 } },
                    axisTick: { show: true, lineStyle: { color: '#000' } },
                    nameTextStyle: { fontSize: latestCs.axisLabelSize || 12, color: latestCs.axisLabelColor || '#000' },
                    axisLabel: { fontSize: latestCs.axisTickSize || 10, color: latestCs.axisTickColor || '#000' }
                },
                yAxis: data.yAxis.map((y, i) => ({ 
                    type: 'value', 
                    name: y.name, 
                    position: y.position,
                    nameLocation: 'middle',
                    nameGap: latestCs.yNameGap || 40,
                    triggerEvent: true,
                    axisLine: { show: true, lineStyle: { color: '#000', width: 1 } },
                    axisTick: { show: true, lineStyle: { color: '#000' } },
                    splitLine: { show: false },
                    nameTextStyle: { fontSize: latestCs.axisLabelSize || 12, color: latestCs.axisLabelColor || '#000' },
                    axisLabel: { fontSize: latestCs.axisTickSize || 10, color: latestCs.axisTickColor || '#000' }
                })),
                series: data.series.map(s => ({ name: s.name, type: s.type, yAxisIndex: s.yAxisIndex, data: s.data, itemStyle: { color: s.color } }))
            });
            // 点击编辑功能
            chartInstances[key].on('click', function(params) {
                // 点击X轴刻度标签
                if (params.componentType === 'xAxis' && params.targetType === 'axisLabel') {
                    const labelIdx = matricesData[key].data.xAxis.indexOf(params.value);
                    if (labelIdx >= 0) {
                        showTextEditor(params.value, function(newLabel) {
                            matricesData[key].data.xAxis[labelIdx] = newLabel;
                            saveAllToStorage();
                            renderChart(key);
                        });
                    }
                }
                // 点击Y轴名称
                else if (params.componentType === 'yAxis' && params.targetType === 'axisName') {
                    const yAxisIdx = params.yAxisIndex || 0;
                    showTextEditor(matricesData[key].data.yAxis[yAxisIdx].name, function(newName) {
                        matricesData[key].data.yAxis[yAxisIdx].name = newName;
                        saveAllToStorage();
                        renderChart(key);
                    });
                }
                // 点击数据点(柱子或折线)
                else if (params.componentType === 'series') {
                    const seriesIdx = params.seriesIndex;
                    const dataIdx = params.dataIndex;
                    const currentVal = matricesData[key].data.series[seriesIdx].data[dataIdx];
                    const seriesName = matricesData[key].data.series[seriesIdx].name;
                    const xLabel = matricesData[key].data.xAxis[dataIdx];
                    showValueEditor('编辑 ' + seriesName + ' [' + xLabel + ']', currentVal, function(newVal) {
                        matricesData[key].data.series[seriesIdx].data[dataIdx] = newVal;
                        saveAllToStorage();
                        renderChart(key);
                    });
                }
            });
        }
        
        // 图10: 面积图
        function renderAreaChart(key) {
            const info = matricesData[key];
            const data = info.data;
            const mainTitle = customTitles[key] || info.name;
            const subtitle = customSubtitles[key] || info.subtitle || '';
            
            let html = `<div class="card-header"><div><span class="card-title">${mainTitle}</span><span style="font-size:12px;color:#666;margin-left:8px">${subtitle}</span></div>
                <div class="card-controls"><button class="ctrl-btn" onclick="exportChart('${key}','png')">PNG</button><button class="ctrl-btn" onclick="toggleStylePanel('${key}')">🎨</button></div></div>
                <div id="chart_${key}" style="width:600px;height:400px"></div>`;
            document.getElementById('card_' + key).innerHTML = html;
            
            const chartDiv = document.getElementById('chart_' + key);
            if (chartInstances[key]) chartInstances[key].dispose();
            chartInstances[key] = echarts.init(chartDiv);
            chartInstances[key].setOption({
                tooltip: { trigger: 'axis' },
                legend: { data: data.series.map(s => s.name) },
                xAxis: { type: 'category', boundaryGap: false, data: data.xAxis },
                yAxis: { type: 'value' },
                series: data.series.map(s => ({ name: s.name, type: 'line', areaStyle: {}, data: s.data, itemStyle: { color: s.color } }))
            });
        }
        
        // 图11: 带状图
        function renderBandChart(key) {
            const info = matricesData[key];
            const data = info.data;
            const mainTitle = customTitles[key] || info.name;
            const subtitle = customSubtitles[key] || info.subtitle || '';
            
            let html = `<div class="card-header"><div><span class="card-title">${mainTitle}</span><span style="font-size:12px;color:#666;margin-left:8px">${subtitle}</span></div>
                <div class="card-controls"><button class="ctrl-btn" onclick="exportChart('${key}','png')">PNG</button><button class="ctrl-btn" onclick="toggleStylePanel('${key}')">🎨</button></div></div>
                <div id="chart_${key}" style="width:600px;height:400px"></div>`;
            document.getElementById('card_' + key).innerHTML = html;
            
            const chartDiv = document.getElementById('chart_' + key);
            if (chartInstances[key]) chartInstances[key].dispose();
            chartInstances[key] = echarts.init(chartDiv);
            chartInstances[key].setOption({
                tooltip: { trigger: 'axis' },
                legend: { data: data.series.map(s => s.name) },
                xAxis: { type: 'category', data: data.xAxis },
                yAxis: { type: 'value' },
                series: data.series.map((s, i) => ({ 
                    name: s.name, type: 'line', 
                    areaStyle: i === 0 ? { opacity: 0.3 } : { opacity: 0.5 },
                    stack: 'band',
                    data: s.data, itemStyle: { color: s.color } 
                }))
            });
        }
        
        // 图12: 等高线图(热力图展示)
        function renderContourChart(key) {
            const info = matricesData[key];
            const data = info.data;
            const mainTitle = customTitles[key] || info.name;
            const subtitle = customSubtitles[key] || info.subtitle || '';
            
            let html = `<div class="card-header"><div><span class="card-title">${mainTitle}</span><span style="font-size:12px;color:#666;margin-left:8px">${subtitle}</span></div>
                <div class="card-controls"><button class="ctrl-btn" onclick="exportChart('${key}','png')">PNG</button><button class="ctrl-btn" onclick="toggleStylePanel('${key}')">🎨</button></div></div>
                <div id="chart_${key}" style="width:600px;height:400px"></div>`;
            document.getElementById('card_' + key).innerHTML = html;
            
            const chartDiv = document.getElementById('chart_' + key);
            if (chartInstances[key]) chartInstances[key].dispose();
            chartInstances[key] = echarts.init(chartDiv);
            chartInstances[key].setOption({
                tooltip: { position: 'top' },
                xAxis: { type: 'category', data: [...new Set(data.data.map(d => d[0]))] },
                yAxis: { type: 'category', data: [...new Set(data.data.map(d => d[1]))] },
                visualMap: { min: -10, max: 10, calculable: true, orient: 'horizontal', left: 'center' },
                series: [{ type: 'heatmap', data: data.data, emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.5)' } } }]
            });
        }
        
        // 图13: 极坐标图
        function renderPolarChart(key) {
            const info = matricesData[key];
            const data = info.data;
            const mainTitle = customTitles[key] || info.name;
            const subtitle = customSubtitles[key] || info.subtitle || '';
            
            let html = `<div class="card-header"><div><span class="card-title">${mainTitle}</span><span style="font-size:12px;color:#666;margin-left:8px">${subtitle}</span></div>
                <div class="card-controls"><button class="ctrl-btn" onclick="exportChart('${key}','png')">PNG</button><button class="ctrl-btn" onclick="toggleStylePanel('${key}')">🎨</button></div></div>
                <div id="chart_${key}" style="width:600px;height:400px"></div>`;
            document.getElementById('card_' + key).innerHTML = html;
            
            const chartDiv = document.getElementById('chart_' + key);
            if (chartInstances[key]) chartInstances[key].dispose();
            chartInstances[key] = echarts.init(chartDiv);
            chartInstances[key].setOption({
                polar: { radius: [30, '80%'] },
                angleAxis: { type: 'value', startAngle: 0 },
                radiusAxis: { min: 0 },
                series: data.series.map(s => ({ 
                    type: 'line', 
                    coordinateSystem: 'polar',
                    name: s.name,
                    data: s.data,
                    itemStyle: { color: s.color }
                }))
            });
        }
        
        // 图14: 3D曲面图(用热力图展示)
        function renderSurface3DChart(key) {
            const info = matricesData[key];
            const data = info.data;
            const mainTitle = customTitles[key] || info.name;
            const subtitle = customSubtitles[key] || info.subtitle || '';
            
            let html = `<div class="card-header"><div><span class="card-title">${mainTitle}</span><span style="font-size:12px;color:#666;margin-left:8px">${subtitle}</span></div>
                <div class="card-controls"><button class="ctrl-btn" onclick="exportChart('${key}','png')">PNG</button><button class="ctrl-btn" onclick="toggleStylePanel('${key}')">🎨</button></div></div>
                <div id="chart_${key}" style="width:600px;height:400px"></div>`;
            document.getElementById('card_' + key).innerHTML = html;
            
            const chartDiv = document.getElementById('chart_' + key);
            if (chartInstances[key]) chartInstances[key].dispose();
            chartInstances[key] = echarts.init(chartDiv);
            chartInstances[key].setOption({
                tooltip: { position: 'top' },
                visualMap: { min: -10, max: 10, calculable: true, orient: 'horizontal', left: 'center' },
                xAxis: { type: 'category', data: [...new Set(data.data.map(d => d[0]))] },
                yAxis: { type: 'category', data: [...new Set(data.data.map(d => d[1]))] },
                series: [{ type: 'heatmap', data: data.data.map(d => [d[0]+10, d[1]+10, d[2]]) }]
            });
        }
        
        // 图15: 3D散点图(用散点图展示)
        function renderScatter3DChart(key) {
            const info = matricesData[key];
            const data = info.data;
            const mainTitle = customTitles[key] || info.name;
            const subtitle = customSubtitles[key] || info.subtitle || '';
            
            let html = `<div class="card-header"><div><span class="card-title">${mainTitle}</span><span style="font-size:12px;color:#666;margin-left:8px">${subtitle}</span></div>
                <div class="card-controls"><button class="ctrl-btn" onclick="exportChart('${key}','png')">PNG</button><button class="ctrl-btn" onclick="toggleStylePanel('${key}')">🎨</button></div></div>
                <div id="chart_${key}" style="width:600px;height:400px"></div>`;
            document.getElementById('card_' + key).innerHTML = html;
            
            const chartDiv = document.getElementById('chart_' + key);
            if (chartInstances[key]) chartInstances[key].dispose();
            chartInstances[key] = echarts.init(chartDiv);
            chartInstances[key].setOption({
                tooltip: {},
                xAxis: { type: 'value', name: 'X' },
                yAxis: { type: 'value', name: 'Y' },
                visualMap: { min: -15, max: 15, dimension: 2, inRange: { color: ['#50a3ba', '#eac736', '#d94e5d'] } },
                series: data.series.map(s => ({ 
                    type: 'scatter', 
                    symbolSize: 10,
                    data: s.data.map(d => [d[0], d[1], d[2]]),
                    itemStyle: { color: s.color }
                }))
            });
        }
        
        // 图16: 3D条形图(用柱状图展示)
        function renderBar3DChart(key) {
            const info = matricesData[key];
            const data = info.data;
            const mainTitle = customTitles[key] || info.name;
            const subtitle = customSubtitles[key] || info.subtitle || '';
            
            let html = `<div class="card-header"><div><span class="card-title">${mainTitle}</span><span style="font-size:12px;color:#666;margin-left:8px">${subtitle}</span></div>
                <div class="card-controls"><button class="ctrl-btn" onclick="exportChart('${key}','png')">PNG</button><button class="ctrl-btn" onclick="toggleStylePanel('${key}')">🎨</button></div></div>
                <div id="chart_${key}" style="width:600px;height:400px"></div>`;
            document.getElementById('card_' + key).innerHTML = html;
            
            const chartDiv = document.getElementById('chart_' + key);
            if (chartInstances[key]) chartInstances[key].dispose();
            chartInstances[key] = echarts.init(chartDiv);
            chartInstances[key].setOption({
                tooltip: {},
                visualMap: { min: 0, max: 25, inRange: { color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026'] } },
                xAxis: { type: 'category', data: ['A', 'B', 'C', 'D', 'E'] },
                yAxis: { type: 'category', data: ['1', '2', '3', '4', '5'] },
                series: [{ type: 'heatmap', data: data.data }]
            });
        }
        
        // 图17: 直方图
        function renderHistogramChart(key) {
            const info = matricesData[key];
            const data = info.data;
            const mainTitle = customTitles[key] || info.name;
            const subtitle = customSubtitles[key] || info.subtitle || '';
            
            // 计算直方图bins
            const rawData = data.data;
            const min = Math.min(...rawData), max = Math.max(...rawData);
            const binCount = data.bins || 10;
            const binWidth = (max - min) / binCount;
            const bins = new Array(binCount).fill(0);
            rawData.forEach(v => {
                const idx = Math.min(Math.floor((v - min) / binWidth), binCount - 1);
                bins[idx]++;
            });
            const binLabels = bins.map((_, i) => (min + i * binWidth).toFixed(1));
            
            let html = `<div class="card-header"><div><span class="card-title">${mainTitle}</span><span style="font-size:12px;color:#666;margin-left:8px">${subtitle}</span></div>
                <div class="card-controls"><button class="ctrl-btn" onclick="exportChart('${key}','png')">PNG</button><button class="ctrl-btn" onclick="toggleStylePanel('${key}')">🎨</button></div></div>
                <div id="chart_${key}" style="width:600px;height:400px"></div>`;
            document.getElementById('card_' + key).innerHTML = html;
            
            const chartDiv = document.getElementById('chart_' + key);
            if (chartInstances[key]) chartInstances[key].dispose();
            chartInstances[key] = echarts.init(chartDiv);
            chartInstances[key].setOption({
                tooltip: {},
                xAxis: { type: 'category', data: binLabels },
                yAxis: { type: 'value', name: 'Frequency' },
                series: [{ type: 'bar', data: bins, itemStyle: { color: '#5470c6' } }]
            });
        }
        
        // 图18: 小提琴图(用箱线图展示)
        function renderViolinChart(key) {
            const info = matricesData[key];
            const data = info.data;
            const mainTitle = customTitles[key] || info.name;
            const subtitle = customSubtitles[key] || info.subtitle || '';
            
            let html = `<div class="card-header"><div><span class="card-title">${mainTitle}</span><span style="font-size:12px;color:#666;margin-left:8px">${subtitle}</span></div>
                <div class="card-controls"><button class="ctrl-btn" onclick="exportChart('${key}','png')">PNG</button><button class="ctrl-btn" onclick="toggleStylePanel('${key}')">🎨</button></div></div>
                <div id="chart_${key}" style="width:600px;height:400px"></div>`;
            document.getElementById('card_' + key).innerHTML = html;
            
            const chartDiv = document.getElementById('chart_' + key);
            if (chartInstances[key]) chartInstances[key].dispose();
            chartInstances[key] = echarts.init(chartDiv);
            
            // 转换为箱线图数据
            const boxData = data.data.map(arr => {
                const sorted = arr.slice().sort((a,b) => a-b);
                const n = sorted.length;
                return [sorted[0], sorted[Math.floor(n*0.25)], sorted[Math.floor(n*0.5)], sorted[Math.floor(n*0.75)], sorted[n-1]];
            });
            
            chartInstances[key].setOption({
                tooltip: {},
                xAxis: { type: 'category', data: data.categories },
                yAxis: { type: 'value' },
                series: [{ type: 'boxplot', data: boxData }]
            });
        }
        
        // 图19: 成对关系图(散点矩阵)
        function renderPairplotChart(key) {
            const info = matricesData[key];
            const data = info.data;
            const mainTitle = customTitles[key] || info.name;
            const subtitle = customSubtitles[key] || info.subtitle || '';
            
            let html = `<div class="card-header"><div><span class="card-title">${mainTitle}</span><span style="font-size:12px;color:#666;margin-left:8px">${subtitle}</span></div>
                <div class="card-controls"><button class="ctrl-btn" onclick="exportChart('${key}','png')">PNG</button><button class="ctrl-btn" onclick="toggleStylePanel('${key}')">🎨</button></div></div>
                <div id="chart_${key}" style="width:600px;height:400px"></div>`;
            document.getElementById('card_' + key).innerHTML = html;
            
            const chartDiv = document.getElementById('chart_' + key);
            if (chartInstances[key]) chartInstances[key].dispose();
            chartInstances[key] = echarts.init(chartDiv);
            
            // 简化为散点图展示前两个变量
            const scatterData = data.data.map(row => [row[0], row[1]]);
            
            chartInstances[key].setOption({
                tooltip: {},
                xAxis: { type: 'value', name: data.variables[0] || 'Var1' },
                yAxis: { type: 'value', name: data.variables[1] || 'Var2' },
                series: [{ type: 'scatter', data: scatterData, symbolSize: 8 }]
            });
        }
        
        // 图20: Facet Grid图
        function renderFacetChart(key) {
            const info = matricesData[key];
            const data = info.data;
            const mainTitle = customTitles[key] || info.name;
            const subtitle = customSubtitles[key] || info.subtitle || '';
            
            let html = `<div class="card-header"><div><span class="card-title">${mainTitle}</span><span style="font-size:12px;color:#666;margin-left:8px">${subtitle}</span></div>
                <div class="card-controls"><button class="ctrl-btn" onclick="exportChart('${key}','png')">PNG</button><button class="ctrl-btn" onclick="toggleStylePanel('${key}')">🎨</button></div></div>
                <div id="chart_${key}" style="width:700px;height:500px"></div>`;
            document.getElementById('card_' + key).innerHTML = html;
            
            const chartDiv = document.getElementById('chart_' + key);
            if (chartInstances[key]) chartInstances[key].dispose();
            chartInstances[key] = echarts.init(chartDiv);
            
            // 创建网格
            const grids = [], xAxes = [], yAxes = [], series = [];
            let gridIdx = 0;
            data.categories.forEach((cat, i) => {
                data.groups.forEach((grp, j) => {
                    grids.push({ left: (j * 50 + 10) + '%', top: (i * 45 + 10) + '%', width: '35%', height: '35%' });
                    xAxes.push({ type: 'value', gridIndex: gridIdx, name: cat + '-' + grp, nameLocation: 'middle', nameGap: 20 });
                    yAxes.push({ type: 'value', gridIndex: gridIdx });
                    series.push({ type: 'scatter', xAxisIndex: gridIdx, yAxisIndex: gridIdx, data: data.data[cat]?.[grp] || [] });
                    gridIdx++;
                });
            });
            
            chartInstances[key].setOption({ grid: grids, xAxis: xAxes, yAxis: yAxes, series: series });
        }
        
        // 图21: 热力图
        function renderHeatmapChart(key) {
            const info = matricesData[key];
            const data = info.data;
            const mainTitle = customTitles[key] || info.name;
            const subtitle = customSubtitles[key] || info.subtitle || '';
            
            let html = `<div class="card-header"><div><span class="card-title">${mainTitle}</span><span style="font-size:12px;color:#666;margin-left:8px">${subtitle}</span></div>
                <div class="card-controls"><button class="ctrl-btn" onclick="exportChart('${key}','png')">PNG</button><button class="ctrl-btn" onclick="toggleStylePanel('${key}')">🎨</button></div></div>
                <div id="chart_${key}" style="width:600px;height:400px"></div>`;
            document.getElementById('card_' + key).innerHTML = html;
            
            const chartDiv = document.getElementById('chart_' + key);
            if (chartInstances[key]) chartInstances[key].dispose();
            chartInstances[key] = echarts.init(chartDiv);
            chartInstances[key].setOption({
                tooltip: { position: 'top' },
                xAxis: { type: 'category', data: data.xAxis },
                yAxis: { type: 'category', data: data.yAxis },
                visualMap: { min: 0, max: 100, calculable: true },
                series: [{ type: 'heatmap', data: data.data, label: { show: true } }]
            });
        }
        
        // 图7: 真正的箱线图(参考图2样式)
        function renderRealBoxplot(key) {
            const info = matricesData[key];
            const data = info.data;
            const mainTitle = customTitles[key] || info.name;
            const subtitle = customSubtitles[key] || info.subtitle || '';
            const cs = chartStyles[key] || {};
            
            // vlag调色板颜色(seaborn vlag)
            const vlagColors = ['#2166ac', '#67a9cf', '#d1e5f0', '#fddbc7', '#ef8a62', '#b2182b'];
            
            let html = `
                <div class="card-header">
                    <div><span class="card-title">${mainTitle}</span><span style="font-size:12px;color:#666;margin-left:8px">${subtitle}</span></div>
                    <div class="card-controls">
                        <button class="ctrl-btn" onclick="exportChart('${key}','png')">PNG</button>
                        <button class="ctrl-btn" onclick="exportChart('${key}','jpg')">JPG</button>
                        <button class="ctrl-btn" onclick="exportChart('${key}','tif')">TIF</button>
                        <button class="ctrl-btn" onclick="toggleStylePanel('${key}')">🎨</button>
                    </div>
                </div>
                <div class="style-panel" id="panel_${key}">
                    <div class="style-panel-header">
                        <span>📊 ${mainTitle} 样式设置</span>
                        <button class="style-panel-close" onclick="cancelStylePanel('${key}')">✕</button>
                    </div>
                    <div class="style-row"><label>主标题:</label><input type="text" value="${mainTitle}" onchange="setCustomTitle('${key}', this.value)" style="flex:1;padding:4px 8px;border:1px solid #ddd;border-radius:4px;font-size:12px"></div>
                    <div class="style-row"><label>副标题:</label><input type="text" value="${subtitle}" onchange="setCustomSubtitle('${key}', this.value)" style="flex:1;padding:4px 8px;border:1px solid #ddd;border-radius:4px;font-size:12px"></div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>轴标签</strong> 字号:<input type="number" value="${cs.axisLabelSize||12}" onchange="setChartStyle('${key}','axisLabelSize',+this.value);renderChart('${key}')" style="width:40px" min="8" max="20">
                    颜色:<input type="color" value="${cs.axisLabelColor||'#000000'}" onchange="setChartStyle('${key}','axisLabelColor',this.value);renderChart('${key}')" style="width:30px;height:22px"></div>
                    <div class="style-row"><strong>刻度</strong> 字号:<input type="number" value="${cs.axisTickSize||10}" onchange="setChartStyle('${key}','axisTickSize',+this.value);renderChart('${key}')" style="width:40px" min="8" max="18">
                    颜色:<input type="color" value="${cs.axisTickColor||'#000000'}" onchange="setChartStyle('${key}','axisTickColor',this.value);renderChart('${key}')" style="width:30px;height:22px"></div>
                    <div class="style-row"><strong>刻度像素间距</strong> X轴:<input type="number" value="${cs.xTickGapPx||80}" onchange="setChartStyle('${key}','xTickGapPx',+this.value)" style="width:50px" min="30" max="200" step="10">px
                        Y轴:<input type="number" value="${cs.yTickGapPx||50}" onchange="setChartStyle('${key}','yTickGapPx',+this.value)" style="width:50px" min="20" max="150" step="10">px</div>
                    <div class="style-row"><strong>X轴名偏移</strong> 
                        <button onclick="adjustAxisOffset('${key}','xNameGap',-5)" style="padding:2px 4px">◀</button>
                        <input type="number" id="xNameGap_${key}" value="${cs.xNameGap||25}" onchange="setChartStyle('${key}','xNameGap',+this.value)" style="width:40px" min="-50" max="80">px
                        <button onclick="adjustAxisOffset('${key}','xNameGap',5)" style="padding:2px 4px">▶</button>
                        <strong style="margin-left:10px">Y轴名偏移</strong>
                        <button onclick="adjustAxisOffset('${key}','yNameGap',-5)" style="padding:2px 4px">◀</button>
                        <input type="number" id="yNameGap_${key}" value="${cs.yNameGap||35}" onchange="setChartStyle('${key}','yNameGap',+this.value)" style="width:40px" min="-50" max="80">px
                        <button onclick="adjustAxisOffset('${key}','yNameGap',5)" style="padding:2px 4px">▶</button>
                    </div>
                    <div class="style-row"><strong>X刻度标签偏移</strong> 
                        <button onclick="adjustAxisOffset('${key}','xTickMargin',-1)" style="padding:2px 4px">▲</button>
                        <input type="number" id="xTickMargin_${key}" value="${cs.xTickMargin||8}" onchange="setChartStyle('${key}','xTickMargin',+this.value)" style="width:40px" min="-20" max="30">px
                        <button onclick="adjustAxisOffset('${key}','xTickMargin',1)" style="padding:2px 4px">▼</button>
                        <strong style="margin-left:10px">Y刻度标签偏移</strong>
                        <button onclick="adjustAxisOffset('${key}','yTickMargin',-1)" style="padding:2px 4px">◀</button>
                        <input type="number" id="yTickMargin_${key}" value="${cs.yTickMargin||8}" onchange="setChartStyle('${key}','yTickMargin',+this.value)" style="width:40px" min="-20" max="30">px
                        <button onclick="adjustAxisOffset('${key}','yTickMargin',1)" style="padding:2px 4px">▶</button>
                    </div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>图表尺寸</strong> 宽×高(px):
                        <input type="number" value="${cs.chartWidth||600}" onchange="setChartStyle('${key}','chartWidth',+this.value);renderChart('${key}')" style="width:60px" min="300" max="1200">×
                        <input type="number" value="${cs.chartHeight||400}" onchange="setChartStyle('${key}','chartHeight',+this.value);renderChart('${key}')" style="width:60px" min="200" max="800">
                    </div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>数据编辑</strong>
                        <button onclick="openRealBoxplotEditor('${key}')" style="font-size:10px;padding:2px 6px;margin-left:10px;background:#3498db;color:#fff;border:none;border-radius:4px">📝 编辑数据</button>
                        <button onclick="openChartImport('${key}','realboxplot')" style="font-size:10px;padding:2px 6px;margin-left:5px;background:#27ae60;color:#fff;border:none;border-radius:4px">📥 AI导入</button>
                    </div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>节点管理</strong>
                        <button onclick="openChartSaveModal('${key}')" style="font-size:10px;padding:3px 8px;margin-left:10px;background:#27ae60;color:#fff;border:none;border-radius:4px">💾 保存</button>
                        <button onclick="openChartRestoreModal('${key}')" style="font-size:10px;padding:3px 8px;margin-left:5px;background:#3498db;color:#fff;border:none;border-radius:4px">🔄 恢复</button>
                    </div>
                    <hr style="margin:10px 0;border:none;border-top:1px solid #ddd;">
                    <div class="style-row" style="justify-content:flex-end;gap:10px">
                        <button onclick="cancelStylePanel('${key}')" style="padding:6px 16px;cursor:pointer;border:1px solid #ddd;border-radius:4px;background:#fff">取消</button>
                        <button onclick="saveStylePanel('${key}')" style="padding:6px 16px;cursor:pointer;border:none;border-radius:4px;background:#667eea;color:#fff">保存</button>
                    </div>
                </div>
                <div id="chart_${key}" style="width:${cs.chartWidth||600}px;height:${cs.chartHeight||400}px">`;
            document.getElementById('card_' + key).innerHTML = html;
            
            const chartDiv = document.getElementById('chart_' + key);
            if (chartInstances[key]) chartInstances[key].dispose();
            
            // ★★★ 根据刻度像素间距计算容器尺寸 ★★★
            const xLabels = data.categories || [];
            const xTickGapPx = cs.xTickGapPx || 80;
            const yTickGapPx = cs.yTickGapPx || 50;
            const yTickCount = 6;
            const chartWidth = Math.max(xLabels.length * xTickGapPx + 150, 400);
            const chartHeight = Math.max(yTickCount * yTickGapPx + 120, 300);
            chartDiv.style.width = chartWidth + 'px';
            chartDiv.style.height = chartHeight + 'px';
            
            chartInstances[key] = echarts.init(chartDiv);
            
            // 计算箱线图数据: [min, Q1, median, Q3, max] + 异常值
            const boxData = [];
            const outliers = [];
            data.data.forEach((arr, idx) => {
                const sorted = arr.slice().sort((a,b) => a-b);
                const n = sorted.length;
                const q1 = sorted[Math.floor(n*0.25)];
                const median = sorted[Math.floor(n*0.5)];
                const q3 = sorted[Math.floor(n*0.75)];
                const iqr = q3 - q1;
                const whisLower = q1 - 1.5 * iqr;  // whis=1.5
                const whisUpper = q3 + 1.5 * iqr;
                
                // 找出须范围内的最小/最大值
                const lower = sorted.find(v => v >= whisLower) || sorted[0];
                const upper = sorted.filter(v => v <= whisUpper).pop() || sorted[n-1];
                
                // 异常值(红色+标记)
                sorted.forEach(v => {
                    if (v < whisLower || v > whisUpper) {
                        outliers.push([idx, v]);
                    }
                });
                
                boxData.push({
                    value: [lower, q1, median, q3, upper],
                    itemStyle: { color: vlagColors[idx % vlagColors.length], borderColor: '#333', borderWidth: 1 }
                });
            });
            
            const seriesList = [{ 
                name: 'boxplot', 
                type: 'boxplot', 
                data: boxData,
                boxWidth: ['30%', '50%']  // widths=0.5
            }];
            
            // 添加异常值散点(sym='r+')
            if (outliers.length > 0) {
                seriesList.push({
                    name: 'outlier',
                    type: 'scatter',
                    data: outliers,
                    itemStyle: { color: '#ff0000' },
                    symbol: 'cross',
                    symbolSize: 10
                });
            }
            
            const latestCs = chartStyles[key] || {};
            const xAxisName = latestCs.xAxisName || 'Groups';
            const yAxisName = latestCs.yAxisName || 'Values';
            chartInstances[key].setOption({
                title: { text: mainTitle, left: 'center', textStyle: { fontSize: 15 }, triggerEvent: true },
                tooltip: { trigger: 'item', axisPointer: { type: 'shadow' } },
                grid: { left: 60, right: 30, top: 50, bottom: 50, borderColor: '#000', borderWidth: 1, show: true },
                xAxis: { 
                    type: 'category', 
                    data: data.categories, 
                    name: xAxisName, 
                    nameLocation: 'middle', 
                    nameGap: latestCs.xNameGap || 25, 
                    boundaryGap: true, 
                    triggerEvent: true,
                    axisLine: { show: true, lineStyle: { color: '#000', width: 1 } },
                    axisTick: { show: true, lineStyle: { color: '#000' } },
                    nameTextStyle: { fontSize: latestCs.axisLabelSize || 12, color: latestCs.axisLabelColor || '#000' },
                    axisLabel: { fontSize: latestCs.axisTickSize || 10, color: latestCs.axisTickColor || '#000' }
                },
                yAxis: { 
                    type: 'value', 
                    name: yAxisName,
                    nameLocation: 'middle',
                    nameGap: latestCs.yNameGap || 35,
                    min: latestCs.yAxisMin || null,
                    max: latestCs.yAxisMax || null,
                    triggerEvent: true,
                    axisLine: { show: true, lineStyle: { color: '#000', width: 1 } },
                    axisTick: { show: true, lineStyle: { color: '#000' } },
                    splitLine: { show: false },
                    nameTextStyle: { fontSize: latestCs.axisLabelSize || 12, color: latestCs.axisLabelColor || '#000' },
                    axisLabel: { fontSize: latestCs.axisTickSize || 10, color: latestCs.axisTickColor || '#000' }
                },
                series: seriesList
            });
            // 点击编辑功能
            chartInstances[key].on('click', function(params) {
                // 点击标题编辑
                if (params.componentType === 'title') {
                    showTextEditor(customTitles[key] || matricesData[key].name, function(newTitle) {
                        customTitles[key] = newTitle;
                        saveAllToStorage();
                        renderChart(key);
                    });
                }
                // 点击X轴刻度标签
                else if (params.componentType === 'xAxis' && params.targetType === 'axisLabel') {
                    const labelIdx = matricesData[key].data.categories.indexOf(params.value);
                    if (labelIdx >= 0) {
                        showTextEditor(params.value, function(newLabel) {
                            matricesData[key].data.categories[labelIdx] = newLabel;
                            saveAllToStorage();
                            renderChart(key);
                        });
                    }
                }
                // 点击X轴名称(Groups)
                else if (params.componentType === 'xAxis' && params.targetType === 'axisName') {
                    showTextEditor(chartStyles[key]?.xAxisName || 'Groups', function(newName) {
                        if (!chartStyles[key]) chartStyles[key] = {};
                        chartStyles[key].xAxisName = newName;
                        saveAllToStorage();
                        renderChart(key);
                    });
                }
                // 点击Y轴名称(Values)
                else if (params.componentType === 'yAxis' && params.targetType === 'axisName') {
                    showTextEditor(chartStyles[key]?.yAxisName || 'Values', function(newName) {
                        if (!chartStyles[key]) chartStyles[key] = {};
                        chartStyles[key].yAxisName = newName;
                        saveAllToStorage();
                        renderChart(key);
                    });
                }
                // 点击Y轴刻度标签 - 设置Y轴范围
                else if (params.componentType === 'yAxis' && params.targetType === 'axisLabel') {
                    const currentMin = chartStyles[key]?.yAxisMin || 'auto';
                    const currentMax = chartStyles[key]?.yAxisMax || 'auto';
                    const newRange = prompt('设置Y轴范围\\n\\n当前: [' + currentMin + ', ' + currentMax + ']\\n格式: 最小值,最大值 (留空为自动)', currentMin + ',' + currentMax);
                    if (newRange !== null) {
                        const parts = newRange.split(',');
                        if (!chartStyles[key]) chartStyles[key] = {};
                        chartStyles[key].yAxisMin = parts[0]?.trim() === 'auto' || parts[0]?.trim() === '' ? null : parseFloat(parts[0]);
                        chartStyles[key].yAxisMax = parts[1]?.trim() === 'auto' || parts[1]?.trim() === '' ? null : parseFloat(parts[1]);
                        saveAllToStorage();
                        renderChart(key);
                    }
                }
                // 点击箱体编辑数据
                else if (params.componentType === 'series' && params.seriesType === 'boxplot') {
                    const dataIdx = params.dataIndex;
                    openBoxplotDataEditor(key, dataIdx);
                }
            });
        }
        
        // 图7箱线图单组数据编辑弹窗
        let boxplotDataEditorDiv = null;
        function openBoxplotDataEditor(key, groupIdx) {
            const info = matricesData[key];
            const data = info.data.data[groupIdx] || [];
            const groupName = info.data.categories[groupIdx] || ('组' + (groupIdx+1));
            
            if (boxplotDataEditorDiv) boxplotDataEditorDiv.remove();
            boxplotDataEditorDiv = document.createElement('div');
            boxplotDataEditorDiv.style.cssText = 'position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:#fff;padding:20px;border-radius:12px;box-shadow:0 10px 40px rgba(0,0,0,0.3);z-index:2000;max-height:80vh;overflow:auto;min-width:300px';
            
            let html = '<div style="font-size:16px;font-weight:bold;margin-bottom:15px">📊 编辑箱线图数据 - ' + groupName + '</div>';
            html += '<div style="margin-bottom:10px"><label>组名:</label><input type="text" id="boxplotGroupName" value="' + groupName + '" style="margin-left:5px;padding:4px;border:1px solid #ddd;border-radius:4px"></div>';
            html += '<div style="margin-bottom:5px"><strong>数据点(每行一个数值):</strong></div>';
            html += '<textarea id="boxplotDataValues" style="width:100%;height:150px;padding:8px;border:1px solid #ddd;border-radius:4px;font-family:monospace">' + data.join('\\n') + '</textarea>';
            html += '<div style="margin-top:15px;text-align:right"><button onclick="closeBoxplotDataEditor()" style="padding:8px 16px;margin-right:10px;border:1px solid #ddd;border-radius:4px;background:#fff;cursor:pointer">取消</button>';
            html += '<button onclick="saveBoxplotDataEditor(\\'' + key + '\\',' + groupIdx + ')" style="padding:8px 16px;border:none;border-radius:4px;background:#667eea;color:#fff;cursor:pointer">保存</button></div>';
            
            boxplotDataEditorDiv.innerHTML = html;
            document.body.appendChild(boxplotDataEditorDiv);
        }
        function closeBoxplotDataEditor() { if (boxplotDataEditorDiv) { boxplotDataEditorDiv.remove(); boxplotDataEditorDiv = null; } }
        function saveBoxplotDataEditor(key, groupIdx) {
            const groupName = document.getElementById('boxplotGroupName').value;
            const valuesText = document.getElementById('boxplotDataValues').value;
            const values = valuesText.split('\\n').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
            
            matricesData[key].data.categories[groupIdx] = groupName;
            matricesData[key].data.data[groupIdx] = values;
            saveAllToStorage();
            closeBoxplotDataEditor();
            renderChart(key);
        }
        
        // 图10: 饼图(参考图2样式)
        function renderPieChart(key) {
            const info = matricesData[key];
            const data = info.data;
            const mainTitle = customTitles[key] || info.name;
            const subtitle = customSubtitles[key] || info.subtitle || '';
            const cs = chartStyles[key] || {};
            
            let html = `
                <div class="card-header">
                    <div><span class="card-title">${mainTitle}</span><span style="font-size:12px;color:#666;margin-left:8px">${subtitle}</span></div>
                    <div class="card-controls">
                        <button class="ctrl-btn" onclick="exportChart('${key}','png')">PNG</button>
                        <button class="ctrl-btn" onclick="exportChart('${key}','jpg')">JPG</button>
                        <button class="ctrl-btn" onclick="exportChart('${key}','tif')">TIF</button>
                        <button class="ctrl-btn" onclick="toggleStylePanel('${key}')">🎨</button>
                    </div>
                </div>
                <div class="style-panel" id="panel_${key}">
                    <div class="style-panel-header">
                        <span>📊 ${mainTitle} 样式设置</span>
                        <button class="style-panel-close" onclick="cancelStylePanel('${key}')">✕</button>
                    </div>
                    <div class="style-row"><label>主标题:</label><input type="text" value="${mainTitle}" onchange="setCustomTitle('${key}', this.value)" style="flex:1;padding:4px 8px;border:1px solid #ddd;border-radius:4px;font-size:12px"></div>
                    <div class="style-row"><label>副标题:</label><input type="text" value="${subtitle}" onchange="setCustomSubtitle('${key}', this.value)" style="flex:1;padding:4px 8px;border:1px solid #ddd;border-radius:4px;font-size:12px"></div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>标签字体</strong> 字号:<input type="number" value="${cs.labelSize||12}" onchange="setChartStyle('${key}','labelSize',+this.value);renderChart('${key}')" style="width:40px" min="8" max="20">
                    颜色:<input type="color" value="${cs.labelColor||'#000000'}" onchange="setChartStyle('${key}','labelColor',this.value);renderChart('${key}')" style="width:30px;height:22px"></div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>图表尺寸</strong> 宽×高(px):
                        <input type="number" value="${cs.chartWidth||600}" onchange="setChartStyle('${key}','chartWidth',+this.value);renderChart('${key}')" style="width:60px" min="300" max="1200">×
                        <input type="number" value="${cs.chartHeight||400}" onchange="setChartStyle('${key}','chartHeight',+this.value);renderChart('${key}')" style="width:60px" min="200" max="800">
                    </div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>数据编辑</strong>
                        <button onclick="openPieEditor('${key}')" style="font-size:10px;padding:2px 6px;margin-left:10px;background:#3498db;color:#fff;border:none;border-radius:4px">📝 编辑数据</button>
                        <button onclick="openChartImport('${key}','pie')" style="font-size:10px;padding:2px 6px;margin-left:5px;background:#27ae60;color:#fff;border:none;border-radius:4px">📥 AI导入</button>
                    </div>
                    <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
                    <div class="style-row"><strong>节点管理</strong>
                        <button onclick="openChartSaveModal('${key}')" style="font-size:10px;padding:3px 8px;margin-left:10px;background:#27ae60;color:#fff;border:none;border-radius:4px">💾 保存</button>
                        <button onclick="openChartRestoreModal('${key}')" style="font-size:10px;padding:3px 8px;margin-left:5px;background:#3498db;color:#fff;border:none;border-radius:4px">🔄 恢复</button>
                    </div>
                    <hr style="margin:10px 0;border:none;border-top:1px solid #ddd;">
                    <div class="style-row" style="justify-content:flex-end;gap:10px">
                        <button onclick="cancelStylePanel('${key}')" style="padding:6px 16px;cursor:pointer;border:1px solid #ddd;border-radius:4px;background:#fff">取消</button>
                        <button onclick="saveStylePanel('${key}')" style="padding:6px 16px;cursor:pointer;border:none;border-radius:4px;background:#667eea;color:#fff">保存</button>
                    </div>
                </div>
                <div id="chart_${key}" style="width:${cs.chartWidth||600}px;height:${cs.chartHeight||400}px;border:1px solid #000"></div>`;
            document.getElementById('card_' + key).innerHTML = html;
            
            const chartDiv = document.getElementById('chart_' + key);
            if (chartInstances[key]) chartInstances[key].dispose();
            chartInstances[key] = echarts.init(chartDiv);
            const latestCs = chartStyles[key] || {};
            chartInstances[key].setOption({
                tooltip: { trigger: 'item' },
                legend: { orient: 'vertical', left: 'left', textStyle: { fontSize: latestCs.labelSize || 12, color: latestCs.labelColor || '#000' } },
                series: [{ 
                    type: 'pie', 
                    radius: '60%', 
                    data: data.series, 
                    label: { fontSize: latestCs.labelSize || 12, color: latestCs.labelColor || '#000' },
                    emphasis: { itemStyle: { shadowBlur: 10, shadowOffsetX: 0, shadowColor: 'rgba(0,0,0,0.5)' } } 
                }]
            });
            // 点击编辑功能
            chartInstances[key].on('click', function(params) {
                if (params.componentType === 'series') {
                    const dataIdx = params.dataIndex;
                    const item = matricesData[key].data.series[dataIdx];
                    // 点击扇区 - 只编辑数值
                    showValueEditor('编辑 ' + item.name + ' 数值', item.value, function(newVal) {
                        matricesData[key].data.series[dataIdx].value = newVal;
                        saveAllToStorage();
                        renderChart(key);
                    });
                }
                // 点击图例标签 - 编辑名称
                else if (params.componentType === 'legend') {
                    const dataIdx = matricesData[key].data.series.findIndex(s => s.name === params.name);
                    if (dataIdx >= 0) {
                        showTextEditor(params.name, function(newName) {
                            matricesData[key].data.series[dataIdx].name = newName;
                            saveAllToStorage();
                            renderChart(key);
                        });
                    }
                }
            });
        }
        
        // 图23: 瀑布图
        function renderWaterfallChart(key) {
            const info = matricesData[key];
            const data = info.data;
            const mainTitle = customTitles[key] || info.name;
            const subtitle = customSubtitles[key] || info.subtitle || '';
            
            // 计算瀑布图数据
            const rawData = data.data;
            const labels = data.xAxis;
            let cumulative = 0;
            const barData = [], helperData = [];
            rawData.forEach((val, i) => {
                if (i === 0 || i === rawData.length - 1) {
                    helperData.push(0);
                    barData.push(val);
                } else {
                    helperData.push(cumulative);
                    barData.push(val);
                    cumulative += val;
                }
            });
            
            let html = `<div class="card-header"><div><span class="card-title">${mainTitle}</span><span style="font-size:12px;color:#666;margin-left:8px">${subtitle}</span></div>
                <div class="card-controls"><button class="ctrl-btn" onclick="exportChart('${key}','png')">PNG</button><button class="ctrl-btn" onclick="toggleStylePanel('${key}')">🎨</button></div></div>
                <div id="chart_${key}" style="width:600px;height:400px"></div>`;
            document.getElementById('card_' + key).innerHTML = html;
            
            const chartDiv = document.getElementById('chart_' + key);
            if (chartInstances[key]) chartInstances[key].dispose();
            chartInstances[key] = echarts.init(chartDiv);
            chartInstances[key].setOption({
                tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
                xAxis: { type: 'category', data: labels },
                yAxis: { type: 'value' },
                series: [
                    { type: 'bar', stack: 'total', itemStyle: { borderColor: 'transparent', color: 'transparent' }, data: helperData },
                    { type: 'bar', stack: 'total', label: { show: true, position: 'top' }, data: barData.map((v, i) => ({ value: Math.abs(v), itemStyle: { color: i === 0 || i === rawData.length - 1 ? '#5470c6' : (v > 0 ? '#91cc75' : '#ee6666') } })) }
                ]
            });
        }
        
        // ============ 分类数量切换 ============
        let currentClassCount = 5;
        const fullClassNames = ['A', 'B', 'C', 'D', 'E', 'F'];
        
        function setClassCount(count) {
            currentClassCount = count;
            // 调整customLabels长度
            while (customLabels.length < count) customLabels.push(fullClassNames[customLabels.length] || 'Class' + (customLabels.length + 1));
            customLabels = customLabels.slice(0, count);
            // 调整矩阵数据
            Object.keys(matricesData).forEach(key => {
                const data = matricesData[key].data;
                // 扩展行
                while (data.length < count) data.push(new Array(data[0]?.length || count).fill(0));
                // 裁剪行
                while (data.length > count) data.pop();
                // 扩展/裁剪列
                data.forEach((row, i) => {
                    while (row.length < count) row.push(0);
                    while (row.length > count) row.pop();
                });
            });
            saveAllToStorage();
            renderAll();
        }
        
        // ============ AI导入功能 ============
        let currentImportKey = null;
        let currentImportType = null;
        
        function openChartImport(key, chartType) {
            currentImportKey = key;
            currentImportType = chartType;
            const typeNames = { confusion: '混淆矩阵', bar: '柱状图', multibar: '对比图', boxplot: '箱线图', roc: 'ROC曲线' };
            document.getElementById('importModalTitle').textContent = '📥 AI导入 - ' + (typeNames[chartType] || chartType);
            document.getElementById('importTargetRow').style.display = 'none';  // 隐藏目标选择
            document.getElementById('importDataText').value = '';
            document.getElementById('importFile').value = '';
            document.getElementById('importStatus').style.display = 'none';
            document.getElementById('pasteArea').innerHTML = '点击此处或按Ctrl+V粘贴截图/文本';
            document.getElementById('pasteArea').style.lineHeight = '100px';
            document.getElementById('imagePreview').style.display = 'none';
            importImageBase64 = null;
            document.getElementById('importModal').classList.add('show');
        }
        
        function openImportModal() {
            currentImportKey = 'fig1';
            currentImportType = 'confusion';
            document.getElementById('importModalTitle').textContent = '📥 AI导入数据';
            document.getElementById('importTargetRow').style.display = 'none';
            document.getElementById('importDataText').value = '';
            document.getElementById('importFile').value = '';
            document.getElementById('importStatus').style.display = 'none';
            document.getElementById('pasteArea').innerHTML = '点击此处或按Ctrl+V粘贴截图/文本';
            document.getElementById('pasteArea').style.lineHeight = '100px';
            document.getElementById('imagePreview').style.display = 'none';
            importImageBase64 = null;
            document.getElementById('importModal').classList.add('show');
        }
        
        function closeImportModal() { document.getElementById('importModal').classList.remove('show'); }
        
        let importImageBase64 = null;  // 存储图片Base64
        
        // 粘贴事件处理
        document.addEventListener('paste', function(e) {
            if (!document.getElementById('importModal').classList.contains('show')) return;
            
            const items = e.clipboardData?.items;
            if (!items) return;
            
            for (let item of items) {
                if (item.type.startsWith('image/')) {
                    e.preventDefault();
                    const file = item.getAsFile();
                    const reader = new FileReader();
                    reader.onload = ev => {
                        importImageBase64 = ev.target.result;
                        document.getElementById('pasteArea').innerHTML = '<img src="' + importImageBase64 + '" style="max-width:100%;max-height:150px;border-radius:4px">';
                        document.getElementById('pasteArea').style.lineHeight = 'normal';
                        document.getElementById('importDataText').value = '[已粘贴截图]';
                        showImportStatus('✅ 截图已粘贴，点击AI识别', false);
                    };
                    reader.readAsDataURL(file);
                    return;
                }
            }
        });
        
        function handleImportFile(input) {
            if (input.files[0]) {
                const file = input.files[0];
                const ext = file.name.split('.').pop().toLowerCase();
                
                // 隐藏图片预览
                document.getElementById('imagePreview').style.display = 'none';
                importImageBase64 = null;
                
                if (['png', 'jpg', 'jpeg', 'tif', 'tiff'].includes(ext)) {
                    // 图片文件处理
                    const reader = new FileReader();
                    reader.onload = e => {
                        importImageBase64 = e.target.result;
                        document.getElementById('previewImg').src = importImageBase64;
                        document.getElementById('imagePreview').style.display = 'block';
                        document.getElementById('importDataText').value = '[已上传图片: ' + file.name + ']';
                        showImportStatus('✅ 图片已加载，点击AI识别将分析图片内容', false);
                    };
                    reader.readAsDataURL(file);
                } else if (ext === 'xlsx' || ext === 'xls') {
                    // Excel文件处理
                    const reader = new FileReader();
                    reader.onload = e => {
                        try {
                            const workbook = XLSX.read(e.target.result, { type: 'array' });
                            const sheet = workbook.Sheets[workbook.SheetNames[0]];
                            const data = XLSX.utils.sheet_to_json(sheet, { header: 1 });
                            const text = data.map(row => row.join('\\t')).join('\\n');
                            document.getElementById('importDataText').value = text;
                            showImportStatus('✅ Excel文件已读取，共' + data.length + '行', false);
                        } catch (err) {
                            showImportStatus('❌ Excel解析失败: ' + err.message, true);
                        }
                    };
                    reader.readAsArrayBuffer(file);
                } else {
                    // 文本文件处理
                    const reader = new FileReader();
                    reader.onload = e => {
                        document.getElementById('importDataText').value = e.target.result;
                    };
                    reader.readAsText(file);
                }
            }
        }
        
        function showImportStatus(msg, isError) {
            const status = document.getElementById('importStatus');
            status.textContent = msg;
            status.style.display = 'block';
            status.style.background = isError ? '#ffe6e6' : '#e6ffe6';
            status.style.color = isError ? '#c00' : '#060';
        }
        
        async function processAIImport() {
            const apiKey = document.getElementById('aiApiKey').value.trim();
            if (!apiKey) {
                showImportStatus('❌ 请先在右下角AI助手中配置API Key', true);
                return;
            }
            
            const dataText = document.getElementById('importDataText').value.trim();
            if (!dataText && !importImageBase64) {
                showImportStatus('❌ 请输入或上传数据', true);
                return;
            }
            
            const targetKey = currentImportKey || 'fig1';
            const chartType = currentImportType || 'confusion';
            const btn = document.getElementById('importBtn');
            btn.disabled = true;
            btn.textContent = '🔄 AI识别中...';
            
            let basePrompt;
            if (chartType === 'confusion') {
                basePrompt = `请从数据中提取混淆矩阵的数值。
- 数据可能使用任意类别标签，请智能识别
- 请根据数据自动检测分类数量（2-6类）
返回JSON：{"type":"confusion","size":N,"labels":["类别1",...],"matrix":[[...],...]}`; 
            } else if (chartType === 'bar') {
                basePrompt = `请从数据中提取柱状图数据（分类准确率）。
- 识别每个类别的名称和对应数值（百分比）
返回JSON：{"type":"bar","labels":["类别1","类别2",...],"values":[数值1,数值2,...]}`;
            } else if (chartType === 'multibar') {
                basePrompt = `请从数据中提取多系列柱状图数据（对比图）。
- 识别X轴标签和多个系列的名称及数值
返回JSON：{"type":"multibar","labels":["X标签1",...],"series":[{"name":"系列1","values":[...]},{"name":"系列2","values":[...]},...]}`;
            } else if (chartType === 'roc') {
                basePrompt = `请从数据中提取ROC曲线数据。
- 识别每条曲线的名称和点集(FPR,TPR)
返回JSON：{"type":"roc","curves":[{"name":"曲线1","points":[[fpr,tpr],...]},...],"labels":["曲线名1",...]}`;
            } else if (chartType === 'boxplot') {
                basePrompt = `请从数据中提取箱线图数据（带误差线）。
- 识别X轴标签和每个系列的平均值、最小值、最大值
返回JSON：{"type":"boxplot","labels":["X标签1",...],"series":[{"name":"系列1","avg":[...],"min":[...],"max":[...],"color":"#颜色"},...]}`; 
            } else {
                basePrompt = `请识别图表数据并返回JSON格式。`;
            }
            basePrompt += `\\n只返回JSON对象，不要其他文字。`;
            
            const platform = document.getElementById('aiPlatform').value;
            const model = document.getElementById('aiModel').value;
            let messages, apiUrl;
            
            if (importImageBase64) {
                if (platform === 'deepseek') {
                    showImportStatus('⚠️ DeepSeek不支持图片，请切换到"硅基流动"或"OpenAI"平台', true);
                    btn.disabled = false;
                    btn.textContent = '🤖 AI识别并填充';
                    return;
                }
                // 视觉模式
                showImportStatus('⏳ 正在调用视觉模型识别图片...', false);
                messages = [{
                    role: 'user',
                    content: [
                        { type: 'text', text: basePrompt },
                        { type: 'image_url', image_url: { url: importImageBase64 } }
                    ]
                }];
            } else {
                // 文本模式
                showImportStatus('⏳ 正在调用AI识别数据...', false);
                messages = [{ role: 'user', content: basePrompt + '\\n\\n数据内容：\\n' + dataText }];
            }
            
            // 根据平台选择API URL
            if (platform === 'openai') {
                apiUrl = 'https://api.openai.com/v1/chat/completions';
            } else if (platform === 'siliconflow') {
                apiUrl = 'https://api.siliconflow.cn/v1/chat/completions';
            } else {
                apiUrl = 'https://api.deepseek.com/chat/completions';
            }
            
            try {
                const response = await fetch(apiUrl, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + apiKey },
                    body: JSON.stringify({
                        model: model,
                        messages: messages,
                        temperature: 0.1,
                        max_tokens: 1000
                    })
                });
                
                if (!response.ok) throw new Error('API请求失败: ' + response.status);
                
                const apiResult = await response.json();
                const content = apiResult.choices[0].message.content.trim();
                
                // 尝试提取JSON
                let parsed;
                try {
                    parsed = JSON.parse(content);
                } catch (e) {
                    // 尝试从文本中提取JSON对象或数组
                    const objMatch = content.match(/\\{[\\s\\S]*\\}/);
                    const arrMatch = content.match(/\\[\\[.*\\]\\]/s);
                    if (objMatch) parsed = JSON.parse(objMatch[0]);
                    else if (arrMatch) parsed = { matrix: JSON.parse(arrMatch[0]) };
                    else throw new Error('无法从AI响应中提取数据');
                }
                
                // 根据图表类型处理返回数据
                const detectedType = parsed.type || chartType;
                
                if (detectedType === 'confusion' || parsed.matrix) {
                    // 混淆矩阵
                    let matrix = parsed.matrix;
                    let detectedSize = parsed.size || matrix?.length;
                    let detectedLabels = parsed.labels || null;
                    
                    if (!matrix && Array.isArray(parsed)) {
                        matrix = parsed;
                        detectedSize = parsed.length;
                    }
                    
                    if (detectedSize >= 2 && detectedSize <= 6 && detectedSize !== currentClassCount) {
                        setClassCount(detectedSize);
                    }
                    
                    if (detectedLabels && Array.isArray(detectedLabels)) {
                        for (let i = 0; i < Math.min(detectedLabels.length, 6); i++) {
                            customLabels[i] = detectedLabels[i] || customLabels[i];
                        }
                    }
                    
                    const size = Math.min(detectedSize, 6);
                    for (let i = 0; i < size; i++) {
                        for (let j = 0; j < size; j++) {
                            matricesData[targetKey].data[i][j] = Number(matrix[i]?.[j]) || 0;
                        }
                    }
                } else if (detectedType === 'bar' || parsed.values) {
                    // 柱状图
                    matricesData[targetKey].data.labels = parsed.labels || [];
                    matricesData[targetKey].data.values = parsed.values || [];
                } else if (detectedType === 'multibar' || parsed.series) {
                    // 对比图
                    matricesData[targetKey].data.labels = parsed.labels || [];
                    if (parsed.series) {
                        const values = [];
                        parsed.labels?.forEach((_, xi) => {
                            values[xi] = parsed.series.map(s => s.values?.[xi] || 0);
                        });
                        matricesData[targetKey].data.values = values;
                        // 更新系列配置
                        if (boxplotSeries[targetKey]) {
                            boxplotSeries[targetKey].count = parsed.series.length;
                            boxplotSeries[targetKey].names = parsed.series.map(s => s.name);
                        }
                    }
                } else if (detectedType === 'roc' || parsed.curves) {
                    // ROC曲线
                    matricesData[targetKey].data.curves = parsed.curves?.map(c => c.points || c) || [];
                    matricesData[targetKey].data.labels = parsed.labels || parsed.curves?.map(c => c.name) || [];
                } else if (detectedType === 'boxplot') {
                    // 箱线图
                    matricesData[targetKey].data.labels = parsed.labels || [];
                    matricesData[targetKey].data.series = parsed.series || [];
                } else {
                    throw new Error('AI返回格式无效或图表类型不支持');
                }
                
                saveAllToStorage();
                renderChart(targetKey);
                showImportStatus('✅ 成功导入并填充到 ' + matricesData[targetKey].name, false);
                
            } catch (error) {
                showImportStatus('❌ ' + error.message, true);
            } finally {
                btn.disabled = false;
                btn.textContent = '🤖 AI识别并填充';
            }
        }
        
        // ============ AI聊天功能 ============
        const AI_CONFIG_KEY = 'deepseek_ai_config';
        let aiMessages = [];
        
        function toggleAI() { 
            document.getElementById('aiPanel').classList.toggle('show');
            updateApiKeyWarning();
        }
        
        // 拖动功能和悬停展开
        (function() {
            let isDragging = false;
            let hasMoved = false;
            let dragTarget = null;
            let dragOffsetX = 0, dragOffsetY = 0;
            let hoverTimer = null;
            
            // AI按钮
            const btn = document.getElementById('aiToggleBtn');
            
            // 悬停2秒展开
            btn.addEventListener('mouseenter', function() {
                if (!isDragging) {
                    hoverTimer = setTimeout(function() {
                        if (!isDragging && !hasMoved) {
                            btn.classList.add('expanded');
                        }
                    }, 2000);
                }
            });
            btn.addEventListener('mouseleave', function() {
                clearTimeout(hoverTimer);
                btn.classList.remove('expanded');
            });
            
            // 拖动
            btn.addEventListener('mousedown', function(e) {
                clearTimeout(hoverTimer);
                btn.classList.remove('expanded');
                isDragging = true;
                hasMoved = false;
                dragTarget = 'btn';
                dragOffsetX = e.clientX - btn.offsetLeft;
                dragOffsetY = e.clientY - btn.offsetTop;
            });
            btn.addEventListener('click', function(e) {
                if (!hasMoved) toggleAI();
            });
            
            // AI面板拖动
            const panel = document.getElementById('aiPanel');
            const header = document.getElementById('aiHeader');
            header.addEventListener('mousedown', function(e) {
                if (e.target.classList.contains('ai-close')) return;
                isDragging = true;
                hasMoved = false;
                dragTarget = 'panel';
                dragOffsetX = e.clientX - panel.offsetLeft;
                dragOffsetY = e.clientY - panel.offsetTop;
                panel.style.transition = 'none';
            });
            
            document.addEventListener('mousemove', function(e) {
                if (!isDragging) return;
                hasMoved = true;
                clearTimeout(hoverTimer);
                btn.classList.remove('expanded');
                e.preventDefault();
                const x = e.clientX - dragOffsetX;
                const y = e.clientY - dragOffsetY;
                if (dragTarget === 'btn') {
                    btn.style.right = 'auto';
                    btn.style.bottom = 'auto';
                    btn.style.left = Math.max(0, Math.min(window.innerWidth - 60, x)) + 'px';
                    btn.style.top = Math.max(0, Math.min(window.innerHeight - 60, y)) + 'px';
                } else {
                    panel.style.right = 'auto';
                    panel.style.bottom = 'auto';
                    panel.style.left = Math.max(0, Math.min(window.innerWidth - 350, x)) + 'px';
                    panel.style.top = Math.max(0, Math.min(window.innerHeight - 400, y)) + 'px';
                }
            });
            
            document.addEventListener('mouseup', function() {
                isDragging = false;
                dragTarget = null;
                panel.style.transition = '';
            });
        })();
        
        function toggleApiKeyVisibility() {
            const input = document.getElementById('aiApiKey');
            const btn = document.getElementById('toggleKeyBtn');
            if (input.type === 'password') {
                input.type = 'text';
                btn.textContent = '🙈';
            } else {
                input.type = 'password';
                btn.textContent = '👁️';
            }
        }
        
        function updateApiKeyWarning() {
            const apiKey = document.getElementById('aiApiKey').value.trim();
            const warning = document.getElementById('apiKeyWarning');
            warning.style.display = apiKey ? 'none' : 'block';
        }
        
        function saveAIConfig() {
            const config = {
                apiKey: document.getElementById('aiApiKey').value,
                platform: document.getElementById('aiPlatform').value,
                model: document.getElementById('aiModel').value
            };
            localStorage.setItem(AI_CONFIG_KEY, JSON.stringify(config));
            updateApiKeyWarning();
        }
        
        function loadAIConfig() {
            const saved = localStorage.getItem(AI_CONFIG_KEY);
            if (saved) {
                try {
                    const config = JSON.parse(saved);
                    document.getElementById('aiApiKey').value = config.apiKey || '';
                    document.getElementById('aiPlatform').value = config.platform || 'deepseek';
                    updateModelOptions();
                    document.getElementById('aiModel').value = config.model || 'deepseek-chat';
                } catch (e) {}
            }
            updateApiKeyWarning();
        }
        
        function updateModelOptions() {
            const platform = document.getElementById('aiPlatform').value;
            const modelSelect = document.getElementById('aiModel');
            const link = document.getElementById('apiKeyLink');
            
            if (platform === 'openai') {
                modelSelect.innerHTML = '<option value="gpt-4o">GPT-4o (视觉)</option><option value="gpt-4o-mini">GPT-4o-mini</option>';
                link.href = 'https://platform.openai.com/api-keys';
                link.textContent = '获取OpenAI Key →';
            } else if (platform === 'siliconflow') {
                modelSelect.innerHTML = '<option value="Qwen/Qwen2.5-VL-72B-Instruct">Qwen2.5-VL-72B ¥4.13/M</option><option value="deepseek-ai/deepseek-vl2">DeepSeek-VL2 ¥1.33/M</option><option value="Qwen/Qwen3-VL-8B-Instruct">Qwen3-VL-8B ¥0.3/M</option><option value="deepseek-ai/DeepSeek-V3">DeepSeek-V3 ¥2/M(文本)</option>';
                link.href = 'https://cloud.siliconflow.cn/account/ak';
                link.textContent = '获取硅基流动Key →';
            } else {
                modelSelect.innerHTML = '<option value="deepseek-chat">DeepSeek V3</option><option value="deepseek-reasoner">DeepSeek R1</option>';
                link.href = 'https://platform.deepseek.com/api_keys';
                link.textContent = '获取DeepSeek Key →';
            }
        }
        
        function addAIMessage(role, content) {
            const container = document.getElementById('aiMessages');
            const div = document.createElement('div');
            div.className = 'ai-msg ' + role;
            div.innerHTML = '<div class="bubble">' + formatAIContent(content) + '</div>';
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
            return div;
        }
        
        function formatAIContent(content) {
            // 简单的代码块处理
            return content.replace(/```([\\s\\S]*?)```/g, '<pre>$1</pre>')
                         .replace(/`([^`]+)`/g, '<code style="background:#f0f0f0;padding:2px 4px;border-radius:3px">$1</code>')
                         .replace(/\\n/g, '<br>');
        }
        
        async function sendAI() {
            const apiKey = document.getElementById('aiApiKey').value.trim();
            const model = document.getElementById('aiModel').value;
            const input = document.getElementById('aiInput');
            const sendBtn = document.getElementById('aiSend');
            const question = input.value.trim();
            
            if (!apiKey) { alert('请先输入 DeepSeek API Key'); return; }
            if (!question) return;
            
            // 添加用户消息
            addAIMessage('user', question);
            aiMessages.push({ role: 'user', content: question });
            input.value = '';
            sendBtn.disabled = true;
            
            // 添加AI消息占位
            const aiDiv = addAIMessage('ai', '思考中...');
            const bubble = aiDiv.querySelector('.bubble');
            
            try {
                const response = await fetch('https://api.deepseek.com/chat/completions', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': 'Bearer ' + apiKey
                    },
                    body: JSON.stringify({
                        model: model,
                        messages: [
                            { role: 'system', content: '你是一个数据分析助手，帮助用户理解和分析混淆矩阵数据。当前有' + currentClassCount + '分类问题，类别为：' + customLabels.join(', ') },
                            ...aiMessages
                        ],
                        stream: true
                    })
                });
                
                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.error?.message || '请求失败');
                }
                
                // 流式读取
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let fullContent = '';
                bubble.innerHTML = '';
                
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\\n').filter(line => line.trim().startsWith('data:'));
                    
                    for (const line of lines) {
                        const data = line.slice(5).trim();
                        if (data === '[DONE]') continue;
                        try {
                            const json = JSON.parse(data);
                            const content = json.choices?.[0]?.delta?.content || '';
                            if (content) {
                                fullContent += content;
                                bubble.innerHTML = formatAIContent(fullContent);
                                document.getElementById('aiMessages').scrollTop = document.getElementById('aiMessages').scrollHeight;
                            }
                        } catch (e) {}
                    }
                }
                
                aiMessages.push({ role: 'assistant', content: fullContent });
                
            } catch (err) {
                bubble.innerHTML = '<span style="color:red">错误: ' + err.message + '</span>';
            }
            
            sendBtn.disabled = false;
        }
        
        // 初始化：加载保存的设置
        loadSettings();
        loadAIConfig();
        renderAll();
    </script>
</body>
</html>'''
    
    return html


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("HTML 交互式可视化生成")
    print("=" * 60)
    
    generate_confusion_html()
    
    print("\n" + "=" * 60)
    print("完成! HTML已保存到 docs/index.html")
    print("公开地址: https://zcq991029.github.io/zcq-visualization/")
    print("=" * 60)


if __name__ == '__main__':
    main()

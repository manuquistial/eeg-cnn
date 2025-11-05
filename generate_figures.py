#!/usr/bin/env python3
"""
Script para generar gráficas de resultados para el artículo.
Genera:
1. Matriz de confusión para BoF-SVM
2. Comparación de métricas entre modelos (bar chart) - ACTUALIZADO
3. Heatmap de resultados del grid search
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json

# Configuración
ROOT_DIR = Path(__file__).parent
RESULTS_DIR = ROOT_DIR / 'results'
BOF_DIR = RESULTS_DIR / 'bof_svm'
CNN_DIR = RESULTS_DIR / 'deepconvnet'
FIGS_DIR = RESULTS_DIR / 'figures'
FIGS_DIR.mkdir(exist_ok=True, parents=True)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.style.use('seaborn-v0_8-whitegrid')

# 1. Matriz de confusión BoF-SVM
print("Generando matriz de confusión BoF-SVM...")
try:
    cm = np.load(BOF_DIR / 'confusion_matrix.npy')
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Etiquetas
    classes = ['MI-L (Left)', 'MI-R (Right)']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Matriz de Confusión - BoF-SVM',
           ylabel='Etiqueta Real',
           xlabel='Etiqueta Predicha')

    # Rotar etiquetas
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Agregar valores en las celdas
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(int(cm[i, j]), 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'confusion_matrix_bof_svm.pdf', bbox_inches='tight')
    plt.savefig(FIGS_DIR / 'confusion_matrix_bof_svm.png', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Guardado: {FIGS_DIR / 'confusion_matrix_bof_svm.pdf'}")
except Exception as e:
    print(f"  ✗ Error: {e}")

# 2. Comparación de métricas (ACTUALIZADO con nuevos valores DeepConvNet)
print("\nGenerando comparación de métricas...")
try:
    # Cargar resultados BoF-SVM desde grid_search_results.csv
    grid_df = pd.read_csv(BOF_DIR / 'grid_search_results.csv')
    best_row = grid_df.loc[grid_df['mean_f1'].idxmax()]
    bof_results = {
        'mean_accuracy': best_row['mean_accuracy'],
        'mean_precision': best_row['mean_precision'],
        'mean_recall': best_row['mean_recall'],
        'mean_f1': best_row['mean_f1']
    }

    # Cargar resultados DeepConvNet desde CSV
    cnn_df = pd.read_csv(CNN_DIR / 'deepconvnet_metrics.csv')
    cnn_results = {
        'accuracy': cnn_df['accuracy'].iloc[0],
        'precision': cnn_df['precision'].iloc[0],
        'recall': cnn_df['recall'].iloc[0],
        'f1': cnn_df['f1_score'].iloc[0]
    }

    print(f"  DeepConvNet - Accuracy: {cnn_results['accuracy']:.4f}")

    # Preparar datos
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    bof_values = [
        bof_results['mean_accuracy'],
        bof_results['mean_precision'],
        bof_results['mean_recall'],
        bof_results['mean_f1']
    ]
    cnn_values = [
        cnn_results['accuracy'],
        cnn_results['precision'],
        cnn_results['recall'],
        cnn_results['f1']
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, bof_values, width, label='BoF-SVM',
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, cnn_values, width, label='DeepConvNet',
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)

    # Agregar valores en las barras
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=9)

    autolabel(bars1)
    autolabel(bars2)

    ax.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax.set_title('Comparación de Métricas entre Modelos', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='upper left', fontsize=11)
    ax.set_ylim([0, max(max(bof_values), max(cnn_values)) * 1.2])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Límite aleatorio')

    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'metrics_comparison.pdf', bbox_inches='tight')
    plt.savefig(FIGS_DIR / 'metrics_comparison.png', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Guardado: {FIGS_DIR / 'metrics_comparison.pdf'}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# 3. Heatmap de Grid Search
print("\nGenerando heatmap de Grid Search...")
try:
    grid_df = pd.read_csv(BOF_DIR / 'grid_search_results.csv')

    # Crear matriz para el heatmap
    k_values = sorted(grid_df['k_clusters'].unique())
    c_values = sorted(grid_df['svm_c'].unique())

    # Matriz de F1-scores
    f1_matrix = np.zeros((len(k_values), len(c_values)))
    for idx, row in grid_df.iterrows():
        k_idx = k_values.index(row['k_clusters'])
        c_idx = c_values.index(row['svm_c'])
        f1_matrix[k_idx, c_idx] = row['mean_f1']

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(f1_matrix, cmap='YlOrRd', aspect='auto', vmin=f1_matrix.min(), vmax=f1_matrix.max())

    # Etiquetas
    ax.set_xticks(np.arange(len(c_values)))
    ax.set_yticks(np.arange(len(k_values)))
    ax.set_xticklabels([f'C={c}' for c in c_values])
    ax.set_yticklabels([f'K={k}' for k in k_values])
    ax.set_xlabel('Parámetro C (SVM)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Número de Clusters (K)', fontweight='bold', fontsize=12)
    ax.set_title('Grid Search: F1-Score por Combinación de Hiperparámetros',
                fontweight='bold', fontsize=12)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('F1-Score', fontweight='bold', fontsize=11)

    # Agregar valores en las celdas
    for i in range(len(k_values)):
        for j in range(len(c_values)):
            text = ax.text(j, i, f'{f1_matrix[i, j]:.3f}',
                         ha="center", va="center",
                         color="black", fontweight='bold', fontsize=10)

    # Marcar el mejor resultado
    best_idx = np.unravel_index(np.argmax(f1_matrix), f1_matrix.shape)
    rect = plt.Rectangle((best_idx[1]-0.5, best_idx[0]-0.5), 1, 1,
                        fill=False, edgecolor='blue', linewidth=3)
    ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'grid_search_heatmap.pdf', bbox_inches='tight')
    plt.savefig(FIGS_DIR / 'grid_search_heatmap.png', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Guardado: {FIGS_DIR / 'grid_search_heatmap.pdf'}")
except Exception as e:
    print(f"  ✗ Error: {e}")

print(f"\n Todas las gráficas generadas en: {FIGS_DIR}")

# -*- coding: utf-8 -*-
"""
DeepConvNet para Clasificación de Señales EEG de Imaginación Motora

Implementación basada en:
Schirrmeister et al. (2017) "Deep learning with convolutional neural networks 
for EEG decoding and visualization"

Esta arquitectura fue específicamente diseñada para señales EEG y ha demostrado
excelente rendimiento en tareas de clasificación de imaginación motora.

Autor: [Tu nombre]
Fecha: Octubre 2025
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# MNE para procesamiento EEG
import mne
from mne.io import read_raw_eeglab

# PyTorch para DeepConvNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim

# Sklearn para métricas
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)

# Utilidades
from glob import glob
from tqdm import tqdm
import re

# ----------------------------
# CONFIGURACIÓN
# ----------------------------
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Parámetros de preprocesamiento
LOW_FREQ = 8.0
HIGH_FREQ = 30.0
EXPECTED_TRIAL_SEC = 9.0

# Parámetros del modelo
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
PATIENCE = 10  # Para early stopping

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Usando dispositivo: {DEVICE}")

# ----------------------------
# UTILIDADES DE CARGA
# ----------------------------
def subject_from_fname(fname: str) -> str:
    """Extrae el ID del sujeto del nombre del archivo"""
    m = re.search(r"(S\d{3})", os.path.basename(fname))
    return m.group(1) if m else os.path.basename(fname)


def load_eeg_file(fname: str, label: int) -> Tuple[np.ndarray, int, str]:
    """
    Carga un archivo EEG y devuelve los datos, etiqueta y sujeto
    
    Args:
        fname: Ruta al archivo .set
        label: 0 para left, 1 para right
    
    Returns:
        data: Array (n_epochs, n_channels, n_timepoints)
        label: Etiqueta de clase
        subject: ID del sujeto
    """
    subject = subject_from_fname(fname)
    
    # Intentar cargar como epochs
    try:
        epochs = mne.read_epochs_eeglab(fname, verbose='ERROR')
        data = epochs.get_data()  # (n_epochs, n_channels, n_timepoints)
        return data, label, subject
    except:
        pass
    
    # Si falla, cargar como raw y segmentar
    raw = read_raw_eeglab(fname, preload=True, verbose='ERROR')
    
    # Filtrar
    try:
        raw.filter(LOW_FREQ, HIGH_FREQ, verbose='ERROR')
    except:
        pass
    
    # Segmentar en ventanas de 9 segundos
    sfreq = float(raw.info['sfreq'])
    n_win = int(np.floor(raw.times[-1] / EXPECTED_TRIAL_SEC))
    
    if n_win < 1:
        data = raw.get_data()
        data = np.expand_dims(data, axis=0)  # (1, channels, timepoints)
        return data, label, subject
    
    samps = int(EXPECTED_TRIAL_SEC * sfreq)
    data_list = []
    
    for i in range(n_win):
        start, end = i * samps, (i + 1) * samps
        if end <= raw.n_times:
            segment = raw.get_data()[:, start:end]
            data_list.append(segment)
    
    data = np.stack(data_list, axis=0)  # (n_epochs, channels, timepoints)
    return data, label, subject


def load_dataset(data_root: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Carga todo el dataset de left_imag y right_imag
    
    Returns:
        X: Array (N, channels, timepoints)
        y: Array (N,) con labels 0=left, 1=right
        subjects: Lista de IDs de sujetos
    """
    X_list = []
    y_list = []
    subjects_list = []
    
    # Cargar left_imag (label 0)
    left_dir = data_root / "left_imag"
    if left_dir.exists():
        print(f"📂 Cargando datos de {left_dir}")
        for set_file in tqdm(sorted(glob(str(left_dir / "*.set"))), desc="Left"):
            try:
                data, label, subject = load_eeg_file(set_file, label=0)
                for epoch in data:
                    X_list.append(epoch)
                    y_list.append(label)
                    subjects_list.append(subject)
            except Exception as e:
                print(f"⚠️  Error cargando {set_file}: {e}")
    
    # Cargar right_imag (label 1)
    right_dir = data_root / "right_imag"
    if right_dir.exists():
        print(f"📂 Cargando datos de {right_dir}")
        for set_file in tqdm(sorted(glob(str(right_dir / "*.set"))), desc="Right"):
            try:
                data, label, subject = load_eeg_file(set_file, label=1)
                for epoch in data:
                    X_list.append(epoch)
                    y_list.append(label)
                    subjects_list.append(subject)
            except Exception as e:
                print(f"⚠️  Error cargando {set_file}: {e}")
    
    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    
    print(f"\n✅ Dataset cargado:")
    print(f"   Shape: {X.shape}")
    print(f"   Labels: {np.bincount(y)}")
    print(f"   Sujetos: {len(set(subjects_list))}")
    
    return X, y, subjects_list


# ----------------------------
# DATASET DE PYTORCH
# ----------------------------
class EEGDataset(Dataset):
    """Dataset de PyTorch para señales EEG"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, normalize: bool = True):
        """
        Args:
            X: Array (N, channels, timepoints)
            y: Array (N,) con labels
            normalize: Si True, normaliza cada canal con z-score
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
        if normalize:
            # Normalización z-score por canal
            mean = self.X.mean(dim=2, keepdim=True)
            std = self.X.std(dim=2, keepdim=True)
            self.X = (self.X - mean) / (std + 1e-8)
        
        # DeepConvNet espera input (N, 1, channels, timepoints)
        # Agregamos dimensión de "features" (como canales en imágenes)
        self.X = self.X.unsqueeze(1)  # (N, 1, channels, timepoints)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ----------------------------
# ARQUITECTURA DEEPCONVNET
# ----------------------------
class DeepConvNet(nn.Module):
    """
    DeepConvNet para clasificación de EEG
    
    Arquitectura basada en Schirrmeister et al. (2017):
    "Deep learning with convolutional neural networks for EEG decoding and visualization"
    
    La arquitectura consiste en:
    1. Bloque temporal-espacial (captura patrones temporales y espaciales)
    2. Múltiples bloques convolucionales profundos (4 bloques)
    3. Clasificador denso
    
    Diferencias clave con CNNs estándar:
    - Primera capa: convolución 2D sobre (canales × tiempo)
    - Dropout más agresivo (0.5)
    - Activación ELU en lugar de ReLU
    - Max pooling con strides mayores
    """
    
    def __init__(self, n_channels: int = 64, n_timepoints: int = 1152, 
                 n_classes: int = 2, dropout: float = 0.5):
        super(DeepConvNet, self).__init__()
        
        # Bloque 1: Convolución temporal (sobre el tiempo)
        # Input: (batch, 1, channels, timepoints)
        self.conv1 = nn.Conv2d(1, 25, kernel_size=(1, 10), stride=1)
        
        # Bloque 2: Convolución espacial (sobre los canales)
        self.conv2 = nn.Conv2d(25, 25, kernel_size=(n_channels, 1), stride=1)
        self.bn1 = nn.BatchNorm2d(25)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        
        # Bloque 3: Convolución profunda
        self.conv3 = nn.Conv2d(25, 50, kernel_size=(1, 10), stride=1)
        self.bn2 = nn.BatchNorm2d(50)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        
        # Bloque 4: Convolución profunda
        self.conv4 = nn.Conv2d(50, 100, kernel_size=(1, 10), stride=1)
        self.bn3 = nn.BatchNorm2d(100)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        
        # Bloque 5: Convolución profunda final
        self.conv5 = nn.Conv2d(100, 200, kernel_size=(1, 10), stride=1)
        self.bn4 = nn.BatchNorm2d(200)
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        
        # Calcular tamaño después de convoluciones y pooling
        # Esto es aproximado y se ajusta dinámicamente en forward
        self.dropout = nn.Dropout(dropout)
        
        # Se calculará dinámicamente el tamaño para FC
        self.fc_input_size = None
        self.fc = None
        
    def forward(self, x):
        # x shape: (batch, 1, channels, timepoints)
        
        # Bloque 1: Temporal
        x = self.conv1(x)
        # shape: (batch, 25, channels, timepoints-9)
        
        # Bloque 2: Espacial
        x = self.conv2(x)
        # shape: (batch, 25, 1, timepoints-9)
        x = self.bn1(x)
        x = F.elu(x)  # ELU activation (mejor que ReLU para EEG)
        x = self.pool1(x)
        x = self.dropout(x)
        
        # Bloque 3
        x = self.conv3(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout(x)
        
        # Bloque 4
        x = self.conv4(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool3(x)
        x = self.dropout(x)
        
        # Bloque 5
        x = self.conv5(x)
        x = self.bn4(x)
        x = F.elu(x)
        x = self.pool4(x)
        x = self.dropout(x)
        
        # Flatten
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Crear capa FC dinámicamente si no existe
        if self.fc is None:
            self.fc_input_size = x.size(1)
            self.fc = nn.Linear(self.fc_input_size, 2).to(x.device)
        
        x = self.fc(x)
        
        return x


# ----------------------------
# ENTRENAMIENTO
# ----------------------------
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Entrena una época"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Métricas
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def validate_epoch(model, dataloader, criterion, device):
    """Valida una época"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, all_preds, all_labels


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, patience, device, output_dir):
    """Entrena el modelo con early stopping"""
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0
    patience_counter = 0
    best_model_path = output_dir / 'best_deepconvnet_model.pth'
    
    print("\n🚀 Iniciando entrenamiento de DeepConvNet...\n")
    
    for epoch in range(num_epochs):
        # Entrenar
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validar
        val_loss, val_acc, _, _ = validate_epoch(model, val_loader, criterion, device)
        
        # Guardar historial
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Imprimir progreso
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✅ Mejor modelo guardado (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"  ⏳ Paciencia: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"\n⛔ Early stopping en epoch {epoch+1}")
            break
        
        print()
    
    # Cargar mejor modelo
    model.load_state_dict(torch.load(best_model_path))
    print(f"\n✅ Mejor modelo cargado (Val Acc: {best_val_acc:.4f})")
    
    return model, history


# ----------------------------
# EVALUACIÓN
# ----------------------------
def evaluate_model(model, test_loader, device, output_dir):
    """Evalúa el modelo y genera métricas detalladas"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calcular métricas
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print("\n" + "="*50)
    print("📊 RESULTADOS FINALES - DeepConvNet")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("="*50 + "\n")
    
    # Reporte de clasificación
    print("📋 Classification Report:")
    print(classification_report(all_labels, all_preds, 
                                target_names=['Left', 'Right']))
    
    # Matriz de confusión
    cm = confusion_matrix(all_labels, all_preds)
    
    # Guardar métricas
    metrics = {
        'model': 'DeepConvNet',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / 'deepconvnet_metrics.csv', index=False)
    print(f"✅ Métricas guardadas en {output_dir / 'deepconvnet_metrics.csv'}")
    
    return metrics, cm, all_labels, all_preds


# ----------------------------
# VISUALIZACIÓN
# ----------------------------
def plot_training_history(history, output_dir):
    """Grafica la historia de entrenamiento"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('DeepConvNet - Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('DeepConvNet - Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'deepconvnet_training_history.png', dpi=150)
    plt.close()
    print(f"✅ Historia de entrenamiento guardada en {output_dir / 'deepconvnet_training_history.png'}")


def plot_confusion_matrix(cm, output_dir):
    """Grafica la matriz de confusión"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Left', 'Right'],
                yticklabels=['Left', 'Right'])
    plt.title('DeepConvNet - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / 'deepconvnet_confusion_matrix.png', dpi=150)
    plt.close()
    print(f"✅ Matriz de confusión guardada en {output_dir / 'deepconvnet_confusion_matrix.png'}")


# ----------------------------
# MAIN
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description='DeepConvNet para clasificación de EEG MI')
    parser.add_argument('--data-root', type=str, 
                       default=str(Path(__file__).resolve().parent),
                       help='Directorio con left_imag/ y right_imag/')
    parser.add_argument('--output-dir', type=str, default='deepconvnet_results',
                       help='Directorio de salida')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    args = parser.parse_args()
    
    # Crear directorios
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*50)
    print("🧠 DeepConvNet para Clasificación de EEG - Imaginación Motora")
    print("="*50)
    print(f"📂 Data root: {data_root}")
    print(f"📁 Output dir: {output_dir.resolve()}")
    print(f"🖥️  Device: {DEVICE}")
    print("="*50 + "\n")
    
    # 1. Cargar datos
    X, y, subjects = load_dataset(data_root)
    
    # 2. Split train/val/test (70/15/15)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp
    )
    
    print(f"\n📊 Splits:")
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Val:   {X_val.shape[0]} samples")
    print(f"   Test:  {X_test.shape[0]} samples\n")
    
    # 3. Crear datasets y dataloaders
    train_dataset = EEGDataset(X_train, y_train, normalize=True)
    val_dataset = EEGDataset(X_val, y_val, normalize=True)
    test_dataset = EEGDataset(X_test, y_test, normalize=True)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 4. Crear modelo DeepConvNet
    n_channels = X.shape[1]
    n_timepoints = X.shape[2]
    model = DeepConvNet(n_channels=n_channels, n_timepoints=n_timepoints, 
                        n_classes=2, dropout=args.dropout)
    model = model.to(DEVICE)
    
    print(f"🏗️  Arquitectura del modelo: DeepConvNet")
    print(f"   Canales: {n_channels}")
    print(f"   Timepoints: {n_timepoints}")
    print(f"   Dropout: {args.dropout}")
    
    # Contar parámetros (aproximado, FC se crea dinámicamente)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parámetros entrenables: ~{total_params:,}\n")
    
    # 5. Configurar entrenamiento
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 6. Entrenar
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=args.epochs, patience=PATIENCE, device=DEVICE, 
        output_dir=output_dir
    )
    
    # 7. Evaluar en test set
    metrics, cm, y_true, y_pred = evaluate_model(model, test_loader, DEVICE, output_dir)
    
    # 8. Visualizaciones
    plot_training_history(history, output_dir)
    plot_confusion_matrix(cm, output_dir)
    
    print(f"\n✅ Proceso completado. Resultados en: {output_dir.resolve()}\n")
    print("="*50)
    print("📝 NOTAS SOBRE DeepConvNet:")
    print("="*50)
    print("✓ Arquitectura específica para EEG (Schirrmeister et al., 2017)")
    print("✓ Convoluciones temporal-espaciales separadas")
    print("✓ Activación ELU (mejor que ReLU para EEG)")
    print("✓ 4 bloques convolucionales profundos")
    print("✓ Dropout 0.5 para regularización")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
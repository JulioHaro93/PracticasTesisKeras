import pandas as pd
import glob
import numpy as np
import os
import matplotlib.pyplot as plt

def createCSV():
    carp_path = 'history/5clases/k4'

    archivos_csv = glob.glob(os.path.join(carp_path, '*.csv'))

    # 3. Lee y concatena los DataFrames
    # pd.concat toma una lista de DataFrames para unirlos
    df_combinado = pd.concat([pd.read_csv(f) for f in archivos_csv], ignore_index=True)

    # 4. Guarda el DataFrame combinado en un nuevo archivo CSV
    # index=False para no escribir el índice del DataFrame en el archivo
    df_combinado.to_csv('historico_completo.csv', index=False, encoding='utf-8-sig')

    print("Archivos CSV unidos exitosamente en 'historico_completo.csv'")
    df = pd.read_csv('historico_completo.csv')
    df.head()
    return df
def grapher(history_df):
    # Crear figura con 2 subplots (2 filas, 1 columna)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # --- Subplot 1: Pérdida ---
    axes[0].plot(history_df['loss'], label='Pérdida de entrenamiento')
    if 'val_loss' in history_df.columns:
        axes[0].plot(history_df['val_loss'], label='Pérdida de validación')
    axes[0].set_xlabel('Épocas')
    axes[0].set_ylabel('Pérdida')
    axes[0].set_title('Evolución de la Pérdida')
    axes[0].legend()
    axes[0].grid(True)

    # --- Subplot 2: Precisión ---
    if 'sparse_categorical_accuracy' in history_df.columns:
        axes[1].plot(history_df['sparse_categorical_accuracy'], label='Precisión de entrenamiento')
    if 'val_sparse_categorical_accuracy' in history_df.columns:
        axes[1].plot(history_df['val_sparse_categorical_accuracy'], label='Precisión de validación')
    axes[1].set_xlabel('Épocas')
    axes[1].set_ylabel('Precisión')
    axes[1].set_title('Evolución de la Precisión')
    axes[1].legend()
    axes[1].grid(True)

    # Ajustar diseño para que no se encimen
    plt.tight_layout()
    plt.show()


def lectorCSV():
    df = pd.read_csv('historico_completo.csv')
    return df
import yfinance as yf
import pandas as pd
from pathlib import Path

def extract_stock_data(ticker, period):
    data = yf.Ticker(ticker)
    all_data = data.history(period = period)
    return all_data

def extract_all_afp_data():
    """
    Extrae datos de TODOS los fondos AFP disponibles automáticamente
    
    Returns:
    pd.DataFrame: DataFrame con los valores de todas las AFP por fondo

    ../src/preprocessing/data/raw
    """
    base_path = Path("data/raw")
    
    # Configuración de todos los fondos
    fondos_config = {
        'fondo0': {
            'file': base_path / "Fondo0-ag2025.XLS",
            'posiciones': {'habitat': (20, 1), 'integra': (21, 1), 'prima': (22, 1), 'profuturo': (23, 1)}
        },
        'fondo1': {
            'file': base_path / "Fondo1-ag2025.XLS",
            'posiciones': {'habitat': (20, 1), 'integra': (21, 1), 'prima': (22, 1), 'profuturo': (23, 1)}
        },
        'fondo2': {
            'file': base_path / "Fondo2-ag2025.XLS", 
            'posiciones': {'habitat': (19, 1), 'integra': (20, 1), 'prima': (21, 1), 'profuturo': (22, 1)}
        },
        'fondo3': {
            'file': base_path / "Fondo3-ag2025.XLS",
            'posiciones': {'habitat': (20, 1), 'integra': (21, 1), 'prima': (22, 1), 'profuturo': (23, 1)}
        }
    }
    
    # Diccionario para almacenar todos los datos
    all_data = {}
    
    for fondo_name, config in fondos_config.items():
        try:
            # Leer el archivo Excel
            df = pd.read_excel(config['file'])
            
            # Extraer datos para cada AFP en este fondo
            datos_fondo = {}
            for afp, (fila, columna) in config['posiciones'].items():
                datos_fondo[afp] = df.iloc[fila, columna]
            
            all_data[fondo_name] = datos_fondo
            print(f"{fondo_name} extraído correctamente")
            
        except Exception as e:
            print(f"Error en {fondo_name}: {e}")
            all_data[fondo_name] = None
    
    # Convertir a DataFrame (estructura correcta)
    df_result = pd.DataFrame(all_data)
    
    return df_result
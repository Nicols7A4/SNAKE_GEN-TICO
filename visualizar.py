import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def graficar_ultimos_resultados():
    # 1. Buscar la carpeta de la sesión más reciente en 'data/'
    # Esto evita que tengas que escribir el nombre de la carpeta manualmente cada vez
    try:
        lista_sesiones = glob.glob(os.path.join('data', 'session_*'))
        if not lista_sesiones:
            print("No se encontraron sesiones de entrenamiento en 'data/'.")
            return
            
        ultima_sesion = max(lista_sesiones, key=os.path.getctime)
        archivo_csv = os.path.join(ultima_sesion, 'stats.csv')
        
        print(f"Graficando datos de: {archivo_csv}")
        
        # 2. Leer el CSV con Pandas
        data = pd.read_csv(archivo_csv)
        
    except Exception as e:
        print(f"Error leyendo el archivo: {e}")
        return

    # 3. Configurar los Gráficos (2 gráficos en una ventana)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # --- GRÁFICO 1: PUNTAJE (SCORE - MANZANAS) ---
    # Eje X: Generación, Eje Y: Score
    ax1.plot(data['Generacion'], data['Score_Mejor'], label='Mejor Score (Gen)', color='blue', marker='o', markersize=3, alpha=0.6)
    ax1.plot(data['Generacion'], data['Record_Global'], label='Récord Histórico', color='red', linestyle='--', linewidth=2)
    
    ax1.set_ylabel('Manzanas Comidas')
    ax1.set_title('Evolución del Aprendizaje: Manzanas')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- GRÁFICO 2: FITNESS (APTITUD MATEMÁTICA) ---
    # Comparar qué tan "inteligente" es el mejor vs el promedio de la población
    ax2.plot(data['Generacion'], data['Fitness_Mejor'], label='Fitness del Mejor', color='green')
    ax2.plot(data['Generacion'], data['Promedio_Fitness'], label='Fitness Promedio', color='orange', linestyle='-.')
    
    ax2.set_ylabel('Fitness (Puntos)')
    ax2.set_xlabel('Generación')
    ax2.set_title('Salud de la Población (Fitness)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    graficar_ultimos_resultados()
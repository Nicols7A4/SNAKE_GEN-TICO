import random
import os
import csv
from datetime import datetime
import numpy as np
from settings import *
from snake import Serpiente
from brain import Cerebro

class Poblacion:
    def __init__(self):
        self.individuos = [Serpiente() for _ in range(POBLACION_TAMANO)]
        self.generacion = 1
        self.mejor_score_hist = 0  # Récord histórico de manzanas (Score)
        
        # --- CONFIGURACIÓN DE CARPETAS ---
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path_session = os.path.join("data", f"session_{self.timestamp}")
        self.path_checkpoints = "checkpoints"
        
        os.makedirs(self.path_session, exist_ok=True)
        os.makedirs(self.path_checkpoints, exist_ok=True)
        
        # Inicializar archivo de LOG (CSV)
        self.path_log = os.path.join(self.path_session, "stats.csv")
        self.path_session = os.path.join("data", f"session_{self.timestamp}/generaciones/")
        
        # Escribir cabeceras del CSV
        with open(self.path_log, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Generacion",           # Número de gen actual
                "ID_Mejor_Serpiente",   # Cuál serpiente fue (0-49)
                "Score_Mejor",          # Manzanas que comió la mejor
                "Record_Global",        # Récord histórico de manzanas
                "Fitness_Mejor",        # Valor matemático ((Score*500) + Pasos)
                "Promedio_Fitness"      # Salud general de la población
            ])

    def hay_vivos(self):
        for s in self.individuos:
            if s.vivo: return True
        return False

    def update_todos(self):
        for s in self.individuos:
            if s.vivo:
                s.pensar()
                s.update()

    def evolucionar(self):
        ranking_temporal = sorted(self.individuos, key=lambda s: s.calcular_fitness(), reverse=True)
        
        mejor_1 = ranking_temporal[0]
        mejor_2 = ranking_temporal[1]
        
        # Encontramos sus IDs originales (dónde estaban sentados en la clase)
        id_1 = self.individuos.index(mejor_1)
        id_2 = self.individuos.index(mejor_2)
        
        print(f"--- PODIO GEN {self.generacion} ---")
        print(f"🥇 1er Lugar: ID {id_1} | Score: {mejor_1.score} | Fit: {mejor_1.calcular_fitness():.0f}")
        print(f"🥈 2do Lugar: ID {id_2} | Score: {mejor_2.score} | Fit: {mejor_2.calcular_fitness():.0f}")
        
        # 1. IDENTIFICAR AL MEJOR (Usando Fitness estrictamente)
        # Usamos calcular_fitness() para elegir al mejor, respetando tu lógica de evolución.
        mejor_ind = max(self.individuos, key=lambda s: s.calcular_fitness())
        
        # Capturar datos estadísticos
        id_mejor = self.individuos.index(mejor_ind) # Guardar el ID antes de ordenar
        score_mejor = mejor_ind.score
        fitness_mejor = mejor_ind.calcular_fitness()
        
        # Calcular promedio de la población
        promedio_fitness = sum(s.calcular_fitness() for s in self.individuos) / len(self.individuos)
        
        # Actualizar Récord Histórico (Basado en Score visible)
        if score_mejor > self.mejor_score_hist:
            self.mejor_score_hist = score_mejor

        print(f"Gen {self.generacion} | ID: {id_mejor} | Score: {score_mejor} | Récord: {self.mejor_score_hist}")

        # --- 2. GUARDADO DE DATOS ---
        self.guardar_datos(mejor_ind, id_mejor, score_mejor, fitness_mejor, promedio_fitness)

        # 3. SELECCIÓN (Ordenamos la lista para el cruce)
        self.individuos.sort(key=lambda x: x.calcular_fitness(), reverse=True)

        # 4. REPRODUCCIÓN (Elitismo + Cruce)
        nueva_gen = []
        
        # Elitismo: Los 2 mejores pasan intactos a la siguiente ronda
        nueva_gen.append(Serpiente(self.individuos[0].cerebro)) 
        nueva_gen.append(Serpiente(self.individuos[1].cerebro)) 

        # Cruce de los mejores (Top 50%)
        pool_padres = self.individuos[:POBLACION_TAMANO // 2]

        while len(nueva_gen) < POBLACION_TAMANO:
            padre_a = random.choice(pool_padres)
            padre_b = random.choice(pool_padres)
            
            hijo_cerebro = self.cruce(padre_a.cerebro, padre_b.cerebro)
            self.mutacion(hijo_cerebro)
            
            nueva_gen.append(Serpiente(hijo_cerebro))

        self.individuos = nueva_gen
        self.generacion += 1

    def guardar_datos(self, mejor_serpiente, id_mejor, score_mejor, fitness_mejor, avg_fit):
        """Gestiona logs CSV, Checkpoints y Data Cruda"""
        
        # A. Escribir en CSV
        with open(self.path_log, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                self.generacion,
                id_mejor,
                score_mejor,
                self.mejor_score_hist,
                f"{fitness_mejor:.2f}",
                f"{avg_fit:.2f}"
            ])

        # B. Guardar Checkpoint (Solo si iguala/supera récord o cada 10 gens)
        if score_mejor >= self.mejor_score_hist or self.generacion % 10 == 0:
            nombre = f"best_gen_{self.timestamp}_{self.generacion}_id_{id_mejor}_score_{score_mejor}.txt"
            ruta = os.path.join(self.path_checkpoints, nombre)
            mejor_serpiente.cerebro.guardar(ruta)

        # C. Guardar TODOS los cromosomas (Data cruda en TXT)
        path_gen = os.path.join(self.path_session, f"gen_{self.generacion}")
        os.makedirs(path_gen, exist_ok=True)
        
        for i, ind in enumerate(self.individuos):
            ind.cerebro.guardar(os.path.join(path_gen, f"snake_{i}.txt"))

    def cruce(self, cerebro_a, cerebro_b):
        genes_a = cerebro_a.genes
        genes_b = cerebro_b.genes
        filas, cols = genes_a.shape
        hijo_genes = np.zeros((filas, cols))
        
        for i in range(filas):
            for j in range(cols):
                if random.random() > 0.5:
                    hijo_genes[i][j] = genes_a[i][j]
                else:
                    hijo_genes[i][j] = genes_b[i][j]
        return Cerebro(cerebro_a.n_inputs, cerebro_a.n_outputs, hijo_genes)

    def mutacion(self, cerebro):
        filas, cols = cerebro.genes.shape
        for i in range(filas):
            for j in range(cols):
                if random.random() < TASA_MUTACION:
                    cerebro.genes[i][j] += np.random.normal(0, 0.5)
                    cerebro.genes[i][j] = np.clip(cerebro.genes[i][j], -1, 1)
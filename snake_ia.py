import pygame
import numpy as np
import random

# --- CONFIGURACIÓN ---
ANCHO_VENTANA = 600
ALTO_VENTANA = 600
TAM_CELDA = 20
FPS = 1000  # Rápido para entrenar, bájalo a 30 para verlos jugar lento

POBLACION_TAMANO = 50
TASA_MUTACION = 0.05
TIEMPO_VIDA_INICIAL = 100  # Pasos antes de morir de hambre si no come

# Colores
NEGRO = (0, 0, 0)
BLANCO = (255, 255, 255)
ROJO = (255, 0, 0)
VERDE = (0, 255, 0)
GRIS = (50, 50, 50)

# Direcciones
ARRIBA = (0, -1)
ABAJO = (0, 1)
IZQUIERDA = (-1, 0)
DERECHA = (1, 0)

class Serpiente:
    def __init__(self, cerebro=None):
        # Posición inicial al centro
        self.cuerpo = [(10, 10), (10, 11), (10, 12)]
        self.direccion = ARRIBA
        self.vivo = True
        self.hambre = TIEMPO_VIDA_INICIAL
        self.score = 0
        self.pasos = 0
        self.color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        
        # Generar comida aleatoria
        self.comida = self.nueva_comida()

        # --- CEREBRO (GENES) ---
        # INPUTS (6): [ComidaX, ComidaY, ObstaculoArriba, ObstaculoAbajo, ObstaculoIzq, ObstaculoDer]
        # OUTPUTS (4): [Arriba, Abajo, Izq, Der]
        self.n_inputs = 6
        self.n_outputs = 4
        
        if cerebro is None:
            # Inicialización aleatoria [cite: 108]
            self.genes = np.random.uniform(-1, 1, (self.n_inputs, self.n_outputs))
        else:
            self.genes = cerebro
            
    def nueva_comida(self):
        while True:
            x = random.randint(0, (ANCHO_VENTANA // TAM_CELDA) - 1)
            y = random.randint(0, (ALTO_VENTANA // TAM_CELDA) - 1)
            if (x, y) not in self.cuerpo:
                return (x, y)

    def pensar(self):
        # 1. Obtener Inputs (Sensores)
        cabeza_x, cabeza_y = self.cuerpo[0]
        comida_x, comida_y = self.comida
        
        # Normalizamos inputs entre -1 y 1 para facilitar el cálculo
        input_comida_x = (comida_x - cabeza_x) / (ANCHO_VENTANA // TAM_CELDA)
        input_comida_y = (comida_y - cabeza_y) / (ALTO_VENTANA // TAM_CELDA)
        
        # Sensores de obstaculos (1 si hay peligro, 0 si no)
        obs_arriba = 1 if self.verificar_colision((cabeza_x, cabeza_y - 1)) else 0
        obs_abajo  = 1 if self.verificar_colision((cabeza_x, cabeza_y + 1)) else 0
        obs_izq    = 1 if self.verificar_colision((cabeza_x - 1, cabeza_y)) else 0
        obs_der    = 1 if self.verificar_colision((cabeza_x + 1, cabeza_y)) else 0
        
        vision = np.array([input_comida_x, input_comida_y, obs_arriba, obs_abajo, obs_izq, obs_der])
        
        # 2. Proceso Neuronal: Inputs * Genes
        decision = np.dot(vision, self.genes)
        
        # 3. Tomar decisión (el valor más alto gana)
        idx_max = np.argmax(decision)
        
        nuevas_dirs = [ARRIBA, ABAJO, IZQUIERDA, DERECHA]
        nueva_dir = nuevas_dirs[idx_max]
        
        # Evitar giro de 180 grados (suicidio inmediato)
        if (nueva_dir[0] * -1, nueva_dir[1] * -1) != self.direccion:
            self.direccion = nueva_dir

    def verificar_colision(self, punto):
        x, y = punto
        # Chocar paredes
        if x < 0 or x >= ANCHO_VENTANA // TAM_CELDA or y < 0 or y >= ALTO_VENTANA // TAM_CELDA:
            return True
        # Chocar consigo mismo (excluyendo la cola que se moverá)
        if punto in self.cuerpo[:-1]:
            return True
        return False

    def update(self):
        if not self.vivo:
            return

        self.hambre -= 1
        if self.hambre <= 0:
            self.vivo = False
            return

        cabeza_x, cabeza_y = self.cuerpo[0]
        dir_x, dir_y = self.direccion
        nueva_cabeza = (cabeza_x + dir_x, cabeza_y + dir_y)

        # Verificar muerte
        if self.verificar_colision(nueva_cabeza):
            self.vivo = False
        else:
            # Moverse
            self.cuerpo.insert(0, nueva_cabeza)
            
            # ¿Comió?
            if nueva_cabeza == self.comida:
                self.score += 1
                self.hambre += 100 # Gana energía
                self.comida = self.nueva_comida()
            else:
                self.cuerpo.pop() # Eliminar cola si no comió
            
            self.pasos += 1

    # Definición de función Fitness según PDF [cite: 227]
    def calcular_fitness(self):
        # Priorizamos mucho comer manzanas, y un poco sobrevivir
        return (self.score * 500) + self.pasos

def cruce(padre_a, padre_b):
    # Cruce Uniforme [cite: 353]
    hijo_genes = np.zeros(padre_a.genes.shape)
    filas, cols = padre_a.genes.shape
    for i in range(filas):
        for j in range(cols):
            if random.random() > 0.5:
                hijo_genes[i][j] = padre_a.genes[i][j]
            else:
                hijo_genes[i][j] = padre_b.genes[i][j]
    return Serpiente(hijo_genes)

def mutacion(agente):
    # Mutación (ajuste aleatorio de pesos) [cite: 366]
    filas, cols = agente.genes.shape
    for i in range(filas):
        for j in range(cols):
            if random.random() < TASA_MUTACION:
                agente.genes[i][j] += np.random.normal(0, 0.5)

def main():
    pygame.init()
    pantalla = pygame.display.set_mode((ANCHO_VENTANA, ALTO_VENTANA))
    pygame.display.set_caption("Snake AI - Algoritmos Genéticos")
    reloj = pygame.time.Clock()
    fuente = pygame.font.SysFont("Arial", 20)

    # 1. Inicialización de Población [cite: 108]
    poblacion = [Serpiente() for _ in range(POBLACION_TAMANO)]
    generacion = 1
    mejor_record_global = 0
    fps_actual = FPS  # Variable local para controlar velocidad

    corriendo = True
    while corriendo:
        pantalla.fill(NEGRO)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False
            # Control de velocidad con teclas
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: fps_actual = 1000
                if event.key == pygame.K_DOWN: fps_actual = 10

        todos_muertos = True
        
        # --- CICLO DE VIDA ---
        for s in poblacion:
            if s.vivo:
                todos_muertos = False
                s.pensar()
                s.update()
                
                # Dibujar cuerpo
                for i, parte in enumerate(s.cuerpo):
                    rect = (parte[0] * TAM_CELDA, parte[1] * TAM_CELDA, TAM_CELDA, TAM_CELDA)
                    # Dibujamos solo la cabeza diferente o un color tenue
                    color = s.color if i == 0 else (50, 50, 50)
                    pygame.draw.rect(pantalla, color, rect)
                
                # Dibujar comida
                rect_comida = (s.comida[0]*TAM_CELDA, s.comida[1]*TAM_CELDA, TAM_CELDA, TAM_CELDA)
                pygame.draw.rect(pantalla, ROJO, rect_comida)

        # --- EVOLUCIÓN ---
        if todos_muertos:
            # Evaluar Aptitud [cite: 109]
            poblacion.sort(key=lambda x: x.calcular_fitness(), reverse=True)
            mejor_actual = poblacion[0].score
            mejor_record_global = max(mejor_record_global, mejor_actual)
            
            print(f"Gen {generacion} | Mejor Score: {mejor_actual} | Récord: {mejor_record_global}")

            nueva_gen = []
            
            # Elitismo: Pasar los 2 mejores directamente
            nueva_gen.append(Serpiente(poblacion[0].genes))
            nueva_gen.append(Serpiente(poblacion[1].genes))
            
            # Selección y Cruce [cite: 112, 113]
            # Seleccionamos padres del top 50%
            pool_padres = poblacion[:POBLACION_TAMANO // 2]
            
            while len(nueva_gen) < POBLACION_TAMANO:
                padre_a = random.choice(pool_padres)
                padre_b = random.choice(pool_padres)
                hijo = cruce(padre_a, padre_b)
                mutacion(hijo) # [cite: 114]
                nueva_gen.append(hijo)
            
            poblacion = nueva_gen
            generacion += 1

        # Info en pantalla
        texto = fuente.render(f"Gen: {generacion} | Vivos: {sum(s.vivo for s in poblacion)}", True, BLANCO)
        pantalla.blit(texto, (10, 10))

        pygame.display.flip()
        reloj.tick(fps_actual) # Control de velocidad

    pygame.quit()

if __name__ == "__main__":
    main()
import pygame
import numpy as np
import random

# --- CONFIGURACIÓN ---
ANCHO = 800
ALTO = 400
FPS = 60
POBLACION_TAMANO = 20  # [cite: 57] Debe existir una población
TASA_MUTACION = 0.05   # Probabilidad de que un gen cambie

# Colores
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
ROJO = (255, 0, 0)
VERDE = (0, 255, 0)

class Agente(pygame.sprite.Sprite):
    def __init__(self, cerebro=None):
        super().__init__()
        # --- APARIENCIA (MARIO) ---
        self.image = pygame.Surface((30, 30))
        self.image.fill(ROJO)
        self.rect = self.image.get_rect()
        self.rect.center = (50, ALTO - 50)
        
        # --- FÍSICA BÁSICA ---
        self.vel_y = 0
        self.en_suelo = True
        self.vivo = True
        
        # --- CEREBRO (GENES) ---
        # El cromosoma es una serie de números (pesos) [cite: 177]
        # Inputs (Sensores): [Dist_Hueco, Dist_Enemigo, Dist_Plataforma, Altura_Propia]
        # Outputs (Acciones): [Mover_Izq, Mover_Der, Saltar]
        self.n_inputs = 4
        self.n_outputs = 3
        
        if cerebro is None:
            # Inicialización aleatoria de la población [cite: 108]
            # Genes aleatorios entre -1 y 1
            self.genes = np.random.uniform(-1, 1, (self.n_inputs, self.n_outputs))
        else:
            self.genes = cerebro

        # --- FITNESS ---
        # Aptitud: Qué tan lejos llegó [cite: 230]
        self.fitness = 0
        self.distancia_recorrida = 0

    def pensar(self, entorno_inputs):
        """
        El agente percibe su entorno y decide qué botón presionar.
        entorno_inputs: Lista de distancias [d_hueco, d_enemigo, d_plat, y_propia]
        """
        if not self.vivo:
            return

        # Proceso neuronal simple: Inputs * Genes = Decisión
        decision = np.dot(entorno_inputs, self.genes) 
        # Aplicamos función de activación (tanh para tener salida entre -1 y 1)
        decision = np.tanh(decision)

        # Interpretación de la salida (Umbral > 0.5 activa la tecla)
        mover_izq = decision[0] > 0.5
        mover_der = decision[1] > 0.5
        saltar = decision[2] > 0.5

        self.actuar(mover_izq, mover_der, saltar)

    def actuar(self, izq, der, salto):
        if izq:
            self.rect.x -= 5
        if der:
            self.rect.x += 5
            self.distancia_recorrida += 1 # Recompensa por avanzar
            
        if salto and self.en_suelo:
            self.vel_y = -15
            self.en_suelo = False

    def update(self):
        # Física simple de gravedad
        self.vel_y += 1
        self.rect.y += self.vel_y
        
        # Colisión con el suelo (simple por ahora)
        if self.rect.bottom >= ALTO:
            self.rect.bottom = ALTO
            self.vel_y = 0
            self.en_suelo = True
            
        # Actualizar Fitness
        # Se basa en "bueno" (distancia) [cite: 233]
        self.fitness = self.distancia_recorrida

def cruce(padre_a, padre_b):
    """
    Operador de Cruce (Crossover) [cite: 245]
    Genera un hijo combinando genes de dos padres.
    Implementación: Cruce Uniforme [cite: 353]
    """
    genes_a = padre_a.genes
    genes_b = padre_b.genes
    
    # Crear matriz vacía para el hijo
    genes_hijo = np.zeros(genes_a.shape)
    
    # Recorrer cada gen y elegir al azar de cuál padre heredarlo
    filas, cols = genes_a.shape
    for i in range(filas):
        for j in range(cols):
            if random.random() > 0.5:
                genes_hijo[i][j] = genes_a[i][j]
            else:
                genes_hijo[i][j] = genes_b[i][j]
                
    return Agente(genes_hijo)

def mutacion(agente):
    """
    Operador de Mutación [cite: 366]
    Cambio aleatorio en los genes para mantener diversidad.
    Tipo: Similar a Flip Bit pero para números reales (ajuste gaussiano) [cite: 374]
    """
    filas, cols = agente.genes.shape
    for i in range(filas):
        for j in range(cols):
            if random.random() < TASA_MUTACION:
                # Sumar un valor pequeño aleatorio (ruido)
                agente.genes[i][j] += np.random.normal(0, 0.2)
                # Limitar valores entre -1 y 1
                agente.genes[i][j] = np.clip(agente.genes[i][j], -1, 1)

def main():
    pygame.init()
    pantalla = pygame.display.set_mode((ANCHO, ALTO))
    reloj = pygame.time.Clock()
    
    # 1. Inicialización de la población [cite: 108]
    poblacion = pygame.sprite.Group()
    marios = [Agente() for _ in range(POBLACION_TAMANO)]
    for m in marios:
        poblacion.add(m)

    generacion = 1
    corriendo = True

    while corriendo:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False

        pantalla.fill(NEGRO) # Limpiar pantalla

        # --- LÓGICA DE JUEGO ---
        todos_muertos = True
        
        for mario in marios:
            if mario.vivo:
                todos_muertos = False
                
                # SIMULACIÓN DE SENSORES (AQUÍ DEBES CONECTAR CON TU TERRENO REAL)
                # Por ahora simulamos valores aleatorios para probar que piensan
                dist_hueco = random.uniform(0, 1)     # Normalizado
                dist_enemigo = random.uniform(0, 1)
                dist_plat = random.uniform(0, 1)
                altura = mario.rect.y / ALTO
                
                inputs = [dist_hueco, dist_enemigo, dist_plat, altura]
                
                mario.pensar(inputs)
                mario.update()
                
                # Matar si sale de pantalla (ejemplo de muerte)
                if mario.rect.x > ANCHO or mario.rect.x < 0: # Limites simples
                    mario.vivo = False 

        poblacion.draw(pantalla)

        # --- EVOLUCIÓN (CUANDO TODOS MUEREN O ACABA EL TIEMPO) ---
        if todos_muertos:
            print(f"Generación {generacion} terminada.")
            
            # 2. Selección: Ordenar por Fitness (Mejores primero) [cite: 112]
            marios.sort(key=lambda x: x.fitness, reverse=True)
            
            # Elitismo: Nos quedamos con los 2 mejores sin cambios
            nueva_generacion = [marios[0], marios[1]]
            
            # Reproducción para llenar el resto
            while len(nueva_generacion) < POBLACION_TAMANO:
                padre_a = random.choice(marios[:10]) # Elegir entre los top 10
                padre_b = random.choice(marios[:10])
                
                hijo = cruce(padre_a, padre_b) # [cite: 113]
                mutacion(hijo)                 # [cite: 114]
                nueva_generacion.append(hijo)
            
            marios = nueva_generacion
            poblacion.empty()
            for m in marios:
                # Reiniciar posiciones para nueva ronda
                m.rect.center = (50, ALTO - 50) 
                m.distancia_recorrida = 0
                m.vivo = True
                poblacion.add(m)
                
            generacion += 1

        pygame.display.flip()
        reloj.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
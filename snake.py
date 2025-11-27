import pygame
import random
import numpy as np
from settings import *
from brain import Cerebro

# Direcciones
ARRIBA = (0, -1)
ABAJO = (0, 1)
IZQUIERDA = (-1, 0)
DERECHA = (1, 0)

class Serpiente:
    def __init__(self, cerebro=None):
        self.cuerpo = [(10, 10), (10, 11), (10, 12)]
        self.direccion = ARRIBA
        self.vivo = True
        self.hambre = TIEMPO_VIDA_INICIAL
        self.score = 0
        self.pasos = 0
        self.color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        self.comida = self.nueva_comida()

        # Inputs: [ComidaX, ComidaY, ObsArriba, ObsAbajo, ObsIzq, ObsDer]
        # Outputs: [Arriba, Abajo, Izq, Der]
        if cerebro:
            self.cerebro = cerebro
        else:
            self.cerebro = Cerebro(6, 4)

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
        
        # Normalizamos coordenadas
        input_comida_x = (comida_x - cabeza_x) / (ANCHO_VENTANA // TAM_CELDA)
        input_comida_y = (comida_y - cabeza_y) / (ALTO_VENTANA // TAM_CELDA)
        
        # Sensores de obstáculos
        obs_arriba = 1 if self.verificar_colision((cabeza_x, cabeza_y - 1)) else 0
        obs_abajo  = 1 if self.verificar_colision((cabeza_x, cabeza_y + 1)) else 0
        obs_izq    = 1 if self.verificar_colision((cabeza_x - 1, cabeza_y)) else 0
        obs_der    = 1 if self.verificar_colision((cabeza_x + 1, cabeza_y)) else 0
        
        vision = np.array([input_comida_x, input_comida_y, obs_arriba, obs_abajo, obs_izq, obs_der])
        
        # 2. Consultar al cerebro
        decision = self.cerebro.predecir(vision)
        
        # 3. Interpretar decisión
        idx_max = np.argmax(decision)
        nuevas_dirs = [ARRIBA, ABAJO, IZQUIERDA, DERECHA]
        nueva_dir = nuevas_dirs[idx_max]
        
        # Evitar giro de 180 grados (suicidio)
        if (nueva_dir[0] * -1, nueva_dir[1] * -1) != self.direccion:
            self.direccion = nueva_dir

    def verificar_colision(self, punto):
        x, y = punto
        if x < 0 or x >= ANCHO_VENTANA // TAM_CELDA or y < 0 or y >= ALTO_VENTANA // TAM_CELDA:
            return True
        if punto in self.cuerpo[:-1]:
            return True
        return False

    def update(self):
        if not self.vivo: return

        self.hambre -= 1
        if self.hambre <= 0:
            self.vivo = False
            return

        cabeza_x, cabeza_y = self.cuerpo[0]
        dx, dy = self.direccion
        nueva_cabeza = (cabeza_x + dx, cabeza_y + dy)

        if self.verificar_colision(nueva_cabeza):
            self.vivo = False
        else:
            self.cuerpo.insert(0, nueva_cabeza)
            if nueva_cabeza == self.comida:
                self.score += 1
                self.hambre += 100
                self.comida = self.nueva_comida()
            else:
                self.cuerpo.pop()
            self.pasos += 1

    def calcular_fitness(self):
        # Función de evaluación: maximizar comida y tiempo [cite: 233]
        return (self.score * 500) + self.pasos
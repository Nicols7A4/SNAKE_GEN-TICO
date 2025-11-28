# 🐍 Snake AI - Algoritmos Genéticos

Un proyecto de **Inteligencia Artificial** que entrena serpientes para jugar Snake de forma autónoma usando **Algoritmos Genéticos** y redes neuronales simples.

---

## 🎯 ¿De qué va el proyecto?

Este sistema utiliza **evolución artificial** para entrenar agentes (serpientes) que aprenden a:
- ✅ Moverse hacia la comida
- ✅ Evitar paredes y su propio cuerpo
- ✅ Maximizar su puntaje (manzanas comidas)

**Sin programar reglas explícitas**, las serpientes evolucionan generación tras generación mediante:
- **Selección natural** (supervivencia del más apto)
- **Cruce genético** (heredar genes de los mejores)
- **Mutación aleatoria** (exploración de nuevas estrategias)

---

## 📊 Parámetros Principales

Definidos en `settings.py`:

```python
# Dimensiones del juego
ANCHO_VENTANA = 600        # Píxeles
ALTO_VENTANA = 600         
TAM_CELDA = 20             # Grid de 30x30

# Algoritmo Genético
POBLACION_TAMANO = 50      # Serpientes por generación
TASA_MUTACION = 0.05       # 5% probabilidad de mutación por peso
TIEMPO_VIDA_INICIAL = 100  # Pasos máximos sin comer (evita bucles infinitos)

# Velocidad
FPS_ENTRENAMIENTO = 1000   # Modo rápido para entrenar
FPS_VER = 30               # Modo lento para observar, puede modificarse
```

---

## 🧠 Arquitectura del Cerebro

Cada serpiente tiene una **red neuronal simple** (perceptrón) que toma decisiones:

### Inputs (6 sensores):
```python
vision = [
    input_comida_x,  # Dirección X hacia comida (normalizado -1 a 1)
    input_comida_y,  # Dirección Y hacia comida (normalizado -1 a 1)
    obs_arriba,      # 1 si hay peligro arriba, 0 si está libre
    obs_abajo,       # 1 si hay peligro abajo, 0 si está libre
    obs_izquierda,   # 1 si hay peligro a la izquierda
    obs_derecha      # 1 si hay peligro a la derecha
]
```

### Outputs (4 acciones):
```python
decision = np.dot(vision, genes)  # Multiplicación matricial
accion = argmax(decision)         # Elige la dirección con mayor valor

# Mapeo:
# 0 → ARRIBA
# 1 → ABAJO
# 2 → IZQUIERDA
# 3 → DERECHA
```

### Matriz de pesos (genes):
```python
genes = np.random.uniform(-1, 1, (6, 4))  # 24 valores aleatorios
```

---

## 📈 Métricas de Evaluación

### Función de Fitness
Definida en `snake.py`:

```python
def calcular_fitness(self):
    return (self.score * 500) + self.pasos
```

**Interpretación:**
- `self.score`: Manzanas comidas (recompensa principal)
- `self.pasos`: Tiempo sobrevivido (recompensa secundaria)
- **Multiplicador 500**: Prioriza fuertemente comer manzanas sobre solo moverse

**Ejemplo:**
- Serpiente A: 10 manzanas, 200 pasos → `(10*500) + 200 = 5200` fitness
- Serpiente B: 8 manzanas, 400 pasos → `(8*500) + 400 = 4400` fitness
- **Ganadora: A** (aunque B sobrevivió más)

### Métricas Registradas (CSV)
En cada generación se guarda en `data/session_XXXXXX/stats.csv`:

| Columna | Descripción |
|---------|-------------|
| `Generacion` | Número de generación actual |
| `ID_Mejor_Serpiente` | Índice (0-49) de la mejor serpiente |
| `Score_Mejor` | Manzanas comidas por el mejor |
| `Record_Global` | Récord histórico de manzanas |
| `Fitness_Mejor` | Valor de fitness del mejor |
| `Promedio_Fitness` | Salud promedio de la población |

---

## 🧬 Algoritmos Genéticos

### 1. Inicialización (Generación 0)
```python
# En ga.py
def __init__(self):
    self.individuos = [Serpiente() for _ in range(POBLACION_TAMANO)]
```
- Crea 50 serpientes con genes **completamente aleatorios**
- Cada una tiene pesos entre `-1` y `1`

### 2. Evaluación
```python
# Se juega el juego completo con cada serpiente
for serpiente in poblacion:
    while serpiente.vivo:
        serpiente.pensar()  # Decidir movimiento con genes
        serpiente.update()  # Moverse y actualizar estado
```

### 3. Selección (Supervivencia del más apto)
```python
# Ordenar por fitness (de mayor a menor)
self.individuos.sort(key=lambda x: x.calcular_fitness(), reverse=True)

# Solo los mejores 50% se reproducen
pool_padres = self.individuos[:POBLACION_TAMANO // 2]
```

### 4. Cruce (Reproducción)
**Cruce Uniforme** - Cada gen tiene 50% de heredarse de cada padre:

```python
def cruce(self, cerebro_a, cerebro_b):
    genes_a = cerebro_a.genes  # Matriz 6x4 del padre A
    genes_b = cerebro_b.genes  # Matriz 6x4 del padre B
    hijo_genes = np.zeros((6, 4))
    
    for i in range(6):     # Por cada input
        for j in range(4):  # Por cada output
            if random.random() > 0.5:
                hijo_genes[i][j] = genes_a[i][j]  # Gen del padre A
            else:
                hijo_genes[i][j] = genes_b[i][j]  # Gen del padre B
    
    return Cerebro(6, 4, hijo_genes)
```

**Ejemplo visual:**
```
Padre A: [0.5, -0.3, 0.8, ...]
Padre B: [0.2,  0.9, -0.5, ...]
         ↓     ↓     ↓
Hijo:   [0.5,  0.9, 0.8, ...]  (combinación aleatoria)
```

### 5. Mutación (Exploración)
```python
def mutacion(self, cerebro):
    for i in range(6):
        for j in range(4):
            if random.random() < TASA_MUTACION:  # 5% probabilidad
                # Añadir ruido gaussiano (media=0, std=0.5)
                cerebro.genes[i][j] += np.random.normal(0, 0.5)
                # Mantener valores entre -1 y 1
                cerebro.genes[i][j] = np.clip(cerebro.genes[i][j], -1, 1)
```

**Ejemplo:**
```
Antes:  0.75
        ↓ (mutación con ruido +0.12)
Después: 0.87
```

### 6. Elitismo
```python
# Los 2 mejores pasan INTACTOS a la siguiente generación
nueva_gen.append(Serpiente(self.individuos[0].cerebro))  # Mejor
nueva_gen.append(Serpiente(self.individuos[1].cerebro))  # Segundo mejor
```

Esto garantiza que nunca perdamos las mejores soluciones encontradas.

### 7. Ciclo Completo
```python
while len(nueva_gen) < POBLACION_TAMANO:
    padre_a = random.choice(pool_padres)  # Elegir padre del top 50%
    padre_b = random.choice(pool_padres)  # Elegir otro padre
    
    hijo_cerebro = self.cruce(padre_a.cerebro, padre_b.cerebro)
    self.mutacion(hijo_cerebro)
    
    nueva_gen.append(Serpiente(hijo_cerebro))

self.individuos = nueva_gen  # Reemplazar población vieja
self.generacion += 1
```

---

## 🔄 Funcionamiento de las Generaciones

### Diagrama del Proceso

```
Generación N
    │
    ├─► [Jugar] → Todas las serpientes juegan simultáneamente
    │              hasta morir (chocar o hambre)
    │
    ├─► [Evaluar] → Calcular fitness de cada serpiente
    │
    ├─► [Seleccionar] → Ordenar por fitness (mejores primero)
    │
    ├─► [Reproducir]
    │     ├─ Elitismo: Copiar 2 mejores
    │     └─ Cruce + Mutación: Crear 48 hijos del top 50%
    │
    └─► Generación N+1 (nueva población)
```

### Ejemplo de Progreso Real

| Gen | Mejor Score | Récord | Promedio Fitness | Observación |
|-----|-------------|--------|------------------|-------------|
| 1   | 3           | 3      | 450              | Movimientos aleatorios |
| 10  | 8           | 8      | 1200             | Empieza a buscar comida |
| 50  | 25          | 25     | 3800             | Evita paredes básicamente |
| 100 | 45          | 45     | 8200             | Estrategias complejas |
| 200 | 80          | 80     | 15000            | Jugador experto |

---

## 🚀 Instalación y Uso

### 1. Instalar Dependencias
```bash
pip install -r requirements.txt
```

**Librerías necesarias:**
- `pygame` - Motor gráfico del juego
- `numpy` - Operaciones matriciales (redes neuronales)
- `pandas` - Análisis de datos CSV
- `matplotlib` - Gráficos de evolución

### 2. Entrenar el Modelo
```bash
python main.py
```

**Configura el modo en `main.py`:**
```python
# Línea 11
ARCHIVO_REPLAY = ""  # Vacío = ENTRENAMIENTO
```

**Controles durante el entrenamiento:**
- `↑` (Flecha Arriba): Acelerar a 1000 FPS
- `↓` (Flecha Abajo): Ralentizar a 10 FPS

### 3. Ver un Modelo Entrenado (Replay)
```python
# En main.py, línea 11
ARCHIVO_REPLAY = "checkpoints/best_gen_334_id_17_score_125.txt"
```

Luego ejecuta:
```bash
python main.py
```

La serpiente jugará con ese cerebro guardado. Si muere, se reinicia automáticamente.

### 4. Visualizar Resultados
```bash
python visualizar.py
```

**Genera gráficos de:**
- Evolución del Score (manzanas comidas)
- Récord histórico
- Fitness del mejor vs promedio poblacional

---

## 📁 Estructura de Archivos

```
MARIO_G/
│
├── main.py              # Punto de entrada (entrenamiento/replay)
├── snake.py             # Clase Serpiente (lógica del juego)
├── brain.py             # Clase Cerebro (red neuronal)
├── ga.py                # Clase Poblacion (algoritmo genético)
├── settings.py          # Parámetros configurables
├── visualizar.py        # Gráficos de resultados
├── requirements.txt     # Dependencias
│
├── checkpoints/         # Mejores cerebros guardados (.txt)
│   └── best_gen_X_id_Y_score_Z.txt
│
└── data/                # Sesiones de entrenamiento
    └── session_YYYYMMDD_HHMMSS/
        ├── stats.csv    # Métricas por generación
        └── gen_X/       # Todos los cerebros de esa generación
            ├── snake_0.txt
            ├── snake_1.txt
            └── ...
```

---

## 🎓 Conceptos Clave

### ¿Por qué funciona?
1. **Variabilidad**: Mutaciones crean diversidad genética
2. **Selección**: Solo los mejores genes se reproducen
3. **Tiempo**: Después de cientos de generaciones, emergen estrategias complejas

### Limitaciones
- Sin memoria a largo plazo (solo ve el estado actual)
- Puede quedar atrapado en óptimos locales
- Requiere muchas generaciones para converger

### Mejoras Posibles
- [ ] Agregar más sensores (distancia a paredes, ver el cuerpo completo)
- [ ] Redes neuronales profundas (capas ocultas)
- [ ] Algoritmos más avanzados (NEAT, Deep Q-Learning)
- [ ] Guardar mejores estrategias en una "Hall of Fame"

---

## 📝 Referencias

Este proyecto implementa los conceptos de:
- **Algoritmos Genéticos** (Holland, 1975)
- **Perceptrón** (Rosenblatt, 1958)
- **Selección por Torneo** y **Elitismo**
- **Cruce Uniforme** y **Mutación Gaussiana**

---

## 👨‍💻 Autor

Proyecto educativo para aprender sobre:
- Inteligencia Artificial
- Algoritmos Evolutivos
- Pygame y visualización de datos

---

## 🎮 ¡Pruébalo!

```bash
# Paso 1: Instalar
pip install -r requirements.txt

# Paso 2: Entrenar (déjalo correr 30 minutos)
python main.py

# Paso 3: Ver resultados
python visualizar.py
```

**¡Observa cómo las serpientes aprenden de la nada!** 🐍🧬🎯

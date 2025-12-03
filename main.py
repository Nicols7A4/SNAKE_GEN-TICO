import pygame
from settings import *
from ga import Poblacion
from snake import Serpiente
from brain import Cerebro

# --- CONFIGURACIÓN DE MODO ---
# Si está vacío "", entrena normal.
# Si pones un archivo "checkpoints/best_gen_50.txt", JUEGA SOLO con ese cerebro.
ARCHIVO_REPLAY = "" 
# ARCHIVO_REPLAY = "checkpoints/best_gen_334_id_17_score_125.txt" 
# ARCHIVO_REPLAY = "checkpoints/best_gen_350_id_30_score_101.txt" 
ARCHIVO_REPLAY = "checkpoints/best_gen_20251202_200859_430_id_25_score_136.txt" 
ARCHIVO_REPLAY = "checkpoints/best_gen_20251202_200859_360_id_23_score_130.txt" 
ARCHIVO_REPLAY = "checkpoints/best_gen_20251202_200859_440_id_41_score_124.txt" 

def main():
    pygame.init()
    pantalla = pygame.display.set_mode((ANCHO_VENTANA, ALTO_VENTANA))
    pygame.display.set_caption("Snake AI - Training & Replay")
    reloj = pygame.time.Clock()
    fuente = pygame.font.SysFont("Arial", 20)

    # Lógica de Selección de Modo
    if ARCHIVO_REPLAY:
        print(f"--- MODO REPLAY ACTIVADO: Cargando {ARCHIVO_REPLAY} ---")
        cerebro_cargado = Cerebro.cargar(ARCHIVO_REPLAY)
        # En modo replay, solo creamos UNA serpiente con ese cerebro
        poblacion = [Serpiente(cerebro_cargado)]
        es_entrenamiento = False
    else:
        print("--- MODO ENTRENAMIENTO: Iniciando nueva población ---")
        ga_controller = Poblacion() # El controlador genético
        es_entrenamiento = True

    fps_actual = FPS_ENTRENAMIENTO if es_entrenamiento else 240 # Lento para ver replay

    corriendo = True
    while corriendo:
        pantalla.fill(NEGRO)
        
        # Eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: fps_actual = 1000 
                if event.key == pygame.K_DOWN: fps_actual = 10  
        
        if es_entrenamiento:
            # --- LÓGICA GA ---
            if ga_controller.hay_vivos():
                ga_controller.update_todos()
                lista_dibujar = ga_controller.individuos
            else:
                ga_controller.evolucionar()
                lista_dibujar = [] # Breve parpadeo al cambiar gen
                
            # Info texto
            info = f"Gen: {ga_controller.generacion} | Vivos: {sum(s.vivo for s in ga_controller.individuos)}"

        else:
            # --- LÓGICA REPLAY ---
            serpiente = poblacion[0]
            if serpiente.vivo:
                serpiente.pensar()
                serpiente.update()
            else:
                # Si muere en replay, reiniciarla para verla jugar otra vez
                print(f"Murió la serpiente con {serpiente.score} puntos replay. Reiniciando...")
                poblacion = [Serpiente(cerebro_cargado)]
            
            lista_dibujar = poblacion
            info = f"REPLAY MODE | Score: {serpiente.score}"

        # Renderizado común
        for s in lista_dibujar:
            if s.vivo:
                for i, parte in enumerate(s.cuerpo):
                    rect = (parte[0]*TAM_CELDA, parte[1]*TAM_CELDA, TAM_CELDA, TAM_CELDA)
                    color = s.color if i == 0 else GRIS
                    pygame.draw.rect(pantalla, color, rect)
                
                rect_comida = (s.comida[0]*TAM_CELDA, s.comida[1]*TAM_CELDA, TAM_CELDA, TAM_CELDA)
                pygame.draw.rect(pantalla, ROJO, rect_comida)

        texto = fuente.render(info, True, BLANCO)
        pantalla.blit(texto, (10, 10))

        pygame.display.flip()
        reloj.tick(fps_actual)

    pygame.quit()

if __name__ == "__main__":
    main()
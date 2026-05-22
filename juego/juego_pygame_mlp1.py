import os
import csv
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pygame
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    try:
        matplotlib.use("Qt5Agg")
    except Exception:
        pass  
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

plt.ion()

BASE_W, BASE_H = 1080, 720
WINDOW_FRACTION = 0.97
EXTRA_SCALE = 1.1

@dataclass
class Sample:
    velocidad_bala: float
    distancia: float
    bala_y: float  
    h1: int  
    h2: int  
    h3: int  
    h4: int  
    h5: int
    h6: int
    h7: int   
    h8: int   
    accion: int    

class Juego:
    def __init__(self) -> None:
        pygame.init()
        self._flags = 0
        self._fullscreen = False
        start_w, start_h = BASE_W, BASE_H
        self.pantalla = pygame.display.set_mode((start_w, start_h), self._flags)
        pygame.display.set_caption("IA: Clonación con 8 Frames de Memoria")

        self.BLANCO = (255, 255, 255)
        self.NEGRO = (0, 0, 0)
        self.GRIS = (200, 200, 200)
        self.AMARILLO = (255, 220, 120)

        self.corriendo = True
        self.modo_auto = False
        self.en_menu = True  
        self.datos_modelo: List[Sample] = []
        self.modelo: Optional[MLPClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.modelo_entrenado = False
        
        # Historial ajustado a 8 frames para balancear la ecuación
        self.memoria_acciones = [0] * 8

        self.w, self.h = start_w, start_h
        self.scale = 1.0
        self._apply_resolution(start_w, start_h, reset_positions=True)
        self._reset_estado_juego()

    def _apply_resolution(self, w: int, h: int, reset_positions: bool) -> None:
        self.w, self.h = int(w), int(h)
        self.scale = max(1.0, min(self.w / BASE_W, self.h / BASE_H) * EXTRA_SCALE)
        
        self.margin = int(50 * self.scale)
        self.ground_y = self.h - int(100 * self.scale)
        self.player_size_normal = (int(32 * self.scale), int(48 * self.scale))
        self.player_size_agachado = (int(32 * self.scale), int(24 * self.scale))
        self.bullet_size = (int(16 * self.scale), int(16 * self.scale))
        self.ship_size = (int(64 * self.scale), int(64 * self.scale))
        
        self.salto_vel_inicial = 16 * self.scale
        self.gravedad = 1 * self.scale
        self.fuente = pygame.font.SysFont("Arial", int(24 * self.scale))
        self.fuente_chica = pygame.font.SysFont("Arial", int(18 * self.scale))

        self._cargar_assets()

        if reset_positions:
            self.jugador = pygame.Rect(self.margin, self.ground_y, self.player_size_normal[0], self.player_size_normal[1])
            self.bala = pygame.Rect(self.w, self.ground_y, self.bullet_size[0], self.bullet_size[1])
            self.nave = pygame.Rect(self.w - int(100 * self.scale), self.ground_y, self.ship_size[0], self.ship_size[1])

    def _cargar_assets(self) -> None:
        def safe_load(path: str, size: Tuple[int, int], color=(200, 200, 200, 255)) -> pygame.Surface:
            try:
                img = pygame.image.load(path).convert_alpha()
                return pygame.transform.smoothscale(img, size) 
            except:
                s = pygame.Surface(size, pygame.SRCALPHA); s.fill(color); return s

        base = os.path.dirname(__file__)
        self.jugador_frames = [safe_load(os.path.join(base, f"assets/sprites/mario_frame_{i}.png"), self.player_size_normal) for i in range(1, 5)]
        self.jugador_agachado_img = pygame.transform.smoothscale(self.jugador_frames[0], self.player_size_agachado)
        self.bala_img = safe_load(os.path.join(base, "assets/sprites/fire_ball.png"), self.bullet_size, (160, 120, 255))
        self.fondo_img = safe_load(os.path.join(base, "assets/game/fondoMario.png"), (self.w, self.h), (40, 40, 40))
        self.nave_img = safe_load(os.path.join(base, "assets/game/bowser.png"), self.ship_size, (140, 255, 200))

    def _toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        if self._fullscreen:
            info = pygame.display.Info()
            w = info.current_w or self.w
            h = info.current_h or self.h
            self.pantalla = pygame.display.set_mode((w, h), pygame.FULLSCREEN)
            self._apply_resolution(w, h, reset_positions=True)
        else:
            self.pantalla = pygame.display.set_mode((BASE_W, BASE_H), self._flags)
            self._apply_resolution(BASE_W, BASE_H, reset_positions=True)
        self._reset_estado_juego()

    def _reset_estado_juego(self) -> None:
        self.jugador.height = self.player_size_normal[1]
        self.jugador.y = self.ground_y
        self.bala_disparada = False
        self.salto = False
        self.agachado = False
        self.en_suelo = True
        self.salto_vel = self.salto_vel_inicial
        self.fondo_x1, self.fondo_x2 = 0, self.w
        self.memoria_acciones = [0] * 8 

    def _reset_modelo(self) -> None:
        self.modelo = None
        self.scaler = None
        self.modelo_entrenado = False

    def exportar_datos_csv(self) -> str:
        if not self.datos_modelo: return "No hay datos para exportar."
        base = os.path.dirname(__file__)
        ruta = os.path.join(base, "datos_mlp.csv")
        try:
            with open(ruta, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["velocidad_bala", "distancia", "bala_y", "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "accion"])
                for s in self.datos_modelo:
                    writer.writerow([s.velocidad_bala, s.distancia, s.bala_y, s.h1, s.h2, s.h3, s.h4, s.h5, s.h6, s.h7, s.h8, s.accion])
        except Exception as e:
            return f"Error al guardar CSV: {e}"
        return f"CSV guardado en datos_mlp.csv ({len(self.datos_modelo)} filas)."

    def graficar_datos_2d(self) -> str:
        if not self.datos_modelo: return "No hay datos para graficar."
        xs = [s.distancia for s in self.datos_modelo]
        ys = [s.bala_y for s in self.datos_modelo]
        colores = {0: "gray", 1: "red", 2: "green"}
        cs = [colores[s.accion] for s in self.datos_modelo]

        fig_num = plt.figure("Datos MLP - 2D", figsize=(8, 6)).number
        plt.figure(fig_num)
        plt.clf()
        ax = plt.gca()
        ax.scatter(xs, ys, c=cs, alpha=0.6, edgecolors="k", s=30)
        ax.set_xlabel("Distancia jugador-bala")
        ax.set_ylabel("Altura de la bala (Y)")
        ax.set_title("Datos MLP (Gris=Nada, Rojo=Salto, Verde=Agacharse)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show(block=False)
        plt.draw() 
        return "Mostrando gráfica 2D interactiva."

    def graficar_datos_3d(self) -> str:
        if not self.datos_modelo: return "No hay datos para graficar."
        xs = [s.distancia for s in self.datos_modelo]
        ys = [s.velocidad_bala for s in self.datos_modelo]
        zs = [s.bala_y for s in self.datos_modelo]
        colores = {0: "gray", 1: "red", 2: "green"}
        cs = [colores[s.accion] for s in self.datos_modelo]

        fig = plt.figure("Datos MLP - 3D", figsize=(8, 6))
        plt.clf()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(xs, ys, zs, c=cs, alpha=0.6, edgecolors="k", s=30)
        ax.set_xlabel("Distancia")
        ax.set_ylabel("Velocidad bala")
        ax.set_zlabel("Altura bala (Y)")
        ax.set_title("Espacio de Características 3D")
        plt.tight_layout()
        plt.show(block=False)
        plt.draw() 
        return "Mostrando gráfica 3D interactiva."

    def disparar_bala(self) -> None:
        if not self.bala_disparada:
            opciones_velocidad = [-10, -16, -24] 
            self.velocidad_bala = int(random.choice(opciones_velocidad) * self.scale)
            
            tipo_bala = random.randint(1, 3)
            
            if tipo_bala == 1: self.bala.y = self.ground_y + int(32 * self.scale)  
            elif tipo_bala == 2: self.bala.y = self.ground_y + int(8 * self.scale)  
            else: self.bala.y = self.ground_y - int(25 * self.scale)
                
            self.bala.x = self.w
            self.bala_disparada = True

    def iniciar_salto(self) -> None:
        if self.en_suelo and not self.agachado:
            self.salto = True
            self.en_suelo = False

    def manejar_fisica(self) -> None:
        if self.salto:
            self.jugador.y -= int(self.salto_vel)
            self.salto_vel -= self.gravedad
            if self.jugador.y >= self.ground_y:
                self.jugador.y = self.ground_y
                self.salto = False
                self.salto_vel = self.salto_vel_inicial
                self.en_suelo = True
        
        if self.agachado and self.en_suelo:
            self.jugador.height = self.player_size_agachado[1]
            self.jugador.y = self.ground_y + (self.player_size_normal[1] - self.player_size_agachado[1])
        elif not self.salto:
            self.jugador.height = self.player_size_normal[1]
            self.jugador.y = self.ground_y

    def registrar_datos(self, accion_manual: int) -> None:
        if self.bala_disparada:
            vel = float(self.velocidad_bala)
            dist = float(self.bala.x - self.jugador.x)
            alt = float(self.bala.y)
            if dist < -self.w: return 
        else:
            vel = 0.0
            dist = float(self.w * 2) 
            alt = float(self.ground_y)
            
        self.datos_modelo.append(Sample(
            velocidad_bala=vel,
            distancia=dist,
            bala_y=alt,
            h1=float(self.memoria_acciones[0]),
            h2=float(self.memoria_acciones[1]),
            h3=float(self.memoria_acciones[2]),
            h4=float(self.memoria_acciones[3]),
            h5=float(self.memoria_acciones[4]),
            h6=float(self.memoria_acciones[5]),
            h7=float(self.memoria_acciones[6]),
            h8=float(self.memoria_acciones[7]),
            accion=accion_manual
        ))

    def entrenar_modelo(self) -> Tuple[bool, str]:
        if len(self.datos_modelo) < 150:
            return False, f"Faltan datos ({len(self.datos_modelo)}/150). Juega en Manual."
        
        # =========================================================
        # MITIGACIÓN DE EXCEPCIONES Y RUIDO (Umbrales Separados)
        # =========================================================
        # 1. Contamos cuántas veces hiciste cada acción
        conteos = {0: 0, 1: 0, 2: 0}
        for s in self.datos_modelo:
            conteos[s.accion] += 1
            
        # 2. Definimos umbrales independientes para cada acción
        UMBRAL_SALTO = 4    # Mínimo de veces que debes saltar para que no sea ignorado
        UMBRAL_AGACHA = 8   # Mínimo de veces que debes agacharte para que no sea ignorado
        
        # El 0 (no hacer nada) siempre es válido
        acciones_validas = [0]
        
        # Evaluamos el estado de Salto (1)
        if conteos[1] >= UMBRAL_SALTO: 
            acciones_validas.append(1)
        else:
            print(f"Filtro: Ignorando Salto (1) -> Ocurrió {conteos[1]} veces (mínimo {UMBRAL_SALTO}).")
            
        # Evaluamos el estado de Agacharse (2)
        if conteos[2] >= UMBRAL_AGACHA: 
            acciones_validas.append(2)
        else:
            print(f"Filtro: Ignorando Agacharse (2) -> Ocurrió {conteos[2]} veces (mínimo {UMBRAL_AGACHA}).")

        # 3. Filtramos los datos ignorando las acciones que no superaron sus respectivos umbrales
        datos_limpios = [s for s in self.datos_modelo if s.accion in acciones_validas]

        # Preparamos X e y solo con los datos válidos
        X = [[s.velocidad_bala, s.distancia, s.bala_y, s.h1, s.h2, s.h3, s.h4, s.h5, s.h6, s.h7, s.h8] for s in datos_limpios]
        y = [s.accion for s in datos_limpios]
        
        # Verificamos que tras limpiar siga habiendo al menos 2 acciones diferentes para entrenar
        if len(set(y)) < 2:
            return False, "Tras ignorar las excepciones, no hay suficientes acciones distintas para aprender."

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.modelo = MLPClassifier(
            hidden_layer_sizes=(32, 16), 
            activation="relu",
            solver="lbfgs",
            alpha=1e-4, 
            max_iter=2000,
            random_state=42
        )
            
        self.modelo.fit(X_scaled, y)
        self.modelo_entrenado = True
        acc = self.modelo.score(X_scaled, y)
        return True, f"¡Clonación Lista! Similitud: {acc*100:.1f}%"

    def decision_auto(self) -> int:
        if not self.modelo_entrenado: return 0
        
        if self.bala_disparada:
            vel = float(self.velocidad_bala)
            dist = float(self.bala.x - self.jugador.x)
            alt = float(self.bala.y)
        else:
            vel = 0.0
            dist = float(self.w * 2)
            alt = float(self.ground_y)
            
        X = [[vel, dist, alt, 
              float(self.memoria_acciones[0]), 
              float(self.memoria_acciones[1]), 
              float(self.memoria_acciones[2]), 
              float(self.memoria_acciones[3]),
              float(self.memoria_acciones[4]),
              float(self.memoria_acciones[5]),
              float(self.memoria_acciones[6]),
              float(self.memoria_acciones[7])]]
              
        Xs = self.scaler.transform(X)
        return int(self.modelo.predict(Xs)[0])

    def _dibujar_menu(self, msg: str = "") -> None:
        self.pantalla.fill(self.NEGRO)
        txt = self.fuente.render(f"Memoria Grabada: {len(self.datos_modelo)} cuadros", True, self.BLANCO)
        self.pantalla.blit(txt, (50, 50))
        
        opciones = [
            "M - Jugar Manual (Inicia grabación limpia)", 
            "A - Auto (Dejar a tu IA clonada jugar)", 
            "T - Entrenar Clonación Exacta", 
            "C - Exportar datos a CSV",
            "2 - Graficar 2D",
            "3 - Graficar 3D",
            "F - Fullscreen",
            "Q - Salir"
        ]
        for i, op in enumerate(opciones):
            t = self.fuente.render(op, True, self.GRIS)
            self.pantalla.blit(t, (50, 100 + i*35))
            
        if msg:
            m = self.fuente_chica.render(msg, True, self.AMARILLO)
            self.pantalla.blit(m, (50, 400))
        pygame.display.flip()

    def loop(self) -> None:
        reloj = pygame.time.Clock()
        msg = ""
        while self.corriendo:
            if self.en_menu:
                self._dibujar_menu(msg)
                for e in pygame.event.get():
                    if e.type == pygame.QUIT: self.corriendo = False
                    if e.type == pygame.KEYDOWN:
                        if e.key == pygame.K_m: 
                            self.modo_auto = False
                            self.en_menu = False
                            self.datos_modelo.clear()
                            self._reset_modelo()
                            self._reset_estado_juego()
                            msg = ""
                        elif e.key == pygame.K_a:
                            if self.modelo_entrenado: 
                                self.modo_auto = True
                                self.en_menu = False
                                self._reset_estado_juego() 
                                msg = ""
                            else: msg = "Primero entrena el clon (T)."
                        elif e.key == pygame.K_t: ok, msg = self.entrenar_modelo()
                        elif e.key == pygame.K_c: msg = self.exportar_datos_csv()
                        elif e.key == pygame.K_2: msg = self.graficar_datos_2d()
                        elif e.key == pygame.K_3: msg = self.graficar_datos_3d()
                        elif e.key == pygame.K_f: self._toggle_fullscreen()
                        elif e.key == pygame.K_q: self.corriendo = False
                continue 
            
            keys = pygame.key.get_pressed()
            accion_actual = 0 
            
            for e in pygame.event.get():
                if e.type == pygame.QUIT: self.corriendo = False
                if e.type == pygame.KEYDOWN and (e.key == pygame.K_ESCAPE or e.key == pygame.K_p): 
                    self.en_menu = True
                    self._reset_estado_juego()

            if self.modo_auto:
                dec = self.decision_auto()
                if dec == 1: self.iniciar_salto()
                self.agachado = (dec == 2)
                
                self.memoria_acciones.pop(0)
                self.memoria_acciones.append(dec)
            else:
                if keys[pygame.K_SPACE]: self.iniciar_salto(); accion_actual = 1
                elif keys[pygame.K_DOWN] or keys[pygame.K_s]: self.agachado = True; accion_actual = 2
                else: self.agachado = False
                
                self.registrar_datos(accion_actual)
                self.memoria_acciones.pop(0)
                self.memoria_acciones.append(accion_actual)

            self.manejar_fisica()
            if not self.bala_disparada: self.disparar_bala()
            
            self.pantalla.fill(self.NEGRO)
            
            self.fondo_x1 -= int(2 * self.scale); self.fondo_x2 -= int(2 * self.scale)
            if self.fondo_x1 <= -self.w: self.fondo_x1 = self.w
            if self.fondo_x2 <= -self.w: self.fondo_x2 = self.w
            self.pantalla.blit(self.fondo_img, (self.fondo_x1, 0))
            self.pantalla.blit(self.fondo_img, (self.fondo_x2, 0))
            
            self.pantalla.blit(self.nave_img, (self.nave.x, self.nave.y))
            
            if self.agachado and self.en_suelo: self.pantalla.blit(self.jugador_agachado_img, (self.jugador.x, self.jugador.y))
            else:
                frame = (pygame.time.get_ticks() // 100) % 4
                self.pantalla.blit(self.jugador_frames[frame], (self.jugador.x, self.jugador.y))
            
            self.bala.x += self.velocidad_bala
            self.pantalla.blit(self.bala_img, (self.bala.x, self.bala.y))
            if self.bala.x < -self.bullet_size[0]: self.bala_disparada = False
            
            if self.jugador.colliderect(self.bala): self._reset_estado_juego()
            
            pygame.display.flip()
            reloj.tick(60)

        pygame.quit()

def main() -> None:
    Juego().loop()

if __name__ == "__main__":
    main()
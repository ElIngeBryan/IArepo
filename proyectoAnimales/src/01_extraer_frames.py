import cv2
import os
import glob # Librería mágica para buscar archivos múltiples

def extraer_frames(ruta_video, carpeta_salida, frames_por_segundo=2):
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)

    # Novedad: Extraemos el nombre del video (ej. "rana_selva") para usarlo en las fotos
    nombre_video = os.path.basename(ruta_video).replace('.mp4', '')

    cap = cv2.VideoCapture(ruta_video)
    fps_original = cap.get(cv2.CAP_PROP_FPS)
    if fps_original == 0:
        print(f"Error al leer el video: {ruta_video}")
        return

    saltos = int(fps_original / frames_por_segundo)
    count, guardados = 0, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % saltos == 0:
            # Ahora la foto se llamará: "rana_selva_img_0000.jpg" (¡cero riesgo de sobrescribir!)
            nombre_imagen = f"{nombre_video}_img_{guardados:04d}.jpg"
            ruta_completa = os.path.join(carpeta_salida, nombre_imagen)
            
            cv2.imwrite(ruta_completa, frame)
            guardados += 1
        count += 1

    cap.release()
    print(f"[{nombre_video}] -> Se extrajeron {guardados} imágenes en '{carpeta_salida}'")


# --- EJECUCIÓN AUTOMÁTICA ---
if __name__ == "__main__":
    carpeta_videos_originales = "../videos_originales"
    carpeta_dataset = "../dataset"
    clases_animales = ['aranas', 'ranas', 'pajaros', 'ballenas', 'changos']

    print("Iniciando procesamiento de todos los videos...")

    # El código va a iterar carpeta por carpeta
    for clase in clases_animales:
        ruta_origen_clase = os.path.join(carpeta_videos_originales, clase)
        ruta_destino_clase = os.path.join(carpeta_dataset, clase)

        # Si la carpeta original existe, busca todos los videos .mp4 adentro
        if os.path.exists(ruta_origen_clase):
            videos_encontrados = glob.glob(os.path.join(ruta_origen_clase, "*.mp4"))
            
            for video in videos_encontrados:
                extraer_frames(video, ruta_destino_clase, frames_por_segundo=2)
        else:
            print(f"La carpeta '{ruta_origen_clase}' no existe aún. Saltando...")
            
    print("¡Proceso terminado!")
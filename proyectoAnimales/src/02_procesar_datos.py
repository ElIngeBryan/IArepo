import cv2
import os
import glob

def estandarizar_dataset(directorio_base, tamano=(96, 96)):
    clases = ['aranas', 'ranas', 'pajaros', 'ballenas', 'changos']
    imagenes_corruptas = 0
    imagenes_procesadas = 0

    for clase in clases:
        ruta_clase = os.path.join(directorio_base, clase)
        # Buscar todas las imágenes jpg y png
        imagenes = glob.glob(os.path.join(ruta_clase, "*.jpg")) + glob.glob(os.path.join(ruta_clase, "*.png"))
        
        for img_path in imagenes:
            try:
                # Leer imagen
                img = cv2.imread(img_path)
                if img is None:
                    raise Exception("Imagen no leída")
                
                # Redimensionar (el consejo de tu profesor para agilizar el procesamiento)
                img_resized = cv2.resize(img, tamano)
                
                # Sobrescribir la imagen con el tamaño correcto
                cv2.imwrite(img_path, img_resized)
                imagenes_procesadas += 1
                
            except Exception as e:
                # Si la imagen está corrupta, se elimina para no arruinar el entrenamiento
                os.remove(img_path)
                imagenes_corruptas += 1

    print(f"Procesamientso listo. Imágenes válidas: {imagenes_procesadas}. Eliminadas: {imagenes_corruptas}")

if __name__ == "__main__":
    estandarizar_dataset("../dataset", tamano=(96, 96))
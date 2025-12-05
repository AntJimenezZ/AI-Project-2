"""
Interfaz web para clasificación de imágenes y reconocimiento de voz
Usa Gradio para crear una interfaz sencilla con dos pestañas
"""

# Solución para el error de ConnectionResetError en Windows
import sys
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Suprimir el error cosmético de ConnectionResetError
    import logging
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)

import gradio as gr
import tensorflow as tf
import numpy as np
import librosa
from PIL import Image
import os
import cv2
import tempfile
import warnings

# Suprimir warnings de conversión de video
warnings.filterwarnings('ignore', category=UserWarning, module='gradio.components.video')

# ============================================
# CONFIGURACIÓN DE RUTAS
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Rutas de modelos de clasificación de imágenes
RESNET_MODEL_PATH = os.path.join(MODELS_DIR, "best_model_resnet50.keras")
MOBILENET_MODEL_PATH = os.path.join(MODELS_DIR, "best_model_mobilenetv2.keras")

# Ruta del modelo de speech-to-text (cuando esté disponible)
SPEECH_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.keras")

# Clases para clasificación de imágenes
CLASS_NAMES = ['Gato', 'Perro']

# Parámetros de audio (coinciden con el entrenamiento de parte2)
SR = 16000
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 80
MAX_AUDIO_SECONDS = 8.0

# Vocabulario usado en el entrenamiento CTC (parte2)
CHARS = list("abcdefghijklmnopqrstuvwxyzñáéíóúü'.,?¡! ")
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 es blank/padding
IDX_TO_CHAR = {i: c for c, i in CHAR_TO_IDX.items()}

# ============================================
# CARGAR MODELOS DE CLASIFICACIÓN DE IMÁGENES
# ============================================
print("Cargando modelos de clasificación de imágenes...")

try:
    model_resnet = tf.keras.models.load_model(RESNET_MODEL_PATH)
    print("✓ Modelo ResNet50 cargado correctamente")
except Exception as e:
    print(f"✗ Error al cargar ResNet50: {e}")
    model_resnet = None

try:
    model_mobilenet = tf.keras.models.load_model(MOBILENET_MODEL_PATH)
    print("✓ Modelo MobileNetV2 cargado correctamente")
except Exception as e:
    print(f"✗ Error al cargar MobileNetV2: {e}")
    model_mobilenet = None

# Modelo de speech-to-text
speech_model = None
try:
    if os.path.exists(SPEECH_MODEL_PATH):
        speech_model = tf.keras.models.load_model(SPEECH_MODEL_PATH, compile=False)
        print("✓ Modelo de speech-to-text cargado correctamente")
    else:
        print("⚠️ Modelo de speech-to-text no encontrado en 'models/best_model.keras'")
except Exception as e:
    print(f"✗ Error al cargar modelo de speech-to-text: {e}")
    speech_model = None

# ============================================
# FUNCIONES PARA CLASIFICACIÓN DE IMÁGENES
# ============================================

def preprocess_image(image):
    """
    Preprocesa la imagen para los modelos de clasificación
    Args:
        image: PIL Image
    Returns:
        numpy array normalizado y redimensionado
    """
    # Convertir a RGB si es necesario
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Redimensionar a 224x224 (tamaño estándar para ResNet50 y MobileNetV2)
    image = image.resize((224, 224))
    
    # Convertir a array numpy
    img_array = np.array(image)
    
    # Normalizar a [0, 1]
    img_array = img_array.astype(np.float32) / 255.0
    
    # Agregar dimensión de batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


# ============================================
# FUNCIONES PARA AUDIO Y DECODIFICACIÓN CTC
# ============================================


def load_wav(path, sr=SR):
    """Carga y recorta audio a la duración máxima."""
    x, _ = librosa.load(path, sr=sr)
    if x.shape[0] > sr * MAX_AUDIO_SECONDS:
        x = x[: int(sr * MAX_AUDIO_SECONDS)]
    return x


def wav_to_log_mel(x):
    """Convierte audio a espectrograma log-mel normalizado."""
    S = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = (S_db + 80.0) / 80.0  # normalizar a ~[0,1]
    return S_db.T  # frames x mel


def greedy_decode(logits, logit_lengths):
    decoded = tf.nn.ctc_greedy_decoder(tf.transpose(logits, [1, 0, 2]), sequence_length=logit_lengths)[0][0]
    dense = tf.sparse.to_dense(decoded, default_value=0).numpy()
    texts = []
    for row in dense:
        s = ""
        for idx in row:
            if idx == 0:
                continue
            s += IDX_TO_CHAR.get(int(idx), '')
        texts.append(s)
    return texts


def classify_image(image):
    """
    Clasifica una imagen usando ambos modelos
    Args:
        image: PIL Image
    Returns:
        dict con resultados de ambos modelos
    """
    if image is None:
        return "Por favor, sube una imagen"
    
    try:
        # Preprocesar imagen
        img_array = preprocess_image(image)
        
        results = {}
        
        # Predicción con ResNet50
        if model_resnet is not None:
            pred_resnet = model_resnet.predict(img_array, verbose=0)
            pred_class_resnet = np.argmax(pred_resnet[0])
            confidence_resnet = float(pred_resnet[0][pred_class_resnet]) * 100
            
            results["ResNet50"] = {
                "Predicción": CLASS_NAMES[pred_class_resnet],
                "Confianza": f"{confidence_resnet:.2f}%"
            }
            
            # Crear distribución de probabilidades para ResNet50
            resnet_probs = {CLASS_NAMES[i]: float(pred_resnet[0][i]) for i in range(len(CLASS_NAMES))}
        else:
            results["ResNet50"] = {"Error": "Modelo no disponible"}
            resnet_probs = None
        
        # Predicción con MobileNetV2
        if model_mobilenet is not None:
            pred_mobilenet = model_mobilenet.predict(img_array, verbose=0)
            pred_class_mobilenet = np.argmax(pred_mobilenet[0])
            confidence_mobilenet = float(pred_mobilenet[0][pred_class_mobilenet]) * 100
            
            results["MobileNetV2"] = {
                "Predicción": CLASS_NAMES[pred_class_mobilenet],
                "Confianza": f"{confidence_mobilenet:.2f}%"
            }
            
            # Crear distribución de probabilidades para MobileNetV2
            mobilenet_probs = {CLASS_NAMES[i]: float(pred_mobilenet[0][i]) for i in range(len(CLASS_NAMES))}
        else:
            results["MobileNetV2"] = {"Error": "Modelo no disponible"}
            mobilenet_probs = None
        
        # Formatear resultados como texto
        output_text = "RESULTADOS DE CLASIFICACIÓN\n\n"
        
        if model_resnet is not None:
            output_text += f"ResNet50:\n"
            output_text += f"   Predicción: {results['ResNet50']['Predicción']}\n"
            output_text += f"   Confianza: {results['ResNet50']['Confianza']}\n\n"
        
        if model_mobilenet is not None:
            output_text += f"MobileNetV2:\n"
            output_text += f"   Predicción: {results['MobileNetV2']['Predicción']}\n"
            output_text += f"   Confianza: {results['MobileNetV2']['Confianza']}\n"
        
        return output_text, resnet_probs, mobilenet_probs
        
    except Exception as e:
        return f"Error al procesar la imagen: {str(e)}", None, None


def convert_video_format(input_path, output_path):
    """
    Convierte un video a formato compatible usando ffmpeg si está disponible.
    Si no, usa solo OpenCV.
    """
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return False
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        cap.release()
        out.release()
        return True
    except:
        return False


def classify_video(video_path, progress=gr.Progress()):
    """
    Procesa un video extrayendo 1 frame por segundo y clasificando cada uno con ambos modelos.
    Utiliza suavizado de predicciones para mayor estabilidad.
    Retorna un análisis con detecciones por modelo y un video con anotaciones.
    
    Args:
        video_path: ruta al archivo de video
        progress: barra de progreso de Gradio
    Returns:
        tupla con (reporte_texto, video_anotado)
    """
    if video_path is None:
        return "Por favor, sube un video", None
    
    cap = None
    out = None
    
    try:
        # Abrir el video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return "Error: No se pudo abrir el video", None
        
        # Obtener propiedades del video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if fps == 0:
            fps = 30  # valor por defecto
        
        duration = total_frames / fps if fps > 0 else 0
        
        # Limitar la duración del video a 2 minutos para evitar problemas de memoria
        max_duration = 120  # 2 minutos
        if duration > max_duration:
            if cap is not None:
                cap.release()
            return f"⚠️ Video demasiado largo ({duration:.1f}s). Por favor, usa un video de máximo {max_duration} segundos.", None
        
        # Configurar escritor de video con codecs compatibles (orden de preferencia)
        temp_output_path = os.path.join(tempfile.gettempdir(), f"video_clasificado_{os.getpid()}.mp4")
        
        # Intentar con diferentes codecs en orden de compatibilidad
        codecs_to_try = [
            ('mp4v', 'MP4V'),  # MPEG-4 - más compatible
            ('XVID', 'XVID'),  # Xvid - muy compatible
            ('MJPG', 'MJPEG'), # Motion JPEG - siempre disponible
        ]
        
        out = None
        for codec_name, codec_desc in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_name)
                out = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))
                if out.isOpened():
                    print(f"✓ Usando codec {codec_desc} para el video de salida")
                    break
                else:
                    out.release()
                    out = None
            except Exception as e:
                print(f"⚠️ No se pudo usar codec {codec_desc}: {e}")
                continue
        
        if out is None or not out.isOpened():
            if cap is not None:
                cap.release()
            return "Error: No se pudo crear el video de salida con ningún codec disponible. OpenCV puede no estar configurado correctamente.", None
        
        # Variables para análisis
        detections_mobilenet = []
        detections_resnet = []
        frame_count = 0
        processed_frames = 0
        
        # Variables para detección con mejor respuesta a cambios
        last_mobilenet_class = None
        last_mobilenet_conf = 0
        last_resnet_class = None
        last_resnet_conf = 0
        frames_since_update_mb = 0
        frames_since_update_rn = 0
        max_frames_display = fps * 1  # Mostrar predicción durante 1 segundo
        
        # Buffer para detección de cambios (últimas 3 predicciones)
        mobilenet_buffer = []
        resnet_buffer = []
        buffer_size = 3
        
        # Procesar frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Actualizar progreso
            if frame_count % (fps * 5) == 0:  # Actualizar cada 5 segundos
                progress(frame_count / total_frames, desc=f"Procesando: {frame_count}/{total_frames} frames")
            
            # Extraer 1 frame por segundo (cada fps frames)
            if frame_count % fps == 0 or frame_count == 1:
                processed_frames += 1
                
                # Convertir BGR a RGB para PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Clasificar la imagen con ambos modelos
                try:
                    img_array = preprocess_image(pil_image)
                    
                    # Predicción con MobileNetV2
                    if model_mobilenet is not None:
                        pred = model_mobilenet.predict(img_array, verbose=0)
                        pred_class = np.argmax(pred[0])
                        confidence = float(pred[0][pred_class]) * 100
                        mobilenet_class = CLASS_NAMES[pred_class]
                        
                        # Agregar al buffer solo si la confianza es aceptable (>40%)
                        if confidence > 40:
                            mobilenet_buffer.append({
                                "clase": mobilenet_class,
                                "confianza": confidence
                            })
                            
                            # Mantener solo las últimas N predicciones
                            if len(mobilenet_buffer) > buffer_size:
                                mobilenet_buffer.pop(0)
                            
                            # Análisis del buffer para decidir la clase
                            if len(mobilenet_buffer) >= 2:  # Al menos 2 predicciones
                                # Contar clases en el buffer
                                gatos = sum(1 for p in mobilenet_buffer if p["clase"] == "Gato")
                                perros = sum(1 for p in mobilenet_buffer if p["clase"] == "Perro")
                                
                                # Decisión por mayoría simple
                                if gatos > perros:
                                    chosen_class = "Gato"
                                    avg_conf = sum(p["confianza"] for p in mobilenet_buffer if p["clase"] == "Gato") / gatos
                                elif perros > gatos:
                                    chosen_class = "Perro"
                                    avg_conf = sum(p["confianza"] for p in mobilenet_buffer if p["clase"] == "Perro") / perros
                                else:
                                    # Empate: usar la más reciente con mayor confianza
                                    chosen_class = mobilenet_buffer[-1]["clase"]
                                    avg_conf = mobilenet_buffer[-1]["confianza"]
                                
                                # Actualizar solo si hay un cambio significativo O suficiente confianza
                                if chosen_class != last_mobilenet_class or avg_conf > 60:
                                    last_mobilenet_class = chosen_class
                                    last_mobilenet_conf = avg_conf
                                    frames_since_update_mb = 0
                                    
                                    detections_mobilenet.append({
                                        "frame": processed_frames,
                                        "segundo": frame_count / fps,
                                        "clase": chosen_class,
                                        "confianza": f"{avg_conf:.2f}%"
                                    })
                    
                    # Predicción con ResNet50
                    if model_resnet is not None:
                        pred = model_resnet.predict(img_array, verbose=0)
                        pred_class = np.argmax(pred[0])
                        confidence = float(pred[0][pred_class]) * 100
                        resnet_class = CLASS_NAMES[pred_class]
                        
                        # Agregar al buffer solo si la confianza es aceptable (>40%)
                        if confidence > 40:
                            resnet_buffer.append({
                                "clase": resnet_class,
                                "confianza": confidence
                            })
                            
                            # Mantener solo las últimas N predicciones
                            if len(resnet_buffer) > buffer_size:
                                resnet_buffer.pop(0)
                            
                            # Análisis del buffer para decidir la clase
                            if len(resnet_buffer) >= 2:  # Al menos 2 predicciones
                                # Contar clases en el buffer
                                gatos = sum(1 for p in resnet_buffer if p["clase"] == "Gato")
                                perros = sum(1 for p in resnet_buffer if p["clase"] == "Perro")
                                
                                # Decisión por mayoría simple
                                if gatos > perros:
                                    chosen_class = "Gato"
                                    avg_conf = sum(p["confianza"] for p in resnet_buffer if p["clase"] == "Gato") / gatos
                                elif perros > gatos:
                                    chosen_class = "Perro"
                                    avg_conf = sum(p["confianza"] for p in resnet_buffer if p["clase"] == "Perro") / perros
                                else:
                                    # Empate: usar la más reciente con mayor confianza
                                    chosen_class = resnet_buffer[-1]["clase"]
                                    avg_conf = resnet_buffer[-1]["confianza"]
                                
                                # Actualizar solo si hay un cambio significativo O suficiente confianza
                                if chosen_class != last_resnet_class or avg_conf > 60:
                                    last_resnet_class = chosen_class
                                    last_resnet_conf = avg_conf
                                    frames_since_update_rn = 0
                                    
                                    detections_resnet.append({
                                        "frame": processed_frames,
                                        "segundo": frame_count / fps,
                                        "clase": chosen_class,
                                        "confianza": f"{avg_conf:.2f}%"
                                    })
                
                except Exception as e:
                    print(f"Error procesando frame {processed_frames}: {e}")
            
            # Dibujar predicciones en TODOS los frames (no solo en los procesados)
            # Esto crea un efecto de "mantener visible" la predicción durante 2 segundos
            y_offset = 50
            
            if last_mobilenet_class and frames_since_update_mb < max_frames_display:
                color_mb = (0, 255, 0) if last_mobilenet_class == "Gato" else (255, 0, 0)
                label_mb = f"MobileNetV2: {last_mobilenet_class} ({last_mobilenet_conf:.1f}%)"
                
                # Dibujar rectángulo de fondo para mejor legibilidad
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 2
                text_size = cv2.getTextSize(label_mb, font, font_scale, thickness)[0]
                
                # Rectángulo de fondo
                cv2.rectangle(frame, (15, y_offset - text_size[1] - 10), 
                            (25 + text_size[0], y_offset + 10), color_mb, -1)
                
                # Texto blanco
                cv2.putText(frame, label_mb, (20, y_offset), font, font_scale, 
                          (255, 255, 255), thickness, cv2.LINE_AA)
                y_offset += 50
            
            if last_resnet_class and frames_since_update_rn < max_frames_display:
                color_rn = (0, 255, 0) if last_resnet_class == "Gato" else (255, 0, 0)
                label_rn = f"ResNet50: {last_resnet_class} ({last_resnet_conf:.1f}%)"
                
                # Dibujar rectángulo de fondo para mejor legibilidad
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 2
                text_size = cv2.getTextSize(label_rn, font, font_scale, thickness)[0]
                
                # Rectángulo de fondo
                cv2.rectangle(frame, (15, y_offset - text_size[1] - 10), 
                            (25 + text_size[0], y_offset + 10), color_rn, -1)
                
                # Texto blanco
                cv2.putText(frame, label_rn, (20, y_offset), font, font_scale, 
                          (255, 255, 255), thickness, cv2.LINE_AA)
                y_offset += 50
            
            # Mostrar tiempo
            cv2.putText(frame, f"Tiempo: {frame_count/fps:.1f}s", (20, y_offset), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Incrementar contadores
            frames_since_update_mb += 1
            frames_since_update_rn += 1
            
            # Escribir frame en el video de salida
            out.write(frame)
        
        # Liberar recursos inmediatamente
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        
        # Verificar que el archivo se creó correctamente
        if not os.path.exists(temp_output_path) or os.path.getsize(temp_output_path) == 0:
            return "Error: El video procesado no se generó correctamente", None
        
        # Generar reporte
        report = "📊 ANÁLISIS DE CLASIFICACIÓN DE VIDEO\n"
        report += f"{'='*70}\n\n"
        report += f"📹 Información del video:\n"
        report += f"   • Duración: {duration:.2f} segundos\n"
        report += f"   • FPS: {fps}\n"
        report += f"   • Total de frames: {total_frames}\n"
        report += f"   • Frames procesados (1 por segundo): {processed_frames}\n"
        report += f"   • Resolución: {frame_width}x{frame_height}\n\n"
        
        report += f"⚙️ Configuración de procesamiento:\n"
        report += f"   • Umbral de confianza mínima: 40%\n"
        report += f"   • Duración de visualización: 1 segundo por detección\n"
        report += f"   • Buffer de suavizado: 3 frames (mayoría simple)\n"
        report += f"   • Actualización: Cambios detectados o confianza >60%\n\n"
        
        # Análisis MobileNetV2
        if detections_mobilenet:
            report += f"{'='*70}\n"
            report += f"🤖 RESULTADOS - MobileNetV2\n"
            report += f"{'='*70}\n"
            gatos_mb = sum(1 for d in detections_mobilenet if d["clase"] == "Gato")
            perros_mb = sum(1 for d in detections_mobilenet if d["clase"] == "Perro")
            report += f"   • Gatos detectados: {gatos_mb}\n"
            report += f"   • Perros detectados: {perros_mb}\n"
            report += f"   • Total detecciones: {len(detections_mobilenet)}\n\n"
            report += f"{'Frame':<8} {'Segundo':<10} {'Clase':<10} {'Confianza':<12}\n"
            report += f"{'-'*50}\n"
            for det in detections_mobilenet:
                report += f"{det['frame']:<8} {det['segundo']:<10.2f} {det['clase']:<10} {det['confianza']:<12}\n"
        else:
            report += f"{'='*70}\n"
            report += f"🤖 RESULTADOS - MobileNetV2\n"
            report += f"{'='*70}\n"
            report += f"   ⚠️ No se realizaron detecciones con confianza > 40%\n\n"
        
        # Análisis ResNet50
        if detections_resnet:
            report += f"\n{'='*70}\n"
            report += f"🤖 RESULTADOS - ResNet50\n"
            report += f"{'='*70}\n"
            gatos_rn = sum(1 for d in detections_resnet if d["clase"] == "Gato")
            perros_rn = sum(1 for d in detections_resnet if d["clase"] == "Perro")
            report += f"   • Gatos detectados: {gatos_rn}\n"
            report += f"   • Perros detectados: {perros_rn}\n"
            report += f"   • Total detecciones: {len(detections_resnet)}\n\n"
            report += f"{'Frame':<8} {'Segundo':<10} {'Clase':<10} {'Confianza':<12}\n"
            report += f"{'-'*50}\n"
            for det in detections_resnet:
                report += f"{det['frame']:<8} {det['segundo']:<10.2f} {det['clase']:<10} {det['confianza']:<12}\n"
        else:
            report += f"\n{'='*70}\n"
            report += f"🤖 RESULTADOS - ResNet50\n"
            report += f"{'='*70}\n"
            report += f"   ⚠️ No se realizaron detecciones con confianza > 40%\n\n"
        
        report += f"\n{'='*70}\n"
        report += f"✅ Video procesado y guardado con anotaciones\n"
        report += f"{'='*70}\n"
        
        return report, temp_output_path
        
    except Exception as e:
        # Asegurar liberación de recursos en caso de error
        if cap is not None:
            try:
                cap.release()
            except:
                pass
        if out is not None:
            try:
                out.release()
            except:
                pass
        
        return f"Error al procesar el video: {str(e)}\n\nSugerencias:\n- Verifica que el video no esté corrupto\n- Intenta con un video más corto (<2 minutos)\n- Asegúrate de que el formato sea compatible (MP4, AVI, MOV)", None



# ============================================
# FUNCIÓN PARA SPEECH-TO-TEXT (PLACEHOLDER)
# ============================================

def transcribe_audio(audio):
    """Transcribe audio a texto usando el modelo CTC entrenado."""
    if audio is None:
        return "Por favor, graba o sube un audio"

    # Asegurar que el modelo esté cargado
    global speech_model
    if speech_model is None:
        if not os.path.exists(SPEECH_MODEL_PATH):
            return "Modelo de speech-to-text no disponible. Guarda best_model.keras en la carpeta models/"
        try:
            speech_model = tf.keras.models.load_model(SPEECH_MODEL_PATH, compile=False)
        except Exception as e:
            return f"No se pudo cargar el modelo de speech-to-text: {e}"

    try:
        # audio puede ser una ruta (str) o una tupla (sr, datos)
        if isinstance(audio, str):
            wav = load_wav(audio)
        else:
            # audio llega como (sample_rate, np.ndarray)
            sr_in, data = audio
            if sr_in != SR:
                data = librosa.resample(np.array(data, dtype=np.float32), orig_sr=sr_in, target_sr=SR)
            wav = data

        feat = wav_to_log_mel(wav)  # (frames, 80)
        if feat.shape[0] < 2:
            return "Audio demasiado corto para transcripción"

        x = np.expand_dims(feat, axis=0)  # (1, frames, 80)
        logits = speech_model.predict(x, verbose=0)

        # Longitud después del stride=2 de la primera conv del modelo
        logit_len = feat.shape[0] // 2
        input_len = tf.constant([logit_len], dtype=tf.int32)

        preds = greedy_decode(logits, input_len)
        return preds[0] if preds else ""

    except Exception as e:
        return f"Error al transcribir el audio: {str(e)}"


# ============================================
# CREAR INTERFAZ CON GRADIO
# ============================================

# Crear interfaz con pestañas
with gr.Blocks(title="Clasificador y Transcriptor") as app:
    
    gr.Markdown(
        """
        # Sistema de Clasificación y Transcripción
        ### Modelos de Deep Learning para imágenes y audio
        """
    )
    
    with gr.Tabs():
        
        # ============================================
        # PESTAÑA 1: CLASIFICACIÓN DE IMÁGENES
        # ============================================
        with gr.Tab("Clasificación de Imágenes"):
            gr.Markdown(
                """
                Sube una imagen de un **gato** o **perro** para clasificarla.
                Se utilizarán dos modelos diferentes para comparar resultados.
                """
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        type="pil",
                        label="Sube una imagen",
                        height=400
                    )
                    classify_btn = gr.Button("Clasificar Imagen", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    output_text = gr.Textbox(
                        label="Resultados",
                        lines=10,
                        max_lines=15
                    )
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Distribución de probabilidades - ResNet50")
                    resnet_label = gr.Label(num_top_classes=2, label="ResNet50")
                
                with gr.Column():
                    gr.Markdown("### Distribución de probabilidades - MobileNetV2")
                    mobilenet_label = gr.Label(num_top_classes=2, label="MobileNetV2")
            
            # Ejemplos de imágenes (opcional)
            gr.Examples(
                examples=[],  # Puedes agregar rutas de ejemplo aquí
                inputs=image_input,
                label="Ejemplos"
            )
            
            # Conectar el botón con la función
            classify_btn.click(
                fn=classify_image,
                inputs=image_input,
                outputs=[output_text, resnet_label, mobilenet_label]
            )
        
        # ============================================
        # PESTAÑA 1.5: CLASIFICACIÓN DE VIDEOS
        # ============================================
        with gr.Tab("Clasificación de Videos"):
            gr.Markdown(
                """
                Sube un video para extraer frames y clasificar gatos o perros.
                Se extrae 1 frame por segundo y se genera un video anotado con los resultados.
                """
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(
                        label="Sube un video",
                        format="mp4"
                    )
                    video_btn = gr.Button("Procesar Video", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    video_report = gr.Textbox(
                        label="Análisis del Video",
                        lines=15,
                        max_lines=20
                    )
            
            with gr.Row():
                video_output = gr.Video(
                    label="Video Clasificado (Con Anotaciones)",
                    format="mp4"
                )
            
            gr.Markdown(
                """
                **Cómo funciona:**
                1. Sube un archivo de video en formato MP4 (máximo 2 minutos)
                2. El sistema extrae 1 frame por segundo
                3. Cada frame se clasifica usando ambos modelos
                4. Se genera un video con anotaciones mostrando la clase detectada y confianza
                5. Se proporciona un análisis detallado de todas las detecciones
                
                **Leyenda de colores:**
                - 🟢 Verde: Gato detectado
                - 🔵 Azul: Perro detectado
                
                **Límites:**
                - Duración máxima: 2 minutos
                - Formatos soportados: MP4, AVI, MOV
                """
            )
            
            # Conectar el botón con la función
            video_btn.click(
                fn=classify_video,
                inputs=video_input,
                outputs=[video_report, video_output]
            )
        
        # ============================================
        # PESTAÑA 2: SPEECH-TO-TEXT
        # ============================================
        with gr.Tab("Reconocimiento de Voz"):
            gr.Markdown(
                """
                Graba o sube un archivo de audio en español para transcribirlo a texto.
                """
            )
            
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="Graba o sube un audio"
                    )
                    transcribe_btn = gr.Button("Transcribir Audio", variant="primary", size="lg")
                
                with gr.Column():
                    transcription_output = gr.Textbox(
                        label="Transcripción",
                        lines=10,
                        max_lines=15,
                        placeholder="La transcripción aparecerá aquí..."
                    )
            
            gr.Markdown(
                """
                ---
                **Nota:** Esta funcionalidad requiere que primero entrenes el modelo en `parte2.ipynb` 
                y guardes el archivo `best_model.keras` en la carpeta `models/`.
                """
            )
            
            # Conectar el botón con la función
            transcribe_btn.click(
                fn=transcribe_audio,
                inputs=audio_input,
                outputs=transcription_output
            )
    
    # Pie de página
    gr.Markdown(
        """
        ---
        **Nota:** Los modelos están optimizados para imágenes de 224x224 píxeles y audios de hasta 8 segundos.
        """
    )

# ============================================
# LANZAR LA APLICACIÓN
# ============================================

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Iniciando la aplicación...")
    print("="*50 + "\n")
    
    # Verificar que estamos usando la política correcta de event loop
    if sys.platform == 'win32':
        print("✓ Usando WindowsSelectorEventLoopPolicy para evitar errores de conexión")
    
    try:
        app.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            inbrowser=True,
            quiet=False,
            show_error=True
        )
    except Exception as e:
        print(f"\n❌ Error al iniciar la aplicación: {e}")
        print("\nIntentando con configuración alternativa...")
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            inbrowser=False
        )

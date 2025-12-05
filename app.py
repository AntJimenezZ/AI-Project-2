"""
Interfaz web para clasificación de imágenes y reconocimiento de voz
Usa Gradio para crear una interfaz sencilla con dos pestañas
"""

import gradio as gr
import tensorflow as tf
import numpy as np
import librosa
from PIL import Image
import os

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
    

    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True 
    )

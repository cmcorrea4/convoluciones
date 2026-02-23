"""
Convoluciones Visualizadas — Aplicación Streamlit
Explora filtros convolucionales sobre cualquier imagen
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import io
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

# ── Configuración de la página ────────────────────────────────────────────────
st.set_page_config(
    page_title="Convoluciones Visualizadas",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Estilos CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 1rem 0 0.5rem 0;
    }
    .info-box {
        background-color: #f0f4ff;
        border-left: 4px solid #667eea;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stSelectbox label, .stSlider label { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Título principal ──────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🔬 Convoluciones Visualizadas</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Explora cómo los filtros convolucionales transforman imágenes — fundamento de las CNN</p>', unsafe_allow_html=True)

# ── Cache para el modelo VGG16 ────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo_vgg():
    modelo = VGG16(weights='imagenet', include_top=False)
    return modelo

# ── Definición de kernels ─────────────────────────────────────────────────────
KERNELS = {
    "Sobel X — Bordes verticales": {
        "kernel": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
        "descripcion": "Detecta cambios de intensidad en dirección horizontal → bordes verticales",
        "emoji": "↕️",
        "color": "#3498db"
    },
    "Sobel Y — Bordes horizontales": {
        "kernel": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
        "descripcion": "Detecta cambios de intensidad en dirección vertical → bordes horizontales",
        "emoji": "↔️",
        "color": "#2ecc71"
    },
    "Sobel combinado — Todos los bordes": {
        "kernel": "sobel_combinado",
        "descripcion": "Magnitud del gradiente en todas las direcciones",
        "emoji": "⬛",
        "color": "#e74c3c"
    },
    "Laplaciano — Bordes finos": {
        "kernel": np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32),
        "descripcion": "Segunda derivada — muy sensible a bordes y ruido",
        "emoji": "🔲",
        "color": "#9b59b6"
    },
    "Nitidez (Sharpen)": {
        "kernel": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
        "descripcion": "Aumenta el contraste local — hace la imagen más nítida",
        "emoji": "✨",
        "color": "#f1c40f"
    },
    "Desenfoque Gaussiano 5×5": {
        "kernel": cv2.getGaussianKernel(5, 1) @ cv2.getGaussianKernel(5, 1).T,
        "descripcion": "Suaviza la imagen — elimina ruido de alta frecuencia",
        "emoji": "🌫️",
        "color": "#95a5a6"
    },
    "Emboss — Relieve": {
        "kernel": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32),
        "descripcion": "Crea efecto de relieve tridimensional",
        "emoji": "🗿",
        "color": "#e67e22"
    },
    "Realce diagonal": {
        "kernel": np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]], dtype=np.float32),
        "descripcion": "Detecta bordes en dirección diagonal",
        "emoji": "↗️",
        "color": "#1abc9c"
    },
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    
    st.markdown("### 🖼️ Imagen")
    fuente_imagen = st.radio(
        "Fuente:",
        ["📤 Subir mi imagen", "🌐 URL de internet", "🎨 Imagen de prueba"],
        index=2
    )
    
    st.markdown("---")
    st.markdown("### 🎛️ Filtro")
    filtro_seleccionado = st.selectbox(
        "Selecciona un filtro:",
        list(KERNELS.keys()),
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 🔬 Feature Maps CNN")
    mostrar_cnn = st.checkbox("Mostrar feature maps de VGG16", value=False)
    
    if mostrar_cnn:
        capa_cnn = st.selectbox(
            "Capa de VGG16:",
            ["block1_conv1", "block1_conv2", "block2_conv1", "block3_conv1", "block4_conv1"],
            index=0
        )
        n_filtros_mostrar = st.slider("Número de filtros a mostrar:", 4, 32, 16, step=4)
    
    st.markdown("---")
    st.markdown("### 📐 Opciones")
    mostrar_diferencia = st.checkbox("Mostrar mapa de activaciones", value=True)
    colormap = st.selectbox("Colormap:", ["gray", "viridis", "plasma", "hot", "RdBu_r"], index=0)

# ── Carga de imagen ───────────────────────────────────────────────────────────
img_array = None

if fuente_imagen == "📤 Subir mi imagen":
    archivo = st.file_uploader(
        "Sube una imagen (JPG, PNG, WEBP):",
        type=["jpg", "jpeg", "png", "webp"],
        help="La imagen se redimensionará a 224×224 píxeles"
    )
    if archivo:
        img_pil = Image.open(archivo).convert("RGB").resize((224, 224))
        img_array = np.array(img_pil)

elif fuente_imagen == "🌐 URL de internet":
    url = st.text_input("URL de la imagen:", placeholder="https://ejemplo.com/imagen.jpg")
    if url:
        try:
            import requests
            from io import BytesIO
            resp = requests.get(url, timeout=10)
            img_pil = Image.open(BytesIO(resp.content)).convert("RGB").resize((224, 224))
            img_array = np.array(img_pil)
            st.success("✅ Imagen cargada desde URL")
        except Exception as e:
            st.error(f"❌ Error al cargar la imagen: {e}")

else:  # Imagen de prueba sintética
    # Crear imagen de prueba con formas geométricas
    img_test = np.zeros((224, 224, 3), dtype=np.uint8) + 30
    # Rectángulo
    cv2.rectangle(img_test, (20, 20), (100, 100), (200, 80, 80), -1)
    # Círculo
    cv2.circle(img_test, (160, 60), 50, (80, 200, 80), -1)
    # Triángulo
    pts = np.array([[60, 200], [160, 130], [220, 200]], np.int32)
    cv2.fillPoly(img_test, [pts], (80, 80, 200))
    # Líneas diagonales
    for i in range(0, 224, 15):
        cv2.line(img_test, (0, i), (224, i+224), (220, 220, 100), 1)
    img_array = img_test
    st.info("🎨 Usando imagen de prueba con formas geométricas (ideal para ver bordes)")

# ── Procesamiento principal ───────────────────────────────────────────────────
if img_array is not None:
    img_gris = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    # Obtener kernel y datos del filtro seleccionado
    info_filtro = KERNELS[filtro_seleccionado]
    
    if isinstance(info_filtro["kernel"], str) and info_filtro["kernel"] == "sobel_combinado":
        sx = cv2.Sobel(img_gris, cv2.CV_32F, 1, 0)
        sy = cv2.Sobel(img_gris, cv2.CV_32F, 0, 1)
        resultado = np.sqrt(sx**2 + sy**2)
        resultado = np.clip(resultado, 0, 255)
        kernel_display = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
    else:
        kernel = info_filtro["kernel"]
        resultado = cv2.filter2D(img_gris, -1, kernel)
        resultado = np.clip(resultado, 0, 255)
        kernel_display = kernel
    
    # ── Sección 1: Visualización principal ───────────────────────────────────
    st.markdown(f'<div class="section-header">① Resultado del Filtro: {info_filtro["emoji"]} {filtro_seleccionado}</div>', 
                unsafe_allow_html=True)
    
    st.markdown(f'<div class="info-box">📖 <b>¿Qué hace este filtro?</b> {info_filtro["descripcion"]}</div>', 
                unsafe_allow_html=True)
    
    # Grid de visualización
    n_cols = 4 if mostrar_diferencia else 3
    cols = st.columns(n_cols)
    
    fig1, axes1 = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    fig1.patch.set_facecolor('#f8f9fa')
    
    axes1[0].imshow(img_array)
    axes1[0].set_title('① Imagen original (RGB)', fontweight='bold', fontsize=11)
    axes1[0].axis('off')
    
    im_k = axes1[1].imshow(kernel_display, cmap='RdBu_r', aspect='auto')
    axes1[1].set_title(f'② Kernel {kernel_display.shape[0]}×{kernel_display.shape[1]}', 
                       fontweight='bold', fontsize=11)
    plt.colorbar(im_k, ax=axes1[1], fraction=0.046, pad=0.04)
    for i in range(kernel_display.shape[0]):
        for j in range(kernel_display.shape[1]):
            axes1[1].text(j, i, f'{kernel_display[i,j]:.1f}', ha='center', va='center',
                         fontsize=10, fontweight='bold',
                         color='white' if abs(kernel_display[i,j]) > 1 else 'black')
    axes1[1].set_xticks([]); axes1[1].set_yticks([])
    
    axes1[2].imshow(resultado, cmap=colormap)
    axes1[2].set_title('③ Feature Map resultante', fontweight='bold', fontsize=11)
    axes1[2].axis('off')
    
    if mostrar_diferencia:
        diff = np.abs(resultado.astype(float) - img_gris.astype(float))
        im_d = axes1[3].imshow(diff, cmap='hot')
        axes1[3].set_title('④ Zonas activadas\n(diferencia)', fontweight='bold', fontsize=11)
        axes1[3].axis('off')
        plt.colorbar(im_d, ax=axes1[3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)
    
    # ── Métricas ──────────────────────────────────────────────────────────────
    st.markdown("---")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    activacion_media = float(np.mean(resultado))
    activacion_max = float(np.max(resultado))
    zonas_activas = float(np.mean(resultado > np.mean(resultado)) * 100)
    contraste = float(np.std(resultado))
    
    with col_m1:
        st.metric("Activación media", f"{activacion_media:.1f}", help="Valor medio del feature map")
    with col_m2:
        st.metric("Activación máxima", f"{activacion_max:.1f}", help="Pico de activación")
    with col_m3:
        st.metric("Zonas activas", f"{zonas_activas:.1f}%", help="% de píxeles sobre la media")
    with col_m4:
        st.metric("Contraste (std)", f"{contraste:.1f}", help="Desviación estándar del feature map")
    
    # ── Sección 2: Comparación de todos los filtros ───────────────────────────
    st.markdown('<div class="section-header">② Comparativa — Todos los filtros a la vez</div>', 
                unsafe_allow_html=True)
    
    fig2, axes2 = plt.subplots(2, 4, figsize=(18, 9))
    fig2.patch.set_facecolor('#f8f9fa')
    fig2.suptitle('Todos los filtros sobre la misma imagen', fontsize=14, fontweight='bold')
    
    nombres_filtros = list(KERNELS.keys())
    for idx in range(8):
        ax = axes2[idx // 4][idx % 4]
        nombre = nombres_filtros[idx]
        info = KERNELS[nombre]
        
        if isinstance(info["kernel"], str) and info["kernel"] == "sobel_combinado":
            sx = cv2.Sobel(img_gris, cv2.CV_32F, 1, 0)
            sy = cv2.Sobel(img_gris, cv2.CV_32F, 0, 1)
            fm = np.clip(np.sqrt(sx**2 + sy**2), 0, 255)
        else:
            fm = np.clip(cv2.filter2D(img_gris, -1, info["kernel"]), 0, 255)
        
        ax.imshow(fm, cmap='gray')
        titulo = nombre.split("—")[0].strip()
        if len(titulo) > 20:
            titulo = titulo[:20] + "..."
        ax.set_title(f'{info["emoji"]} {titulo}', fontsize=9, fontweight='bold')
        ax.axis('off')
        
        # Borde de color
        for spine in ax.spines.values():
            spine.set_edgecolor(info["color"])
            spine.set_linewidth(3)
            spine.set_visible(True)
    
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)
    
    # ── Sección 3: Feature Maps CNN ───────────────────────────────────────────
    if mostrar_cnn:
        st.markdown('<div class="section-header">③ Feature Maps de VGG16 — Lo que aprende una CNN real</div>', 
                    unsafe_allow_html=True)
        
        with st.spinner("⏳ Cargando VGG16 y calculando activaciones..."):
            try:
                modelo = cargar_modelo_vgg()
                capa = modelo.get_layer(capa_cnn)
                modelo_vis = Model(inputs=modelo.input, outputs=capa.output)
                
                img_prep = tf.keras.applications.vgg16.preprocess_input(
                    np.expand_dims(img_array.astype(np.float32), axis=0)
                )
                activaciones = modelo_vis.predict(img_prep, verbose=0)
                
                n_mostrar = min(n_filtros_mostrar, activaciones.shape[-1])
                cols_grid = 8
                rows_grid = (n_mostrar + cols_grid - 1) // cols_grid
                
                fig3, axes3 = plt.subplots(rows_grid, cols_grid, figsize=(18, rows_grid * 2.2))
                fig3.patch.set_facecolor('#1a1a2e')
                fig3.suptitle(
                    f'Feature Maps — Capa VGG16: {capa_cnn} | Shape: {activaciones.shape} | Mostrando {n_mostrar}/{activaciones.shape[-1]} filtros',
                    fontsize=12, fontweight='bold', color='white'
                )
                
                axes_flat = axes3.flatten() if rows_grid > 1 else [axes3] if rows_grid == 1 and cols_grid == 1 else axes3.flatten()
                
                for i in range(n_mostrar):
                    fm = activaciones[0, :, :, i]
                    axes_flat[i].imshow(fm, cmap='viridis')
                    axes_flat[i].set_title(f'F{i+1}', fontsize=7, color='white')
                    axes_flat[i].axis('off')
                
                for i in range(n_mostrar, len(axes_flat)):
                    axes_flat[i].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig3)
                plt.close(fig3)
                
                profundidad = int(capa_cnn.split("block")[1][0])
                descripciones = {
                    1: "detecta **bordes, gradientes y colores básicos** — similar a Sobel pero aprendido automáticamente",
                    2: "combina bordes para detectar **texturas y patrones simples**",
                    3: "reconoce **formas más complejas** y partes de objetos",
                    4: "captura representaciones de **alto nivel** — partes semánticas de objetos"
                }
                st.info(f"💡 **Capa {capa_cnn}** (profundidad {profundidad}): {descripciones.get(profundidad, 'características abstractas de alto nivel')}")
                
            except Exception as e:
                st.error(f"❌ Error al cargar VGG16: {e}")
                st.info("💡 Asegúrate de tener instalado tensorflow: `pip install tensorflow`")
    
    # ── Sección 4: Teoría ─────────────────────────────────────────────────────
    with st.expander("📚 ¿Cómo funciona una convolución? — Teoría rápida"):
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("""
            ### El proceso paso a paso
            1. **El kernel** (matriz pequeña: 3×3, 5×5...) se desliza sobre la imagen
            2. En cada posición se calcula el **producto punto** entre el kernel y el parche de imagen
            3. El resultado forma el **feature map** (mapa de características)
            4. Múltiples kernels → múltiples feature maps en paralelo
            
            ### Parámetros clave
            - **Stride**: cuántos píxeles avanza el kernel
            - **Padding**: relleno para preservar dimensiones
            - **Profundidad**: cuántos filtros aplica la capa
            """)
        with col_t2:
            st.markdown("""
            ### Lo que aprende una CNN
            | Capa | Detecta |
            |------|---------|
            | 1 | Bordes, colores |
            | 2 | Texturas, patrones |
            | 3 | Formas, partes |
            | 4+ | Conceptos abstractos |
            
            ### ¿Por qué es poderoso?
            Los filtros **se aprenden automáticamente** durante el entrenamiento con backpropagation — 
            no los diseña ningún humano. Las capas profundas aprenden representaciones que serían 
            imposibles de diseñar a mano.
            """)

else:
    # Pantalla de bienvenida
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 12px; border: 2px dashed #ccc;">
            <h2>👆 Carga una imagen para comenzar</h2>
            <p style="color: #666; font-size: 1.1rem;">
                Usa el panel lateral para:<br>
                • Subir tu propia imagen<br>
                • Pegar una URL<br>
                • Usar la imagen de prueba
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Preview de qué hace la app
    st.markdown("### 🎯 ¿Qué puedes explorar?")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        **🔲 Filtros clásicos**
        Sobel, Laplaciano, Gaussian blur, Emboss y más — diseñados por expertos
        """)
    with col_b:
        st.markdown("""
        **📊 Comparativa visual**
        Los 8 filtros aplicados a tu imagen al mismo tiempo
        """)
    with col_c:
        st.markdown("""
        **🤖 Feature Maps VGG16**
        Lo que ve una CNN entrenada en 1.2M de imágenes
        """)

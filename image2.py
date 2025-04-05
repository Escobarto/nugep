import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import plotly.graph_objects as go
from skimage import feature, color, exposure
import pandas as pd
import matplotlib.pyplot as plt
import base64
import json
import datetime
import os
import uuid
from scipy import ndimage
import plotly.express as px
from io import StringIO, BytesIO


st.set_page_config(
    page_title="HoloArt Analyzer",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def add_custom_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .card {
        border-radius: 10px;
        padding: 1.5rem;
        background-color: #F8FAFC;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #475569;
    }
    .highlight {
        font-weight: bold;
        color: #1E40AF;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #64748B;
        font-size: 0.8rem;
    }
    .status-good {
        color: #10B981;
        font-weight: bold;
    }
    .status-warning {
        color: #F59E0B;
        font-weight: bold;
    }
    .status-critical {
        color: #EF4444;
        font-weight: bold;
    }
    .logo-text {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E40AF;
        margin-bottom: 0;
    }
    .tab-content {
        padding: 1rem 0;
    }
    .btn-primary {
        background-color: #2563EB;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        text-align: center;
        margin: 0.5rem 0;
        cursor: pointer;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F1F5F9;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #DBEAFE;
    }
    </style>
    """, unsafe_allow_html=True)

add_custom_css()

if 'history' not in st.session_state:
    st.session_state.history = []
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'comparison_image' not in st.session_state:
    st.session_state.comparison_image = None
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None


def create_holographic_effect(image):
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
   
    gray_filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    sobelx = cv2.Sobel(gray_filtered, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray_filtered, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    gaussian1 = cv2.GaussianBlur(gray_filtered, (5, 5), 0)
    gaussian2 = cv2.GaussianBlur(gray_filtered, (9, 9), 0)
    dog = gaussian1 - gaussian2
    
    combined = magnitude * 0.7 + dog * 0.3
    
    depth_map = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX)
    depth_map = np.uint8(depth_map)
    depth_map = cv2.equalizeHist(depth_map)
    
    depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)
    
    return depth_map


def image_to_3d(image, depth_map):
   
    height, width = image.shape[:2]
    scale = min(1.0, 500 / max(height, width))
    new_height, new_width = int(height * scale), int(width * scale)
    
    img_resized = cv2.resize(image, (new_width, new_height))
    depth_resized = cv2.resize(depth_map, (new_width, new_height))
    
    
    x = np.arange(0, new_width, 1)
    y = np.arange(0, new_height, 1)
    x_mesh, y_mesh = np.meshgrid(x, y)
    
   
    z_mesh = depth_resized / 15.0 
    
    img_rgb = img_resized
    if len(img_resized.shape) == 2:
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    colors = np.zeros((new_height, new_width, 3))
    colors[:, :, 0] = r / 255.0
    colors[:, :, 1] = g / 255.0
    colors[:, :, 2] = b / 255.0
    
    return x_mesh, y_mesh, z_mesh, colors


def analyze_artwork(image):
    results = {}
    
    
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = feature.canny(gray, sigma=2)
    edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
    results["edge_quality"] = edge_density
    
   
    gray_norm = gray / 255.0
    glcm = feature.graycomatrix(
        (gray_norm * 255).astype(np.uint8), 
        distances=[1], 
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
        levels=256,
        symmetric=True, 
        normed=True
    )
    contrast = feature.graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = feature.graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = feature.graycoprops(glcm, 'homogeneity')[0, 0]
    
    results["textura_contraste"] = contrast
    results["textura_dissimilaridade"] = dissimilarity
    results["textura_homogeneidade"] = homogeneity
    
    
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist_hue = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    hist_sat = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    
    results["cor_dominante_h"] = np.argmax(hist_hue)
    results["saturacao_media"] = np.mean(hsv[:, :, 1])
    results["brilho_medio"] = np.mean(hsv[:, :, 2])
    
   
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]
    results["variacao_luminosidade"] = np.std(l_channel)
    
    
    yellowing = np.mean(image[:, :, 2]) / np.mean(image[:, :, 0])
    results["indice_amarelamento"] = yellowing
    
    
    kernel = np.ones((3,3), np.uint8)
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    _, binary = cv2.threshold(gradient, 10, 255, cv2.THRESH_BINARY)
    crack_density = np.sum(binary) / (binary.shape[0] * binary.shape[1])
    results["densidade_rachaduras"] = crack_density
    
   
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    direction = np.arctan2(sobely, sobelx) * 180 / np.pi
    direction_hist = np.histogram(direction, bins=18, range=(-180, 180))[0]
    direction_entropy = -np.sum((direction_hist/np.sum(direction_hist)) * 
                               np.log2(direction_hist/np.sum(direction_hist) + 1e-10))
    results["entropia_pinceladas"] = direction_entropy
    
   
    analysis_results = {
        "Qualidade de Bordas": f"{edge_density:.4f}",
        "Contraste de Textura": f"{contrast:.4f}",
        "Homogeneidade": f"{homogeneity:.4f}",
        "Saturação Média": f"{results['saturacao_media']:.2f}",
        "Variação de Luminosidade": f"{results['variacao_luminosidade']:.2f}",
        "Índice de Amarelamento": f"{results['indice_amarelamento']:.2f}",
        "Densidade de Rachaduras": f"{crack_density:.4f}",
        "Entropia de Pinceladas": f"{direction_entropy:.2f}"
    }
    

    interpretation = {}
    
    if edge_density > 0.2:
        interpretation["Autenticidade"] = "Alta possibilidade de ser autêntica (bordas bem definidas)"
    else:
        interpretation["Autenticidade"] = "Requer análise adicional (bordas menos definidas)"
    
    if results['variacao_luminosidade'] > 50:
        interpretation["Estado de Conservação"] = "Possíveis danos ou restaurações detectados"
    else:
        interpretation["Estado de Conservação"] = "Bom estado de conservação"
    
    if yellowing > 1.4:
        interpretation["Idade Estimada"] = "Possivelmente envelhecida (alto índice de amarelamento)"
    else:
        interpretation["Idade Estimada"] = "Sem sinais claros de envelhecimento"
    
    if crack_density > 0.1:
        interpretation["Integridade Física"] = "Presença de rachaduras ou craquelê detectada"
    else:
        interpretation["Integridade Física"] = "Superfície íntegra, sem rachaduras significativas"
    
    if direction_entropy > 4.0:
        interpretation["Estilo Artístico"] = "Padrões de pincelada complexos e variados (possível impressionismo)"
    else:
        interpretation["Estilo Artístico"] = "Padrões de pincelada mais uniformes e estruturados"
    
    return analysis_results, interpretation, binary, direction

def generate_results_chart(results):
    df = pd.DataFrame({
        'Métrica': list(results.keys()),
        'Valor': [float(v.replace(',', '.')) for v in results.values()]
    })
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(df['Métrica'], df['Valor'], color='steelblue')
    ax.set_xlabel('Valor')
    ax.set_title('Métricas de Análise da Obra')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.01,
            bar.get_y() + bar.get_height()/2,
            f'{width:.3f}',
            ha='left',
            va='center'
        )
    
    plt.tight_layout()
    return fig


def compare_artworks(image1, image2):
    results = {}
    
    
    if image1.shape[:2] != image2.shape[:2]:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    
   
    gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    
    
    from skimage.metrics import structural_similarity as ssim
    ssim_index, ssim_map = ssim(gray1, gray2, full=True)
    results["SSIM"] = ssim_index
    
   
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    hist1_norm = hist1 / np.sum(hist1)
    hist2_norm = hist2 / np.sum(hist2)
    hist_diff = np.sum(np.abs(hist1_norm - hist2_norm)) / 2  
    results["histograma_diff"] = hist_diff
    
   
    edges1 = feature.canny(gray1, sigma=2)
    edges2 = feature.canny(gray2, sigma=2)
    edge_diff = np.sum(np.abs(edges1.astype(int) - edges2.astype(int))) / (edges1.shape[0] * edges1.shape[1])
    results["borda_diff"] = edge_diff
    
   
    glcm1 = feature.graycomatrix(gray1, [1], [0], 256, symmetric=True, normed=True)
    glcm2 = feature.graycomatrix(gray2, [1], [0], 256, symmetric=True, normed=True)
    contrast1 = feature.graycoprops(glcm1, 'contrast')[0, 0]
    contrast2 = feature.graycoprops(glcm2, 'contrast')[0, 0]
    results["contraste_diff"] = abs(contrast1 - contrast2) / max(contrast1, contrast2)
    
    
    diff_map = np.abs(gray1.astype(np.float32) - gray2.astype(np.float32))
    diff_map = diff_map / np.max(diff_map) * 255 
    diff_map = diff_map.astype(np.uint8)
    diff_map_color = cv2.applyColorMap(diff_map, cv2.COLORMAP_JET)
    
    
    results["diferenca_media_percentual"] = np.mean(diff_map) / 255 * 100
    
    
    interpretation = {}
    
    if results["SSIM"] > 0.9:
        interpretation["Semelhança Estrutural"] = "Alta semelhança estrutural entre as imagens"
    elif results["SSIM"] > 0.8:
        interpretation["Semelhança Estrutural"] = "Semelhança estrutural moderada"
    else:
        interpretation["Semelhança Estrutural"] = "Baixa semelhança estrutural"
    
    if results["diferenca_media_percentual"] < 5:
        interpretation["Diferença Global"] = "Diferenças muito sutis"
    elif results["diferenca_media_percentual"] < 15:
        interpretation["Diferença Global"] = "Diferenças moderadas"
    else:
        interpretation["Diferença Global"] = "Diferenças significativas"
    
    if results["borda_diff"] < 0.1:
        interpretation["Contornos e Detalhes"] = "Contornos e detalhes muito semelhantes"
    elif results["borda_diff"] < 0.3:
        interpretation["Contornos e Detalhes"] = "Algumas diferenças em contornos e detalhes"
    else:
        interpretation["Contornos e Detalhes"] = "Contornos e detalhes significativamente diferentes"
    
    return results, interpretation, diff_map_color, ssim_map


def generate_report_pdf(image, depth_map, analysis_results, interpretation):
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RPImage, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    
   
    buffer = BytesIO()
    
   
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    styleH = styles['Heading1']
    styleH2 = styles['Heading2']
    styleN = styles['Normal']
    
    
    story = []
    
   
    story.append(Paragraph("Relatório de Análise Holográfica", styleH))
    story.append(Spacer(1, 12))
    
   
    story.append(Paragraph(f"Data: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}", styleN))
    story.append(Spacer(1, 12))
    
   
    pil_img = Image.fromarray(image)
    img_path = "temp_original.png"
    pil_img.save(img_path)
    
    pil_depth = Image.fromarray(depth_map)
    depth_path = "temp_depth.png"
    pil_depth.save(depth_path)
    
   
    story.append(Paragraph("Imagem Original", styleH2))
    story.append(RPImage(img_path, width=4*inch, height=3*inch))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("Mapa de Profundidade Holográfico", styleH2))
    story.append(RPImage(depth_path, width=4*inch, height=3*inch))
    story.append(Spacer(1, 24))
    
    
    story.append(Paragraph("Resultados da Análise", styleH2))
    story.append(Spacer(1, 12))
    
    
    data = [["Métrica", "Valor"]]
    for key, value in analysis_results.items():
        data.append([key, value])
    
    table = Table(data, colWidths=[3*inch, 2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(table)
    story.append(Spacer(1, 24))
    
    
    story.append(Paragraph("Interpretação dos Resultados", styleH2))
    story.append(Spacer(1, 12))
    
    for key, value in interpretation.items():
        story.append(Paragraph(f"<b>{key}:</b> {value}", styleN))
        story.append(Spacer(1, 6))
    
    
    doc.build(story)
    
    
    os.remove(img_path)
    os.remove(depth_path)
    
    buffer.seek(0)
    return buffer


def login(username, password):
    
    valid_users = {
        "admin": "password123",
        "curador": "museu2025",
        "restaurador": "arte1234"
    }
    
    if username in valid_users and valid_users[username] == password:
        st.session_state.logged_in = True
        st.session_state.username = username
        return True
    return False


def save_to_history(image, results, interpretation):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "timestamp": timestamp,
        "results": results,
        "interpretation": interpretation,
        "thumbnail": image_to_base64(image, size=(100, 100))
    }
    st.session_state.history.append(entry)


def image_to_base64(image, size=None):
    if size:
        
        pil_img = Image.fromarray(image)
        pil_img.thumbnail(size)
        img_arr = np.array(pil_img)
    else:
        img_arr = image
        
    
    is_success, buffer = cv2.imencode(".jpg", img_arr)
    io_buf = BytesIO(buffer)
    encoded_img = base64.b64encode(io_buf.getvalue()).decode("utf-8")
    return encoded_img


def show_login_screen():
    st.markdown("<h2 class='sub-header'>Login do Sistema</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>Acesso ao HoloArt Analyzer</h3>", unsafe_allow_html=True)
        username = st.text_input("Nome de Usuário")
        password = st.text_input("Senha", type="password")
        login_button = st.button("Entrar")
        
        if login_button:
            if login(username, password):
                st.success("Login realizado com sucesso!")
                st.rerun()
            else:
                st.error("Nome de usuário ou senha incorretos!")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>Informações</h3>", unsafe_allow_html=True)
        st.markdown("""
        Para demonstração, utilize as seguintes credenciais:
        
        **Usuários:**
        - admin / password123
        - curador / museu2025
        - restaurador / arte1234
        
        O sistema completo inclui gerenciamento de permissões para diferentes 
        tipos de usuários e acesso a funcionalidades específicas.
        """)
        st.markdown("</div>", unsafe_allow_html=True)


def show_history():
    st.markdown("<h2 class='sub-header'>Histórico de Análises</h2>", unsafe_allow_html=True)
    
    if not st.session_state.history:
        st.info("Nenhuma análise foi realizada ainda.")
        return
    
    for i, entry in enumerate(st.session_state.history):
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown(f"<img src='data:image/jpeg;base64,{entry['thumbnail']}' width='100'>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"<h3>Análise {i+1} - {entry['timestamp']}</h3>", unsafe_allow_html=True)
            
            with st.expander("Ver detalhes"):
                st.markdown("<h4>Resultados</h4>", unsafe_allow_html=True)
                for key, value in entry['results'].items():
                    st.markdown(f"**{key}:** {value}")
                
                st.markdown("<h4>Interpretação</h4>", unsafe_allow_html=True)
                for key, value in entry['interpretation'].items():
                    st.markdown(f"**{key}:** {value}")


def show_comparison(original_image):
    st.markdown("<h2 class='sub-header'>Comparação de Obras</h2>", unsafe_allow_html=True)
    
    uploaded_comparison = st.file_uploader("Carregar Obra para Comparação", type=["jpg", "jpeg", "png"])
    
    if uploaded_comparison:
        comparison_bytes = uploaded_comparison.read()
        comparison_pil = Image.open(io.BytesIO(comparison_bytes))
        comparison_array = np.array(comparison_pil)
        
        if len(comparison_array.shape) == 2:
            comparison_array = cv2.cvtColor(comparison_array, cv2.COLOR_GRAY2RGB)
        elif comparison_array.shape[2] == 4:
            comparison_array = comparison_array[:, :, :3]
        
        st.session_state.comparison_image = comparison_array
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("<h3>Obra Original</h3>", unsafe_allow_html=True)
            st.image(original_image, use_column_width=True)
        
        with col2:
            st.markdown("<h3>Obra para Comparação</h3>", unsafe_allow_html=True)
            st.image(comparison_array, use_column_width=True)
        
        if st.button("Realizar Análise Comparativa"):
            results, interpretation, diff_map, ssim_map = compare_artworks(original_image, comparison_array)
            st.session_state.comparison_results = {
                "results": results,
                "interpretation": interpretation,
                "diff_map": diff_map,
                "ssim_map": ssim_map
            }
    
    if st.session_state.comparison_results:
        st.markdown("<h3 class='sub-header'>Resultados da Comparação</h3>", unsafe_allow_html=True)
        
        results = st.session_state.comparison_results["results"]
        interpretation = st.session_state.comparison_results["interpretation"]
        diff_map = st.session_state.comparison_results["diff_map"]
        ssim_map = st.session_state.comparison_results["ssim_map"]
        
        col3, col4, col5 = st.columns([1, 1, 1])
        
        with col3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h4>Métricas de Semelhança</h4>", unsafe_allow_html=True)
            st.markdown(f"**Índice SSIM:** {results['SSIM']:.4f}")
            st.markdown(f"**Diferença de Histograma:** {results['histograma_diff']:.4f}")
            st.markdown(f"**Diferença de Bordas:** {results['borda_diff']:.4f}")
            st.markdown(f"**Diferença de Contraste:** {results['contraste_diff']:.4f}")
            st.markdown(f"**Diferença Média (%):** {results['diferenca_media_percentual']:.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col4:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h4>Interpretação</h4>", unsafe_allow_html=True)
            for key, value in interpretation.items():
                st.markdown(f"**{key}:** {value}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col5:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h4>Conclusão</h4>", unsafe_allow_html=True)
            
            if results['SSIM'] > 0.9 and results['diferenca_media_percentual'] < 5:
                st.markdown("<p class='status-good'>As obras são extremamente similares, possivelmente cópias ou versões.</p>", unsafe_allow_html=True)
            elif results['SSIM'] > 0.8 and results['diferenca_media_percentual'] < 15:
                st.markdown("<p class='status-warning'>As obras têm similaridades substanciais, possivelmente do mesmo autor ou escola.</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p class='status-critical'>As obras apresentam diferenças significativas, provavelmente de autores ou períodos diferentes.</p>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        col6, col7 = st.columns([1, 1])
        
        with col6:
            st.markdown("<h4>Mapa de Diferenças</h4>", unsafe_allow_html=True)
            st.image(diff_map, caption="Áreas com maior diferença aparecem em vermelho/amarelo")
        
        with col7:
            st.markdown("<h4>Mapa SSIM</h4>", unsafe_allow_html=True)
            ssim_display = (ssim_map * 255).astype(np.uint8)
            st.image(ssim_display, caption="Áreas com maior similaridade aparecem mais claras")


st.markdown("<h1 class='main-header'>HoloArt Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='info-text'>Sistema avançado de análise holográfica para acervos museológicos</p>", unsafe_allow_html=True)


if not st.session_state.logged_in:
    show_login_screen()
else:
   
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3750/3750176.png", width=100)
        st.markdown(f"<p>Bem-vindo, <b>{st.session_state.username}</b></p>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Controles</h2>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Carregar Obra de Arte", type=["jpg", "jpeg", "png"])
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>Funcionalidades</h3>", unsafe_allow_html=True)
        st.markdown("""
        - Visualização 3D holográfica
        - Detecção de falsificações
        - Análise de conservação
        - Identificação de estilos
        - Mapeamento de danos
        - Comparação de obras
        - Exportação de relatórios
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("Sair"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()

    
    if uploaded_file is not None:
       
        image_bytes = uploaded_file.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(pil_image)
        
        if len(image_array.shape) == 2:
           
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif image_array.shape[2] == 4:
           
            image_array = image_array[:, :, :3]
        
        depth_map = create_holographic_effect(image_array)
        
       
        tabs = st.tabs(["Análise Holográfica", "Detalhes Técnicos", "Comparação", "Relatório", "Histórico"])
        
       
        with tabs[0]:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("<h2 class='sub-header'>Imagem Original</h2>", unsafe_allow_html=True)
                st.image(image_array, caption="Obra de Arte Original", use_column_width=True)
                
                st.markdown("<h2 class='sub-header'>Mapa de Profundidade</h2>", unsafe_allow_html=True)
                st.image(depth_map, caption="Mapa de Profundidade para Holografia", use_column_width=True)
            
            with col2:
                st.markdown("<h2 class='sub-header'>Representação Holográfica 3D</h2>", unsafe_allow_html=True)
                
               
                x_mesh, y_mesh, z_mesh, colors = image_to_3d(image_array, depth_map)
                
                
                fig = go.Figure(data=[go.Surface(
                    z=z_mesh,
                    surfacecolor=colors,
                    colorscale=[[0, 'rgb(0,0,0)'], [1, 'rgb(255,255,255)']],
                    showscale=False,
                    opacity=0.9,
                    cmin=0,
                    cmax=1
                )])
                
                fig.update_layout(
                    title="Visualização Holográfica Interativa",
                    scene=dict(
                        xaxis_title="",
                        yaxis_title="",
                        zaxis_title="Profundidade",
                        aspectmode='manual',
                        aspectratio=dict(x=1, y=1, z=0.5),
                        xaxis=dict(showticklabels=False),
                        yaxis=dict(showticklabels=False),
                        zaxis=dict(showticklabels=False),
                    ),
                    width=600,
                    height=500,
                    margin=dict(l=0, r=0, b=0, t=30)
                )
                
                st.plotly_chart(fig)
                
              
                st.markdown("<h3>Controles de Visualização</h3>", unsafe_allow_html=True)
                col_ctrl1, col_ctrl2 = st.columns([1, 1])
                
                with col_ctrl1:
                    st.button("Rotação Automática", key="auto_rotate")
                    st.button("Destacar Textura", key="highlight_texture")
                
                with col_ctrl2:
                    st.button("Vista Superior", key="top_view")
                    st.button("Ampliar Detalhes", key="zoom_details")
            
           
            st.markdown("<h2 class='sub-header'>Análise da Obra</h2>", unsafe_allow_html=True)
            
            col3, col4 = st.columns([1, 1])
            
            with col3:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                analysis_results, interpretation, crack_map, direction_map = analyze_artwork(image_array)
                
                st.markdown("<h3>Métricas Extraídas</h3>", unsafe_allow_html=True)
                for key, value in analysis_results.items():
                    st.markdown(f"<p><span class='highlight'>{key}:</span> {value}</p>", unsafe_allow_html=True)
                
                st.markdown("<h3>Interpretação dos Resultados</h3>", unsafe_allow_html=True)
                for key, value in interpretation.items():
                    st.markdown(f"<p><span class='highlight'>{key}:</span> {value}</p>", unsafe_allow_html=True)
                
                
                if st.button("Salvar Análise"):
                    save_to_history(image_array, analysis_results, interpretation)
                    st.success("Análise salva no histórico!")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col4:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                chart = generate_results_chart(analysis_results)
                st.pyplot(chart)
                st.markdown("</div>", unsafe_allow_html=True)
            
            
            st.markdown("<h2 class='sub-header'>Recomendações</h2>", unsafe_allow_html=True)
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            
           
            if float(analysis_results["Variação de Luminosidade"].replace(",", ".")) > 40:
                st.markdown("""
                - <span class='status-warning'>**Conservação Prioritária**</span>: A obra apresenta variações significativas de luminosidade, indicando possíveis danos estruturais ou deterioração.
                - Recomenda-se análise por especialista em restauração para avaliação detalhada.
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                - <span class='status-good'>**Conservação Preventiva**</span>: A obra parece estar em bom estado, mas se beneficiaria de condições controladas de umidade e temperatura.
                """, unsafe_allow_html=True)
            
            if float(analysis_results["Índice de Amarelamento"].replace(",", ".")) > 1.3:
                st.markdown("""
                - <span class='status-warning'>**Proteção UV**</span>: Recomenda-se exposição com filtros UV para prevenir amarelamento adicional.
                - Considerar digitalização de alta resolução para preservação digital.
                """, unsafe_allow_html=True)
            
            if float(analysis_results["Densidade de Rachaduras"].replace(",", ".")) > 0.05:
                st.markdown("""
                - <span class='status-critical'>**Restauração Recomendada**</span>: A análise detectou possíveis rachaduras ou craquelê que podem requerer intervenção.
                - Minimizar manipulação física da obra até avaliação especializada.
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        
        with tabs[1]:
            st.markdown("<h2 class='sub-header'>Detalhes Técnicos da Análise</h2>", unsafe_allow_html=True)
            
            col_tech1, col_tech2 = st.columns([1, 1])
            
            with col_tech1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h3>Detecção de Rachaduras</h3>", unsafe_allow_html=True)
                st.image(crack_map, caption="Mapa de Rachaduras e Craquelê", use_column_width=True)
                st.markdown("""
                A detecção de rachaduras utiliza técnicas de morfologia matemática e realce 
                de gradientes para identificar padrões de craquelê e fissuras na superfície da obra.
                """)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col_tech2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h3>Análise de Pinceladas</h3>", unsafe_allow_html=True)
                
                
                direction_colored = cv2.applyColorMap(
                    cv2.normalize(direction_map.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                    cv2.COLORMAP_HSV
                )
                st.image(direction_colored, caption="Mapa de Direção de Pinceladas", use_column_width=True)
                
                st.markdown("""
                A análise de pinceladas detecta padrões e direções predominantes nos traços, 
                informação importante para identificação de autoria e estilo artístico.
                """)
                st.markdown("</div>", unsafe_allow_html=True)
            
            
            st.markdown("<h3 class='sub-header'>Análise Espectral</h3>", unsafe_allow_html=True)
            
            col_spec1, col_spec2 = st.columns([1, 1])
            
            with col_spec1:
                
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if len(image_array.shape) == 3 else image_array
                f_transform = np.fft.fft2(gray)
                f_shift = np.fft.fftshift(f_transform)
                magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
                
               
                magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                st.image(magnitude_spectrum, caption="Espectro de Frequência (FFT)", use_column_width=True)
            
            with col_spec2:
               
                hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
                hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
                hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
                hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
                
               
                fig, ax = plt.subplots(3, 1, figsize=(10, 8))
                ax[0].plot(hist_h, color='red')
                ax[0].set_title('Distribuição de Matiz (H)')
                ax[1].plot(hist_s, color='green')
                ax[1].set_title('Distribuição de Saturação (S)')
                ax[2].plot(hist_v, color='blue')
                ax[2].set_title('Distribuição de Valor (V)')
                plt.tight_layout()
                
                st.pyplot(fig)
            
            
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3>Notas Técnicas</h3>", unsafe_allow_html=True)
            st.markdown("""
            - **Reconstrução Holográfica**: A transformação de imagens 2D em representações 3D é baseada na extração de informações de gradiente e textura.
            - **Identificação de Autenticidade**: A análise de padrões microscópicos, como pinceladas e composição espectral, pode auxiliar na autenticação.
            - **Limitações**: Esta análise digital não substitui exames físico-químicos completos para autenticação definitiva.
            - **Precisão do Mapeamento 3D**: A precisão do relevo depende da qualidade da imagem original e de suas características de iluminação e contraste.
            """)
            st.markdown("</div>", unsafe_allow_html=True)
        
        
        with tabs[2]:
            show_comparison(image_array)
        
       
        with tabs[3]:
            st.markdown("<h2 class='sub-header'>Geração de Relatório</h2>", unsafe_allow_html=True)
            
            col_report1, col_report2 = st.columns([1, 2])
            
            with col_report1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h3>Opções de Relatório</h3>", unsafe_allow_html=True)
                
                include_original = st.checkbox("Incluir imagem original", value=True)
                include_3d = st.checkbox("Incluir visualização 3D", value=True)
                include_analysis = st.checkbox("Incluir análise detalhada", value=True)
                include_recommendations = st.checkbox("Incluir recomendações", value=True)
                
                report_format = st.radio("Formato do relatório", ["PDF", "HTML"])
                
                if st.button("Gerar Relatório"):
                   
                    pdf_buffer = generate_report_pdf(image_array, depth_map, analysis_results, interpretation)
                    
                    
                    st.download_button(
                        label="Baixar Relatório PDF",
                        data=pdf_buffer,
                        file_name=f"relatorio_holografico_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col_report2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h3>Prévia do Relatório</h3>", unsafe_allow_html=True)
                
                st.markdown(f"""
                # Relatório de Análise Holográfica
                
                **Data:** {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}
                
                ## Informações Gerais
                - **Técnica de Análise:** Reconstrução Holográfica Digital
                - **Resolução da Imagem:** {image_array.shape[1]}x{image_array.shape[0]}
                
                ## Resultados Principais
                - **Índice de Autenticidade:** {analysis_results['Qualidade de Bordas']}
                - **Estado de Conservação:** {interpretation['Estado de Conservação']}
                - **Possível Período:** {interpretation['Idade Estimada']}
                
                ## Detalhes da Análise
                Foram identificadas características específicas relacionadas à técnica, 
                materiais e estado de conservação da obra, conforme métricas apresentadas 
                na análise completa.
                
                ## Recomendações
                Baseado na análise digital realizada, recomenda-se atenção aos 
                aspectos de conservação identificados e possivelmente análises 
                complementares para confirmação de autenticidade.
                """)
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        
        with tabs[4]:
            show_history()
        
    else:
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("""
        ## Bem-vindo ao HoloArt Analyzer!
        
        Este sistema de análise holográfica permite examinar obras de arte em detalhes sem precedentes através de:
        
        - **Transformação 3D**: Visualize cada detalhe da superfície da obra
        - **Análise de Autenticidade**: Identifique padrões consistentes com falsificações
        - **Avaliação de Conservação**: Detecte danos microscópicos e necessidades de restauração
        - **Identificação de Técnicas**: Reconheça padrões de pinceladas e técnicas de diferentes artistas
        
        Para começar, carregue uma imagem usando o painel à esquerda.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
       
        st.markdown("<h2 class='sub-header'>Exemplos de Análise</h2>", unsafe_allow_html=True)
        
        col_ex1, col_ex2, col_ex3 = st.columns([1, 1, 1])
        
        with col_ex1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/300px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg")
            st.markdown("""
            **Análise de Técnica de Pinceladas**
            
            A análise holográfica revela padrões característicos de Van Gogh, com texturas tridimensionais distintas.
            """)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_ex2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/Mona_Lisa.jpg/250px-Mona_Lisa.jpg")
            st.markdown("""
            **Detecção de Restaurações**
            
            A visualização 3D destaca áreas de intervenção e restauração invisíveis ao olho nu.
            """)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_ex3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Michelangelo_-_Creation_of_Adam_%28cropped%29.jpg/320px-Michelangelo_-_Creation_of_Adam_%28cropped%29.jpg")
            st.markdown("""
            **Análise de Deterioração**
            
            O mapeamento 3D revela microfissuras e áreas de possível desprendimento de tinta.
            """)
            st.markdown("</div>", unsafe_allow_html=True)
        
        
        st.markdown("<h2 class='sub-header'>Recursos do Sistema</h2>", unsafe_allow_html=True)
        
        col_feat1, col_feat2, col_feat3, col_feat4 = st.columns([1, 1, 1, 1])
        
        with col_feat1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("""
            ### 🔍 Análise Visual Avançada
            
            - Detecção de padrões microscópicos
            - Identificação de técnicas de pintura
            - Análise espectral e cromática
            """)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_feat2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("""
            ### 🖌️ Autenticação de Obras
            
            - Comparação com banco de dados
            - Análise de assinaturas e estilos
            - Detecção de anomalias estruturais
            """)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_feat3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("""
            ### 🗂️ Gestão de Acervo
            
            - Catalogação digital avançada
            - Histórico de conservação
            - Recomendações de preservação
            """)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_feat4:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("""
            ### 📊 Relatórios e Exportação
            
            - Geração de relatórios técnicos
            - Visualizações para apresentações
            - Exportação de dados para pesquisa
            """)
            st.markdown("</div>", unsafe_allow_html=True)


st.markdown("<div class='footer'>© 2025 HoloArt Analyzer | Desenvolvido para Conservação e Análise de Patrimônio Cultural</div>", unsafe_allow_html=True)


import streamlit.components.v1 as components

# Carregar Folium via CDN
components.html(
    """
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
    """,
    height=0,
)

import streamlit as st
import pandas as pd
import branca as folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import sqlite3
from datetime import datetime
import hashlib
import base64
from io import BytesIO
import requests
from PIL import Image
from typing import Tuple, Optional
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AppConfig:
    """Application configuration settings"""
    PAGE_TITLE = "Sistema de Catalogação de Acervos"
    DB_NAME = 'acervos.db'
    MAX_IMAGE_SIZE = (800, 800)
    ADMIN_USERNAME = 'admin'
    ADMIN_PASSWORD = 'admin123'

def hash_password(password: str) -> str:
    return hashlib.sha256(str.encode(password)).hexdigest()

class DatabaseManager:
    def __init__(self, db_name: str):
        self.db_name = db_name

    def get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_name)

    def initialize_database(self):
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                self._create_tables(c)
                self._create_admin_user(c)
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise

    def _create_tables(self, cursor: sqlite3.Cursor):
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS usuarios (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE,
                password TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS acervos (
                id INTEGER PRIMARY KEY,
                titulo TEXT NOT NULL,
                descricao TEXT,
                data TEXT,
                localizacao TEXT,
                latitude REAL,
                longitude REAL,
                data_cadastro TEXT,
                imagem BLOB
            )
        ''')

    def _create_admin_user(self, cursor: sqlite3.Cursor):
        cursor.execute('SELECT * FROM usuarios WHERE username = ?', (AppConfig.ADMIN_USERNAME,))
        if not cursor.fetchone():
            hashed_password = hash_password(AppConfig.ADMIN_PASSWORD)
            cursor.execute(
                'INSERT INTO usuarios (username, password) VALUES (?, ?)',
                (AppConfig.ADMIN_USERNAME, hashed_password)
            )

class ImageProcessor:
    @staticmethod
    def process_image(image_data, is_url: bool = False) -> Optional[bytes]:
        try:
            if is_url:
                response = requests.get(image_data)
                img = Image.open(BytesIO(response.content))
            else:
                img = Image.open(image_data)

            img.thumbnail(AppConfig.MAX_IMAGE_SIZE, Image.LANCZOS)
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return buffered.getvalue()
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return None

class GeocodingService:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="meu_app")

    def geocode(self, address: str) -> Tuple[Optional[float], Optional[float]]:
        try:
            location = self.geolocator.geocode(address)
            return (location.latitude, location.longitude) if location else (None, None)
        except GeocoderTimedOut:
            logger.warning(f"Geocoding timeout for address: {address}")
            return None, None

class AcervoManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.geocoding_service = GeocodingService()

    def save_acervo(self, titulo: str, descricao: str, data: str, 
                    localizacao: str, imagem=None, imagem_url: str = None):
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                lat, lon = self.geocoding_service.geocode(localizacao)
                
                imagem_bytes = None
                if imagem or imagem_url:
                    image_processor = ImageProcessor()
                    imagem_bytes = image_processor.process_image(
                        imagem_url if imagem_url else imagem,
                        is_url=bool(imagem_url)
                    )

                cursor.execute('''
                    INSERT INTO acervos (titulo, descricao, data, localizacao, 
                                       latitude, longitude, data_cadastro, imagem)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (titulo, descricao, data, localizacao, lat, lon,
                     datetime.now().strftime("%Y-%m-%d %H:%M:%S"), imagem_bytes))
        except Exception as e:
            logger.error(f"Error saving acervo: {e}")
            raise

class StreamlitInterface:
    def __init__(self):
        self.db_manager = DatabaseManager(AppConfig.DB_NAME)
        self.acervo_manager = AcervoManager(self.db_manager)
        
    def run(self):
        st.set_page_config(page_title=AppConfig.PAGE_TITLE, layout="wide")
        self.db_manager.initialize_database()
        
        if 'logged_in' not in st.session_state:
            st.session_state.logged_in = False

        if not st.session_state.logged_in:
            self._show_login_page()
        else:
            self._show_main_interface()

    def _show_login_page(self):
        st.title("Login")
        username = st.text_input("Usuário")
        password = st.text_input("Senha", type="password")

        if st.button("Entrar"):
            if self._verify_login(username, password):
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Credenciais inválidas!")

    def _show_main_interface(self):
        st.title("Sistema de Catalogação e Georreferenciamento de Acervos")

        with st.sidebar:
            self._show_sidebar_menu()

        if st.session_state.get('menu_atual') == "criar_ficha":
            self._show_create_acervo_form()
        elif st.session_state.get('menu_atual') == "ver_galeria":
            self._show_acervo_gallery()
        elif st.session_state.get('menu_atual') == "ver_mapa":
            self._show_acervo_map()
        elif st.session_state.get('menu_atual') == "dashboard":
            self._show_dashboard()
        else:
            self._show_dashboard()  # Mostrar dashboard como página inicial

    def _show_sidebar_menu(self):
        with st.expander("GERAR FICHA DE ACERVO", expanded=False):
            if st.button("Criar Nova Ficha", key="criar_ficha"):
                st.session_state.menu_atual = "criar_ficha"

        with st.expander("VISUALIZAR ACERVO", expanded=False):
            if st.button("Ver Galeria", key="ver_galeria"):
                st.session_state.menu_atual = "ver_galeria"
            if st.button("Ver Mapa", key="ver_mapa"):
                st.session_state.menu_atual = "ver_mapa"

        with st.expander("ANÁLISE DE DADOS", expanded=False):
            if st.button("Dashboard", key="dashboard"):
                st.session_state.menu_atual = "dashboard"

        with st.expander("EXPORTAR/IMPORTAR", expanded=False):
            self._show_export_import_options()

        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

    def _show_create_acervo_form(self):
        st.header("Nova Ficha de Acervo")
        titulo = st.text_input("Título")
        descricao = st.text_area("Descrição")
        data = st.date_input("Data")
        localizacao = st.text_input("Localização")

        opcao_imagem = st.radio("Adicionar imagem via:", ("Upload", "URL"))

        imagem = None
        imagem_url = None

        if opcao_imagem == "Upload":
            imagem = st.file_uploader("Escolha uma imagem", type=['jpg', 'jpeg', 'png'])
        else:
            imagem_url = st.text_input("URL da imagem")

        if st.button("Salvar"):
            if titulo and descricao and localizacao:
                self.acervo_manager.save_acervo(titulo, descricao, str(data), localizacao, imagem, imagem_url)
                st.success("Acervo salvo com sucesso!")
            else:
                st.warning("Preencha todos os campos obrigatórios!")

    def _show_acervo_gallery(self):
        st.header("Galeria de Acervos")
        termo_busca = st.text_input("Buscar acervos")
        acervos = self._buscar_acervos(termo_busca)

        cols = st.columns(3)
        for idx, acervo in enumerate(acervos):
            col = cols[idx % 3]
            with col:
                st.subheader(acervo['titulo'])
                if 'imagem' in acervo and acervo['imagem']:
                    try:
                        img = Image.open(BytesIO(acervo['imagem']))
                        st.image(img)
                    except Exception as e:
                        st.error(f"Erro ao carregar imagem: {e}")
                st.write(f"**Descrição:** {acervo['descricao']}")
                st.write(f"**Data:** {acervo['data']}")
                st.write(f"**Localização:** {acervo['localizacao']}")
                
                # Adicionar botões de edição e exclusão
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Editar #{acervo['id']}"):
                        self._show_edit_acervo_form(acervo['id'])
                with col2:
                    if st.button(f"Excluir #{acervo['id']}"):
                        self._delete_acervo(acervo['id'])
                        st.rerun()
                
                st.markdown("---")

    def _show_acervo_map(self):
        st.header("Mapa de Acervos")
        acervos = self._buscar_acervos()
        m = folium.Map(location=[0, 0], zoom_start=2)
        for acervo in acervos:
            if acervo['latitude'] and acervo['longitude']:
                folium.Marker(
                    [acervo['latitude'], acervo['longitude']],
                    popup=f"{acervo['titulo']}<br>{acervo['localizacao']}"
                ).add_to(m)
        folium_static(m)

    def _show_dashboard(self):
        st.header("Dashboard de Análise de Dados")

        # Carregar dados
        acervos = self._buscar_acervos()
        df = pd.DataFrame(acervos)

        if df.empty:
            st.warning("Não há dados disponíveis para exibir no dashboard.")
            return

        # Verificar e converter a coluna 'data' para datetime, se existir
        if 'data' in df.columns:
            df['data'] = pd.to_datetime(df['data'], errors='coerce')
        else:
            st.warning("A coluna 'data' não está presente nos dados. Algumas análises podem não estar disponíveis.")

        # Layout de duas colunas
        col1, col2 = st.columns(2)

        with col1:
            if 'data' in df.columns:
                # Gráfico de barras: Contagem de acervos por ano
                df_year = df.groupby(df['data'].dt.year).size().reset_index(name='count')
                fig_year = px.bar(df_year, x='data', y='count', title='Acervos por Ano')
                st.plotly_chart(fig_year)

            # Gráfico de pizza: Distribuição de acervos por localização
            if 'localizacao' in df.columns:
                df_location = df['localizacao'].value_counts().reset_index()
                df_location.columns = ['localizacao', 'count']  # Renomeando as colunas
                fig_location = px.pie(df_location, values='count', names='localizacao',
                                      title='Distribuição de Acervos por Localização')
                st.plotly_chart(fig_location)
            else:
                st.warning("A coluna 'localizacao' não está presente nos dados.")

        with col2:
            if 'data' in df.columns:
                # Gráfico de linha: Acumulado de acervos ao longo do tempo
                df_cumsum = df.sort_values('data').groupby('data').size().cumsum().reset_index(name='cumsum')
                fig_cumsum = px.line(df_cumsum, x='data', y='cumsum',
                                     title='Acumulado de Acervos ao Longo do Tempo')
                st.plotly_chart(fig_cumsum)

            # Estatísticas gerais
            st.subheader("Estatísticas Gerais")
            st.write(f"Total de Acervos: {len(df)}")
            if 'data' in df.columns:
                st.write(f"Acervo mais antigo: {df['data'].min().strftime('%d/%m/%Y')}")
                st.write(f"Acervo mais recente: {df['data'].max().strftime('%d/%m/%Y')}")
            if 'localizacao' in df.columns:
                st.write(f"Número de localizações únicas: {df['localizacao'].nunique()}")

        # Mapa de calor das localizações
        if 'latitude' in df.columns and 'longitude' in df.columns:
            st.subheader("Mapa de Calor das Localizações")
            fig_heatmap = px.density_mapbox(df, lat='latitude', lon='longitude', zoom=3,
                                            mapbox_style="stamen-terrain")
            st.plotly_chart(fig_heatmap)
        else:
            st.warning("As colunas 'latitude' e 'longitude' não estão presentes nos dados. Não é possível exibir o mapa de calor.")


    def _show_edit_acervo_form(self, acervo_id):
        acervo = self._get_acervo_by_id(acervo_id)
        if acervo:
            st.header(f"Editar Acervo #{acervo_id}")
            titulo = st.text_input("Título", value=acervo['titulo'])
            descricao = st.text_area("Descrição", value=acervo['descricao'])
            data = st.date_input("Data", value=datetime.strptime(acervo['data'], '%Y-%m-%d').date())
            localizacao = st.text_input("Localização", value=acervo['localizacao'])

            if st.button("Atualizar"):
                self._update_acervo(acervo_id, titulo, descricao, str(data), localizacao)
                st.success("Acervo atualizado com sucesso!")
                st.rerun()

    def _get_acervo_by_id(self, acervo_id):
        with self.db_manager.get_connection() as conn:
            query = f"SELECT * FROM acervos WHERE id = {acervo_id}"
            result = pd.read_sql_query(query, conn).to_dict('records')
            return result[0] if result else None

    def _update_acervo(self, acervo_id, titulo, descricao, data, localizacao):
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE acervos
                SET titulo = ?, descricao = ?, data = ?, localizacao = ?
                WHERE id = ?
            ''', (titulo, descricao, data, localizacao, acervo_id))

    def _delete_acervo(self, acervo_id):
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM acervos WHERE id = ?', (acervo_id,))

    def _show_export_import_options(self):
        if st.button("Exportar para Excel"):
            excel_data = self._exportar_excel()
            st.download_button(
                label="Download Excel",
                data=excel_data,
                file_name="acervos.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        uploaded_file = st.file_uploader("Importar do Excel", type=['xlsx'])
        if uploaded_file is not None:
            if st.button("Processar Importação"):
                self._importar_excel(uploaded_file)
                st.success("Importação concluída!")

    def _verify_login(self, username, password):
        with self.db_manager.get_connection() as conn:
            c = conn.cursor()
            hashed_pw = hash_password(password)
            c.execute('SELECT * FROM usuarios WHERE username=? AND password=?', (username, hashed_pw))
            return c.fetchone() is not None

    def _buscar_acervos(self, termo_busca=None):
        with self.db_manager.get_connection() as conn:
            if termo_busca:
                query = f"""SELECT id, titulo, descricao, data, localizacao, latitude, longitude, data_cadastro, imagem
                           FROM acervos 
                           WHERE titulo LIKE '%{termo_busca}%' 
                           OR descricao LIKE '%{termo_busca}%'
                           OR localizacao LIKE '%{termo_busca}%'"""
            else:
                query = "SELECT id, titulo, descricao, data, localizacao, latitude, longitude, data_cadastro, imagem FROM acervos"
            return pd.read_sql_query(query, conn).to_dict('records')

    def _exportar_excel(self):
        with self.db_manager.get_connection() as conn:
            df = pd.read_sql_query("SELECT id, titulo, descricao, data, localizacao, latitude, longitude, data_cadastro FROM acervos", conn)
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        return output.getvalue()

    def _importar_excel(self, file):
        df = pd.read_excel(file)
        with self.db_manager.get_connection() as conn:
            c = conn.cursor()
            for _, row in df.iterrows():
                c.execute('''INSERT INTO acervos
                             (titulo, descricao, data, localizacao, latitude, longitude, data_cadastro)
                             VALUES (?, ?, ?, ?, ?, ?, ?)''',
                          (row['titulo'], row['descricao'], row['data'], 
                           row['localizacao'], row['latitude'], row['longitude'],
                           datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

def main():
    app = StreamlitInterface()
    app.run()

if __name__ == "__main__":
    main()

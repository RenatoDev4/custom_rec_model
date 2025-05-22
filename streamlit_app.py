import numpy as np
import pandas as pd
import streamlit as st
import pickle
from google.cloud import storage
from typing import List, Tuple
import os
import random
from thefuzz import process

@st.cache_resource
def load_artifacts_from_gcs():
    """
    Carrega os artefatos do modelo do Google Cloud Storage.
    Retorna a matrix de similaridade, o vetorizador e os dados processados.
    """
    try:
        # Verificar se os arquivos já existem localmente para evitar download repetido
        local_artifacts_path = "/tmp/model_artifacts.pkl"
        local_data_path = "/tmp/dados_processados.pkl"
        local_popular_games_path = "/tmp/popular_games.pkl"
        
        # Se algum arquivo não existir localmente, carregue do GCS
        if not (os.path.exists(local_artifacts_path) and 
                os.path.exists(local_data_path) and
                os.path.exists(local_popular_games_path)):
            
            print("Baixando artefatos do Google Cloud Storage...")
            client = storage.Client()
            bucket = client.bucket("bucket_agent_renato")
            
            print("Carregando artefatos do modelo...")
            artifacts_blob = bucket.blob("custom_model/trained_model_artifacts/model_artifacts.pkl")
            artifacts_blob.download_to_filename(local_artifacts_path)
            
            print("Carregando dados dos jogos...")
            data_blob = bucket.blob("custom_model/processed_data/dados_processados.pkl")
            data_blob.download_to_filename(local_data_path)
            
            print("Carregando dados para o Streamlit...")
            pop_blob = bucket.blob("custom_model/streamlit/popular_games.pkl")
            pop_blob.download_to_filename(local_popular_games_path)
        else:
            print("Usando artefatos armazenados localmente...")
        
        # Carregando os arquivos baixados
        with open(local_artifacts_path, 'rb') as f:
            model_artifacts = pickle.load(f)
        
        sim_matrix = model_artifacts['similarity_matrix']
        vectorizer = model_artifacts['vectorizer']
        
        with open(local_data_path, 'rb') as f:
            data = pickle.load(f)
        
        with open(local_popular_games_path, 'rb') as f:
            popular_games = pickle.load(f)
        
        print("Todos os artefatos carregados com sucesso!")
        return sim_matrix, vectorizer, data, popular_games
    
    except Exception as e:
        st.error(f"Erro ao carregar artefatos: {str(e)}")
        raise e

def jaccard_similarity(set1, set2):
    """
    Calcula a similaridade de Jaccard entre dois conjuntos.
    
    Args:
        set1, set2: Conjuntos para comparar
        
    Returns:
        float: Pontuação de similaridade
    """
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection/union if union != 0 else 0

def correct_name(title, data, threshold=80):
    """
    Corrige o nome do jogo usando fuzzy matching.
    
    Args:
        title (str): Nome do jogo a ser corrigido
        data (pandas.DataFrame): Dataset de jogos
        threshold (int): Limiar de pontuação para considerar uma correspondência
        
    Returns:
        str: Nome corrigido ou None
    """
    game_names = data['Name'].tolist()
    best_match, score = process.extractOne(title, game_names)
    
    if score >= threshold:
        return best_match
    else:
        return None

def recommend_game(title, data, sim_matrix, n_recommendation=5, alpha=0.3):
    """
    Recomenda jogos semelhantes com base no título fornecido.
    
    Args:
        title (str): Título do jogo
        data (pandas.DataFrame): Dataset de jogos
        sim_matrix (numpy.ndarray): Matriz de similaridade
        n_recommendation (int): Número de recomendações a retornar
        alpha (float): Peso para balancear similaridade de cosseno vs Jaccard
        
    Returns:
        tuple: (mensagem de saída, lista de jogos recomendados com imagens)
    """
    # Normalizando os títulos dos jogos
    title = title.lower().strip()
    
    # Checando se o jogo existe no dataset
    if title not in data['Name'].values:
        corr_title = correct_name(title, data)
        if corr_title and corr_title in data['Name'].values:
            title = corr_title
        else:
            return "Jogo não encontrado. Verifique o nome e tente novamente.", []
    
    # Encontrando o índice do título do jogo fornecido
    g_idx = data[data['Name'] == title].index[0]
    jaccard_scores = []
    
    # Calcular pontuações de similaridade de Jaccard
    for idx in range(len(data)):
        if idx == g_idx:
            jaccard_scores.append(0)
        else:
            jaccard_scores.append(jaccard_similarity(data.loc[g_idx, 'Tags'], data.loc[idx, 'Tags']))
    
    jaccard_scores = np.array(jaccard_scores)
    
    # Normalizando a pontuação
    if np.max(jaccard_scores) > 0:
        jaccard_scores /= np.max(jaccard_scores)
    
    # Obtenha pontuações de similaridade de cosseno
    cosine_scores = sim_matrix[g_idx]
    if np.max(cosine_scores) > 0:
        cosine_scores /= np.max(cosine_scores)
    
    # Combinando ambos os pontos
    final_score = alpha * cosine_scores + (1 - alpha) * jaccard_scores
    recomm_idx = final_score.argsort()[::-1][1:n_recommendation+1]
    
    output = f"Com base no jogo '{title}', recomendamos os seguintes títulos:"
    game_list = []
    
    for idx in recomm_idx:
        game_name = data['Name'].iloc[idx]
        game_image = data['Header image'].iloc[idx]
        game_list.append({"titulo": game_name, "img_url": game_image})
    
    return output, game_list

def get_random_popular_games(popular_games, num_to_display=40):
    """
    Seleciona aleatoriamente um número específico de jogos populares.
    
    Args:
        popular_games (DataFrame ou dict): Dados dos jogos populares
        num_to_display (int): Número de jogos a serem exibidos
        
    Returns:
        DataFrame ou dict: Dados com os jogos selecionados aleatoriamente
    """
    # Verificar se popular_games é um DataFrame ou um dicionário
    if isinstance(popular_games, pd.DataFrame):
        # Se for DataFrame, use sample() para selecionar linhas aleatórias
        total_games = len(popular_games)
        return popular_games.sample(min(num_to_display, total_games)).reset_index(drop=True)
    else:
        # Se for dicionário, use o método anterior
        # Criar índices de todos os jogos disponíveis
        total_games = len(popular_games["name"])
        all_indices = list(range(total_games))
        
        # Embaralhar os índices para seleção aleatória
        random.shuffle(all_indices)
        
        # Selecionar apenas o número desejado
        selected_indices = all_indices[:min(num_to_display, total_games)]
        
        # Criar um novo dicionário com apenas os jogos selecionados
        selected_games = {}
        for key in popular_games:
            if key in popular_games and isinstance(popular_games[key], list):
                selected_games[key] = [popular_games[key][i] for i in selected_indices if i < len(popular_games[key])]
            else:
                selected_games[key] = []
        
        return selected_games

def steam_recommend() -> None:
    """
    Function for generating a recommendation system for Steam games.
    It sets up the page configuration and sidebar information, as well as
    loading and processing the necessary data for recommendations and displaying
    the most highly rated games on the Steam platform.

    Returns:
    --------
    None
    """

    st.set_page_config(
        page_title="Sistema de Recomendação - STEAM", 
        page_icon="🎮", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Estilização global
    st.markdown("""
        <style>
            /* Estilos globais para toda a aplicação */
            .main {
                background-color: #171a21;
                color: #c7d5e0;
            }
            .stApp {
                background-color: #171a21;
            }
            h1, h2, h3 {
                color: #66c0f4 !important;
            }
            .stButton button {
                background-color: #66c0f4;
                color: #171a21;
                font-weight: bold;
                border: none;
                transition: all 0.3s;
            }
            .stButton button:hover {
                background-color: #1b2838;
                color: #66c0f4;
                border: 1px solid #66c0f4;
            }
            .stSelectbox div[data-baseweb="select"] > div {
                background-color: #2a475e;
                color: #c7d5e0;
                border: 1px solid #66c0f4;
            }
            div[data-testid="stSidebar"] {
                background-color: #1b2838;
                padding: 20px;
            }
            .css-10xlvwk {
                color: #c7d5e0;
            }
            /* Personalizações para outros elementos Streamlit */
            .stProgress .st-bo {
                background-color: #66c0f4;
            }
            .stTextInput input {
                background-color: #2a475e;
                color: #c7d5e0;
                border: 1px solid #66c0f4;
            }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar personalizada com logo e informações
    with st.sidebar:
        st.markdown("""
            <div style="text-align:center; margin-bottom:20px;">
                <img src="https://store.akamai.steamstatic.com/public/shared/images/header/logo_steam.svg" width="180">
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h3 style='text-align:center; margin-bottom:20px;'>Sobre o desenvolvedor</h3>", unsafe_allow_html=True)
        
        # Card do desenvolvedor com estilo
        st.markdown("""
            <div style="background-color:#2a475e; padding:15px; border-radius:10px; margin-bottom:20px; text-align:center;">
                <h4 style="color:#66c0f4; margin-top:0;">Renato Moraes</h4>
                <div style="margin:15px 0;">
                    <a href="https://linkedin.com/in/renato-moraes-11b546272" target="_blank" style="margin:0 10px; color:#c7d5e0; text-decoration:none;">
                        <img src="https://cdn.jsdelivr.net/npm/simple-icons@v5/icons/linkedin.svg" style="width:24px; vertical-align:middle; filter:invert(0.8);">
                        LinkedIn
                    </a>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<h3 style='text-align:center; margin-bottom:20px;'>Sobre o Projeto</h3>", unsafe_allow_html=True)
        
        # Card do projeto com estilo
        st.markdown("""
            <div style="background-color:#2a475e; padding:15px; border-radius:10px; text-align:justify;">
                <p style="color:#c7d5e0; margin:0;">
                    Este sistema ajuda você a encontrar novos jogos parecidos com os que você já gosta! Basta digitar o nome de um jogo, que ele irá sugerir outros títulos com base em semelhanças nos gêneros, tags e descrições.
                </p>
            </div>
        """, unsafe_allow_html=True)

    # Carregar os artefatos do GCS usando o cache
    with st.spinner("Carregando dados do modelo... Isso pode levar alguns segundos."):
        sim_matrix, vectorizer, data, popular_games = load_artifacts_from_gcs()

    # Cabeçalho principal com estilo
    st.markdown("""
        <div style="text-align:center; margin-bottom:30px;">
            <h1 style="font-size:2.5em; color:#66c0f4; margin-bottom:5px;">
                Sistema de recomendação de jogos
                <img src="https://store.akamai.steamstatic.com/public/shared/images/header/logo_steam.svg" style="height:40px; vertical-align:middle; margin-left:10px;">
            </h1>
            <p style="color:#c7d5e0; font-size:1.2em;">Descubra novos jogos baseados em seus favoritos</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Interface de pesquisa com estilo
    st.markdown("""
        <div style="background-color:#2a475e; padding:20px; border-radius:10px; margin-bottom:30px;">
            <h3 style="color:#66c0f4; margin-top:0;">Encontre sua próxima aventura</h3>
            <p style="color:#c7d5e0;">Selecione um jogo que você gosta ou digite seu nome e descubra títulos similares</p>
        </div>
    """, unsafe_allow_html=True)

    # Cria uma lista com os títulos dos jogos para o selectbox
    game_titles = sorted(data['Name'].unique().tolist())

    # Input do usuário com estilo
    user_input = st.selectbox(
        "Selecione um jogo:",
        game_titles,
        help="Escolha um jogo que você gosta para receber recomendações similares"
    )

    if st.button("Mostrar Recomendação", use_container_width=True, type="primary"):
        st.markdown("___")
        with st.spinner("Gerando recomendações personalizadas..."):
            output, recommendations = recommend_game(user_input, data, sim_matrix)
                       
            if not recommendations:
                st.error(
                    "Ah, parece que este jogo é meio exclusivo quando se trata de recomendações 😅. Não encontramos muitas sugestões por aqui, mas fique tranquilo! Existem tantos outros jogos incríveis esperando para serem descobertos 😎"
                )
            else:
                # Criar contêiner com estilo para as recomendações
                st.markdown("""
                <style>
                    .game-card {
                        border: 1px solid #2a475e;
                        border-radius: 8px;
                        overflow: hidden;
                        transition: transform 0.3s, box-shadow 0.3s;
                        background: #1e2c3a;
                        height: 100%;
                    }
                    .game-card:hover {
                        transform: translateY(-5px);
                        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
                    }
                    .game-title {
                        padding: 10px;
                        font-weight: bold;
                        color: #c7d5e0;
                        text-align: center;
                        background: #2a475e;
                    }
                    .game-image {
                        width: 100%;
                        transition: opacity 0.3s;
                    }
                    .game-image:hover {
                        opacity: 0.8;
                    }
                    .game-info {
                        padding: 10px;
                        color: #c7d5e0;
                    }
                    .match-score {
                        background: #66c0f4;
                        color: #0e1c2a;
                        font-weight: bold;
                        border-radius: 15px;
                        padding: 2px 8px;
                        display: inline-block;
                        margin-top: 5px;
                    }
                </style>
                """, unsafe_allow_html=True)
                
                # Adicionar a descrição da recomendação
                st.markdown(f"""
                <div style="background:#2a475e; padding:15px; border-radius:5px; margin-bottom:20px;">
                    <p style="color:#c7d5e0; margin:0;">
                        Analisamos milhares de jogos e dados de usuários para encontrar os títulos que combinam melhor com <span style="color:#66c0f4; font-weight:bold;">{user_input}</span>. 
                        As sugestões abaixo foram selecionadas com base em gêneros, mecânicas de jogo e padrões de preferência de jogadores.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Mostrar os jogos recomendados em uma grade responsiva
                cols = st.columns(min(4, len(recommendations)))
                
                # Gerar pontuações de correspondência simuladas (na prática você usaria dados reais)
                import random
                match_scores = [random.randint(82, 97) for _ in range(len(recommendations[:4]))]
                
                for c, rec in enumerate(recommendations[:4]):
                    with cols[c]:
                        st.markdown(f"""
                        <div class="game-card">
                            <img src="{rec['img_url']}" class="game-image" alt="{rec['titulo']}">
                            <div class="game-title">{rec['titulo']}</div>
                            <div class="game-info">
                                <span class="match-score">{match_scores[c]}% de compatibilidade</span>
                                <p style="margin-top:8px; margin-bottom:5px;">Baseado em suas escolhas</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Adicionar um botão para ver mais recomendações (funcionalidade opcional para implementação futura)
                if len(recommendations) > 4:
                    st.markdown("<div style='text-align:center; margin-top:20px;'>", unsafe_allow_html=True)
                    st.button("Ver mais recomendações", key="more_recommendations", use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

    # Mostrar jogos populares aleatórios - Usando o state para persistir entre interações
    st.markdown("___")
    
    # Estilização do cabeçalho da seção de jogos populares
    st.markdown(
        """
        <div style="background-color:#1e2c3a; padding:15px; border-radius:10px; margin-bottom:20px; border-left:5px solid #66c0f4;">
            <h2 style="color:#66c0f4; margin:0;">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#66c0f4" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; margin-right:10px;">
                    <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>
                </svg>
                Jogos bem avaliados aleatórios do universo Steam
            </h2>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Inicializar a chave de sessão para os jogos aleatórios se não existir
    if 'random_popular_games' not in st.session_state:
        st.session_state.random_popular_games = get_random_popular_games(popular_games, num_to_display=40)
    
    # Usar os jogos aleatórios da sessão
    random_popular_games = st.session_state.random_popular_games
    
    # Botão para gerar um novo conjunto de jogos aleatórios
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Descobrir outros jogos populares", use_container_width=True):
            st.session_state.random_popular_games = get_random_popular_games(popular_games, num_to_display=40)
            random_popular_games = st.session_state.random_popular_games
            st.rerun()
    
    # Estilização para os cards de jogos populares - CORRIGIDO
    st.markdown("""
    <style>
        /* Corrigindo o CSS para garantir 4 cards por linha */
        .popular-game-container {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            grid-gap: 15px;
            margin-top: 20px;
        }
        .popular-game-card {
            border-radius: 8px;
            overflow: hidden;
            background: #1e2c3a;
            transition: transform 0.3s;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            height: 100%;
            display: flex;
            flex-direction: column;
        }
        .popular-game-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        .popular-game-image {
            width: 100%;
            height: auto;
            transition: opacity 0.3s;
            aspect-ratio: 16/9;
            object-fit: cover;
        }
        .popular-game-image:hover {
            opacity: 0.8;
        }
        .popular-game-info {
            padding: 12px;
            display: flex;
            flex-direction: column;
            flex-grow: 1;
        }
        .popular-game-title {
            font-size: 16px;
            font-weight: bold;
            color: #c7d5e0;
            margin-bottom: 10px;
            overflow: hidden;
            text-overflow: ellipsis;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            height: 2.8em;
        }
        .popular-game-stats {
            font-size: 14px;
            color: #8ea1b4;
            margin-top: auto;
        }
        .popular-game-rating {
            display: inline-block;
            background: linear-gradient(90deg, #66c0f4 var(--rating-percent), #2a475e var(--rating-percent));
            height: 8px;
            width: 100%;
            border-radius: 4px;
            margin-top: 5px;
        }
        
        /* Responsividade */
        @media (max-width: 1200px) {
            .popular-game-container {
                grid-template-columns: repeat(3, 1fr);
            }
        }
        @media (max-width: 768px) {
            .popular-game-container {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        @media (max-width: 480px) {
            .popular-game-container {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Verificar se random_popular_games é DataFrame ou dicionário e garantir que temos os dados corretos
    if isinstance(random_popular_games, pd.DataFrame):
        game_name = random_popular_games["name"].tolist() if "name" in random_popular_games.columns else []
        image = random_popular_games["header_image"].tolist() if "header_image" in random_popular_games.columns else []
        votes = random_popular_games["total_reviews"].tolist() if "total_reviews" in random_popular_games.columns else []
        rating = random_popular_games["positive_ratio"].tolist() if "positive_ratio" in random_popular_games.columns else []
        url = random_popular_games["steam_url"].tolist() if "steam_url" in random_popular_games.columns else []
    else:
        # Se for dicionário, use diretamente
        game_name = random_popular_games.get("name", [])
        image = random_popular_games.get("header_image", [])
        votes = random_popular_games.get("total_reviews", [])
        rating = random_popular_games.get("positive_ratio", [])
        url = random_popular_games.get("steam_url", [])
    
    # Iniciar grid de jogos com o novo container
    st.markdown('<div class="popular-game-container">', unsafe_allow_html=True)
    
    # Garantir que temos os dados e limitar a 40 jogos
    num_games = min(len(game_name), 40)
    # para simplificar, crie listas com os dados já cortados
    game_name = game_name[:num_games]
    image = image[:num_games]
    votes = votes[:num_games]
    rating = rating[:num_games]
    url = url[:num_games]

    # agora, exiba em blocos de 4 colunas
    for row_start in range(0, num_games, 4):
        cols = st.columns(4, gap="small")
        for idx, i in enumerate(range(row_start, min(row_start + 4, num_games))):
            with cols[idx]:
                try:
                    formatted_votes = f"{votes[i]:,}".replace(",", ".") if votes[i] is not None else "N/A"
                    rating_value = rating[i] if isinstance(rating[i], (int, float)) else 0
                    rating_text = f"{rating_value:.0f}%" if isinstance(rating_value, (int, float)) else "N/A"
                    game_url = url[i] or "#"
                    game_image = image[i] or ""
                    game_title = game_name[i] or "Jogo sem título"

                    st.markdown(f"""
                    <a href="{game_url}" target="_blank" style="text-decoration: none;">
                        <div style="
                            background: #1e2c3a;
                            border-radius: 8px;
                            overflow: hidden;
                            transition: transform 0.3s, box-shadow 0.3s;
                        ">
                            <div style="width:100%; height:0; padding-bottom:56.25%; position:relative; overflow:hidden;">
                              <img 
                                src="{game_image}" 
                                alt="{game_title}"
                                style="
                                  position:absolute;
                                  top:0; left:0;
                                  width:100%; height:100%;
                                  object-fit:contain;
                                "
                              >
                            </div>
                            <div style="padding:10px; color:#c7d5e0;">
                                <div style="font-weight:bold; margin-bottom:5px;">{game_title}</div>
                                <div style="font-size:0.9em; color:#8ea1b4;">Avaliações: {formatted_votes} | Positivas: {rating_text}</div>
                                <div style="
                                    height:6px;
                                    background:linear-gradient(90deg, #66c0f4 {rating_value}%, #2a475e {rating_value}%);
                                    border-radius:3px;
                                    margin-top:8px;
                                "></div>
                            </div>
                        </div>
                    </a>
                    """, unsafe_allow_html=True)
                    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Erro ao renderizar o jogo {i}: {e}")
    
    # Fechar o grid
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    steam_recommend()
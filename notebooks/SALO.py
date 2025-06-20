import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from geopy.distance import geodesic
import folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime
import os
warnings.filterwarnings('ignore')

class LogisticsOptimizer:
    def __init__(self):
        self.df = None
        self.clusters = None
        self.simulation_results = None

    def generate_sample_data(self, n_points=150):
        """Gera dados simulados realistas para demonstração"""
        np.random.seed(42)
        
        # Coordenadas aproximadas de São Paulo e região metropolitana
        center_lat, center_lon = -23.5505, -46.6333
        
        # Criar 3-4 concentrações regionais
        concentrations = [
            (-23.5505, -46.6333, 40),  # Centro SP
            (-23.6821, -46.8755, 30),  # Osasco
            (-23.4538, -46.5333, 25),  # Guarulhos
            (-23.7937, -46.6816, 35),  # São Bernardo
            (-23.3200, -46.7311, 20),  # Jundiaí
        ]
        
        data = []
        seller_id = 1000
        
        for i, (lat_center, lon_center, n_stores) in enumerate(concentrations):
            for _ in range(n_stores):
                # Variação aleatória ao redor do centro
                lat = lat_center + np.random.normal(0, 0.05)  # ~5km de variação
                lon = lon_center + np.random.normal(0, 0.05)
                
                # Volume de pacotes com distribuição realista
                volume = max(1, int(np.random.lognormal(3, 1)))  # Média ~20, max ~200
                
                data.append({
                    'EXPECTPICKUP': pd.Timestamp.now().date(),
                    'CLIENTEseller_id': f'SELLER_{seller_id}',
                    'ADDRESS': f'Rua Exemplo {seller_id % 1000}',
                    'CITY': 'São Paulo' if i < 2 else ['Osasco', 'Guarulhos', 'São Bernardo', 'Jundiaí'][i-2],
                    'STATE': 'SP',
                    'ZIP_CODE': f'{8000 + i}{seller_id % 1000:03d}',
                    'AREA': f'Região {i+1}',
                    'ROTA': f'ROTA_{i+1}',
                    'VOLUME_TIKTOK_PICKUP': volume,
                    'ENDERECO_COMPLETO': f'Rua Exemplo {seller_id % 1000}, São Paulo, SP',
                    'LATITUDE': lat,
                    'LONGITUDE': lon
                })
                seller_id += 1
        
        self.df = pd.DataFrame(data)
        return self.df
    
    def load_data(self, df=None, filepath=None):
        """Carrega dados do usuário"""
        if df is not None:
            self.df = df.copy()
        elif filepath:
            self.df = pd.read_csv(filepath)
        else:
            self.df = self.generate_sample_data()
        
        # Limpeza básica
        self.df = self.df.dropna(subset=['LATITUDE', 'LONGITUDE', 'VOLUME_TIKTOK_PICKUP'])
        self.df = self.df[self.df['VOLUME_TIKTOK_PICKUP'] > 0]
        
        # Verificações de segurança
        if len(self.df) == 0:
            raise ValueError("Nenhum dado válido encontrado após limpeza. Verifique as colunas LATITUDE, LONGITUDE e VOLUME_TIKTOK_PICKUP")
        
        # Verificar se as coordenadas são válidas
        invalid_coords = (
            (self.df['LATITUDE'].abs() > 90) | 
            (self.df['LONGITUDE'].abs() > 180) |
            (self.df['LATITUDE'].isna()) |
            (self.df['LONGITUDE'].isna())
        )
        
        if invalid_coords.sum() > 0:
            print(f"⚠️  Removendo {invalid_coords.sum()} pontos com coordenadas inválidas")
            self.df = self.df[~invalid_coords]
        
        print(f"Dataset carregado: {len(self.df)} pontos de coleta")
        print(f"Coordenadas - Lat: {self.df['LATITUDE'].min():.4f} a {self.df['LATITUDE'].max():.4f}")
        print(f"Coordenadas - Lon: {self.df['LONGITUDE'].min():.4f} a {self.df['LONGITUDE'].max():.4f}")
        print(f"Volume - Min: {self.df['VOLUME_TIKTOK_PICKUP'].min()}, Max: {self.df['VOLUME_TIKTOK_PICKUP'].max()}")
        
        return self.df
    
    def exploratory_analysis(self):
        """Análise exploratória dos dados"""
        print("=== ANÁLISE EXPLORATÓRIA ===")
        print(f"Total de pontos: {len(self.df)}")
        print(f"Volume total de pacotes: {self.df['VOLUME_TIKTOK_PICKUP'].sum()}")
        print(f"Volume médio por ponto: {self.df['VOLUME_TIKTOK_PICKUP'].mean():.1f}")
        print(f"Volume mediano: {self.df['VOLUME_TIKTOK_PICKUP'].median():.1f}")
        print(f"Desvio padrão do volume: {self.df['VOLUME_TIKTOK_PICKUP'].std():.1f}")
        
        # Estatísticas geográficas
        lat_range = self.df['LATITUDE'].max() - self.df['LATITUDE'].min()
        lon_range = self.df['LONGITUDE'].max() - self.df['LONGITUDE'].min()
        print(f"Dispersão geográfica - Lat: {lat_range:.4f}°, Lon: {lon_range:.4f}°")
        
        return {
            'total_points': len(self.df),
            'total_volume': self.df['VOLUME_TIKTOK_PICKUP'].sum(),
            'avg_volume': self.df['VOLUME_TIKTOK_PICKUP'].mean(),
            'geographic_spread': (lat_range, lon_range)
        }
    
    def calculate_distance_matrix(self, sample_size=None):
        """Calcula matriz de distâncias geográficas"""
        df_sample = self.df.sample(n=min(sample_size or len(self.df), 200))  # Limita para performance
        
        n = len(df_sample)
        distance_matrix = np.zeros((n, n))
        
        coords = df_sample[['LATITUDE', 'LONGITUDE']].values
        
        for i in range(n):
            for j in range(i+1, n):
                dist = geodesic(coords[i], coords[j]).kilometers
                distance_matrix[i][j] = dist
                distance_matrix[j][i] = dist
        
        return distance_matrix, df_sample
    
    def find_optimal_clusters(self, max_clusters=12):
        """Encontra número ótimo de clusters usando método do cotovelo e silhueta"""
        X = self.df[['LATITUDE', 'LONGITUDE']].values
        weights = self.df['VOLUME_TIKTOK_PICKUP'].values
        
        # Normalização ponderada
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Adicionar peso como terceira dimensão (normalizado)
        weights_scaled = (weights - weights.min()) / (weights.max() - weights.min())
        X_weighted = np.column_stack([X_scaled, weights_scaled * 0.5])  # Peso reduzido para não dominar
        
        inertias = []
        silhouette_scores = []
        K_range = range(2, max_clusters + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_weighted)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_weighted, cluster_labels))
        
        # Encontrar cotovelo
        deltas = np.diff(inertias)
        second_deltas = np.diff(deltas)
        elbow_idx = np.argmax(second_deltas) + 2  # +2 devido aos diffs
        optimal_k_elbow = K_range[elbow_idx] if elbow_idx < len(K_range) else K_range[-1]
        
        # Melhor silhueta
        optimal_k_silhouette = K_range[np.argmax(silhouette_scores)]
        
        # Compromisso entre os dois métodos
        optimal_k = int(np.mean([optimal_k_elbow, optimal_k_silhouette]))
        
        print(f"Número ótimo de clusters:")
        print(f"  - Método do cotovelo: {optimal_k_elbow}")
        print(f"  - Melhor silhueta: {optimal_k_silhouette}")
        print(f"  - Escolhido (média): {optimal_k}")
        
        return optimal_k, (K_range, inertias, silhouette_scores)
    
    def perform_clustering(self, n_clusters=None):
        """Executa clustering geográfico ponderado"""
        if n_clusters is None:
            n_clusters, _ = self.find_optimal_clusters()
        
        X = self.df[['LATITUDE', 'LONGITUDE']].values
        weights = self.df['VOLUME_TIKTOK_PICKUP'].values
        
        # Preparação dos dados
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        weights_scaled = (weights - weights.min()) / (weights.max() - weights.min())
        X_weighted = np.column_stack([X_scaled, weights_scaled * 0.3])
        
        # K-means principal
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        cluster_labels = kmeans.fit_predict(X_weighted)
        
        # Adicionar clusters ao dataframe
        self.df['CLUSTER'] = cluster_labels
        
        # Calcular estatísticas por cluster
        cluster_stats = []
        for cluster_id in range(n_clusters):
            cluster_data = self.df[self.df['CLUSTER'] == cluster_id]
            
            # Centróide geográfico
            centroid_lat = cluster_data['LATITUDE'].mean()
            centroid_lon = cluster_data['LONGITUDE'].mean()
            
            # Raio máximo do cluster
            max_distance = 0
            for _, row in cluster_data.iterrows():
                dist = geodesic((centroid_lat, centroid_lon), 
                              (row['LATITUDE'], row['LONGITUDE'])).kilometers
                max_distance = max(max_distance, dist)
            
            stats = {
                'cluster_id': cluster_id,
                'n_points': len(cluster_data),
                'total_volume': cluster_data['VOLUME_TIKTOK_PICKUP'].sum(),
                'avg_volume_per_point': cluster_data['VOLUME_TIKTOK_PICKUP'].mean(),
                'centroid_lat': centroid_lat,
                'centroid_lon': centroid_lon,
                'max_radius_km': max_distance,
                'cities': cluster_data['CITY'].value_counts().to_dict()
            }
            cluster_stats.append(stats)
        
        self.clusters = pd.DataFrame(cluster_stats)
        
        print("\n=== RESULTADOS DO CLUSTERING ===")
        for _, cluster in self.clusters.iterrows():
            print(f"Cluster {cluster['cluster_id']}:")
            print(f"  - Pontos: {cluster['n_points']}")
            print(f"  - Volume total: {cluster['total_volume']}")
            print(f"  - Raio máximo: {cluster['max_radius_km']:.1f} km")
            print(f"  - Principais cidades: {list(cluster['cities'].keys())[:3]}")
            print()
        
        return self.clusters
    
    def monte_carlo_simulation(self, n_simulations=5000):
        """Simulação Monte Carlo para dimensionamento de frota"""
        if self.clusters is None:
            raise ValueError("Execute o clustering primeiro!")
        
        # Parâmetros da simulação
        vehicle_capacities = [100, 150, 300]
        trips_per_day = [1, 2]
        
        # Parâmetros de tempo (em horas) - mais conservadores
        base_pickup_time = 0.25  # 15 min por ponto
        travel_time_per_km = 0.05  # 3 min por km
        working_hours_per_day = 8
        max_reasonable_time = working_hours_per_day * 0.8  # 80% do tempo máximo
        
        simulation_results = []
        
        for _, cluster in self.clusters.iterrows():
            cluster_id = cluster['cluster_id']
            total_volume = cluster['total_volume']
            n_points = cluster['n_points']
            max_radius = cluster['max_radius_km']
            
            print(f"Simulando Cluster {cluster_id} (Volume: {total_volume}, Pontos: {n_points})...")
            
            for capacity in vehicle_capacities:
                for trips in trips_per_day:
                    daily_capacity = capacity * trips
                    vehicles_needed = []
                    
                    for sim in range(n_simulations):
                        # Variações aleatórias mais conservadoras
                        volume_variation = np.random.uniform(0.9, 1.1)    # ±10%
                        time_variation = np.random.uniform(0.9, 1.1)     # ±10%
                        traffic_factor = np.random.uniform(1.0, 1.3)     # Trânsito mais moderado
                        
                        adjusted_volume = max(1, int(total_volume * volume_variation))
                        
                        # Tempo estimado por veículo
                        pickup_time = n_points * base_pickup_time * time_variation
                        travel_time = max_radius * 2 * travel_time_per_km * traffic_factor
                        total_time_per_trip = pickup_time + travel_time
                        
                        # Capacidade diária inicial
                        current_daily_capacity = daily_capacity
                        
                        # Verificar se cabe no dia de trabalho
                        total_daily_time = total_time_per_trip * trips
                        if total_daily_time > max_reasonable_time:
                            # Reduzir eficiência se não couber no tempo
                            efficiency_penalty = total_daily_time / max_reasonable_time
                            current_daily_capacity = max(10, int(current_daily_capacity / efficiency_penalty))
                        
                        # Garantir que daily_capacity nunca seja zero ou muito baixo
                        current_daily_capacity = max(5, current_daily_capacity)
                        
                        # Calcular veículos necessários
                        vehicles = max(1, int(np.ceil(adjusted_volume / current_daily_capacity)))
                        
                        # Limite máximo razoável de veículos por cluster
                        vehicles = min(vehicles, max(10, n_points))
                        
                        vehicles_needed.append(vehicles)
                    
                    # Estatísticas da simulação
                    result = {
                        'cluster_id': cluster_id,
                        'capacity': capacity,
                        'trips': trips,
                        'daily_capacity': capacity * trips,
                        'min_vehicles': int(np.percentile(vehicles_needed, 5)),
                        'median_vehicles': int(np.percentile(vehicles_needed, 50)),
                        'max_vehicles': int(np.percentile(vehicles_needed, 95)),
                        'mean_vehicles': np.mean(vehicles_needed),
                        'std_vehicles': np.std(vehicles_needed)
                    }
                    simulation_results.append(result)
        
        self.simulation_results = pd.DataFrame(simulation_results)
        print(f"✅ Simulação concluída para {len(self.clusters)} clusters")
        return self.simulation_results
    
    def generate_fleet_recommendations(self):
        """Gera recomendações finais de frota"""
        if self.simulation_results is None:
            raise ValueError("Execute a simulação Monte Carlo primeiro!")
        
        recommendations = []
        
        for cluster_id in self.clusters['cluster_id'].unique():
            cluster_sims = self.simulation_results[
                self.simulation_results['cluster_id'] == cluster_id
            ]
            
            # Cenários
            conservative = cluster_sims[
                (cluster_sims['capacity'] == 100) & (cluster_sims['trips'] == 1)
            ].iloc[0]
            
            moderate = cluster_sims[
                (cluster_sims['capacity'] == 150) & (cluster_sims['trips'] == 1)
            ].iloc[0]
            
            optimistic = cluster_sims[
                (cluster_sims['capacity'] == 300) & (cluster_sims['trips'] == 2)
            ].iloc[0]
            
            cluster_info = self.clusters[self.clusters['cluster_id'] == cluster_id].iloc[0]
            
            recommendation = {
                'cluster_id': cluster_id,
                'total_volume': cluster_info['total_volume'],
                'n_points': cluster_info['n_points'],
                'conservative_range': f"{int(conservative['min_vehicles'])}-{int(conservative['max_vehicles'])}",
                'moderate_range': f"{int(moderate['min_vehicles'])}-{int(moderate['max_vehicles'])}",
                'optimistic_range': f"{int(optimistic['min_vehicles'])}-{int(optimistic['max_vehicles'])}",
                'recommended_scenario': 'moderate',
                'recommended_vehicles': f"{int(moderate['min_vehicles'])}-{int(moderate['max_vehicles'])}",
                'main_cities': ', '.join(list(cluster_info['cities'].keys())[:3])
            }
            recommendations.append(recommendation)
        
        recommendations_df = pd.DataFrame(recommendations)
        
        print("\n=== RECOMENDAÇÕES DE FROTA ===")
        print(recommendations_df.to_string(index=False))
        
        return recommendations_df
    
    def create_interactive_map(self):
        """Cria mapa interativo com clusters"""
        if self.df is None or 'CLUSTER' not in self.df.columns:
            raise ValueError("Execute o clustering primeiro!")
        
        # Centro do mapa
        center_lat = self.df['LATITUDE'].mean()
        center_lon = self.df['LONGITUDE'].mean()
        
        # Criar mapa
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles='OpenStreetMap'
        )
        
        # Cores para clusters
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
                 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 
                 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen']
        
        # Adicionar pontos por cluster
        for cluster_id in sorted(self.df['CLUSTER'].unique()):
            cluster_data = self.df[self.df['CLUSTER'] == cluster_id]
            color = colors[cluster_id % len(colors)]
            
            for _, row in cluster_data.iterrows():
                folium.CircleMarker(
                    location=[row['LATITUDE'], row['LONGITUDE']],
                    radius=max(3, min(15, row['VOLUME_TIKTOK_PICKUP'] / 10)),
                    popup=f"""
                    <b>Cluster {cluster_id}</b><br>
                    Seller: {row['CLIENTEseller_id']}<br>
                    Volume: {row['VOLUME_TIKTOK_PICKUP']} pacotes<br>
                    Cidade: {row['CITY']}<br>
                    Endereço: {row['ADDRESS']}
                    """,
                    color=color,
                    fill=True,
                    fillOpacity=0.6
                ).add_to(m)
        
        # Adicionar centróides dos clusters
        for _, cluster in self.clusters.iterrows():
            folium.Marker(
                location=[cluster['centroid_lat'], cluster['centroid_lon']],
                popup=f"""
                <b>Centro do Cluster {cluster['cluster_id']}</b><br>
                Pontos: {cluster['n_points']}<br>
                Volume Total: {cluster['total_volume']}<br>
                Raio: {cluster['max_radius_km']:.1f} km
                """,
                icon=folium.Icon(color='black', icon='info-sign')
            ).add_to(m)
            
            # Círculo de cobertura
            folium.Circle(
                location=[cluster['centroid_lat'], cluster['centroid_lon']],
                radius=cluster['max_radius_km'] * 1000,  # metros
                color=colors[cluster['cluster_id'] % len(colors)],
                fill=False,
                opacity=0.3
            ).add_to(m)
        
        return m
    
    def run_complete_analysis(self, df=None, n_clusters=None):
        """Executa análise completa"""
        print("🚚 INICIANDO ANÁLISE DE OTIMIZAÇÃO LOGÍSTICA")
        print("=" * 50)
        
        # 1. Carregar dados
        self.load_data(df)
        
        # 2. Análise exploratória
        self.exploratory_analysis()
        
        # 3. Clustering
        print("\n📍 EXECUTANDO CLUSTERING...")
        self.perform_clustering(n_clusters)
        
        # 4. Simulação Monte Carlo
        print("\n🎲 EXECUTANDO SIMULAÇÃO MONTE CARLO...")
        self.monte_carlo_simulation()
        
        # 5. Recomendações
        print("\n📊 GERANDO RECOMENDAÇÕES...")
        recommendations = self.generate_fleet_recommendations()
        
        # 6. Mapa
        print("\n🗺️  CRIANDO MAPA INTERATIVO...")
        map_viz = self.create_interactive_map()
        
        return {
            'data': self.df,
            'clusters': self.clusters,
            'simulation_results': self.simulation_results,
            'recommendations': recommendations,
            'map': map_viz
        }

    def export_clustered_data(self, filename=None, add_timestamp=True):
        """Exporta o DataFrame com a coluna de clusters para Excel"""
        if self.df is None or 'CLUSTER' not in self.df.columns:
            raise ValueError("Execute o clustering primeiro!")
        
        # Definir nome do arquivo
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else ""
            filename = f"dados_com_clusters_{timestamp}.xlsx" if timestamp else "dados_com_clusters.xlsx"
        elif add_timestamp and not filename.endswith(('.xlsx', '.xls')):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename}_{timestamp}.xlsx"
        elif not filename.endswith(('.xlsx', '.xls')):
            filename = f"{filename}.xlsx"
        
        try:
            # Criar uma cópia organizada dos dados
            export_df = self.df.copy()
            
            # Reorganizar colunas para ter CLUSTER no início
            cols = ['CLUSTER'] + [col for col in export_df.columns if col != 'CLUSTER']
            export_df = export_df[cols]
            
            # Ordenar por cluster
            export_df = export_df.sort_values(['CLUSTER', 'VOLUME_TIKTOK_PICKUP'], ascending=[True, False])
            
            # Adicionar estatísticas por cluster
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Aba principal com dados
                export_df.to_excel(writer, sheet_name='Dados_com_Clusters', index=False)
                
                # Aba com resumo dos clusters
                if self.clusters is not None:
                    cluster_summary = self.clusters.copy()
                    cluster_summary.to_excel(writer, sheet_name='Resumo_Clusters', index=False)
                
                # Aba com estatísticas gerais
                stats_data = []
                for cluster_id in sorted(export_df['CLUSTER'].unique()):
                    cluster_data = export_df[export_df['CLUSTER'] == cluster_id]
                    stats_data.append({
                        'Cluster': cluster_id,
                        'Total_Pontos': len(cluster_data),
                        'Total_Volume': cluster_data['VOLUME_TIKTOK_PICKUP'].sum(),
                        'Volume_Medio': cluster_data['VOLUME_TIKTOK_PICKUP'].mean(),
                        'Principais_Cidades': ', '.join(cluster_data['CITY'].value_counts().head(3).index.tolist())
                    })
                
                stats_df = pd.DataFrame(stats_data)
                stats_df.to_excel(writer, sheet_name='Estatisticas_Clusters', index=False)
            
            print(f"✅ Dados com clusters exportados: {filename}")
            print(f"   📊 {len(export_df)} pontos distribuídos em {export_df['CLUSTER'].nunique()} clusters")
            return filename
            
        except Exception as e:
            print(f"❌ Erro ao exportar dados: {e}")
            return None

    def export_fleet_recommendations(self, filename=None, add_timestamp=True):
        """Exporta as recomendações de frota para Excel"""
        if self.simulation_results is None:
            raise ValueError("Execute a simulação Monte Carlo primeiro!")
        
        # Definir nome do arquivo
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else ""
            filename = f"recomendacoes_frota_{timestamp}.xlsx" if timestamp else "recomendacoes_frota.xlsx"
        elif add_timestamp and not filename.endswith(('.xlsx', '.xls')):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename}_{timestamp}.xlsx"
        elif not filename.endswith(('.xlsx', '.xls')):
            filename = f"{filename}.xlsx"
        
        try:
            # Gerar recomendações se ainda não foi feito
            recommendations = self.generate_fleet_recommendations()
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Aba 1: Recomendações resumidas
                recommendations.to_excel(writer, sheet_name='Recomendacoes_Resumo', index=False)
                
                # Aba 2: Resultados detalhados da simulação
                simulation_detailed = self.simulation_results.copy()
                simulation_detailed.to_excel(writer, sheet_name='Simulacao_Detalhada', index=False)
                
                # Aba 3: Resumo executivo
                total_volume = self.df['VOLUME_TIKTOK_PICKUP'].sum()
                total_points = len(self.df)
                n_clusters = self.df['CLUSTER'].nunique()
                
                executive_summary = pd.DataFrame([
                    {'Metrica': 'Total de Pontos de Coleta', 'Valor': total_points},
                    {'Metrica': 'Volume Total de Pacotes', 'Valor': total_volume},
                    {'Metrica': 'Número de Clusters', 'Valor': n_clusters},
                    {'Metrica': 'Volume Médio por Ponto', 'Valor': f"{total_volume/total_points:.1f}"},
                    {'Metrica': 'Data da Análise', 'Valor': datetime.now().strftime("%d/%m/%Y %H:%M")},
                    {'Metrica': 'Cenário Recomendado', 'Valor': 'Moderado (150 pacotes/veículo, 1 viagem/dia)'},
                ])
                executive_summary.to_excel(writer, sheet_name='Resumo_Executivo', index=False)
                
                # Aba 4: Análise por cluster
                cluster_analysis = []
                for _, rec in recommendations.iterrows():
                    cluster_info = self.clusters[self.clusters['cluster_id'] == rec['cluster_id']].iloc[0]
                    cluster_analysis.append({
                        'Cluster': rec['cluster_id'],
                        'Pontos': rec['n_points'],
                        'Volume_Total': rec['total_volume'],
                        'Cidades_Principais': rec['main_cities'],
                        'Raio_Cobertura_km': f"{cluster_info['max_radius_km']:.1f}",
                        'Veiculos_Conservador': rec['conservative_range'],
                        'Veiculos_Moderado': rec['moderate_range'],
                        'Veiculos_Otimista': rec['optimistic_range'],
                        'Recomendacao': rec['recommended_vehicles']
                    })
                
                cluster_analysis_df = pd.DataFrame(cluster_analysis)
                cluster_analysis_df.to_excel(writer, sheet_name='Analise_por_Cluster', index=False)
            
            print(f"✅ Recomendações de frota exportadas: {filename}")
            print(f"   🚛 Análise de {n_clusters} clusters com {total_volume} pacotes")
            return filename
            
        except Exception as e:
            print(f"❌ Erro ao exportar recomendações: {e}")
            return None

    def export_interactive_map(self, filename=None, add_timestamp=True, add_info_panel=True):
        """Exporta o mapa interativo para HTML"""
        if self.df is None or 'CLUSTER' not in self.df.columns:
            raise ValueError("Execute o clustering primeiro!")
        
        # Definir nome do arquivo
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else ""
            filename = f"mapa_clusters_{timestamp}.html" if timestamp else "mapa_clusters.html"
        elif add_timestamp and not filename.endswith('.html'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename}_{timestamp}.html"
        elif not filename.endswith('.html'):
            filename = f"{filename}.html"
        
        try:
            # Criar mapa base
            m = self.create_interactive_map()
            
            if add_info_panel:
                # Adicionar painel de informações
                total_volume = self.df['VOLUME_TIKTOK_PICKUP'].sum()
                total_points = len(self.df)
                n_clusters = self.df['CLUSTER'].nunique()
                
                info_html = f'''
                <div style="position: fixed; 
                            top: 10px; right: 10px; width: 320px; 
                            background-color: white; border: 2px solid #333; 
                            border-radius: 10px; z-index: 9999; 
                            font-family: Arial, sans-serif; font-size: 12px; 
                            padding: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
                <h3 style="margin-top: 0; color: #333; border-bottom: 2px solid #007cba; padding-bottom: 5px;">
                    📊 Análise de Clusters Logísticos
                </h3>
                <div style="margin: 10px 0;">
                    <p><b>📍 Total de Clusters:</b> {n_clusters}</p>
                    <p><b>📦 Pontos de Coleta:</b> {total_points}</p>
                    <p><b>📋 Volume Total:</b> {total_volume:,} pacotes</p>
                    <p><b>📊 Média por Ponto:</b> {total_volume/total_points:.1f} pacotes</p>
                    <p><b>📅 Data da Análise:</b> {datetime.now().strftime('%d/%m/%Y')}</p>
                </div>
                <hr style="border: 1px solid #ddd;">
                <div style="font-size: 10px; color: #666;">
                    <p><b>Legenda:</b></p>
                    <p>🔵 Pontos de coleta (tamanho = volume)</p>
                    <p>📍 Centróides dos clusters</p>
                    <p>⭕ Área de cobertura</p>
                </div>
                </div>
                '''
                m.get_root().html.add_child(folium.Element(info_html))
                
                # Adicionar título
                title_html = '''
                <h2 align="center" style="font-size: 24px; font-family: Arial, sans-serif; 
                                          color: #333; margin: 20px 0;">
                    🚚 Mapa de Clusters para Otimização Logística
                </h2>
                '''
                m.get_root().html.add_child(folium.Element(title_html))
            
            # Salvar mapa
            m.save(filename)
            
            print(f"✅ Mapa interativo exportado: {filename}")
            print(f"   🗺️  {n_clusters} clusters com {total_points} pontos")
            print(f"   💡 Abra o arquivo em qualquer navegador para visualizar")
            return filename
            
        except Exception as e:
            print(f"❌ Erro ao exportar mapa: {e}")
            return None

    def export_all_results(self, base_filename=None, add_timestamp=True):
        """Exporta todos os resultados (dados, recomendações e mapa)"""
        if self.df is None or 'CLUSTER' not in self.df.columns:
            raise ValueError("Execute a análise completa primeiro!")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else ""
        
        print("📤 EXPORTANDO TODOS OS RESULTADOS...")
        print("=" * 40)
        
        results = {}
        
        # 1. Exportar dados com clusters
        print("1️⃣ Exportando dados com clusters...")
        data_file = base_filename + "_dados" if base_filename else "dados_com_clusters"
        results['data_file'] = self.export_clustered_data(data_file, add_timestamp)
        
        # 2. Exportar recomendações
        print("\n2️⃣ Exportando recomendações de frota...")
        rec_file = base_filename + "_recomendacoes" if base_filename else "recomendacoes_frota"
        results['recommendations_file'] = self.export_fleet_recommendations(rec_file, add_timestamp)
        
        # 3. Exportar mapa
        print("\n3️⃣ Exportando mapa interativo...")
        map_file = base_filename + "_mapa" if base_filename else "mapa_clusters"
        results['map_file'] = self.export_interactive_map(map_file, add_timestamp)
        
        print("\n✅ EXPORTAÇÃO CONCLUÍDA!")
        print(f"📁 Arquivos gerados:")
        for key, filename in results.items():
            if filename:
                print(f"   - {filename}")
        
        return results


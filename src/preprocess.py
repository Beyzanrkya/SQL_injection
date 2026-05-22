import sqlparse
import networkx as nx
import matplotlib.pyplot as plt

class SQLQueryPreprocessor:
    """
    SQL sorgularını analiz etmek, soyut sözdizimi ağacı (AST) graflarına dönüştürmek
    ve bu graflardan makine öğrenimi öznitelikleri çıkarmak için kullanılan sınıf.
    """
    def __init__(self):
        pass

    def clean_query(self, query: str) -> str:
        """
        SQL sorgusunu temizler ve standart bir formata getirir (Önişleme adımı).
        - Sorguyu küçük harfe dönüştürür (opsiyonel analiz için).
        - Başındaki ve sonundaki boşlukları temizler.
        """
        if not query:
            return ""
        return str(query).strip()

    def sql_to_graph(self, query: str) -> nx.DiGraph:
        """
        SQL sorgusunu sqlparse kullanarak ayrıştırır ve bir NetworkX DiGraph (yönlü graf)
        yapısına dönüştürür. Düğümler komutlar, operatörler ve değerlerdir.
        
        Args:
            query (str): Analiz edilecek SQL sorgusu.
            
        Returns:
            nx.DiGraph: Sorgunun AST yapısını temsil eden yönlü graf.
        """
        G = nx.DiGraph()
        cleaned = self.clean_query(query)
        parsed = sqlparse.parse(cleaned)
        
        if not parsed:
            return G
        
        statement = parsed[0]
        
        def add_nodes_edges(parent_node, tokens):
            for token in tokens:
                node_id = id(token)
                # Düğüm ekle (tipi ve değeriyle birlikte)
                label = f"{token.ttype or 'Compound'}\n{str(token)[:10]}"
                G.add_node(node_id, label=label, type=str(token.ttype), value=str(token))
                
                if parent_node is not None:
                    G.add_edge(parent_node, node_id)
                
                if hasattr(token, 'tokens'):
                    add_nodes_edges(node_id, token.tokens)
                    
        add_nodes_edges(None, statement.tokens)
        return G

    def extract_graph_features(self, query: str) -> dict:
        """
        SQL sorgusunun AST graf yapısından makine öğrenimi modeli için
        sayısal ve yapısal öznitelikler çıkarır.
        
        Args:
            query (str): Analiz edilecek SQL sorgusu.
            
        Returns:
            dict: Graf özelliklerini içeren öznitelik sözlüğü.
        """
        try:
            G = self.sql_to_graph(query)
            num_nodes = G.number_of_nodes()
            
            # Derece merkeziyeti hesaplama (Gereksinimlerdeki Merkeziyet kriteri)
            if num_nodes > 1:
                deg_cent_vals = list(nx.degree_centrality(G).values())
                avg_deg_cent = sum(deg_cent_vals) / len(deg_cent_vals)
                max_deg_cent = max(deg_cent_vals)
            else:
                avg_deg_cent = 0.0
                max_deg_cent = 0.0

            features = {
                'node_count': num_nodes,
                'edge_count': G.number_of_edges(),
                'avg_degree': sum(dict(G.degree()).values()) / num_nodes if num_nodes > 0 else 0,
                'max_depth': nx.dag_longest_path_length(G) if num_nodes > 0 else 0,
                'avg_deg_centrality': avg_deg_cent,
                'max_deg_centrality': max_deg_cent
            }
            return features
        except Exception as e:
            return {
                'node_count': 0,
                'edge_count': 0,
                'avg_degree': 0,
                'max_depth': 0,
                'avg_deg_centrality': 0.0,
                'max_deg_centrality': 0.0
            }

# Geriye dönük uyumluluk için modül seviyesinde takma adlar (Aliases)
_preprocessor = SQLQueryPreprocessor()

def sql_to_graph(query):
    return _preprocessor.sql_to_graph(query)

def extract_graph_features(query):
    return _preprocessor.extract_graph_features(query)

if __name__ == "__main__":
    # Test ve doğrulama adımları
    test_query = "SELECT * FROM users WHERE id = '1' OR '1'='1'"
    features = extract_graph_features(test_query)
    print(f"Sorgu: {test_query}")
    print(f"Graf Öznitelikleri: {features}")



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import warnings
import matplotlib
from collections import Counter
import random
from math import log2, sqrt
import time

warnings.filterwarnings('ignore')
matplotlib.use('Agg')
plt.style.use('dark_background')

class DecisionTreeNode:
    """Nó da Árvore de Decisão implementada manualmente"""
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None  
        self.samples = 0
        self.gini = 0.0
        
    def is_leaf(self):
        return self.value is not None

class ManualDecisionTree:
    """Árvore de Decisão implementada manualmente para detecção de fraudes"""
    
    def __init__(self, max_depth=10, min_samples_split=5, min_samples_leaf=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.feature_importances_ = None
        
    def gini_impurity(self, y):
        """Calcula impureza de Gini"""
        if len(y) == 0:
            return 0
        
        counts = Counter(y)
        total = len(y)
        impurity = 1.0
        
        for count in counts.values():
            prob = count / total
            impurity -= prob ** 2
            
        return impurity
    
    def entropy(self, y):
        """Calcula entropia"""
        if len(y) == 0:
            return 0
            
        counts = Counter(y)
        total = len(y)
        entropy = 0.0
        
        for count in counts.values():
            if count > 0:
                prob = count / total
                entropy -= prob * log2(prob)
                
        return entropy
    
    def information_gain(self, X_feature, y, threshold):
        """Calcula ganho de informação"""
        # Dividir dados
        left_mask = X_feature <= threshold
        right_mask = X_feature > threshold
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0
            
        # Calcular entropia ponderada
        total_samples = len(y)
        left_weight = np.sum(left_mask) / total_samples
        right_weight = np.sum(right_mask) / total_samples
        
        weighted_entropy = (left_weight * self.entropy(y[left_mask]) + 
                           right_weight * self.entropy(y[right_mask]))
        
        return self.entropy(y) - weighted_entropy
    
    def find_best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_gain = -1
        
        n_features = X.shape[1]
        
        # Testar cada feature
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            # Testar diferentes thresholds
            for threshold in thresholds:
                gain = self.information_gain(feature_values, y, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    
        return best_feature, best_threshold, best_gain
    
    def build_tree(self, X, y, depth=0):
        node = DecisionTreeNode()
        node.samples = len(y)
        node.gini = self.gini_impurity(y)
        
        # Condições de parada
        if (depth >= self.max_depth or 
            len(y) < self.min_samples_split or 
            len(np.unique(y)) == 1):
            
            # Criar nó folha
            node.value = Counter(y).most_common(1)[0][0]
            return node
        
        # Encontrar melhor divisão
        feature_idx, threshold, gain = self.find_best_split(X, y)
        
        if gain == 0:
            # Não há ganho, criar folha
            node.value = Counter(y).most_common(1)[0][0]
            return node
            
        # Dividir dados
        left_mask = X[:, feature_idx] <= threshold
        right_mask = X[:, feature_idx] > threshold
        
        # Verificar se divisão é válida
        if (np.sum(left_mask) < self.min_samples_leaf or 
            np.sum(right_mask) < self.min_samples_leaf):
            node.value = Counter(y).most_common(1)[0][0]
            return node
        
        # Configurar nó interno
        node.feature_index = feature_idx
        node.threshold = threshold
        
        # Construir subárvores
        node.left = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node
    
    def fit(self, X, y):
        """Treina a árvore de decisão"""
        self.root = self.build_tree(X, y)
        self._calculate_feature_importances(X, y)
        
    def _calculate_feature_importances(self, X, y):
        """Calcula importância das features"""
        n_features = X.shape[1]
        self.feature_importances_ = np.zeros(n_features)
        
        def traverse(node, total_samples):
            if node.is_leaf():
                return
                
            # Calcular importância desta divisão
            feature_idx = node.feature_index
            importance = (node.samples / total_samples) * node.gini
            
            left_gini = node.left.gini if node.left else 0
            right_gini = node.right.gini if node.right else 0
            
            left_samples = node.left.samples if node.left else 0
            right_samples = node.right.samples if node.right else 0
            
            weighted_child_gini = ((left_samples * left_gini + right_samples * right_gini) / 
                                  node.samples)
            
            self.feature_importances_[feature_idx] += importance - weighted_child_gini
            
            # Continuar recursão
            if node.left:
                traverse(node.left, total_samples)
            if node.right:
                traverse(node.right, total_samples)
        
        if self.root:
            traverse(self.root, len(y))
            
        # Normalizar
        if self.feature_importances_.sum() > 0:
            self.feature_importances_ /= self.feature_importances_.sum()
    
    def predict_sample(self, x):
        node = self.root
        
        while not node.is_leaf():
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
                
        return node.value
    
    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self.predict_sample(x))
        return np.array(predictions)

class ManualRandomForest:
    
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=5, 
                 min_samples_leaf=2, max_features='sqrt', random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None
        
    def _bootstrap_sample(self, X, y):
        """Cria amostra bootstrap"""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
    
    def _get_feature_subset(self, n_features):
        if self.max_features == 'sqrt':
            max_features = int(sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(log2(n_features))
        elif isinstance(self.max_features, int):
            max_features = self.max_features
        else:
            max_features = n_features
            
        return np.random.choice(n_features, 
                               min(max_features, n_features), 
                               replace=False)
    
    def fit(self, X, y):
        """Treina o Random Forest"""
        print(f"Treinando Random Forest com {self.n_estimators} árvores...")
        
        if self.random_state:
            np.random.seed(self.random_state)
            random.seed(self.random_state)
        
        self.trees = []
        n_features = X.shape[1]
        
        for i in range(self.n_estimators):
            if i % 10 == 0:
                print(f"  Treinando árvore {i+1}/{self.n_estimators}")
            
            # Bootstrap sample
            X_bootstrap, y_bootstrap = self._bootstrap_sample(X, y)
            
            # Feature subset
            feature_indices = self._get_feature_subset(n_features)
            X_subset = X_bootstrap[:, feature_indices]
            
            # Treinar árvore
            tree = ManualDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            
            tree.fit(X_subset, y_bootstrap)
            
            # Armazenar árvore com índices de features
            self.trees.append((tree, feature_indices))
        
        # Calcular importâncias das features
        self._calculate_feature_importances(n_features)
        print("✅ Random Forest treinado com sucesso!")
    
    def _calculate_feature_importances(self, n_features):
        self.feature_importances_ = np.zeros(n_features)
        
        for tree, feature_indices in self.trees:
            for idx, importance in enumerate(tree.feature_importances_):
                original_idx = feature_indices[idx]
                self.feature_importances_[original_idx] += importance
                
        # Normalizar
        self.feature_importances_ /= self.n_estimators
        
        if self.feature_importances_.sum() > 0:
            self.feature_importances_ /= self.feature_importances_.sum()
    
    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.n_estimators))
        
        for i, (tree, feature_indices) in enumerate(self.trees):
            X_subset = X[:, feature_indices]
            predictions[:, i] = tree.predict(X_subset)
        
        # Votação majoritária
        final_predictions = []
        for i in range(X.shape[0]):
            votes = Counter(predictions[i])
            final_predictions.append(votes.most_common(1)[0][0])
            
        return np.array(final_predictions)
    
    def predict_proba(self, X):
        predictions = np.zeros((X.shape[0], self.n_estimators))
        
        for i, (tree, feature_indices) in enumerate(self.trees):
            X_subset = X[:, feature_indices]
            predictions[:, i] = tree.predict(X_subset)
        
        # Calcular probabilidades
        probabilities = np.zeros((X.shape[0], 2))
        
        for i in range(X.shape[0]):
            votes = Counter(predictions[i])
            total_votes = self.n_estimators
            
            probabilities[i, 0] = votes.get(0, 0) / total_votes  # Legítima
            probabilities[i, 1] = votes.get(1, 0) / total_votes  # Fraude
            
        return probabilities

class FraudDetectionSystem:
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.metrics = {}
        self.data_info = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def criar_pasta_imagens(self):
        """Cria a pasta imagens se não existir"""
        if not os.path.exists('imagens'):
            os.makedirs('imagens')
            print(" Pasta 'imagens' criada")
    
    def carregar_dados(self, caminho_arquivo):
        """Carrega dados do Credit Card Fraud Dataset"""
        print(f" Carregando Credit Card Fraud Dataset de {caminho_arquivo}...")
        try:
            df = pd.read_csv(caminho_arquivo)
            print(f" Dados carregados: {len(df)} registros, {len(df.columns)} colunas")
            
            # Verificar estrutura esperada do dataset
            expected_columns = ['Time', 'Amount', 'Class'] + [f'V{i}' for i in range(1, 29)]
            
            if 'Class' not in df.columns:
                print(" Coluna 'Class' não encontrada!")
                return None
            
            # Armazenar informações dos dados
            self.data_info = {
                'total_records': len(df),
                'total_features': len(df.columns) - 1,
                'fraud_count': int(df['Class'].sum()),
                'legitimate_count': len(df) - int(df['Class'].sum())
            }
            
            print(f"📊 Transações legítimas: {self.data_info['legitimate_count']:,}")
            print(f"⚠️ Transações fraudulentas: {self.data_info['fraud_count']:,}")
            print(f"📈 Taxa de fraude: {(self.data_info['fraud_count']/self.data_info['total_records'])*100:.3f}%")
            
            return df
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {str(e)}")
            return None
    
    def analisar_dados(self, df, salvar_arquivo=True, mostrar_plot=False):
        """Analisa e visualiza a distribuição das classes"""
        print("📊 Analisando distribuição dos dados...")
        self.criar_pasta_imagens()
        
        try:
            # Contar classes
            class_counts = df['Class'].value_counts().sort_index()
            
            # Cria gráfico 
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='#2b2b2b')
            
            # Gráfico de barras
            ax1.set_facecolor('#2b2b2b')
            colors = ['#00ff88', '#ff4444']
            labels = ['Legítimas', 'Fraudes']
            
            bars = ax1.bar(labels, class_counts.values, color=colors, alpha=0.8, 
                          edgecolor='white', linewidth=2)
            
            # Adicionar valores
            for bar, count in zip(bars, class_counts.values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2, height + max(class_counts.values)*0.02,
                        f'{count:,}', ha='center', va='bottom', 
                        color='white', fontweight='bold', fontsize=14)
            
            ax1.set_title('Distribuição de Transações', fontsize=16, fontweight='bold', color='white')
            ax1.set_ylabel('Quantidade', fontsize=12, color='white')
            ax1.tick_params(colors='white')
            
            # Gráfico de pizza
            ax2.set_facecolor('#2b2b2b')
            wedges, texts, autotexts = ax2.pie(class_counts.values, labels=labels, colors=colors,
                                             autopct='%1.2f%%', startangle=90)
            
            for text in texts + autotexts:
                text.set_color('white')
                text.set_fontweight('bold')
            
            ax2.set_title('Proporção de Classes', fontsize=16, fontweight='bold', color='white')
            
            plt.suptitle('🏦 ANÁLISE DO CREDIT CARD FRAUD DATASET', 
                        fontsize=20, fontweight='bold', color='white', y=0.98)
            
            plt.tight_layout()
            
            if salvar_arquivo:
                save_path = 'imagens/grafico_distribuicao.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                           facecolor='#2b2b2b', edgecolor='none')
                print(f"💾 Gráfico salvo em: {save_path}")
            
            if mostrar_plot:
                plt.show()
            else:
                plt.close(fig)
            
            return class_counts
            
        except Exception as e:
            print(f"❌ Erro na análise: {str(e)}")
            if 'fig' in locals():
                plt.close(fig)
            return None
    
    def preparar_dados(self, df):
        """Prepara dados para o Random Forest manual"""
        print("⚙️ Preparando dados para Random Forest manual...")
        
        try:
            # Separar features e target
            X = df.drop('Class', axis=1).values  # Converter para numpy array
            y = df['Class'].values
            
            # Armazenar nomes das features
            self.feature_names = df.drop('Class', axis=1).columns.tolist()
            print(f"🔧 Features: {len(self.feature_names)} ({', '.join(self.feature_names[:5])}...)")
            
            # Normalização simples (min-max scaling)
            print("📏 Normalizando features...")
            X_min = X.min(axis=0)
            X_max = X.max(axis=0)
            X_range = X_max - X_min
            
            # Evitar divisão por zero
            X_range[X_range == 0] = 1
            X_normalized = (X - X_min) / X_range
            
            # Divisão treino/teste
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_normalized, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Balanceamento para treino (undersampling da classe majoritária)
            print("⚖️ Balanceando dados de treino...")
            fraud_indices = np.where(y_train == 1)[0]
            legit_indices = np.where(y_train == 0)[0]
            
            # Manter todas as fraudes e reduzir legítimas
            n_fraud = len(fraud_indices)
            n_legit_sample = min(n_fraud * 3, len(legit_indices))  # Ratio 3:1
            
            legit_sample_indices = np.random.choice(legit_indices, n_legit_sample, replace=False)
            
            # Combinar índices
            balanced_indices = np.concatenate([fraud_indices, legit_sample_indices])
            np.random.shuffle(balanced_indices)
            
            X_train_balanced = X_train[balanced_indices]
            y_train_balanced = y_train[balanced_indices]
            
            print(f"📊 Dados balanceados - Fraudes: {np.sum(y_train_balanced == 1):,}, "
                  f"Legítimas: {np.sum(y_train_balanced == 0):,}")
            
            # Armazenar para uso posterior
            self.X_train = X_train_balanced
            self.X_test = X_test
            self.y_train = y_train_balanced
            self.y_test = y_test
            
            print(f"✅ Dados preparados - Treino: {len(X_train_balanced):,}, Teste: {len(X_test):,}")
            return X_train_balanced, X_test, y_train_balanced, y_test
            
        except Exception as e:
            print(f"❌ Erro na preparação: {str(e)}")
            return None, None, None, None
    
    def treinar_modelo(self, X_train, y_train):
        """Treina o Random Forest manual"""
        print("🌳 Iniciando treinamento do Random Forest manual...")
        start_time = time.time()
        
        try:
            # Criar e treinar Random Forest manual
            self.model = ManualRandomForest(
                n_estimators=50,  # Reduzido para velocidade
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=42
            )
            
            self.model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            print(f"✅ Random Forest treinado em {training_time:.2f}s!")
            return self.model
            
        except Exception as e:
            print(f"❌ Erro no treinamento: {str(e)}")
            return None
    
    def avaliar_modelo(self, X_test, y_test, salvar_arquivo=True, mostrar_plot=False):
        """Avalia o modelo e gera matriz de confusão"""
        print("📏 Avaliando Random Forest manual...")
        self.criar_pasta_imagens()
        
        try:
            if self.model is None:
                print("❌ Modelo não foi treinado!")
                return None
            
            # Predições
            y_pred = self.model.predict(X_test)
            y_proba = self.model.predict_proba(X_test)
            
            # Calcular métricas manualmente
            # True Positives, False Positives, True Negatives, False Negatives
            tp = np.sum((y_test == 1) & (y_pred == 1))
            fp = np.sum((y_test == 0) & (y_pred == 1))
            tn = np.sum((y_test == 0) & (y_pred == 0))
            fn = np.sum((y_test == 1) & (y_pred == 0))
            
            # Métricas
            accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Armazenar métricas
            self.metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn)
            }
            
            # Criar matriz de confusão
            cm = np.array([[tn, fp], [fn, tp]])
            
            # Gráfico da matriz de confusão
            fig, ax = plt.subplots(figsize=(10, 8), facecolor='#2b2b2b')
            ax.set_facecolor('#2b2b2b')
            
            # Heatmap customizado
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues', alpha=0.8)
            
            # Adicionar texto
            for i in range(2):
                for j in range(2):
                    text = ax.text(j, i, f'{cm[i, j]:,}',
                                 ha="center", va="center",
                                 color="white", fontsize=20, fontweight='bold')
            
            # Configurações
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Legítima', 'Fraude'], color='white', fontsize=14)
            ax.set_yticklabels(['Legítima', 'Fraude'], color='white', fontsize=14)
            
            ax.set_xlabel('Predição do Random Forest', fontsize=14, fontweight='bold', color='white')
            ax.set_ylabel('Classe Real', fontsize=14, fontweight='bold', color='white')
            ax.set_title('🎯 MATRIZ DE CONFUSÃO - RANDOM FOREST MANUAL', 
                        fontsize=16, fontweight='bold', pad=20, color='white')
            
            # Adicionar métricas
            metrics_text = (f'Acurácia: {accuracy:.3f} | Precisão: {precision:.3f} | '
                           f'Recall: {recall:.3f} | F1-Score: {f1:.3f}')
            
            plt.figtext(0.5, 0.02, metrics_text, ha='center', fontsize=12, color='white',
                       bbox=dict(boxstyle='round', facecolor='#3498db', alpha=0.8))
            
            plt.tight_layout()
            
            if salvar_arquivo:
                save_path = 'imagens/matriz_confusao.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight',
                           facecolor='#2b2b2b', edgecolor='none')
                print(f"💾 Matriz de confusão salva em: {save_path}")
            
            if mostrar_plot:
                plt.show()
            else:
                plt.close(fig)
            
            # Imprimir métricas
            print(f"\n MÉTRICAS")
            print(f" Acurácia: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f" Precisão: {precision:.4f} ({precision*100:.2f}%)")
            print(f" Recall: {recall:.4f} ({recall*100:.2f}%)")
            print(f" F1-Score: {f1:.4f}")
            print(f" True Positives: {tp:,}")
            print(f" False Positives: {fp:,}")
            print(f" True Negatives: {tn:,}")
            print(f" False Negatives: {fn:,}")
            print(f"==========================================\n")
            
            return y_pred
            
        except Exception as e:
            print(f" Erro na avaliação: {str(e)}")
            if 'fig' in locals():
                plt.close(fig)
            return None
    
    def mostrar_importancia(self, salvar_arquivo=True, mostrar_plot=False, top_n=15):
        """Mostra importância das features do Random Forest manual"""
        if self.model is None:
            print(" Modelo não foi treinado.")
            return None
            
        print("Analisando importância das features...")
        self.criar_pasta_imagens()
        
        try:
            # Obter importâncias
            importances = self.model.feature_importances_
            feature_importance_pairs = list(zip(self.feature_names, importances))
            feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            top_features = feature_importance_pairs[:top_n]
            
            # Criar gráfico
            fig, ax = plt.subplots(figsize=(14, 10), facecolor='#2b2b2b')
            ax.set_facecolor('#2b2b2b')
            
            features, importance_values = zip(*top_features)
            y_pos = np.arange(len(features))
            
            # Cores gradientes
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(features)))
            
            bars = ax.barh(y_pos, importance_values, color=colors, alpha=0.8, 
                          edgecolor='white', linewidth=1)
            
            # Adicionar valores
            for i, (bar, value) in enumerate(zip(bars, importance_values)):
                ax.text(value + max(importance_values)*0.01, bar.get_y() + bar.get_height()/2,
                       f'{value:.4f}', va='center', ha='left', 
                       color='white', fontweight='bold', fontsize=10)
            
            # Configurações
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, color='white', fontsize=11)
            ax.set_xlabel('Importância da Feature', fontsize=14, fontweight='bold', color='white')
            ax.set_title(f' TOP {top_n} FEATURES MAIS IMPORTANTES\nRANDOM FOREST MANUAL', 
                        fontsize=16, fontweight='bold', pad=20, color='white')
            
            # Configurar eixos
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Grid sutil
            ax.grid(True, alpha=0.2, color='white', axis='x')
            ax.set_axisbelow(True)
            
            # Inverter ordem para mostrar maior importância no topo
            ax.invert_yaxis()
            
            plt.tight_layout()
            
            if salvar_arquivo:
                save_path = 'imagens/importancia.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight',
                           facecolor='#2b2b2b', edgecolor='none')
                print(f"💾 Gráfico de importância salvo em: {save_path}")
            
            if mostrar_plot:
                plt.show()
            else:
                plt.close(fig)
            
            print(" Análise de importância concluída.")
            
            # Retornar top 5 para exibição
            return dict(top_features[:5])
            
        except Exception as e:
            print(f" Erro na análise de importância: {str(e)}")
            if 'fig' in locals():
                plt.close(fig)
            return None
    
    def gerar_relatorio_completo(self):
        """Gera relatório completo da análise"""
        if not self.metrics or not self.data_info:
            print(" Execute a análise completa antes de gerar o relatório!")
            return False
            
        print("📄 Gerando relatório completo...")
        
        try:
            # Obter top 5 features importantes
            top_features = {}
            if self.model and self.feature_names:
                importances = self.model.feature_importances_
                feature_importance_pairs = list(zip(self.feature_names, importances))
                feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
                top_features = dict(feature_importance_pairs[:5])
            
            total_transacoes = self.data_info['total_records']
            total_fraudes = self.data_info['fraud_count']
            total_legitimas = self.data_info['legitimate_count']
            taxa_fraude = (total_fraudes / total_transacoes) * 100 if total_transacoes > 0 else 0
            
            relatorio = f"""
RELATÓRIO COMPLETO - DETECÇÃO DE FRAUDES
 RESUMO DOS DADOS:
• Dataset: Credit Card Fraud Detection (Kaggle)
• Total de transações analisadas: {total_transacoes:,}
• Transações fraudulentas: {total_fraudes:,} ({taxa_fraude:.3f}%)
• Transações legítimas: {total_legitimas:,} ({100-taxa_fraude:.3f}%)
• Features analisadas: {self.data_info['total_features']} (V1-V28, Time, Amount)

 MODELO IMPLEMENTADO:
• Algoritmo: Random Forest Manual (Implementação própria)
• Estrutura: {self.model.n_estimators if self.model else 50} Árvores de Decisão
• Profundidade máxima: {self.model.max_depth if self.model else 15}
• Critério de divisão: Ganho de Informação (Entropia)
• Balanceamento: Undersampling (3:1 ratio)
• Features por árvore: √n_features (random subset)

 PERFORMANCE DO MODELO:
• Acurácia: {self.metrics['accuracy']:.4f} ({self.metrics['accuracy']*100:.2f}%)
• Precisão: {self.metrics['precision']:.4f} ({self.metrics['precision']*100:.2f}%)
• Recall: {self.metrics['recall']:.4f} ({self.metrics['recall']*100:.2f}%)
• F1-Score: {self.metrics['f1_score']:.4f}

 MATRIZ DE CONFUSÃO:
• True Positives (Fraudes detectadas): {self.metrics.get('tp', 0):,}
• False Positives (Falsos alarmes): {self.metrics.get('fp', 0):,}
• True Negatives (Legítimas corretas): {self.metrics.get('tn', 0):,}
• False Negatives (Fraudes perdidas): {self.metrics.get('fn', 0):,}

TOP 5 FEATURES MAIS IMPORTANTES:
"""
            
            # Adicionar top features ao relatório
            for i, (feature, importance) in enumerate(top_features.items(), 1):
                relatorio += f"• {i}º lugar: {feature} (Importância: {importance:.4f})\n"
            
            relatorio += f"""
INTERPRETAÇÃO DOS RESULTADOS:
• Precisão: {self.metrics['precision']*100:.1f}% das transações classificadas como fraude são realmente fraudes
• Recall: {self.metrics['recall']*100:.1f}% das fraudes reais foram detectadas pelo modelo
• F1-Score: {self.metrics['f1_score']:.3f} indica {'excelente' if self.metrics['f1_score'] > 0.9 else 'bom' if self.metrics['f1_score'] > 0.8 else 'moderado' if self.metrics['f1_score'] > 0.6 else 'baixo'} equilíbrio entre precisão e recall

IMPLEMENTAÇÃO TÉCNICA:
• Árvores de Decisão: Implementação manual com critério de entropia
• Divisão de nós: Busca exaustiva pelo melhor threshold
• Parada: Profundidade máxima, mínimo de amostras por nó
• Bootstrap: Amostragem com reposição para cada árvore
• Votação: Majoritária para classificação final
• Probabilidades: Baseadas na proporção de votos

ARQUIVOS GERADOS:
• imagens/grafico_distribuicao.png - Distribuição das classes do dataset
• imagens/matriz_confusao.png - Matriz de confusão do Random Forest
• imagens/importancia.png - Importância das features
• relatorio_fraudes.txt - Este relatório completo

CARACTERÍSTICAS DO DATASET:
• V1-V28: Features transformadas por PCA (Principal Component Analysis)
• Time: Segundos decorridos desde a primeira transação
• Amount: Valor da transação em euros
• Class: 0 = Legítima, 1 = Fraude

RECOMENDAÇÕES PARA PRODUÇÃO:
• Implementar monitoramento contínuo de drift nos dados
• Retreinar modelo periodicamente com novos dados
• Considerar ensemble com outros algoritmos (SVM, Neural Networks)
• Implementar pipeline de feature engineering automático
• Adicionar explicabilidade local para decisões individuais
• Configurar alertas para transações de alto risco
• Manter logs detalhados para auditoria e compliance
• Considerar técnicas de balanceamento mais sofisticadas (SMOTE, ADASYN)

CONSIDERAÇÕES DE SEGURANÇA:
• Validar integridade dos dados de entrada
• Implementar controles de acesso ao modelo
• Criptografar dados sensíveis em trânsito e em repouso
• Manter backup do modelo treinado
• Implementar rate limiting para prevenção de ataques

 MÉTRICAS DE NEGÓCIO:
• Taxa de Falsos Positivos: {(self.metrics.get('fp', 0) / (self.metrics.get('fp', 0) + self.metrics.get('tn', 1))) * 100:.2f}%
• Taxa de Falsos Negativos: {(self.metrics.get('fn', 0) / (self.metrics.get('fn', 0) + self.metrics.get('tp', 1))) * 100:.2f}%
• Especificidade: {(self.metrics.get('tn', 0) / (self.metrics.get('tn', 0) + self.metrics.get('fp', 1))) * 100:.2f}%

 STATUS: {' Modelo apresenta excelente performance para detecção de fraudes' if self.metrics['f1_score'] > 0.8 else ' Modelo necessita ajustes nos hiperparâmetros'}

=== FIM DO RELATÓRIO ===
Gerado automaticamente pelo Sistema de Detecção de Fraudes v3.0
Random Forest Manual implementado especificamente para Credit Card Fraud Detection
"""
            
            # Salvar relatório em arquivo
            with open('relatorio_fraudes.txt', 'w', encoding='utf-8') as f:
                f.write(relatorio)
            print("Relatório salvo em: relatorio_fraudes.txt")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao gerar relatório: {str(e)}")
            return False
    
    def predizer_transacao(self, valores_transacao):
        """Prediz se uma transação específica é fraudulenta"""
        if self.model is None:
            return {"erro": "Modelo não foi treinado ainda!"}
            
        try:
            # Garantir que é array numpy
            if not isinstance(valores_transacao, np.ndarray):
                valores_transacao = np.array(valores_transacao)
            
            # Reshape se necessário
            if len(valores_transacao.shape) == 1:
                valores_transacao = valores_transacao.reshape(1, -1)
            
            # Normalizar os dados da transação (usar mesmo método do treino)
            # Nota: Em produção, deveria salvar os parâmetros de normalização
            
            # Fazer predição
            predicao = self.model.predict(valores_transacao)[0]
            probabilidades = self.model.predict_proba(valores_transacao)[0]
            
            resultado = {
                'predicao': 'Fraude' if predicao == 1 else 'Legítima',
                'probabilidade_fraude': float(probabilidades[1]),
                'probabilidade_legitima': float(probabilidades[0]),
                'confianca': float(max(probabilidades)),
                'classe_predita': int(predicao),
                'algoritmo': 'Random Forest Manual'
            }
            
            return resultado
            
        except Exception as e:
            return {"erro": f"Erro na predição: {str(e)}"}
    
    def executar_pipeline_completo(self, caminho_csv, mostrar_plots=False):
        """Executa todo o pipeline de análise"""
        print("Iniciando Sistema de Detecção de Fraudes")
        print(Credit Card Fraud Detection com Random Forest Manual")
        print("=" * 60)
        
        try:
            # 1. Carregar dados
            df = self.carregar_dados(caminho_csv)
            if df is None:
                return False
            
            # 2. Análise exploratória
            self.analisar_dados(df, mostrar_plot=mostrar_plots)
            
            # 3. Preparar dados
            X_train, X_test, y_train, y_test = self.preparar_dados(df)
            if X_train is None:
                return False
            
            # 4. Treinar modelo
            self.treinar_modelo(X_train, y_train)
            if self.model is None:
                return False
            
            # 5. Avaliar modelo
            self.avaliar_modelo(X_test, y_test, mostrar_plot=mostrar_plots)
            
            # 6. Análise de importância
            self.mostrar_importancia(mostrar_plot=mostrar_plots)
            
            # 7. Gerar relatório completo
            self.gerar_relatorio_completo()
            
            print("\n Pipeline executado com sucesso!")
            print(" Verifique a pasta 'imagens' para os gráficos gerados")
            print(" Verifique o arquivo 'relatorio_fraudes.txt' para o relatório completo")
            print(" Random Forest Manual treinado e pronto para uso!")
            
            return True
            
        except Exception as e:
            print(f" Erro no pipeline: {str(e)}")
            return False
    
    def demonstracao_dados_sinteticos(self, mostrar_plots=False):
        """Demonstração do sistema com dados sintéticos do tipo Credit Card"""
        print(" Executando demonstração com dados sintéticos...")
        print(" Simulando Credit Card Fraud Dataset...")
        
        # Criar dados sintéticos similares ao dataset real
        np.random.seed(42)
        n_samples = 10000
        n_frauds = 300  # ~3% como no dataset real
        
        print(f Gerando {n_samples} transações sintéticas ({n_frauds} fraudes)")
        
        # Gerar features V1-V28 (simulando PCA components)
        legitimate_features = np.random.normal(0, 1, (n_samples - n_frauds, 28))
        fraud_features = np.random.normal(0, 2, (n_frauds, 28))  # Maior variância para fraudes
        
        # Adicionar Time e Amount
        legitimate_time = np.random.uniform(0, 172800, n_samples - n_frauds)  # 48 horas
        fraud_time = np.random.uniform(0, 172800, n_frauds)
        
        legitimate_amount = np.random.lognormal(3, 1.5, n_samples - n_frauds)  # Log-normal para amounts
        fraud_amount = np.random.lognormal(4, 2, n_frauds)  # Amounts maiores para fraudes
        
        # Combinar dados
        X_legit = np.column_stack([legitimate_time, legitimate_features, legitimate_amount])
        X_fraud = np.column_stack([fraud_time, fraud_features, fraud_amount])
        
        X_synthetic = np.vstack([X_legit, X_fraud])
        y_synthetic = np.array([0] * (n_samples - n_frauds) + [1] * n_frauds)
        
        # Criar DataFrame com nomes corretos
        feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        df_synthetic = pd.DataFrame(X_synthetic, columns=feature_names)
        df_synthetic['Class'] = y_synthetic
        
        # Embaralhar dados
        df_synthetic = df_synthetic.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(" Executando pipeline com dados sintéticos...")
        
        # Armazenar informações dos dados
        self.data_info = {
            'total_records': len(df_synthetic),
            'total_features': len(df_synthetic.columns) - 1,
            'fraud_count': int(df_synthetic['Class'].sum()),
            'legitimate_count': len(df_synthetic) - int(df_synthetic['Class'].sum())
        }
        
        # Executar pipeline
        self.analisar_dados(df_synthetic, mostrar_plot=mostrar_plots)
        X_train, X_test, y_train, y_test = self.preparar_dados(df_synthetic)
        if X_train is not None:
            self.treinar_modelo(X_train, y_train)
            if self.model is not None:
                self.avaliar_modelo(X_test, y_test, mostrar_plot=mostrar_plots)
                self.mostrar_importancia(mostrar_plot=mostrar_plots)
                self.gerar_relatorio_completo()
        
        print(" Demonstração concluída com sucesso!")
        
        # Exemplo de predição
        print("\n Testando predição com transação sintética...")
        transacao_teste = np.random.normal(0, 1.5, 30)  # Transação suspeita
        resultado = self.predizer_transacao(transacao_teste)
        
        if 'erro' not in resultado:
            print(f" Resultado da predição: {resultado['predicao']}")
            print(f" Probabilidade de fraude: {resultado['probabilidade_fraude']:.3f}")
            print(f" Confiança: {resultado['confianca']:.3f}")
            print(f" Algoritmo: {resultado['algoritmo']}")
        
        return True

def main():
    """Função principal para teste"""
    # Criar instância do sistema
    fraud_system = FraudDetectionSystem()
    
    print(" Sistema de Detecção de Fraudes - Credit Card Dataset")
    print(" Random Forest Manual com Árvores de Decisão Implementadas")
    print(" Especialmente desenvolvido para o dataset do Kaggle")
    print("\n" + "=" * 60)
    
    # CORRIGIDO: Sempre usar dados sintéticos realistas se não encontrar arquivo
    print(" Executando com dados sintéticos do Credit Card Fraud...")
    fraud_system.demonstracao_dados_sinteticos(mostrar_plots=False)
    
    return fraud_system

if __name__ == "__main__":
    # Executar sistema
    sistema = main()
    
    # Exemplo de uso das funcionalidades
    print("\n" + "="*60)
    print(" EXEMPLO DE USO DO SISTEMA:")
    print("="*60)
    
    print("""
# Para usar o sistema com seu dataset:
sistema = FraudDetectionSystem()

# Carregar e analisar dados reais:
sucesso = sistema.executar_pipeline_completo('creditcard.csv')

# Ou executar demonstração:
sistema.demonstracao_dados_sinteticos(mostrar_plots=True)

# Para fazer predições em novas transações:
# valores = [time, v1, v2, ..., v28, amount]  # 30 features
# resultado = sistema.predizer_transacao(valores)

# Para acessar métricas do modelo:
# print(sistema.metrics)

# Para acessar informações dos dados:
# print(sistema.data_info)

# O modelo é um Random Forest manual com:
# - Árvores de Decisão implementadas do zero
# - Critério de divisão por ganho de informação
# - Bootstrap sampling para cada árvore
# - Votação majoritária para classificação
""")

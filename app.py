import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk
import threading
import pandas as pd
import os
from PIL import Image, ImageTk
import webbrowser
import time
from datetime import datetime
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import numpy as np
import sys
import subprocess

# IMPORT DO BACKEND
from backend import FraudDetectionSystem

# Configuração do tema
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class AnimatedProgressBar(ctk.CTkProgressBar):
    """Barra de progresso animada MELHORADA"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_value = 0
        self.animation_running = False
        self.animation_id = None
    
    def animate_to(self, target_value, duration=0.3):
        # Cancelar animação anterior se existir
        if self.animation_id:
            self.after_cancel(self.animation_id)
            self.animation_running = False
        
        if self.animation_running:
            return
        
        self.target_value = target_value
        self.animation_running = True
        start_value = self.get()
        steps = 20
        step_size = (target_value - start_value) / steps
        delay = int(duration * 1000 / steps)
        
        def animate_step(current_step):
            if current_step <= steps and self.animation_running:
                new_value = start_value + (step_size * current_step)
                self.set(new_value)
                
                if current_step < steps:
                    self.animation_id = self.after(delay, lambda: animate_step(current_step + 1))
                else:
                    self.animation_running = False
                    self.animation_id = None
            else:
                self.animation_running = False
                self.animation_id = None
        
        animate_step(0)
    
    def set_immediate(self, value):
        """Define valor imediatamente sem animação"""
        if self.animation_id:
            self.after_cancel(self.animation_id)
        self.animation_running = False
        self.set(value)

class ModernCard(ctk.CTkFrame):
    """Card moderno com animações"""
    def __init__(self, parent, title, value, icon, color, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.title = title
        self.value = value
        self.icon = icon
        self.color = color
        
        self.setup_ui()
        self.bind_hover_events()
    
    def setup_ui(self):
        # Container principal
        self.configure(height=140, corner_radius=20, fg_color=("#f8f9fa", "#2b2b2b"))
        self.pack_propagate(False)
        
        # Header com ícone
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", padx=20, pady=(20, 5))
        
        self.icon_label = ctk.CTkLabel(
            header_frame,
            text=self.icon,
            font=ctk.CTkFont(size=24),
        )
        self.icon_label.pack(side="left")
        
        # Título
        self.title_label = ctk.CTkLabel(
            header_frame,
            text=self.title,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("#666", "#aaa")
        )
        self.title_label.pack(side="right")
        
        # Valor principal
        self.value_label = ctk.CTkLabel(
            self,
            text=self.value,
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color=self.color
        )
        self.value_label.pack(pady=(5, 20))
    
    def bind_hover_events(self):
        def on_enter(event):
            self.configure(fg_color=("#e8f4f8", "#404040"))
        
        def on_leave(event):
            self.configure(fg_color=("#f8f9fa", "#2b2b2b"))
        
        self.bind("<Enter>", on_enter)
        self.bind("<Leave>", on_leave)
        for child in self.winfo_children():
            child.bind("<Enter>", on_enter)
            child.bind("<Leave>", on_leave)
    
    def update_value(self, new_value):
        self.value_label.configure(text=new_value)

class RealTimeChart(ctk.CTkFrame):
    """Gráfico em tempo real integrado CORRIGIDO"""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.setup_chart()
    
    def setup_chart(self):
        # Configurar matplotlib para tema escuro
        plt.style.use('dark_background')
        
        self.fig, self.ax = plt.subplots(figsize=(6, 3), facecolor='#2b2b2b')
        self.ax.set_facecolor('#2b2b2b')
        
        # Canvas para integrar matplotlib no tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        
        # Dados iniciais vazios
        self.clear_chart()
    
    def clear_chart(self):
        self.ax.clear()
        self.ax.text(0.5, 0.5, '📊 Carregue dados para ver o gráfico', 
                    ha='center', va='center', transform=self.ax.transAxes,
                    fontsize=12, color='#888')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()
    
    def update_chart(self, df):
        self.ax.clear()
        
        # CORRIGIDO: Gráfico de distribuição de fraudes
        fraud_counts = df['Class'].value_counts()
        colors = ['#27ae60', '#e74c3c']
        labels = ['Legítimas', 'Fraudes']
        
        # Garantir que as cores correspondam aos labels corretos
        if len(fraud_counts) >= 2:
            # Ordenar para garantir ordem correta (0=Legítimas, 1=Fraudes)
            fraud_counts = fraud_counts.sort_index()
            
        wedges, texts, autotexts = self.ax.pie(
            fraud_counts.values, 
            labels=labels, 
            colors=colors,
            autopct='%1.1f%%',
            startangle=90
        )
        
        self.ax.set_title('Distribuição de Transações', color='white', fontsize=14, pad=20)
        self.canvas.draw()
    
    def load_image_from_file(self, image_path):
        """NOVO: Carrega imagem de arquivo para exibir no canvas"""
        try:
            if os.path.exists(image_path):
                self.ax.clear()
                self.ax.axis('off')
                
                # Carregar e exibir imagem
                img = plt.imread(image_path)
                self.ax.imshow(img)
                self.ax.set_title(os.path.basename(image_path), color='white', fontsize=12)
                
                self.canvas.draw()
                return True
            else:
                self.show_error_message(f"Arquivo não encontrado: {image_path}")
                return False
        except Exception as e:
            self.show_error_message(f"Erro ao carregar imagem: {str(e)}")
            return False
    
    def show_error_message(self, message):
        """Mostra mensagem de erro no gráfico"""
        self.ax.clear()
        self.ax.text(0.5, 0.5, f'❌ {message}', 
                    ha='center', va='center', transform=self.ax.transAxes,
                    fontsize=10, color='red', wrap=True)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()

class AdvancedDataTable(ctk.CTkScrollableFrame):
    """Tabela de dados avançada com filtros e paginação"""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.df = None
        self.current_page = 0
        self.rows_per_page = 50
        self.setup_ui()
    
    def setup_ui(self):
        # Controles superiores
        controls_frame = ctk.CTkFrame(self)
        controls_frame.pack(fill="x", padx=10, pady=10)
        
        # Filtros
        filter_frame = ctk.CTkFrame(controls_frame, fg_color="transparent")
        filter_frame.pack(side="left", padx=10, pady=10)
        
        ctk.CTkLabel(filter_frame, text="Filtrar:", font=ctk.CTkFont(size=12)).pack(side="left", padx=5)
        
        self.filter_var = ctk.StringVar(value="Todas")
        self.filter_combo = ctk.CTkComboBox(
            filter_frame,
            values=["Todas", "Apenas Fraudes", "Apenas Legítimas"],
            variable=self.filter_var,
            command=self.apply_filter,
            width=150
        )
        self.filter_combo.pack(side="left", padx=5)
        
        # Container da tabela
        self.table_container = ctk.CTkFrame(self)
        self.table_container.pack(fill="both", expand=True, padx=10, pady=10)
    
    def load_data(self, df):
        self.df = df.copy()
        self.current_page = 0
        self.update_table()
    
    def apply_filter(self, choice=None):
        if self.df is not None:
            self.current_page = 0
            self.update_table()
    
    def get_filtered_data(self):
        if self.df is None:
            return pd.DataFrame()
        
        df_filtered = self.df.copy()
        
        # Aplicar filtro de tipo
        filter_type = self.filter_var.get()
        if filter_type == "Apenas Fraudes":
            df_filtered = df_filtered[df_filtered['Class'] == 1]
        elif filter_type == "Apenas Legítimas":
            df_filtered = df_filtered[df_filtered['Class'] == 0]
        
        return df_filtered
    
    def update_table(self):
        # Limpar tabela atual
        for widget in self.table_container.winfo_children():
            widget.destroy()
        
        df_filtered = self.get_filtered_data()
        
        if df_filtered.empty:
            no_data_label = ctk.CTkLabel(
                self.table_container,
                text="📄 Nenhum dado encontrado",
                font=ctk.CTkFont(size=16),
                text_color="#7f8c8d"
            )
            no_data_label.pack(expand=True, pady=50)
            return
        
        # Mostrar apenas alguns registros para exemplo
        display_data = df_filtered.head(20)
        
        # Cabeçalho simplificado
        header_frame = ctk.CTkFrame(self.table_container)
        header_frame.pack(fill="x", padx=5, pady=5)
        
        headers = ["ID", "Status"] + list(df_filtered.columns[:3]) if len(df_filtered.columns) > 3 else ["ID", "Status"] + list(df_filtered.columns)
        
        for i, header in enumerate(headers[:5]):  # Limitar a 5 colunas
            label = ctk.CTkLabel(
                header_frame,
                text=header,
                font=ctk.CTkFont(size=12, weight="bold"),
                width=100
            )
            label.grid(row=0, column=i, padx=2, pady=5, sticky="ew")
        
        # Dados
        for idx, (orig_idx, row) in enumerate(display_data.iterrows()):
            row_frame = ctk.CTkFrame(self.table_container)
            row_frame.pack(fill="x", padx=5, pady=1)
            
            # Cor baseada no status
            is_fraud = row['Class'] == 1
            bg_color = ("#ffe6e6", "#4a2c2c") if is_fraud else ("#e6ffe6", "#2c4a2c")
            status_text = "🔴 FRAUDE" if is_fraud else "🟢 LEGÍTIMA"
            
            row_frame.configure(fg_color=bg_color)
            
            # Valores da linha
            values = [str(orig_idx), status_text]
            for col in df_filtered.columns[:3]:
                if col in row:
                    val = row[col]
                    if isinstance(val, (int, float)):
                        values.append(f"{val:.2f}")
                    else:
                        values.append(str(val)[:10])  # Limitar texto
                else:
                    values.append("N/A")
            
            for j, value in enumerate(values[:5]):  # Limitar a 5 colunas
                label = ctk.CTkLabel(
                    row_frame,
                    text=value,
                    font=ctk.CTkFont(size=10),
                    width=100
                )
                label.grid(row=0, column=j, padx=2, pady=2, sticky="ew")

class FraudDetectionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("🔍 Sistema Avançado de Detecção de Fraudes v3.0")
        self.geometry("1400x900")
        self.minsize(1200, 800)
        
        # CORRIGIDO: Instanciar o sistema de backend
        self.fraud_system = FraudDetectionSystem()
        
        # Variáveis de estado
        self.df = None
        self.model = None
        self.predictions = None
        self.current_file = None
        self.processing_start_time = None
        self.analysis_thread = None
        
        # Configurações da interface
        self.setup_styles()
        self.setup_layout()
        
        # Auto-salvar configurações
        self.load_settings()
        
    def setup_styles(self):
        """Configura estilos customizados"""
        self.colors = {
            'primary': '#3498db',
            'success': '#27ae60',
            'danger': '#e74c3c',
            'warning': '#f39c12',
            'info': '#17a2b8',
            'secondary': '#6c757d'
        }
    
    def setup_layout(self):
        """Configura o layout principal melhorado"""
        # Header moderno
        self.create_modern_header()
        
        # Container principal com navegação lateral
        self.create_main_container()
        
        # Status bar melhorado
        self.create_enhanced_status_bar()
    
    def create_modern_header(self):
        """Cria header moderno com gradiente visual"""
        header_frame = ctk.CTkFrame(self, height=100, corner_radius=15)
        header_frame.pack(fill="x", padx=20, pady=20)
        header_frame.pack_propagate(False)
        
        # Container esquerdo - Título e info
        left_container = ctk.CTkFrame(header_frame, fg_color="transparent")
        left_container.pack(side="left", fill="both", expand=True, padx=20, pady=15)
        
        # Título principal com styling
        title_label = ctk.CTkLabel(
            left_container,
            text="🔍 Sistema Avançado de Detecção de Fraudes",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title_label.pack(anchor="w")
        
        # Subtítulo
        subtitle_label = ctk.CTkLabel(
            left_container,
            text="Análise inteligente e detecção em tempo real | Machine Learning",
            font=ctk.CTkFont(size=14),
            text_color=("#666", "#aaa")
        )
        subtitle_label.pack(anchor="w", pady=(5, 0))
        
        # Container direito - Controles
        right_container = ctk.CTkFrame(header_frame, fg_color="transparent")
        right_container.pack(side="right", padx=20, pady=15)
        
        # Botão de upload melhorado
        self.upload_btn = ctk.CTkButton(
            right_container,
            text="📁 Carregar Dataset",
            command=self.upload_csv,
            font=ctk.CTkFont(size=16, weight="bold"),
            height=45,
            width=180,
            corner_radius=25,
            fg_color=self.colors['primary'],
            hover_color=("#2980b9", "#5dade2")
        )
        self.upload_btn.pack(pady=(0, 10))
        
        # Info do arquivo atual
        self.file_info_label = ctk.CTkLabel(
            right_container,
            text="Nenhum arquivo carregado",
            font=ctk.CTkFont(size=12),
            text_color=("#666", "#aaa")
        )
        self.file_info_label.pack()
    
    def create_main_container(self):
        """Cria container principal com navegação lateral"""
        main_container = ctk.CTkFrame(self, corner_radius=15)
        main_container.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        # Sidebar de navegação
        self.create_sidebar(main_container)
        
        # Área de conteúdo principal
        self.content_area = ctk.CTkFrame(main_container, corner_radius=10)
        self.content_area.pack(side="right", fill="both", expand=True, padx=(10, 20), pady=20)
        
        # Configurar páginas
        self.setup_pages()
        self.show_page("dashboard")
    
    def create_sidebar(self, parent):
        """Cria sidebar de navegação"""
        sidebar = ctk.CTkFrame(parent, width=250, corner_radius=10)
        sidebar.pack(side="left", fill="y", padx=20, pady=20)
        sidebar.pack_propagate(False)
        
        # Título da sidebar
        nav_title = ctk.CTkLabel(
            sidebar,
            text="📊 Navegação",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        nav_title.pack(pady=20)
        
        # Botões de navegação
        self.nav_buttons = {}
        nav_items = [
            ("dashboard", "📊 Dashboard", "Visão geral e estatísticas"),
            ("data", "📋 Dados", "Visualizar dataset"),
            ("reports", "📈 Relatórios", "Gráficos e métricas"),
            ("settings", "⚙️ Configurações", "Preferências do sistema")
        ]
        
        for page_id, title, description in nav_items:
            btn_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
            btn_frame.pack(fill="x", padx=15, pady=5)
            
            btn = ctk.CTkButton(
                btn_frame,
                text=title,
                command=lambda p=page_id: self.show_page(p),
                font=ctk.CTkFont(size=14, weight="bold"),
                height=50,
                anchor="w",
                fg_color="transparent",
                text_color=("gray20", "gray80"),
                hover_color=("gray85", "gray25")
            )
            btn.pack(fill="x")
            
            desc_label = ctk.CTkLabel(
                btn_frame,
                text=description,
                font=ctk.CTkFont(size=11),
                text_color=("gray50", "gray60"),
                anchor="w"
            )
            desc_label.pack(fill="x", padx=20, pady=(0, 5))
            
            self.nav_buttons[page_id] = btn
    
    def setup_pages(self):
        """Configura todas as páginas"""
        self.pages = {}
        
        # Dashboard
        self.pages["dashboard"] = self.create_dashboard_page()
        
        # Análise CORRIGIDA
        self.pages["analysis"] = self.create_analysis_page()
        
        # Dados
        self.pages["data"] = self.create_data_page()
        
        # Relatórios CORRIGIDOS
        self.pages["reports"] = self.create_reports_page()
        
        # Configurações
        self.pages["settings"] = self.create_settings_page()
    
    def create_dashboard_page(self):
        """Cria página do dashboard"""
        page = ctk.CTkFrame(self.content_area, fg_color="transparent")
        
        # Título da página
        title_label = ctk.CTkLabel(
            page,
            text="📊 Dashboard Executivo",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=20)
        
        # CORRIGIDO: Área de progresso com callback
        progress_frame = ctk.CTkFrame(page, height=80)
        progress_frame.pack(fill="x", padx=20, pady=10)
        progress_frame.pack_propagate(False)
        
        self.progress_label = ctk.CTkLabel(
            progress_frame,
            text="Pronto para análise",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.progress_label.pack(pady=5)
        
        self.progress_bar = AnimatedProgressBar(progress_frame, height=20)
        self.progress_bar.pack(fill="x", padx=20, pady=5)
        
        # Grid de cards estatísticos
        cards_container = ctk.CTkFrame(page)
        cards_container.pack(fill="x", padx=20, pady=20)
        
        # Configurar grid 2x2
        cards_container.grid_columnconfigure((0, 1), weight=1)
        cards_container.grid_rowconfigure((0, 1), weight=1)
        
        # Cards
        self.cards = {
            'total': ModernCard(
                cards_container, "Total de Transações", "0", "📊", 
                self.colors['info'], corner_radius=15
            ),
            'fraud': ModernCard(
                cards_container, "Fraudes Detectadas", "0", "⚠️", 
                self.colors['danger'], corner_radius=15
            ),
            'legitimate': ModernCard(
                cards_container, "Transações Legítimas", "0", "✅", 
                self.colors['success'], corner_radius=15
            ),
            'rate': ModernCard(
                cards_container, "Taxa de Fraude", "0%", "📈", 
                self.colors['warning'], corner_radius=15
            )
        }
        
        self.cards['total'].grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.cards['fraud'].grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        self.cards['legitimate'].grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        self.cards['rate'].grid(row=1, column=1, padx=10, pady=10, sticky="ew")
        
        # Gráfico em tempo real
        chart_frame = ctk.CTkFrame(page, height=250)
        chart_frame.pack(fill="x", padx=20, pady=20)
        
        self.realtime_chart = RealTimeChart(chart_frame)
        self.realtime_chart.pack(fill="both", expand=True)
        
        # Botões de ação
        actions_frame = ctk.CTkFrame(page, fg_color="transparent")
        actions_frame.pack(fill="x", padx=20, pady=20)
        
        self.action_buttons = {}
        
        self.action_buttons['analyze'] = ctk.CTkButton(
            actions_frame,
            text="🚀 Iniciar Análise Completa",
            command=self.start_complete_analysis,
            font=ctk.CTkFont(size=16, weight="bold"),
            height=50,
            fg_color=self.colors['success'],
            state="disabled"
        )
        self.action_buttons['analyze'].pack(side="left", padx=10)
        
        self.action_buttons['export'] = ctk.CTkButton(
            actions_frame,
            text="💾 Exportar Resultados",
            command=self.export_results,
            font=ctk.CTkFont(size=16, weight="bold"),
            height=50,
            fg_color=self.colors['info'],
            state="disabled"
        )
        self.action_buttons['export'].pack(side="right", padx=10)
        
        return page
    
    def create_analysis_page(self):
        """CORRIGIDO: Cria página de análise detalhada com gráficos"""
        page = ctk.CTkFrame(self.content_area, fg_color="transparent")
        
        title_label = ctk.CTkLabel(
            page,
            text="🔬 Análise de Machine Learning",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=20)
        
        # Área de informações do modelo
        self.model_info_frame = ctk.CTkScrollableFrame(page, height=300)
        self.model_info_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        # NOVO: Área para gráficos da análise
        self.analysis_charts_frame = ctk.CTkFrame(page, height=400)
        self.analysis_charts_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # CORRIGIDO: Gráfico integrado para análise ML
        self.analysis_chart = RealTimeChart(self.analysis_charts_frame)
        self.analysis_chart.pack(fill="both", expand=True)
        
        # Texto inicial
        initial_text = ctk.CTkLabel(
            self.model_info_frame,
            text="🤖 Carregue um dataset e execute a análise para ver:\n\n"
                 "• Métricas de performance do modelo\n"
                 "• Matriz de confusão interativa\n"
                 "• Importância dos atributos\n"
                 "• Relatório de classificação detalhado\n"
                 "• Gráficos de performance em tempo real",
            font=ctk.CTkFont(size=14),
            justify="left"
        )
        initial_text.pack(pady=30)
        
        return page
    
    def create_reports_page(self):
        """CORRIGIDO: Cria página de relatórios com funcionalidade"""
        page = ctk.CTkFrame(self.content_area, fg_color="transparent")
        
        title_label = ctk.CTkLabel(
            page,
            text="📈 Relatórios e Gráficos",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=20)
        
        # CORRIGIDO: Botões para diferentes tipos de gráficos com funções corretas
        buttons_frame = ctk.CTkFrame(page)
        buttons_frame.pack(fill="x", padx=20, pady=20)
        
        # Container dos botões em grid
        buttons_container = ctk.CTkFrame(buttons_frame, fg_color="transparent")
        buttons_container.pack(expand=True, pady=20)
        
        # CORRIGIDO: Botões com funcionalidades reais
        self.report_buttons = {}
        
        self.report_buttons['distribution'] = ctk.CTkButton(
            buttons_container,
            text="📊 Distribuição",
            command=self.show_distribution_chart_corrected,
            height=40,
            width=180,
            font=ctk.CTkFont(size=14, weight="bold"),
            state="disabled"
        )
        self.report_buttons['distribution'].grid(row=0, column=0, padx=10, pady=10)
        
        self.report_buttons['confusion'] = ctk.CTkButton(
            buttons_container,
            text="🎯 Matriz Confusão",
            command=self.show_confusion_matrix_corrected,
            height=40,
            width=180,
            font=ctk.CTkFont(size=14, weight="bold"),
            state="disabled"
        )
        self.report_buttons['confusion'].grid(row=0, column=1, padx=10, pady=10)
        
        self.report_buttons['importance'] = ctk.CTkButton(
            buttons_container,
            text="📈 Importância",
            command=self.show_feature_importance_corrected,
            height=40,
            width=180,
            font=ctk.CTkFont(size=14, weight="bold"),
            state="disabled"
        )
        self.report_buttons['importance'].grid(row=1, column=0, padx=10, pady=10)
        
        self.report_buttons['full_report'] = ctk.CTkButton(
            buttons_container,
            text="📋 Relatório Completo",
            command=self.generate_full_report_corrected,
            height=40,
            width=180,
            font=ctk.CTkFont(size=14, weight="bold"),
            state="disabled"
        )
        self.report_buttons['full_report'].grid(row=1, column=1, padx=10, pady=10)
        
# NOVO: Área para exibir gráficos diretamente na interface
        self.reports_content = ctk.CTkFrame(page)
        self.reports_content.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Gráfico para relatórios
        self.reports_chart = RealTimeChart(self.reports_content)
        self.reports_chart.pack(fill="both", expand=True)
        
        return page
    
    def create_data_page(self):
        """Cria página de visualização de dados simplificada"""
        page = ctk.CTkFrame(self.content_area, fg_color="transparent")
        
        title_label = ctk.CTkLabel(
            page,
            text="📋 Visualização de Dados",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=20)
        
        # Tabela avançada
        self.data_table = AdvancedDataTable(page)
        self.data_table.pack(fill="both", expand=True, padx=20, pady=20)
        
        return page
    
    def create_settings_page(self):
        """Cria página de configurações"""
        page = ctk.CTkFrame(self.content_area, fg_color="transparent")
        
        title_label = ctk.CTkLabel(
            page,
            text="⚙️ Configurações do Sistema",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=20)
        
        # Configurações gerais
        settings_frame = ctk.CTkFrame(page)
        settings_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Tema
        theme_frame = ctk.CTkFrame(settings_frame)
        theme_frame.pack(fill="x", padx=20, pady=20)
        
        ctk.CTkLabel(theme_frame, text="🎨 Tema da Interface:", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=20, pady=10)
        
        theme_var = ctk.StringVar(value="dark")
        theme_options = ctk.CTkSegmentedButton(
            theme_frame,
            values=["light", "dark", "system"],
            variable=theme_var,
            command=self.change_theme
        )
        theme_options.pack(padx=20, pady=10)
        
        # Configurações de análise
        analysis_frame = ctk.CTkFrame(settings_frame)
        analysis_frame.pack(fill="x", padx=20, pady=20)
        
        ctk.CTkLabel(analysis_frame, text="🔬 Configurações de Análise:", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=20, pady=10)
        
        # Auto-save results
        self.auto_save_var = ctk.BooleanVar(value=True)
        auto_save_check = ctk.CTkCheckBox(
            analysis_frame,
            text="Salvar resultados automaticamente",
            variable=self.auto_save_var
        )
        auto_save_check.pack(anchor="w", padx=20, pady=5)
        
        # Show notifications
        self.notifications_var = ctk.BooleanVar(value=True)
        notifications_check = ctk.CTkCheckBox(
            analysis_frame,
            text="Mostrar notificações",
            variable=self.notifications_var
        )
        notifications_check.pack(anchor="w", padx=20, pady=5)
        
        return page
    
    def upload_csv(self):
        """CORRIGIDO: Carrega arquivo CSV com progresso real"""
        file_path = filedialog.askopenfilename(
            title="Selecionar Dataset CSV",
            filetypes=[
                ("Arquivos CSV", "*.csv"),
                ("Arquivos Excel", "*.xlsx;*.xls"),
                ("Todos os arquivos", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        def load_file_threaded():
            try:
                self.current_file = file_path
                
                # Progresso 1: Iniciando
                self.update_progress_threaded(0.1, "📂 Iniciando carregamento...")
                time.sleep(0.3)
                
                # Progresso 2: Lendo arquivo
                self.update_progress_threaded(0.3, "📊 Lendo arquivo...")
                df = self.fraud_system.carregar_dados(file_path)
                
                if df is None:
                    raise ValueError("Erro ao carregar o dataset")
                
                # Progresso 3: Validando
                self.update_progress_threaded(0.6, "🔍 Validando estrutura...")
                time.sleep(0.3)
                
                # Validar estrutura dos dados
                if 'Class' not in df.columns:
                    self.after(0, lambda: messagebox.showwarning(
                        "Aviso", 
                        "Coluna 'Class' não encontrada. Assumindo última coluna como target."
                    ))
                    df.rename(columns={df.columns[-1]: 'Class'}, inplace=True)
                
                # Progresso 4: Atualizando interface
                self.update_progress_threaded(0.8, "📈 Atualizando interface...")
                
                # Armazenar dados
                self.df = df
                
                # Atualizar interface na thread principal
                def update_interface():
                    try:
                        file_name = os.path.basename(file_path)
                        file_size = len(self.df)
                        self.file_info_label.configure(text=f"{file_name} ({file_size:,} registros)")
                        
                        # Habilitar botões
                        self.action_buttons['analyze'].configure(state="normal")
                        
                        # Atualizar tabela de dados
                        if hasattr(self, 'data_table'):
                            self.data_table.load_data(self.df)
                        
                        # Atualizar gráfico
                        self.realtime_chart.update_chart(self.df)
                        
                        # Mostrar estatísticas básicas
                        self.update_basic_stats()
                        
                        # Progresso final
                        self.update_progress_threaded(1.0, "✅ Dataset carregado com sucesso!")
                        self.update_status(f"✅ Dataset carregado: {file_size:,} registros")
                        
                        # Reset da barra após 2 segundos
                        self.after(2000, lambda: self.progress_bar.set(0))
                        self.after(2000, lambda: self.progress_label.configure(text="Pronto para análise"))
                        
                    except Exception as e:
                        print(f"Erro ao atualizar interface: {e}")
                
                self.after(0, update_interface)
                
            except Exception as e:
                def show_error():
                    messagebox.showerror("Erro", f"Erro ao carregar arquivo:\n{str(e)}")
                    self.update_status("❌ Erro ao carregar dataset")
                    self.progress_bar.set(0)
                    self.progress_label.configure(text="Erro no carregamento")
                
                self.after(0, show_error)
        
        # Executar carregamento em thread separada
        threading.Thread(target=load_file_threaded, daemon=True).start()
    
    def start_complete_analysis(self):
        """CORRIGIDO: Inicia análise completa com progresso real"""
        if self.df is None:
            messagebox.showwarning("Aviso", "Carregue um dataset primeiro!")
            return
        
        self.processing_start_time = time.time()
        self.update_status("🚀 Iniciando análise completa...")
        self.update_progress_threaded(0.0, "🚀 Iniciando análise...")
        
        # Desabilitar botão durante processamento
        self.action_buttons['analyze'].configure(state="disabled")
        
        # Executar em thread
        self.analysis_thread = threading.Thread(target=self.process_analysis_corrected, daemon=True)
        self.analysis_thread.start()
    
    def process_analysis_corrected(self):
        """CORRIGIDO: Processa análise completa com callbacks de progresso FUNCIONAIS"""
        try:
            # Etapa 1: Análise exploratória
            self.update_progress_threaded(0.15, "📊 Realizando análise exploratória...")
            self.fraud_system.analisar_dados(self.df, salvar_arquivo=True, mostrar_plot=False)
            time.sleep(0.5)
            
            # Etapa 2: Preparação dos dados
            self.update_progress_threaded(0.35, "🔧 Preparando dados para ML...")
            X_train, X_test, y_train, y_test = self.fraud_system.preparar_dados(self.df)
            time.sleep(0.5)
            
            # Etapa 3: Treinamento do modelo
            self.update_progress_threaded(0.55, "🤖 Treinando modelo de ML...")
            self.fraud_system.treinar_modelo(X_train, y_train)
            time.sleep(0.5)
            
            # Etapa 4: Avaliação
            self.update_progress_threaded(0.75, "📈 Avaliando performance...")
            self.predictions = self.fraud_system.avaliar_modelo(X_test, y_test, salvar_arquivo=True, mostrar_plot=False)
            time.sleep(0.5)
            
            # Etapa 5: Importância dos atributos
            self.update_progress_threaded(0.90, "🔍 Analisando importância...")
            self.fraud_system.mostrar_importancia(salvar_arquivo=True, mostrar_plot=False)
            time.sleep(0.3)
            
            # Etapa 6: Relatório completo
            self.update_progress_threaded(0.95, "📋 Gerando relatórios...")
            self.fraud_system.gerar_relatorio_completo()
            time.sleep(0.2)
            
            # Finalizar
            self.update_progress_threaded(1.0, "✅ Análise concluída com sucesso!")
            
            # Atualizar interface na thread principal
            self.after(0, self.on_analysis_complete_corrected)
            
        except Exception as e:
            self.after(0, lambda: self.on_analysis_error_corrected(str(e)))
    
    def update_progress_threaded(self, value, text):
        """Atualiza progresso de forma thread-safe"""
        def update_ui():
            try:
                if hasattr(self, 'progress_bar') and hasattr(self, 'progress_label'):
                    self.progress_bar.animate_to(value)
                    self.progress_label.configure(text=text)
                    self.update_status(text)
                    # Forçar atualização da interface
                    self.update_idletasks()
            except Exception as e:
                print(f"Erro ao atualizar interface: {e}")
        
        # Agendar atualização na thread principal
        try:
            self.after(0, update_ui)
        except Exception as e:
            print(f"Erro ao agendar atualização: {e}")
    
    def on_analysis_complete_corrected(self):
        """Callback quando análise é concluída"""
        try:
            processing_time = time.time() - self.processing_start_time if self.processing_start_time else 0
            
            # Habilitar botões
            if hasattr(self, 'action_buttons'):
                self.action_buttons['analyze'].configure(state="normal")
                self.action_buttons['export'].configure(state="normal")
            
            # Habilitar botões de relatórios
            if hasattr(self, 'report_buttons'):
                for btn in self.report_buttons.values():
                    btn.configure(state="normal")
            
            # Atualizar informações do modelo
            self.update_model_info_corrected()
            
            # Carregar primeiro gráfico na aba de análise
            self.load_analysis_chart()
            
            # Notificação de conclusão
            if hasattr(self, 'notifications_var') and self.notifications_var.get():
                accuracy = self.fraud_system.metrics.get('accuracy', 0) * 100 if hasattr(self.fraud_system, 'metrics') and self.fraud_system.metrics else 0
                messagebox.showinfo(
                    "Análise Concluída!", 
                    f"✅ Análise finalizada em {processing_time:.1f}s\n"
                    f"🎯 Acurácia do modelo: {accuracy:.2f}%\n"
                    f"📊 Gráficos gerados e prontos para visualização!"
                )
            
            accuracy = self.fraud_system.metrics.get('accuracy', 0) * 100 if hasattr(self.fraud_system, 'metrics') and self.fraud_system.metrics else 0
            self.update_status(f"🎉 Análise concluída em {processing_time:.1f}s - Acurácia: {accuracy:.1f}%")
            
            # Reset da barra após 3 segundos
            if hasattr(self, 'progress_bar') and hasattr(self, 'progress_label'):
                self.after(3000, lambda: self.progress_bar.set(0))
                self.after(3000, lambda: self.progress_label.configure(text="Análise concluída"))
                
        except Exception as e:
            print(f"Erro ao finalizar análise: {e}")
    
    def on_analysis_error_corrected(self, error_msg):
        """Callback para erros na análise"""
        try:
            if hasattr(self, 'action_buttons'):
                self.action_buttons['analyze'].configure(state="normal")
            
            if hasattr(self, 'progress_bar') and hasattr(self, 'progress_label'):
                self.progress_bar.set(0)
                self.progress_label.configure(text="❌ Erro no processamento")
            
            messagebox.showerror("Erro na Análise", f"❌ Erro durante o processamento:\n\n{error_msg}")
            self.update_status("❌ Erro durante a análise")
            
        except Exception as e:
            print(f"Erro ao processar erro da análise: {e}")
    
    def load_analysis_chart(self):
        """Carrega gráfico na aba de análise ML"""
        try:
            # Tentar carregar matriz de confusão primeiro
            confusion_path = "imagens/matriz_confusao.png"
            if os.path.exists(confusion_path) and hasattr(self, 'analysis_chart'):
                self.analysis_chart.load_image_from_file(confusion_path)
            else:
                if hasattr(self, 'analysis_chart'):
                    self.analysis_chart.show_error_message("Execute a análise para ver os gráficos")
        except Exception as e:
            print(f"Erro ao carregar gráfico de análise: {e}")
    
    def update_model_info_corrected(self):
        """Atualiza informações detalhadas do modelo"""
        try:
            # Limpar conteúdo anterior
            if hasattr(self, 'model_info_frame'):
                for widget in self.model_info_frame.winfo_children():
                    widget.destroy()
            
            if not hasattr(self.fraud_system, 'metrics') or not self.fraud_system.metrics:
                return
            
            # Informações do dataset
            dataset_frame = ctk.CTkFrame(self.model_info_frame)
            dataset_frame.pack(fill="x", padx=10, pady=10)
            
            ctk.CTkLabel(
                dataset_frame,
                text="📊 Informações do Dataset",
                font=ctk.CTkFont(size=18, weight="bold")
            ).pack(pady=10)
            
            # Usar dados do fraud_system
            data_info = getattr(self.fraud_system, 'data_info', {})
            dataset_info = f"""📄 Arquivo: {os.path.basename(self.current_file) if self.current_file else 'N/A'}
📈 Total de registros: {data_info.get('total_records', 0):,}
⚠️ Fraudes detectadas: {data_info.get('fraud_count', 0):,}
✅ Transações legítimas: {data_info.get('legitimate_count', 0):,}
📊 Taxa de fraude: {(data_info.get('fraud_count', 0) / max(data_info.get('total_records', 1), 1)) * 100:.3f}%
🔧 Features utilizadas: {data_info.get('total_features', 0)}"""
            
            ctk.CTkLabel(
                dataset_frame,
                text=dataset_info,
                font=ctk.CTkFont(size=12),
                justify="left"
            ).pack(padx=20, pady=10)
            
            # Informações do modelo
            model_frame = ctk.CTkFrame(self.model_info_frame)
            model_frame.pack(fill="x", padx=10, pady=10)
            
            ctk.CTkLabel(
                model_frame,
                text="🤖 Métricas do Modelo",
                font=ctk.CTkFont(size=18, weight="bold")
            ).pack(pady=10)
            
            # Usar métricas reais do backend
            metrics = getattr(self.fraud_system, 'metrics', {})
            processing_time = time.time() - self.processing_start_time if self.processing_start_time else 0
            
            model_info = f"""🔬 Algoritmo: Random Forest Classifier
📊 Divisão treino/teste: 70/30 (com balanceamento)
🎯 Acurácia: {metrics.get('accuracy', 0)*100:.2f}%
🎱 Precisão: {metrics.get('precision', 0)*100:.2f}%
🎪 Recall: {metrics.get('recall', 0)*100:.2f}%
📈 F1-Score: {metrics.get('f1_score', 0):.3f}
🕐 Tempo de processamento: {processing_time:.1f}s"""
            
            ctk.CTkLabel(
                model_frame,
                text=model_info,
                font=ctk.CTkFont(size=12),
                justify="left"
            ).pack(padx=20, pady=10)
            
        except Exception as e:
            print(f"Erro ao atualizar informações do modelo: {e}")
    
    def show_distribution_chart_corrected(self):
        """CORRIGIDO: Mostra gráfico de distribuição na interface"""
        try:
            image_path = "imagens/grafico_distribuicao.png"
            if self.reports_chart.load_image_from_file(image_path):
                self.update_status("📊 Gráfico de distribuição carregado")
                # Mudar para aba de relatórios se não estiver lá
                self.show_page("reports")
            else:
                self.update_status("❌ Gráfico de distribuição não encontrado")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar gráfico:\n{str(e)}")
    
    def show_confusion_matrix_corrected(self):
        """CORRIGIDO: Mostra matriz de confusão na interface"""
        try:
            image_path = "imagens/matriz_confusao.png"
            if self.reports_chart.load_image_from_file(image_path):
                self.update_status("🎯 Matriz de confusão carregada")
                self.show_page("reports")
            else:
                self.update_status("❌ Matriz de confusão não encontrada")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar matriz:\n{str(e)}")
    
    def show_feature_importance_corrected(self):
        """CORRIGIDO: Mostra importância dos atributos na interface"""
        try:
            image_path = "imagens/importancia.png"
            if self.reports_chart.load_image_from_file(image_path):
                self.update_status("📈 Gráfico de importância carregado")
                self.show_page("reports")
            else:
                self.update_status("❌ Gráfico de importância não encontrado")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar gráfico:\n{str(e)}")
    
    def generate_full_report_corrected(self):
        """CORRIGIDO: Gera e exibe relatório completo"""
        try:
            # Abrir o arquivo de relatório se existir
            report_path = "relatorio_fraudes.txt"
            if os.path.exists(report_path):
                if os.name == 'nt':  # Windows
                    os.startfile(report_path)
                elif os.name == 'posix':  # Linux/Mac
                    subprocess.run(['xdg-open', report_path] if os.uname().sysname == 'Linux' else ['open', report_path])
                
                self.update_status("📋 Relatório completo aberto")
            else:
                messagebox.showwarning("Aviso", "Execute a análise primeiro para gerar o relatório!")
                
            # Também mostrar o primeiro gráfico disponível
            for image_name in ["grafico_distribuicao.png", "matriz_confusao.png", "importancia.png"]:
                image_path = f"imagens/{image_name}"
                if os.path.exists(image_path):
                    self.reports_chart.load_image_from_file(image_path)
                    break
                    
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao abrir relatório:\n{str(e)}")
    
    def export_results(self):
        """Exporta resultados da análise"""
        if self.df is None or not hasattr(self.fraud_system, 'model') or self.fraud_system.model is None:
            messagebox.showwarning("Aviso", "Execute uma análise primeiro!")
            return
        
        try:
            export_path = filedialog.asksaveasfilename(
                title="Salvar Resultados",
                defaultextension=".xlsx",
                filetypes=[
                    ("Excel", "*.xlsx"),
                    ("CSV", "*.csv"),
                    ("JSON", "*.json")
                ]
            )
            
            if not export_path:
                return
            
            # Criar relatório
            if export_path.endswith('.xlsx'):
                self.export_to_excel(export_path)
            elif export_path.endswith('.csv'):
                self.export_to_csv(export_path)
            elif export_path.endswith('.json'):
                self.export_to_json(export_path)
            
            messagebox.showinfo("Sucesso", f"✅ Resultados exportados para:\n{export_path}")
            self.update_status(f"💾 Resultados exportados: {os.path.basename(export_path)}")
            
        except Exception as e:
            messagebox.showerror("Erro", f"❌ Erro ao exportar:\n{str(e)}")
    
    def export_to_excel(self, path):
        """Exporta para Excel"""
        try:
            with pd.ExcelWriter(path, engine='openpyxl') as writer:
                # Dados originais
                self.df.to_excel(writer, sheet_name='Dataset Original', index=False)
                
                # Estatísticas
                if hasattr(self.fraud_system, 'data_info') and hasattr(self.fraud_system, 'metrics'):
                    data_info = self.fraud_system.data_info
                    metrics = self.fraud_system.metrics
                    
                    stats_df = pd.DataFrame({
                        'Métrica': [
                            'Total de Transações', 
                            'Fraudes', 
                            'Legítimas', 
                            'Taxa de Fraude (%)',
                            'Acurácia (%)',
                            'Precisão (%)',
                            'Recall (%)',
                            'F1-Score'
                        ],
                        'Valor': [
                            data_info.get('total_records', 0),
                            data_info.get('fraud_count', 0),
                            data_info.get('legitimate_count', 0),
                            f"{(data_info.get('fraud_count', 0) / max(data_info.get('total_records', 1), 1)) * 100:.2f}",
                            f"{metrics.get('accuracy', 0) * 100:.2f}",
                            f"{metrics.get('precision', 0) * 100:.2f}",
                            f"{metrics.get('recall', 0) * 100:.2f}",
                            f"{metrics.get('f1_score', 0):.3f}"
                        ]
                    })
                    stats_df.to_excel(writer, sheet_name='Estatísticas', index=False)
            print(f"✅ Dados exportados para Excel: {path}")
        except Exception as e:
            print(f"❌ Erro ao exportar Excel: {e}")
            raise
    
    def export_to_csv(self, path):
        """Exporta para CSV"""
        try:
            self.df.to_csv(path, index=False)
            print(f"✅ Dados exportados para CSV: {path}")
        except Exception as e:
            print(f"❌ Erro ao exportar CSV: {e}")
            raise
    
    def export_to_json(self, path):
        """Exporta para JSON"""
        try:
            data_info = getattr(self.fraud_system, 'data_info', {})
            metrics = getattr(self.fraud_system, 'metrics', {})
            
            report = {
                'dataset_info': {
                    'file_name': os.path.basename(self.current_file) if self.current_file else 'N/A',
                    'total_records': data_info.get('total_records', 0),
                    'fraud_count': data_info.get('fraud_count', 0),
                    'legitimate_count': data_info.get('legitimate_count', 0),
                    'fraud_rate': float((data_info.get('fraud_count', 0) / max(data_info.get('total_records', 1), 1)) * 100)
                },
                'model_metrics': {
                    'algorithm': 'Random Forest',
                    'accuracy': float(metrics.get('accuracy', 0)),
                    'precision': float(metrics.get('precision', 0)),
                    'recall': float(metrics.get('recall', 0)),
                    'f1_score': float(metrics.get('f1_score', 0)),
                    'features_count': data_info.get('total_features', 0)
                },
                'analysis_info': {
                    'timestamp': datetime.now().isoformat(),
                    'processing_time': time.time() - self.processing_start_time if self.processing_start_time else 0
                }
            }
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"✅ Dados exportados para JSON: {path}")
        except Exception as e:
            print(f"❌ Erro ao exportar JSON: {e}")
            raise
    
    def update_basic_stats(self):
        """Atualiza estatísticas básicas"""
        if self.df is None:
            return
        
        total = len(self.df)
        frauds = int(self.df['Class'].sum())
        legitimate = total - frauds
        fraud_rate = (frauds / total) * 100 if total > 0 else 0
        
        # Atualizar cards
        self.cards['total'].update_value(f"{total:,}")
        self.cards['fraud'].update_value(f"{frauds:,}")
        self.cards['legitimate'].update_value(f"{legitimate:,}")
        self.cards['rate'].update_value(f"{fraud_rate:.2f}%")
    
    def show_page(self, page_name):
        """Mostra a página especificada"""
        # Esconder todas as páginas
        for page in self.pages.values():
            page.pack_forget()
        
        # Mostrar página selecionada
        if page_name in self.pages:
            self.pages[page_name].pack(fill="both", expand=True)
        
        # Atualizar estilo dos botões de navegação
        for btn_id, btn in self.nav_buttons.items():
            if btn_id == page_name:
                btn.configure(fg_color=self.colors['primary'])
            else:
                btn.configure(fg_color="transparent")
    
    def create_enhanced_status_bar(self):
        """Cria barra de status melhorada"""
        status_frame = ctk.CTkFrame(self, height=40, corner_radius=0)
        status_frame.pack(fill="x", side="bottom")
        
        # Status principal
        self.status_label = ctk.CTkLabel(
            status_frame,
            text="💡 Sistema pronto - Carregue um dataset para começar a análise",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(side="left", padx=15, pady=10)
        
        # Informações do sistema
        system_info = ctk.CTkFrame(status_frame, fg_color="transparent")
        system_info.pack(side="right", padx=15, pady=5)
        
        # Timestamp
        self.timestamp_label = ctk.CTkLabel(
            system_info,
            text=datetime.now().strftime("%H:%M:%S"),
            font=ctk.CTkFont(size=10),
            text_color=("#666", "#aaa")
        )
        self.timestamp_label.pack(side="right", padx=5)
        
        # Versão
        version_label = ctk.CTkLabel(
            system_info,
            text="v3.0 Advanced",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=self.colors['primary']
        )
        version_label.pack(side="right", padx=5)
        
        # Atualizar timestamp periodicamente
        self.update_timestamp()
    
    def update_timestamp(self):
        """Atualiza o timestamp na barra de status"""
        self.timestamp_label.configure(text=datetime.now().strftime("%H:%M:%S"))
        self.after(1000, self.update_timestamp)
    
    def change_theme(self, theme):
        """Muda tema da aplicação"""
        try:
            ctk.set_appearance_mode(theme)
            self.update_status(f"🎨 Tema alterado para: {theme}")
        except Exception as e:
            print(f"Erro ao mudar tema: {e}")
    
    def update_status(self, message):
        """Atualiza barra de status"""
        try:
            if hasattr(self, 'status_label'):
                self.status_label.configure(text=message)
        except Exception as e:
            print(f"Erro ao atualizar status: {e}")
    
    def load_settings(self):
        """Carrega configurações salvas"""
        # Implementação futura se necessário
        pass

if __name__ == "__main__":
    print("🚀 Iniciando Sistema de Detecção de Fraudes v3.0")
    print("=" * 60)
    
    try:
        app = FraudDetectionApp()
        app.mainloop()
    except Exception as e:
        print(f"❌ Erro ao iniciar aplicação: {e}")
        import traceback
        traceback.print_exc()# app.py - ARQUIVO COMPLETO COM TODO
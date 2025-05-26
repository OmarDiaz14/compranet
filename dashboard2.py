import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime, timedelta
import numpy as np
from collections import Counter
import calendar
import ast # Importado para safe_eval_list aunque ya estaba implícito

# --- Carga de datos y preprocesamiento ---
# Cargar el archivo CSV proporcionado
df = pd.read_csv("contratos_tic_consolidados_llama3_colab (9).csv", encoding="utf-8")


# Renombrar columnas para que coincidan con las que el script espera
# (basado en el análisis del CSV y los nombres usados en el script)
column_mapping = {
    'CÃ³digo del contrato': 'Código del contrato',
    'TÃ­tulo del contrato': 'Título del contrato',
    'Tipo de contrataciÃ³n': 'Tipo de contratación',
    'DirecciÃ³n del anuncio': 'Dirección del anuncio',
    'Siglas de la InstituciÃ³n': 'Siglas de la Institución'
    # Columnas como 'Fecha de inicio del contrato', 'Importe DRC', 'Orden de gobierno',
    # 'Proveedor o contratista', 'Tipo Procedimiento', 'clasificacion_llm', 'subcategoria_tic'
    # parecen tener nombres compatibles o se usarán directamente.
}
df.rename(columns=column_mapping, inplace=True)

# Crear la columna 'es_TIC' a partir de 'clasificacion_llm'
# Asumiendo que 'clasificacion_llm' tiene valores como 'TIC', 'NO TIC'
if 'clasificacion_llm' in df.columns:
    df['es_TIC'] = df['clasificacion_llm'].apply(lambda x: True if isinstance(x, str) and x.strip().upper() == 'TIC' else False)
else:
    # Si 'clasificacion_llm' no existe, crear 'es_TIC' como False por defecto para evitar errores,
    # aunque esto significará que el filtro 'tic' no encontrará nada.
    df['es_TIC'] = False
    print("Advertencia: La columna 'clasificacion_llm' no se encontró. 'es_TIC' se ha establecido a False para todos los registros.")

# Crear la columna 'terminos_positivos' a partir de 'subcategoria_tic'
# Tratar cada 'subcategoria_tic' como un único término en una lista.
if 'subcategoria_tic' in df.columns:
    df['terminos_positivos'] = df['subcategoria_tic'].apply(
        lambda x: [x.strip()] if pd.notna(x) and isinstance(x, str) and x.strip() != "" else []
    )
else:
    df['terminos_positivos'] = pd.Series([[] for _ in range(len(df))]) # Series de listas vacías
    print("Advertencia: La columna 'subcategoria_tic' no se encontró. 'terminos_positivos' se ha establecido como listas vacías.")


# Asegurarse de que las fechas estén en formato datetime
df['Fecha de inicio del contrato'] = pd.to_datetime(df['Fecha de inicio del contrato'], dayfirst=True, errors='coerce')
df['Fecha de fin del contrato'] = pd.to_datetime(df['Fecha de fin del contrato'], dayfirst=True, errors='coerce')
df['Fecha de firma del contrato'] = pd.to_datetime(df['Fecha de firma del contrato'], errors='coerce') # dayfirst=False o no especificado es el default

# Crear columnas derivadas para análisis
df['Duración del contrato (días)'] = (df['Fecha de fin del contrato'] - df['Fecha de inicio del contrato']).dt.days
df['Mes de inicio'] = df['Fecha de inicio del contrato'].dt.month
df['Año de inicio'] = df['Fecha de inicio del contrato'].dt.year
df['Mes de fin'] = df['Fecha de fin del contrato'].dt.month
df['Año de fin'] = df['Fecha de fin del contrato'].dt.year
df['Tiempo hasta vencimiento (días)'] = (df['Fecha de fin del contrato'] - datetime.now()).dt.days

# Clasificación de oportunidades basada en contratos próximos a vencer
df['Oportunidad'] = pd.cut(
    df['Tiempo hasta vencimiento (días)'],
    bins=[-float('inf'), 0, 90, 180, 365, float('inf')],
    labels=['Vencido', 'Urgente (< 3 meses)', 'Corto plazo (3-6 meses)', 'Medio plazo (6-12 meses)', 'Largo plazo (> 12 meses)']
)

# Asegurarse de que no haya valores nulos en la columna 'Siglas de la Institución'
# y que la columna exista después del posible renombramiento.
if 'Siglas de la Institución' in df.columns:
    siglas_list = df['Siglas de la Institución'].dropna().unique()
else:
    siglas_list = []
    print("Advertencia: La columna 'Siglas de la Institución' no se encontró después del preprocesamiento.")

orden_gobierno_list = df['Orden de gobierno'].dropna().unique() if 'Orden de gobierno' in df.columns else []

# Crear la aplicación Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Definición de pestañas
app.layout = html.Div([
    html.H1("Dashboard de Oportunidades en Contratos TIC Gubernamentales",
            style={'textAlign': 'center', 'color': '#2c3e50', 'margin-bottom': '20px'}),

    # Filtros generales que aplican a todas las pestañas
    html.Div([
        html.Div([
            html.Label("Filtrar por Clasificación TIC"),
            dcc.Dropdown(
                id='tic-dropdown',
                options=[
                    {'label': 'Todos los contratos', 'value': 'todos'},
                    {'label': 'Solo contratos TIC', 'value': 'tic'},
                    {'label': 'Solo contratos no TIC', 'value': 'no_tic'}
                ],
                value='tic'  # Cambiado a 'tic' para enfocarse en contratos TIC por defecto
            ),
        ], style={'width': '23%', 'display': 'inline-block', 'margin-right': '2%'}),

        html.Div([
            html.Label("Seleccionar Institución"),
            dcc.Dropdown(
                id='siglas-dropdown',
                options=[{'label': sigla, 'value': sigla} for sigla in siglas_list],
                multi=True,
                value=[]
            ),
        ], style={'width': '23%', 'display': 'inline-block', 'margin-right': '2%'}),

        html.Div([
            html.Label("Orden de Gobierno"),
            dcc.Dropdown(
                id='gobierno-dropdown',
                options=[{'label': orden, 'value': orden} for orden in orden_gobierno_list],
                multi=True,
                value=[]
            ),
        ], style={'width': '23%', 'display': 'inline-block', 'margin-right': '2%'}),

        html.Div([
            html.Label("Rango de Importes (MXN)"),
            dcc.RangeSlider(
                id='importe-slider',
                min=0,
                # Usar df['Importe DRC'].min() si quieres que el mínimo sea dinámico y no siempre 0
                max=round(df['Importe DRC'].max()) if 'Importe DRC' in df.columns and not df['Importe DRC'].empty else 1000000,
                step=100000,
                marks={i: f"${i/1000000:.1f}M" for i in range(0,
                    int(round(df['Importe DRC'].max())) if 'Importe DRC' in df.columns and not df['Importe DRC'].empty else 1000000,
                    1000000 if ('Importe DRC' in df.columns and not df['Importe DRC'].empty and round(df['Importe DRC'].max()) > 1000000) else 200000 # Ajustar step de marks si el max es pequeño
                )},
                value=[0, round(df['Importe DRC'].max()) if 'Importe DRC' in df.columns and not df['Importe DRC'].empty else 1000000]
            ),
        ], style={'width': '23%', 'display': 'inline-block'}),
    ], style={'margin-bottom': '20px', 'backgroundColor': '#f9f9f9', 'padding': '15px', 'borderRadius': '5px'}),

    # Pestañas
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Resumen y Tendencias', value='tab-1'),
        dcc.Tab(label='Oportunidades de Negocio', value='tab-2'),
        dcc.Tab(label='Análisis Competitivo', value='tab-3'),
        dcc.Tab(label='Distribución Geográfica', value='tab-4'),
        dcc.Tab(label='Análisis de Términos TIC', value='tab-5'),
        # dcc.Tab(label='Análisis de Proveedores', value='tab-6'),
        dcc.Tab(label='Redes de Contratacion', value='tab-network'), # Redes de Concentracion
    ], style={'margin-bottom': '20px'}),

    # Contenido de pestañas
    html.Div(id='tabs-content')
])

# Callback para cambiar el contenido de las pestañas
@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_tab_content(tab):
    if tab == 'tab-1':
        return html.Div([
            # Métricas generales
            html.Div([
                html.Div(id='metric-tic-percentage', className='metric-box'),
                html.Div(id='metric-total-contratos', className='metric-box'),
                html.Div(id='metric-avg-importe', className='metric-box'),
                html.Div(id='metric-total-value', className='metric-box'),
            ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '20px'}),

            # Tendencias principales
            html.Div([
                html.Div([
                    html.H3("Evolución de Contratos TIC por Año"),
                    dcc.Graph(id='tic-trend-graph'),
                ], style={'width': '49%', 'display': 'inline-block'}),

                html.Div([
                    html.H3("Crecimiento de Importe por Categoría"),
                    dcc.Graph(id='importe-growth-graph'),
                ], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),
            ], style={'margin-bottom': '20px'}),

            html.Div([
                html.H3("Inversión TIC Mensual (Estacionalidad)"),
                dcc.Graph(id='seasonality-graph'),
            ], style={'margin-bottom': '20px'}),

            html.Div([
                html.H3("Top 10 Instituciones por Gasto en TIC"),
                dcc.Graph(id='top-institutions-graph'),
            ], style={'margin-bottom': '20px'}),
        ])

    elif tab == 'tab-2': # *** PESTAÑA MODIFICADA ***
        return html.Div([
            # Row 1: Graphs
            html.Div([ 
                html.Div([ # Left Graph
                    html.H3("Oportunidades por Proximidad de Vencimiento"),
                    dcc.Graph(id='opportunities-expiry-graph'),
                ], style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'top'}),

                html.Div([ # Right Graph
                    html.H3("Valor de Contratos Próximos a Vencer"),
                    dcc.Graph(id='value-expiry-graph'),
                ], style={'width': '49%', 'display': 'inline-block', 'float': 'right', 'vertical-align': 'top'}),
            ], style={'margin-bottom': '20px', 'overflow': 'hidden'}),

            # Row 2: Table for contracts from clicked bar in 'opportunities-expiry-graph'
            html.Div(id='opportunity-category-contracts-table-container', 
                     style={'margin-top': '20px', 'margin-bottom': '20px', 'clear': 'both'}), # NUEVO DIV PARA LA TABLA

            # Row 3: Timeline graph
            html.Div([
                html.H3("Línea de Tiempo de Vencimientos de Contratos TIC"),
                dcc.Graph(id='timeline-graph'),
            ], style={'margin-bottom': '20px'}),

            # Row 4: Existing general opportunity table
            html.Div([
                html.H3("Contratos Próximos a Vencer (Oportunidades de Renovación)"),
                html.Div(id='opportunity-table-container')
            ]),
        ])


    elif tab == 'tab-3':
        return html.Div([
            # Fila 1: Gráfica de Market Share y Gráfica de Especialización
            html.Div([
                html.Div([ # Columna Izquierda
                    html.H3("Market Share de Proveedores TIC"),
                    dcc.Graph(id='provider-share-graph'),
                    # La tabla ya NO va aquí para que ocupe el ancho completo
                ], style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'top'}), # 'vertical-align': 'top' es útil

                html.Div([ # Columna Derecha
                    html.H3("Especialización de Proveedores"),
                    dcc.Graph(id='provider-specialization-graph'),
                ], style={'width': '49%', 'display': 'inline-block', 'float': 'right', 'vertical-align': 'top'}),
            ], style={'margin-bottom': '20px', 'overflow': 'hidden'}), # 'overflow': 'hidden' para contener los floats

            # Fila 2: Tabla de Contratos del Proveedor (ocupando el ancho completo)
            html.Div(
                id='provider-contracts-table-container',
                style={'width': '100%', 'margin-top': '20px'} # Asegura que ocupe el ancho y tenga un margen
            ),

            # Fila 3: Gráfica de Duración Promedio
            html.Div([
                html.H3("Duración Promedio de Contratos por Proveedor"),
                dcc.Graph(id='contract-duration-graph'),
            ], style={'margin-bottom': '20px', 'clear': 'both'}), # 'clear': 'both' por si acaso

            # Fila 4: Tabla de Análisis de Competidores
            html.Div([
                html.H3("Análisis de Competidores"),
                html.Div(id='competitor-table-container') # Esta ya debería ocupar el ancho completo por defecto
            ]),
        ])

    elif tab == 'tab-4':
        return html.Div([
            html.Div([
                html.H3("Distribución Geográfica de Contratos TIC"),
                dcc.Graph(id='geo-distribution-graph'),
            ], style={'margin-bottom': '20px'}),

            html.Div([
                html.Div([
                    html.H3("Gasto TIC por Orden de Gobierno"),
                    dcc.Graph(id='government-level-graph'),
                ], style={'width': '49%', 'display': 'inline-block'}),

                html.Div([
                    html.H3("Tipo de Procedimiento por Región"),
                    dcc.Graph(id='procedure-type-graph'),
                ], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),
            ], style={'margin-bottom': '20px'}),
        ])

    elif tab == 'tab-5':
        return html.Div([
            # Fila 1: Gráfico de Top Términos y Gráfico de Tendencias de Términos
            html.Div([
                html.Div([
                    html.H3("Top Términos TIC Encontrados"),
                    dcc.Graph(id='top-terms-graph'),
                    # El html.Div(id='term-contracts-table-container') YA NO VA AQUÍ
                ], style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                html.Div([
                    html.H3("Tendencias de Términos TIC por Año"),
                    dcc.Graph(id='terms-trend-graph'),
                ], style={'width': '49%', 'display': 'inline-block', 'float': 'right', 'verticalAlign': 'top'}),
            ], style={'margin-bottom': '20px', 'overflow': 'hidden'}), # Contenedor para la primera fila de gráficos

            # Fila 2: Contenedor para la tabla de contratos del término seleccionado (ocupará el ancho completo)
            html.Div(
                id='term-contracts-table-container',
                style={'width': '100%', 'margin-top': '20px', 'margin-bottom': '20px'} # Asegura que ocupe el ancho y tenga margen
            ),

            # Fila 3: Gráfico de Co-ocurrencia y Gráfico de Valor Promedio
            html.Div([
                html.Div([
                    html.H3("Co-ocurrencia de Términos"),
                    dcc.Graph(id='term-cooccurrence-graph'),
                ], style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                html.Div([
                    html.H3("Valor Promedio de Contratos por Término TIC"),
                    dcc.Graph(id='term-value-graph'),
                ], style={'width': '49%', 'display': 'inline-block', 'float': 'right', 'verticalAlign': 'top'}),
            ], style={'margin-bottom': '20px', 'clear': 'both', 'overflow': 'hidden'}), # Contenedor para la segunda fila de gráficos
        ])

    elif tab == 'tab-network':
        return html.Div([
            html.H3("Redes de Relaciones Proveedor-Institución"),
            dcc.Graph(id='network-graph', style={'height': '700px'}), # Aumentar altura si es necesario
            html.P("Visualiza que proveedores contratan con que insituciones. El tamaño del enlace representa el valor total de los contratos entre ellos (Top 50 relaciones mostradas)."),
        ])

# Función auxiliar para filtrar datos
def filter_dataframe(df_original, tic_filter, selected_siglas, selected_gobierno, importe_range):
    # Asegurarse de que df_original no sea None y sea un DataFrame
    if df_original is None or not isinstance(df_original, pd.DataFrame):
        return pd.DataFrame() # Devuelve un DataFrame vacío si la entrada no es válida

    filtered_df = df_original.copy()

    # Filtrar por clasificación TIC (asegurarse de que 'es_TIC' exista)
    if 'es_TIC' in filtered_df.columns:
        if tic_filter == 'tic':
            filtered_df = filtered_df[filtered_df['es_TIC'] == True]
        elif tic_filter == 'no_tic':
            filtered_df = filtered_df[filtered_df['es_TIC'] == False]
        # Si es 'todos', no se aplica filtro por 'es_TIC'
    elif tic_filter != 'todos': # Si se espera filtrar por TIC pero la columna no existe
        print(f"Advertencia: La columna 'es_TIC' no existe, no se puede filtrar por clasificación TIC '{tic_filter}'.")
        return pd.DataFrame() # Podría ser mejor devolver df vacío si el filtro es crucial


    # Filtrar por siglas de institución
    if 'Siglas de la Institución' in filtered_df.columns and selected_siglas and len(selected_siglas) > 0:
        filtered_df = filtered_df[filtered_df['Siglas de la Institución'].isin(selected_siglas)]
    elif selected_siglas and len(selected_siglas) > 0: # Si hay selección pero la columna no existe
        print("Advertencia: La columna 'Siglas de la Institución' no existe, no se puede filtrar por institución.")


    # Filtrar por orden de gobierno
    if 'Orden de gobierno' in filtered_df.columns and selected_gobierno and len(selected_gobierno) > 0:
        filtered_df = filtered_df[filtered_df['Orden de gobierno'].isin(selected_gobierno)]
    elif selected_gobierno and len(selected_gobierno) > 0:
        print("Advertencia: La columna 'Orden de gobierno' no existe, no se puede filtrar por orden de gobierno.")


    # Filtrar por rango de importe
    if 'Importe DRC' in filtered_df.columns and importe_range:
        filtered_df = filtered_df[
            (filtered_df['Importe DRC'] >= importe_range[0]) &
            (filtered_df['Importe DRC'] <= importe_range[1])
        ]
    elif importe_range and ('Importe DRC' not in filtered_df.columns):
         print("Advertencia: La columna 'Importe DRC' no existe, no se puede filtrar por importe.")

    return filtered_df

# Callbacks para actualizar las visualizaciones de la pestaña 1: Resumen y Tendencias
@app.callback(
    [Output('metric-tic-percentage', 'children'),
     Output('metric-total-contratos', 'children'),
     Output('metric-avg-importe', 'children'),
     Output('metric-total-value', 'children'),
     Output('tic-trend-graph', 'figure'),
     Output('importe-growth-graph', 'figure'),
     Output('seasonality-graph', 'figure'),
     Output('top-institutions-graph', 'figure')],
    [Input('tic-dropdown', 'value'),
     Input('siglas-dropdown', 'value'),
     Input('gobierno-dropdown', 'value'),
     Input('importe-slider', 'value')]
)
def update_tab1(tic_filter, selected_siglas, selected_gobierno, importe_range):
    filtered_df = filter_dataframe(df, tic_filter, selected_siglas, selected_gobierno, importe_range)

    # Si no hay datos después de filtrar
    if filtered_df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No hay datos para los filtros seleccionados",
            xaxis=dict(title="", showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(title="", showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )

        metric_empty = html.Div([
            html.H4("Sin datos"),
            html.P("0")
        ], style={'textAlign': 'center', 'padding': '10px', 'background': '#f0f0f0', 'borderRadius': '5px', 'flex': '1', 'margin': '5px'})

        return metric_empty, metric_empty, metric_empty, metric_empty, empty_fig, empty_fig, empty_fig, empty_fig

    # Métricas
    total_contratos = len(filtered_df)
    contratos_tic = 0
    if 'es_TIC' in filtered_df.columns:
        contratos_tic = filtered_df['es_TIC'].sum()
    porcentaje_tic = (contratos_tic / total_contratos) * 100 if total_contratos > 0 else 0

    avg_importe = 0
    total_importe = 0
    if 'Importe DRC' in filtered_df.columns and not filtered_df['Importe DRC'].dropna().empty:
        avg_importe = filtered_df['Importe DRC'].mean()
        total_importe = filtered_df['Importe DRC'].sum()


    # Gráfica de tendencia de contratos TIC por año
    fig_trend = go.Figure().update_layout(title="No hay datos para tendencia de contratos")
    if 'Fecha de inicio del contrato' in filtered_df.columns and 'Código del contrato' in filtered_df.columns and 'es_TIC' in filtered_df.columns:
        # Asegurarse que la fecha de inicio no tenga NaT antes de agrupar
        yearly_df_prep = filtered_df.dropna(subset=['Fecha de inicio del contrato'])
        if not yearly_df_prep.empty:
            yearly_df = yearly_df_prep.groupby([yearly_df_prep['Fecha de inicio del contrato'].dt.year, 'es_TIC']).agg(
                count=('Código del contrato', 'count'),
                total_importe=('Importe DRC', 'sum') if 'Importe DRC' in filtered_df.columns else ('Código del contrato', 'count')
            ).reset_index().rename(columns={'Fecha de inicio del contrato': 'Año de inicio del contrato'})


            df_tic_trend = yearly_df[yearly_df['es_TIC'] == True]
            if not df_tic_trend.empty:
                fig_trend = px.line(
                    df_tic_trend,
                    x='Año de inicio del contrato',
                    y='count',
                    title="Evolución de Contratos TIC por Año",
                    labels={'count': 'Número de Contratos', 'Año de inicio del contrato': 'Año'},
                    markers=True,
                    line_shape='linear'
                )
                fig_trend.update_traces(line=dict(color='#2c3e50', width=3), marker=dict(size=10))
            else:
                 fig_trend.update_layout(title="No hay contratos TIC para mostrar tendencia")
        else:
            fig_trend.update_layout(title="No hay fechas válidas para mostrar tendencia")


    # Gráfica de crecimiento de importe por categoría/año
    fig_growth = go.Figure().update_layout(title="No hay datos para crecimiento de importe")
    if 'Importe DRC' in filtered_df.columns and 'Fecha de inicio del contrato' in filtered_df.columns:
        growth_df_prep = filtered_df.dropna(subset=['Fecha de inicio del contrato', 'Importe DRC'])
        if not growth_df_prep.empty:
            # Agrupar por año y una categoría relevante (por ejemplo, Tipo de contratación)
            group_col = 'Tipo de contratación' if 'Tipo de contratación' in growth_df_prep.columns else None
            
            if group_col:
                growth_df = growth_df_prep.groupby([growth_df_prep['Fecha de inicio del contrato'].dt.year, group_col])['Importe DRC'].sum().reset_index()
                growth_df = growth_df.rename(columns={'Fecha de inicio del contrato': 'Año Contrato'})
                if not growth_df.empty:
                    fig_growth = px.area(
                        growth_df,
                        x='Año Contrato',
                        y='Importe DRC',
                        color=group_col,
                        title=f"Crecimiento de Importe por {group_col}",
                        labels={'Importe DRC': 'Importe Total (MXN)', 'Año Contrato': 'Año'},
                    )
                else:
                    fig_growth.update_layout(title=f"No hay datos para crecimiento de importe por {group_col}")
            else:
                # Si no existe la columna de tipo de contratación, usar solo el total por año
                growth_df = growth_df_prep.groupby(growth_df_prep['Fecha de inicio del contrato'].dt.year)['Importe DRC'].sum().reset_index()
                growth_df = growth_df.rename(columns={'Fecha de inicio del contrato': 'Año Contrato'})
                if not growth_df.empty:
                    fig_growth = px.area(
                        growth_df,
                        x='Año Contrato',
                        y='Importe DRC',
                        title="Crecimiento de Importe Total por Año",
                        labels={'Importe DRC': 'Importe Total (MXN)', 'Año Contrato': 'Año'},
                    )
                else:
                    fig_growth.update_layout(title="No hay datos para crecimiento de importe total por año")
        else:
            fig_growth.update_layout(title="No hay fechas o importes válidos para mostrar crecimiento")


    # Gráfica de estacionalidad (patrones mensuales)
    fig_season = go.Figure().update_layout(title="No hay datos para estacionalidad")
    if 'Fecha de inicio del contrato' in filtered_df.columns and 'Código del contrato' in filtered_df.columns:
        season_df_prep = filtered_df.dropna(subset=['Fecha de inicio del contrato'])
        if not season_df_prep.empty:
            season_df = season_df_prep.groupby(season_df_prep['Fecha de inicio del contrato'].dt.month).agg(
                count=('Código del contrato', 'count'),
                total_importe=('Importe DRC', 'sum') if 'Importe DRC' in filtered_df.columns else ('Código del contrato', 'count')
            ).reset_index().rename(columns={'Fecha de inicio del contrato': 'Mes Numérico'})


            # Asignar nombres de meses
            month_names = {i: calendar.month_name[i] for i in range(1, 13)}
            season_df['Mes'] = season_df['Mes Numérico'].map(month_names)
            season_df = season_df.sort_values('Mes Numérico') # Ordenar por mes numérico

            y_col = 'total_importe' if 'total_importe' in season_df.columns and season_df['total_importe'].sum() > 0 else 'count'
            y_title = 'Importe Total (MXN)' if y_col == 'total_importe' else 'Número de Contratos'

            if not season_df.empty:
                fig_season = px.bar(
                    season_df,
                    x='Mes',
                    y=y_col,
                    title=f"Estacionalidad en la Contratación TIC (Análisis Mensual)",
                    labels={y_col: y_title, 'Mes': ''},
                    color=y_col,
                    color_continuous_scale='Viridis'
                )
                fig_season.update_xaxes(
                    categoryorder='array',
                    categoryarray=[calendar.month_name[i] for i in range(1, 13)]
                )
            else:
                fig_season.update_layout(title="No hay datos agregados para estacionalidad")
        else:
            fig_season.update_layout(title="No hay fechas válidas para estacionalidad")


    # Top 10 instituciones por gasto en TIC
    fig_top = go.Figure().update_layout(title="No hay datos para top instituciones")
    if 'Siglas de la Institución' in filtered_df.columns:
        # Si tenemos importe, usamos esa columna para el ranking
        if 'Importe DRC' in filtered_df.columns and not filtered_df['Importe DRC'].dropna().empty:
            top_inst_prep = filtered_df.dropna(subset=['Siglas de la Institución', 'Importe DRC'])
            top_inst = top_inst_prep.groupby('Siglas de la Institución')['Importe DRC'].sum().reset_index()
            top_inst = top_inst.sort_values('Importe DRC', ascending=False).head(10)
            if not top_inst.empty:
                fig_top = px.bar(
                    top_inst,
                    x='Siglas de la Institución',
                    y='Importe DRC',
                    title="Top 10 Instituciones por Gasto en TIC",
                    labels={'Importe DRC': 'Importe Total (MXN)', 'Siglas de la Institución': 'Institución'},
                    color='Importe DRC',
                    color_continuous_scale='Viridis'
                )
            else:
                fig_top.update_layout(title="No hay datos de importe para top instituciones")

        elif 'Código del contrato' in filtered_df.columns: # Si no tenemos importe, usamos el conteo de contratos
            top_inst_prep = filtered_df.dropna(subset=['Siglas de la Institución'])
            top_inst = top_inst_prep.groupby('Siglas de la Institución')['Código del contrato'].count().reset_index(name='count')
            top_inst = top_inst.sort_values('count', ascending=False).head(10)
            if not top_inst.empty:
                fig_top = px.bar(
                    top_inst,
                    x='Siglas de la Institución',
                    y='count',
                    title="Top 10 Instituciones por Número de Contratos TIC",
                    labels={'count': 'Número de Contratos', 'Siglas de la Institución': 'Institución'},
                    color='count',
                    color_continuous_scale='Viridis'
                )
            else:
                fig_top.update_layout(title="No hay datos de conteo para top instituciones")
        else:
            fig_top.update_layout(title="Faltan columnas 'Importe DRC' o 'Código del contrato' para top instituciones")

    # Formatear las métricas para mostrar
    metric_style = {'textAlign': 'center', 'padding': '10px', 'background': '#f0f0f0', 'borderRadius': '5px', 'flex': '1', 'margin': '5px'}
    metric_tic = html.Div([html.H4("Porcentaje TIC"), html.P(f"{porcentaje_tic:.1f}%")], style=metric_style)
    metric_total = html.Div([html.H4("Total Contratos"), html.P(f"{total_contratos:,}")], style=metric_style)
    metric_avg = html.Div([html.H4("Importe Promedio"), html.P(f"${avg_importe:,.2f} MXN")], style=metric_style)
    metric_total_value = html.Div([html.H4("Valor Total"), html.P(f"${total_importe:,.2f} MXN")], style=metric_style)

    return metric_tic, metric_total, metric_avg, metric_total_value, fig_trend, fig_growth, fig_season, fig_top

# Callbacks para actualizar las visualizaciones de la pestaña 2: Oportunidades de Negocio
@app.callback(
    [Output('opportunities-expiry-graph', 'figure'),
     Output('value-expiry-graph', 'figure'),
     Output('timeline-graph', 'figure'),
     Output('opportunity-table-container', 'children')],
    [Input('tic-dropdown', 'value'),
     Input('siglas-dropdown', 'value'),
     Input('gobierno-dropdown', 'value'),
     Input('importe-slider', 'value')]
)
def update_tab2(tic_filter, selected_siglas, selected_gobierno, importe_range):
    filtered_df = filter_dataframe(df, tic_filter, selected_siglas, selected_gobierno, importe_range)

    # Si no hay datos después de filtrar
    if filtered_df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No hay datos para los filtros seleccionados",
            xaxis=dict(title="", showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(title="", showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        empty_table = html.Div("No hay datos para mostrar en la tabla.", style={'textAlign': 'center', 'padding': '20px'})
        return empty_fig, empty_fig, empty_fig, empty_table

    # Oportunidades por proximidad de vencimiento
    fig_opp = go.Figure().update_layout(title="No hay datos de 'Oportunidad' para gráfico")
    if 'Oportunidad' in filtered_df.columns and 'Código del contrato' in filtered_df.columns:
        opp_counts = filtered_df.groupby('Oportunidad', observed=False)['Código del contrato'].count().reset_index(name='count')
        order = ['Vencido', 'Urgente (< 3 meses)', 'Corto plazo (3-6 meses)',
                 'Medio plazo (6-12 meses)', 'Largo plazo (> 12 meses)']
        
        #Asegurar que todas las categorias estén presentes
        for cat in order:
            if cat not in opp_counts['Oportunidad'].values:
                opp_counts = pd.concat([opp_counts, pd.DataFrame({'Oportunidad': [cat], 'count': [0]})], ignore_index=True)
        
        opp_counts['Oportunidad'] = pd.Categorical(opp_counts['Oportunidad'], categories=order, ordered=True)
        opp_counts = opp_counts.sort_values('Oportunidad')

        if not opp_counts.empty:
            fig_opp = px.bar(
                opp_counts,
                x='Oportunidad',
                y='count',
                title="Distribución de Contratos por Proximidad de Vencimiento",
                labels={'count': 'Número de Contratos', 'Oportunidad': ''},
                color='Oportunidad',
                color_discrete_map={'Vencido': 'grey', 'Urgente (< 3 meses)': 'red',
                                    'Corto plazo (3-6 meses)': 'orange',
                                    'Medio plazo (6-12 meses)': 'blue',
                                    'Largo plazo (> 12 meses)': 'green'})
        else:
            fig_opp.update_layout(title="No hay datos agregados para oportunidades por vencimiento")


    # Valor de contratos próximos a vencer
    fig_value = go.Figure().update_layout(title="No hay datos de 'Importe DRC' u 'Oportunidad' para gráfico")
    if 'Oportunidad' in filtered_df.columns and 'Importe DRC' in filtered_df.columns:
        value_by_expiry_prep = filtered_df.dropna(subset=['Importe DRC'])
        if not value_by_expiry_prep.empty:
            value_by_expiry = value_by_expiry_prep.groupby('Oportunidad', observed=False)['Importe DRC'].sum().reset_index()
            order = ['Vencido', 'Urgente (< 3 meses)', 'Corto plazo (3-6 meses)',
                     'Medio plazo (6-12 meses)', 'Largo plazo (> 12 meses)']

            for cat in order:
                if cat not in value_by_expiry['Oportunidad'].values:
                    value_by_expiry = pd.concat([value_by_expiry, pd.DataFrame({'Oportunidad': [cat], 'Importe DRC': [0]})], ignore_index=True)

            value_by_expiry['Oportunidad'] = pd.Categorical(value_by_expiry['Oportunidad'], categories=order, ordered=True)
            value_by_expiry = value_by_expiry.sort_values('Oportunidad')

            if not value_by_expiry.empty:
                fig_value = px.bar(value_by_expiry, x='Oportunidad', y='Importe DRC',
                                   title="Valor Total de Contratos por Proximidad de Vencimiento",
                                   labels={'Importe DRC': 'Importe Total (MXN)', 'Oportunidad': ''},
                                   color='Oportunidad',
                                   color_discrete_map={'Vencido': 'grey', 'Urgente (< 3 meses)': 'red',
                                                       'Corto plazo (3-6 meses)': 'orange',
                                                       'Medio plazo (6-12 meses)': 'blue',
                                                       'Largo plazo (> 12 meses)': 'green'})
            else:
                fig_value.update_layout(title="No hay datos agregados para valor por vencimiento")
        else:
            fig_value.update_layout(title="No hay importes válidos para valor por vencimiento")

    # Línea de tiempo de vencimientos
    fig_timeline = go.Figure().update_layout(title="No hay datos para línea de tiempo de vencimientos")
    if 'Fecha de fin del contrato' in filtered_df.columns:
        now = datetime.now()
        # Considerar contratos que vencen en el futuro o vencieron recientemente (ej. últimos 90 días)
        timeline_df_prep = filtered_df[
            (filtered_df['Fecha de fin del contrato'] > now - timedelta(days=90)) & # Vencieron en los últimos 90 días
            (filtered_df['Fecha de fin del contrato'].notna()) # Asegurar que la fecha no sea NaT
        ].copy()


        if not timeline_df_prep.empty:
            # Crear tooltip_info de forma segura
            def create_tooltip(row):
                titulo = row.get('Título del contrato', 'N/A')
                siglas = row.get('Siglas de la Institución', 'N/A')
                proveedor = row.get('Proveedor o contratista', 'N/A')
                importe_val = row.get('Importe DRC', None)
                importe_str = f"${importe_val:,.2f} MXN" if pd.notna(importe_val) else "Importe: N/A"
                return f"Contrato: {titulo}<br>Institución: {siglas}<br>Proveedor: {proveedor}<br>{importe_str}"

            timeline_df_prep['tooltip_info'] = timeline_df_prep.apply(create_tooltip, axis=1)
            
            y_col_timeline = 'Importe DRC' if 'Importe DRC' in timeline_df_prep.columns and timeline_df_prep['Importe DRC'].notna().any() else 'Siglas de la Institución'
            size_col_timeline = 'Importe DRC' if 'Importe DRC' in timeline_df_prep.columns and timeline_df_prep['Importe DRC'].notna().any() else None
            color_col_timeline = 'Oportunidad' if 'Oportunidad' in timeline_df_prep.columns else None
            hover_name_col = 'Título del contrato' if 'Título del contrato' in timeline_df_prep.columns else None


            fig_timeline = px.scatter(
                timeline_df_prep,
                x='Fecha de fin del contrato',
                y=y_col_timeline,
                size=size_col_timeline,
                color=color_col_timeline,
                hover_name=hover_name_col,
                hover_data=['tooltip_info'],
                title="Línea de Tiempo de Vencimientos de Contratos",
                labels={
                    'Fecha de fin del contrato': 'Fecha de Vencimiento',
                    y_col_timeline: 'Importe (MXN)' if y_col_timeline == 'Importe DRC' else 'Institución',
                    'Oportunidad': 'Categoría Oportunidad'
                },
                color_discrete_map={ # Asegúrate que estos labels coincidan con los de 'Oportunidad'
                    'Vencido': 'grey',
                    'Urgente (< 3 meses)': 'red',
                    'Corto plazo (3-6 meses)': 'orange',
                    'Medio plazo (6-12 meses)': 'blue',
                    'Largo plazo (> 12 meses)': 'green'
                }
            )
            fig_timeline.add_vline(x=now, line_dash="dash", line_color="red")
            fig_timeline.add_annotation(
                x=now, y=1, yref="paper", text="Hoy", showarrow=False, yshift=10,
            )
        else:
            fig_timeline.update_layout(title="No hay contratos próximos a vencer o vencidos recientemente")


    # Tabla de oportunidades
    opportunity_table_content = html.Div("No hay datos disponibles para mostrar oportunidades en tabla.", style={'textAlign': 'center', 'padding': '20px'})
    if 'Tiempo hasta vencimiento (días)' in filtered_df.columns:
        # Filtrar solo contratos que vencen en los próximos 12 meses (o ya vencieron y están en 'Oportunidad')
        opp_df = filtered_df[
            (filtered_df['Tiempo hasta vencimiento (días)'] <= 365) # Incluye vencidos y hasta 1 año
        ].copy()

        if not opp_df.empty:
            opp_df = opp_df.sort_values('Tiempo hasta vencimiento (días)')
            display_cols = ['Siglas de la Institución', 'Título del contrato', 'Proveedor o contratista',
                            'Fecha de fin del contrato', 'Tiempo hasta vencimiento (días)', 'Oportunidad']
            if 'Importe DRC' in opp_df.columns:
                display_cols.append('Importe DRC')

            table_cols_present = [col for col in display_cols if col in opp_df.columns]
            opp_table_df = opp_df[table_cols_present].head(20).copy() # Usar .copy() para evitar SettingWithCopyWarning

            if 'Fecha de fin del contrato' in opp_table_df.columns:
                opp_table_df['Fecha de fin del contrato'] = opp_table_df['Fecha de fin del contrato'].dt.strftime('%d/%m/%Y')
            if 'Importe DRC' in opp_table_df.columns:
                # Primero, genera la serie de strings formateados
                formatted_importes = opp_table_df['Importe DRC'].apply(lambda x: f"${x:,.2f} MXN" if pd.notna(x) else "N/A")
                # Luego, asigna los valores como un array de NumPy para ser más explícito
                opp_table_df.loc[:, 'Importe DRC'] = formatted_importes.to_numpy()



            opportunity_table_content = dash_table.DataTable(
                id='opportunity-table',
                columns=[{"name": col.replace('_', ' ').title(), "id": col} for col in opp_table_df.columns],
                data=opp_table_df.to_dict('records'),
                style_table={'overflowX': 'auto', 'width': '100%'},
                style_cell={'textAlign': 'left', 'padding': '10px', 'whiteSpace': 'normal', 'height': 'auto', 'minWidth': '100px'},
                style_header={'backgroundColor': '#2c3e50', 'color': 'white', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {'if': {'row_index': 'odd'}, 'backgroundColor': '#f9f9f9'},
                    {'if': {'column_id': 'Oportunidad', 'filter_query': '{Oportunidad} contains "Urgente"'}, 'backgroundColor': '#ffcccc', 'color': 'black'},
                    {'if': {'column_id': 'Oportunidad', 'filter_query': '{Oportunidad} contains "Corto plazo"'}, 'backgroundColor': '#fff2cc', 'color': 'black'},
                    {'if': {'column_id': 'Oportunidad', 'filter_query': '{Oportunidad} contains "Vencido"'}, 'backgroundColor': '#dddddd', 'color': 'black'}
                ],
                page_size=10
            )
        else:
            opportunity_table_content = html.Div("No hay contratos próximos a vencer o vencidos recientemente en los filtros seleccionados.", style={'textAlign': 'center', 'padding': '20px'})

    return fig_opp, fig_value, fig_timeline, opportunity_table_content

#Callbacks para ;a tabla de contratos de la pestana 2: Oportunidades de Negocio
@app.callback(
    Output('opportunity-category-contracts-table-container', 'children'),
    [Input('opportunities-expiry-graph', 'clickData'),
     Input('tic-dropdown', 'value'), 
     Input('siglas-dropdown', 'value'),
     Input('gobierno-dropdown', 'value'),
     Input('importe-slider', 'value')]
)
def display_opportunity_category_contracts_table(click_data, tic_filter, selected_siglas, selected_gobierno, importe_range):
    ctx = dash.callback_context
    if not click_data or not click_data['points']:
        return html.P("Haz clic en una barra del gráfico 'Oportunidades por Proximidad de Vencimiento' para ver los contratos asociados.",
                      style={'textAlign': 'center', 'marginTop': '20px'})

    try:
        selected_category = click_data['points'][0]['x']
    except (KeyError, IndexError, TypeError) as e:
        print(f"Error al extraer la categoría de oportunidad: {e}, click_data: {click_data}")
        return html.P("No se pudo obtener la categoría seleccionada del gráfico.",
                      style={'textAlign': 'center', 'marginTop': '20px', 'color': 'red'})

    # Aplicar filtros globales
    filtered_df_global = filter_dataframe(df.copy(), tic_filter, selected_siglas, selected_gobierno, importe_range)

    if filtered_df_global.empty:
        return html.P(f"No hay datos para los filtros generales seleccionados al buscar contratos para la categoría '{selected_category}'.",
                      style={'textAlign': 'center', 'marginTop': '20px'})

    if 'Oportunidad' not in filtered_df_global.columns:
        return html.P("La columna 'Oportunidad' no se encuentra en los datos filtrados.",
                      style={'color': 'red', 'textAlign': 'center', 'marginTop': '20px'})

    # Filtrar por la categoría de oportunidad seleccionada
    category_contracts_df = filtered_df_global[
        filtered_df_global['Oportunidad'] == selected_category
    ].copy()

    if category_contracts_df.empty:
        return html.P(f"No se encontraron contratos para la categoría '{selected_category}' con los filtros actuales.",
                      style={'textAlign': 'center', 'marginTop': '20px'})

    # Definir columnas y formatear
    cols_for_table_display = []
    
    col_titulo_script = 'Título del contrato'
    col_institucion_script = 'Siglas de la Institución'
    col_proveedor_script = 'Proveedor o contratista'
    col_importe_script = 'Importe DRC'
    col_fecha_inicio_script = 'Fecha de inicio del contrato'
    col_fecha_fin_script = 'Fecha de fin del contrato'
    col_anuncio_script = 'Dirección del anuncio'
    col_oportunidad_script = 'Oportunidad' # Puede mostrarse para confirmar

    current_cols_for_selection = []

    if col_titulo_script in category_contracts_df.columns:
        current_cols_for_selection.append(col_titulo_script)
        cols_for_table_display.append({"name": "Título del Contrato", "id": col_titulo_script})
    
    if col_institucion_script in category_contracts_df.columns:
        current_cols_for_selection.append(col_institucion_script)
        cols_for_table_display.append({"name": "Institución", "id": col_institucion_script})

    if col_proveedor_script in category_contracts_df.columns:
        current_cols_for_selection.append(col_proveedor_script)
        cols_for_table_display.append({"name": "Proveedor", "id": col_proveedor_script})

    if col_importe_script in category_contracts_df.columns:
        current_cols_for_selection.append(col_importe_script)
        category_contracts_df.loc[:, col_importe_script] = pd.to_numeric(category_contracts_df[col_importe_script], errors='coerce')
        cols_for_table_display.append({
            "name": "Importe DRC", "id": col_importe_script, "type": "numeric",
            "format": dash_table.Format.Format(scheme=dash_table.Format.Scheme.fixed, precision=2, group=True, symbol=dash_table.Format.Symbol.yes, symbol_prefix='$')
        })
    
    if col_fecha_inicio_script in category_contracts_df.columns:
        current_cols_for_selection.append(col_fecha_inicio_script)
        category_contracts_df.loc[:, col_fecha_inicio_script] = pd.to_datetime(category_contracts_df[col_fecha_inicio_script], errors='coerce').dt.strftime('%d/%m/%Y')
        cols_for_table_display.append({"name": "Fecha Inicio", "id": col_fecha_inicio_script})

    if col_fecha_fin_script in category_contracts_df.columns:
        current_cols_for_selection.append(col_fecha_fin_script)
        category_contracts_df.loc[:, col_fecha_fin_script] = pd.to_datetime(category_contracts_df[col_fecha_fin_script], errors='coerce').dt.strftime('%d/%m/%Y')
        cols_for_table_display.append({"name": "Fecha Fin", "id": col_fecha_fin_script})

    if col_oportunidad_script in category_contracts_df.columns: # Opcional, ya que todos serán de la misma categoría
        current_cols_for_selection.append(col_oportunidad_script)
        cols_for_table_display.append({"name": "Categoría Oportunidad", "id": col_oportunidad_script})


    warning_message_anuncio = None
    md_col_name_oportunidad = 'Enlace_Anuncio_Oportunidad_MD' # Nombre único para esta tabla
    if col_anuncio_script in category_contracts_df.columns:
        category_contracts_df[md_col_name_oportunidad] = category_contracts_df[col_anuncio_script].apply(
            lambda x: f"[Ver Anuncio]({x})" if pd.notna(x) and str(x).strip().lower().startswith('http') else "N/A"
        )
        current_cols_for_selection.append(md_col_name_oportunidad)
        cols_for_table_display.append({"name": "Anuncio", "id": md_col_name_oportunidad, 'presentation': 'markdown'})
    else:
        warning_message_anuncio = html.P(f"Advertencia: La columna '{col_anuncio_script}' no fue encontrada.", style={'color': 'orange', 'textAlign': 'center', 'fontSize': '0.9em'})

    if not current_cols_for_selection:
         return html.P(f"No hay columnas con datos para mostrar para la categoría '{selected_category}'.",
                       style={'textAlign': 'center', 'marginTop': '20px', 'color': 'red'})
    
    # Ordenar por 'Tiempo hasta vencimiento (días)' para mostrar los más urgentes primero dentro de la categoría
    # Esto es útil si la categoría no es 'Vencido', donde el tiempo es negativo o cero.
    data_for_table = category_contracts_df.copy()
    if 'Tiempo hasta vencimiento (días)' in data_for_table.columns:
        data_for_table = data_for_table.sort_values(by='Tiempo hasta vencimiento (días)', ascending=True)
    
    data_for_table = data_for_table[current_cols_for_selection]


    table = dash_table.DataTable(
        id='opportunity-category-specific-contracts-table', # ID único para esta tabla
        columns=cols_for_table_display,
        data=data_for_table.to_dict('records'),
        style_table={'overflowX': 'auto', 'marginTop': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'width':'100%'},
        style_header={'backgroundColor': '#34495e', 'color': 'white', 'fontWeight': 'bold', 'textAlign': 'center', 'padding': '10px'},
        style_cell={
            'textAlign': 'left', 'padding': '8px',
            'minWidth': '100px', 'maxWidth': '350px',
            'whiteSpace': 'normal', 'height': 'auto',
            'border': '1px solid #eee',
            'fontFamily': 'Arial, sans-serif', 'fontSize': '13px',
            'verticalAlign': 'middle'
        },
        style_cell_conditional=[
            {'if': {'column_id': col_titulo_script}, 'minWidth': '200px', 'fontWeight': 'bold'},
            {'if': {'column_id': col_importe_script}, 'textAlign': 'right', 'minWidth': '130px'},
            {'if': {'column_id': md_col_name_oportunidad if col_anuncio_script in category_contracts_df.columns else ''}, 
             'textAlign': 'center', 'minWidth': '100px'}
        ],
        style_data_conditional=[
            {'if': {'row_index': 'odd'}, 'backgroundColor': '#f9f9f9'}
        ],
        page_size=10,
        filter_action='native',
        sort_action='native',
    )

    return html.Div([
        html.H4(f"Contratos en la categoría de oportunidad: '{selected_category}'",
                style={'marginTop': '20px', 'marginBottom': '10px', 'textAlign': 'center', 'color': '#2c3e50'}),
        warning_message_anuncio if warning_message_anuncio else "",
        table
    ])


# Callbacks para actualizar las visualizaciones de la pestaña 3: Análisis Competitivo
@app.callback(
    [Output('provider-share-graph', 'figure'),
     Output('provider-specialization-graph', 'figure'),
     Output('contract-duration-graph', 'figure'),
     Output('competitor-table-container', 'children')],
    [Input('tic-dropdown', 'value'),
     Input('siglas-dropdown', 'value'),
     Input('gobierno-dropdown', 'value'),
     Input('importe-slider', 'value')]
)
def update_tab3(tic_filter, selected_siglas, selected_gobierno, importe_range):
    filtered_df = filter_dataframe(df, tic_filter, selected_siglas, selected_gobierno, importe_range)

    if filtered_df.empty:
        empty_fig = go.Figure().update_layout(title="No hay datos para los filtros seleccionados", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        empty_table = html.Div("No hay datos para mostrar en la tabla.", style={'textAlign': 'center', 'padding': '20px'})
        return empty_fig, empty_fig, empty_fig, empty_table

    # Market Share de Proveedores (Top 10)
    fig_share = go.Figure().update_layout(title="No hay datos de proveedores")
    if 'Proveedor o contratista' in filtered_df.columns:
        provider_col = 'Proveedor o contratista'
        if 'Importe DRC' in filtered_df.columns and not filtered_df[provider_col].dropna().empty and not filtered_df['Importe DRC'].dropna().empty:
            provider_share = filtered_df.groupby(provider_col)['Importe DRC'].sum().reset_index()
            provider_share = provider_share.sort_values('Importe DRC', ascending=False).head(10)
            if not provider_share.empty:
                fig_share = px.pie(
                    provider_share, values='Importe DRC', names=provider_col,
                    title="Market Share de Proveedores TIC (por Importe)", hole=0.4
                )
        elif 'Código del contrato' in filtered_df.columns and not filtered_df[provider_col].dropna().empty : # Por número de contratos
            provider_share = filtered_df.groupby(provider_col)['Código del contrato'].count().reset_index(name='count')
            provider_share = provider_share.sort_values('count', ascending=False).head(10)
            if not provider_share.empty:
                fig_share = px.pie(
                    provider_share, values='count', names=provider_col,
                    title="Market Share de Proveedores TIC (por Número de Contratos)", hole=0.4
                )


    # Especialización de Proveedores
    fig_spec = go.Figure().update_layout(title="No hay datos para especialización de proveedores")
    if 'Proveedor o contratista' in filtered_df.columns and 'terminos_positivos' in filtered_df.columns:
        # Asegurarse que terminos_positivos es una lista
        df_spec_prep = filtered_df[filtered_df['terminos_positivos'].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()
        if not df_spec_prep.empty:
            df_spec_prep['primer_termino'] = df_spec_prep['terminos_positivos'].apply(lambda x: x[0] if x else None) # Tomar el primer término como especialización
            
            # Contar frecuencia de (Proveedor, primer_termino)
            # Usaremos el valor del contrato para determinar los "top proveedores" y luego su especialización
            if 'Importe DRC' in df_spec_prep.columns:
                top_providers_by_value = df_spec_prep.groupby('Proveedor o contratista')['Importe DRC'].sum().nlargest(15).index
                spec_df_focus = df_spec_prep[df_spec_prep['Proveedor o contratista'].isin(top_providers_by_value)]
                
                # Para cada uno de estos proveedores, encontrar su término más frecuente o el de su contrato más grande
                # Simplificamos: tomamos el primer término del primer contrato (o el más común si hay muchos)
                provider_specialization_list = []
                for provider in top_providers_by_value:
                    provider_contracts = spec_df_focus[spec_df_focus['Proveedor o contratista'] == provider]
                    if not provider_contracts.empty:
                        all_terms_for_provider = [term for sublist in provider_contracts['terminos_positivos'] for term in sublist]
                        if all_terms_for_provider:
                            term_counts = Counter(all_terms_for_provider)
                            most_common = term_counts.most_common(1)[0]
                            provider_specialization_list.append({
                                'Proveedor': provider,
                                'Término Especialización': most_common[0],
                                'Frecuencia': most_common[1] # Frecuencia de ese término para ese proveedor
                            })
                if provider_specialization_list:
                    spec_df_viz = pd.DataFrame(provider_specialization_list)
                    fig_spec = px.bar(
                        spec_df_viz.sort_values('Frecuencia', ascending=False),
                        x='Proveedor', y='Frecuencia', color='Término Especialización',
                        title="Especialización de Proveedores TIC (Término Más Común para Top Proveedores)",
                        labels={'Frecuencia': 'Frecuencia del Término Principal', 'Proveedor': ''}
                    )
                    fig_spec.update_layout(xaxis={'categoryorder':'total descending'})


    # Duración Promedio de Contratos por Proveedor
    fig_duration = go.Figure().update_layout(title="No hay datos para duración de contratos")
    if 'Proveedor o contratista' in filtered_df.columns and 'Duración del contrato (días)' in filtered_df.columns:
        duration_df_prep = filtered_df.dropna(subset=['Proveedor o contratista', 'Duración del contrato (días)'])
        if not duration_df_prep.empty:
            duration_df = duration_df_prep.groupby('Proveedor o contratista')['Duración del contrato (días)'].agg(
                ['mean', 'count']).reset_index()
            duration_df.columns = ['Proveedor o contratista', 'Duración Promedio', 'Número de Contratos']
            duration_df = duration_df[duration_df['Número de Contratos'] >= 2].sort_values(
                'Duración Promedio', ascending=False).head(15)
            if not duration_df.empty:
                fig_duration = px.bar(
                    duration_df, x='Proveedor o contratista', y='Duración Promedio', color='Número de Contratos',
                    title="Duración Promedio de Contratos por Proveedor (en días, Top 15)",
                    labels={'Duración Promedio': 'Días', 'Proveedor o contratista': ''}, color_continuous_scale='Viridis'
                )
                fig_duration.update_layout(xaxis={'categoryorder':'total descending'})

    # Tabla de análisis de competidores
    competitor_table_content = html.Div("No hay datos para análisis de competidores.", style={'textAlign': 'center', 'padding': '20px'})
    if 'Proveedor o contratista' in filtered_df.columns and 'Código del contrato' in filtered_df.columns:
        agg_dict = {
            'total_contratos': ('Código del contrato', 'count'),
            'instituciones_unicas': ('Siglas de la Institución', lambda x: x.nunique() if 'Siglas de la Institución' in filtered_df.columns else 0)
        }
        if 'Importe DRC' in filtered_df.columns:
            agg_dict['valor_total'] = ('Importe DRC', 'sum')
            agg_dict['valor_promedio'] = ('Importe DRC', 'mean')
        if 'Duración del contrato (días)' in filtered_df.columns:
            agg_dict['duracion_promedio'] = ('Duración del contrato (días)', 'mean')

        competitor_analysis = filtered_df.groupby('Proveedor o contratista').agg(**agg_dict).reset_index()

        sort_col = 'valor_total' if 'valor_total' in competitor_analysis.columns else 'total_contratos'
        competitor_analysis = competitor_analysis.sort_values(sort_col, ascending=False).head(15).copy()


        if 'valor_total' in competitor_analysis.columns:
            formatted_valor_total = competitor_analysis['valor_total'].apply(lambda x: f"${x:,.2f} MXN" if pd.notna(x) else "N/A")
            competitor_analysis['valor_total'] = formatted_valor_total.to_numpy()
        if 'valor_promedio' in competitor_analysis.columns:
            formatted_valor_promedio = competitor_analysis['valor_promedio'].apply(lambda x: f"${x:,.2f} MXN" if pd.notna(x) else "N/A")
            competitor_analysis['valor_promedio'] = formatted_valor_promedio.to_numpy()
        if 'duracion_promedio' in competitor_analysis.columns:
            formatted_duracion_promedio = competitor_analysis['duracion_promedio'].apply(lambda x: f"{x:.0f} días" if pd.notna(x) else "N/A")
            competitor_analysis['duracion_promedio'] = formatted_duracion_promedio.to_numpy()

        
        if not competitor_analysis.empty:
            competitor_table_content = dash_table.DataTable(
                id='competitor-table',
                columns=[{"name": col.replace('_', ' ').title(), "id": col} for col in competitor_analysis.columns],
                data=competitor_analysis.to_dict('records'),
                style_table={'overflowX': 'auto', 'width': '100%'},
                style_cell={'textAlign': 'left', 'padding': '8px', 'whiteSpace': 'normal', 'minWidth': '100px'},
                style_header={'backgroundColor': '#2c3e50', 'color': 'white', 'fontWeight': 'bold'},
                style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#f9f9f9'}],
                page_size=15
            )
            
    return fig_share, fig_spec, fig_duration, competitor_table_content


####### Callbacks para mostrar los contratos al hacer clic en el gráfico de proveedores #######
@app.callback(
    Output('provider-contracts-table-container', 'children'),
    [Input('provider-share-graph', 'clickData'),
     Input('tic-dropdown', 'value'), # Pasar filtros globales
     Input('siglas-dropdown', 'value'),
     Input('gobierno-dropdown', 'value'),
     Input('importe-slider', 'value')]
)
def display_provider_contracts_table(click_data, tic_filter, selected_siglas, selected_gobierno, importe_range):
    ctx = dash.callback_context
    if not click_data or not click_data['points']:
        return html.P("Haga clic en un proveedor en el gráfico de Market Share para ver sus contratos.",
                      style={'textAlign': 'center', 'marginTop': '20px'})

    try:
        provider_name = click_data['points'][0]['label']
    except (KeyError, IndexError, TypeError):
        return html.P("No se pudo obtener el proveedor seleccionado del gráfico.",
                      style={'textAlign': 'center', 'marginTop': '20px', 'color': 'red'})

    # Aplicar los filtros globales al DataFrame original 'df'
    # Es importante usar el 'df' global que ya tiene las columnas preprocesadas
    filtered_df_for_provider = filter_dataframe(df, tic_filter, selected_siglas, selected_gobierno, importe_range)

    if filtered_df_for_provider.empty:
        return html.P(f"No hay datos para los filtros generales seleccionados al buscar contratos de '{provider_name}'.",
                      style={'textAlign': 'center', 'marginTop': '20px'})

    if 'Proveedor o contratista' not in filtered_df_for_provider.columns:
        return html.P("La columna 'Proveedor o contratista' no se encuentra en los datos filtrados.",
                      style={'color': 'red', 'textAlign': 'center', 'marginTop': '20px'})

    provider_contracts_df = filtered_df_for_provider[
        filtered_df_for_provider['Proveedor o contratista'] == provider_name
    ].copy() # Usar .copy() para evitar SettingWithCopyWarning

    if provider_contracts_df.empty:
        return html.P(f"No se encontraron contratos para '{provider_name}' con los filtros actuales.",
                      style={'textAlign': 'center', 'marginTop': '20px'})

    # Definir columnas y formatear
    cols_to_display_data = pd.DataFrame()
    cols_for_table = []
    
    col_titulo_script = 'Título del contrato' # Nombre esperado por el script
    col_importe_script = 'Importe DRC'
    col_fecha_inicio_script = 'Fecha de inicio del contrato'
    col_fecha_fin_script = 'Fecha de fin del contrato'
    col_anuncio_script = 'Dirección del anuncio' # Nombre esperado por el script

    # Preparar columnas para la tabla
    current_cols = []
    if col_titulo_script in provider_contracts_df.columns:
        current_cols.append(col_titulo_script)
        cols_for_table.append({"name": "Título del Contrato", "id": col_titulo_script})
    if col_importe_script in provider_contracts_df.columns:
        current_cols.append(col_importe_script)
        # Formatear Importe DRC como numérico en la tabla
        provider_contracts_df.loc[:, col_importe_script] = pd.to_numeric(provider_contracts_df[col_importe_script], errors='coerce')
        cols_for_table.append({
            "name": "Importe DRC", "id": col_importe_script, "type": "numeric",
            "format": dash_table.Format.Format(scheme=dash_table.Format.Scheme.fixed, precision=2, group=True, symbol=dash_table.Format.Symbol.yes, symbol_prefix='$')
        })
    if col_fecha_inicio_script in provider_contracts_df.columns:
        current_cols.append(col_fecha_inicio_script)
        provider_contracts_df.loc[:, col_fecha_inicio_script] = pd.to_datetime(provider_contracts_df[col_fecha_inicio_script], errors='coerce').dt.strftime('%d/%m/%Y')
        cols_for_table.append({"name": "Fecha Inicio", "id": col_fecha_inicio_script})
    if col_fecha_fin_script in provider_contracts_df.columns:
        current_cols.append(col_fecha_fin_script)
        provider_contracts_df.loc[:, col_fecha_fin_script] = pd.to_datetime(provider_contracts_df[col_fecha_fin_script], errors='coerce').dt.strftime('%d/%m/%Y')
        cols_for_table.append({"name": "Fecha Fin", "id": col_fecha_fin_script})

    warning_message = None
    if col_anuncio_script in provider_contracts_df.columns:
        provider_contracts_df['Enlace_Anuncio_MD'] = provider_contracts_df[col_anuncio_script].apply(
            lambda x: f"[Ver Anuncio]({x})" if pd.notna(x) and str(x).strip().lower().startswith('http') else "N/A"
        )
        current_cols.append('Enlace_Anuncio_MD')
        cols_for_table.append({"name": "Anuncio", "id": "Enlace_Anuncio_MD", 'presentation': 'markdown'})
    else:
        warning_message = html.P("Advertencia: La columna 'Dirección del anuncio' no fue encontrada.", style={'color': 'orange', 'textAlign': 'center'})

    cols_to_display_data = provider_contracts_df[current_cols]

    if not cols_for_table:
        return html.P(f"No hay columnas válidas para mostrar para el proveedor '{provider_name}'.",
                      style={'textAlign': 'center', 'marginTop': '20px', 'color': 'red'})

    table = dash_table.DataTable(
        id='provider-specific-contracts-table',
        columns=cols_for_table,
        data=cols_to_display_data.to_dict('records'),
        style_table={'overflowX': 'auto', 'marginTop': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'width':'100%'},
        style_header={'backgroundColor': '#34495e', 'color': 'white', 'fontWeight': 'bold', 'textAlign': 'center', 'padding': '10px'},
        style_cell={'textAlign': 'left', 'padding': '8px', 'minWidth': '120px', 'maxWidth': '400px', 'whiteSpace': 'normal', 'height': 'auto', 'border': '1px solid #eee', 'fontFamily': 'Arial, sans-serif', 'fontSize': '13px', 'verticalAlign': 'middle'},
        style_cell_conditional=[
            {'if': {'column_id': col_titulo_script}, 'minWidth': '250px', 'fontWeight': 'bold'},
            {'if': {'column_id': col_importe_script}, 'textAlign': 'right', 'minWidth': '150px'},
            {'if': {'column_id': 'Enlace_Anuncio_MD'}, 'textAlign': 'center', 'minWidth': '100px'}
        ],
        style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#f9f9f9'}],
        page_size=5, # Menos filas para esta tabla específica
        filter_action='native',
        sort_action='native',
    )
    return html.Div([
        html.H4(f"Contratos para: {provider_name}", style={'marginTop': '20px', 'marginBottom': '5px', 'textAlign': 'center'}),
        warning_message if warning_message else "",
        table
    ])

# Callbacks para actualizar las visualizaciones de la pestaña 4: Distribución Geográfica
@app.callback(
    [Output('geo-distribution-graph', 'figure'),
     Output('government-level-graph', 'figure'),
     Output('procedure-type-graph', 'figure')],
    [Input('tic-dropdown', 'value'),
     Input('siglas-dropdown', 'value'),
     Input('gobierno-dropdown', 'value'),
     Input('importe-slider', 'value')]
)
def update_tab4(tic_filter, selected_siglas, selected_gobierno, importe_range):
    filtered_df = filter_dataframe(df, tic_filter, selected_siglas, selected_gobierno, importe_range)

    if filtered_df.empty:
        empty_fig = go.Figure().update_layout(title="No hay datos para los filtros seleccionados", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        return empty_fig, empty_fig, empty_fig

    # Mapa de distribución geográfica (simplificado por la falta de coordenadas)
    fig_geo = go.Figure().update_layout(title="No hay datos para distribución por orden de gobierno")
    if 'Orden de gobierno' in filtered_df.columns and 'Código del contrato' in filtered_df.columns:
        geo_df = filtered_df.groupby('Orden de gobierno')['Código del contrato'].count().reset_index(name='count')
        if not geo_df.empty:
            fig_geo = px.bar(
                geo_df, x='Orden de gobierno', y='count',
                title="Distribución de Contratos por Orden de Gobierno (Simulado)",
                labels={'count': 'Número de Contratos', 'Orden de gobierno': ''}, color='Orden de gobierno'
            )
            fig_geo.add_annotation(
                x=0.5, y=-0.2, xref="paper", yref="paper", align="center",
                text="Nota: Visualización por Orden de Gobierno. Para mapa geográfico se requieren datos de ubicación.",
                showarrow=False, font=dict(size=10, color="gray")
            )

    # Gasto TIC por Orden de Gobierno
    fig_gov = go.Figure().update_layout(title="No hay datos para gasto por orden de gobierno")
    if 'Orden de gobierno' in filtered_df.columns and 'Importe DRC' in filtered_df.columns:
        gov_spending_prep = filtered_df.dropna(subset=['Importe DRC'])
        if not gov_spending_prep.empty:
            gov_spending = gov_spending_prep.groupby('Orden de gobierno')['Importe DRC'].sum().reset_index()
            gov_spending = gov_spending.sort_values('Importe DRC', ascending=False)
            if not gov_spending.empty and gov_spending['Importe DRC'].sum() > 0 :
                fig_gov = px.pie(
                    gov_spending, values='Importe DRC', names='Orden de gobierno',
                    title="Distribución del Gasto TIC por Orden de Gobierno", hole=0.4
                )

    # Tipo de Procedimiento por Región/Gobierno
    fig_proc = go.Figure().update_layout(title="No hay datos para tipo de procedimiento")
    if 'Tipo Procedimiento' in filtered_df.columns and 'Orden de gobierno' in filtered_df.columns and 'Código del contrato' in filtered_df.columns:
        proc_df_prep = filtered_df.dropna(subset=['Tipo Procedimiento', 'Orden de gobierno'])
        if not proc_df_prep.empty:
            proc_df = proc_df_prep.groupby(['Orden de gobierno', 'Tipo Procedimiento'])['Código del contrato'].count().reset_index(name='count')
            if not proc_df.empty:
                fig_proc = px.bar(
                    proc_df, x='Orden de gobierno', y='count', color='Tipo Procedimiento',
                    title="Tipos de Procedimiento por Orden de Gobierno",
                    labels={'count': 'Número de Contratos', 'Orden de gobierno': ''}, barmode='group'
                )
    return fig_geo, fig_gov, fig_proc


# Callbacks para actualizar las visualizaciones de la pestaña 5: Análisis de Términos TIC
@app.callback(
    [Output('top-terms-graph', 'figure'),
     Output('terms-trend-graph', 'figure'),
     Output('term-cooccurrence-graph', 'figure'),
     Output('term-value-graph', 'figure')],
    [Input('tic-dropdown', 'value'),
     Input('siglas-dropdown', 'value'),
     Input('gobierno-dropdown', 'value'),
     Input('importe-slider', 'value')]
)
def update_tab5(tic_filter, selected_siglas, selected_gobierno, importe_range):
    filtered_df = filter_dataframe(df, tic_filter, selected_siglas, selected_gobierno, importe_range)

    empty_fig_layout = dict(
        title="No hay datos para los filtros seleccionados",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    fig_terms = go.Figure().update_layout(**empty_fig_layout)
    fig_trend = go.Figure().update_layout(**empty_fig_layout)
    fig_cooccur = go.Figure().update_layout(**empty_fig_layout)
    fig_value_term = go.Figure().update_layout(**empty_fig_layout)

    if filtered_df.empty or 'terminos_positivos' not in filtered_df.columns:
        return fig_terms, fig_trend, fig_cooccur, fig_value_term

    # Función segura para obtener listas de términos (ya definida globalmente, pero podría redefinirse o asegurarse que está en scope si es necesario)
    def safe_eval_list(val): # Redefinida aquí para asegurar que el callback la tenga, aunque en el script original estaba global
        if isinstance(val, list): return val
        if isinstance(val, str):
            try:
                # import ast # ya importado arriba
                parsed = ast.literal_eval(val) # Esto no se usará si 'terminos_positivos' ya es lista
                if isinstance(parsed, list): return parsed
            except (ValueError, SyntaxError): pass
        return []

    # Como 'terminos_positivos' ya es una lista de strings por el preprocesamiento, no necesitamos safe_eval_list
    # Necesitamos aplanar la lista de listas de términos
    all_terms_series = filtered_df['terminos_positivos'].dropna()
    all_terms_flat = [term for sublist in all_terms_series for term in sublist if term.strip() != ""]


    if not all_terms_flat: # Si no hay términos después de filtrar y aplanar
        fig_terms.update_layout(title="No se encontraron términos TIC")
        fig_trend.update_layout(title="No se encontraron términos TIC para tendencias")
        fig_cooccur.update_layout(title="No se encontraron términos TIC para co-ocurrencia")
        fig_value_term.update_layout(title="No se encontraron términos TIC para análisis de valor")
        return fig_terms, fig_trend, fig_cooccur, fig_value_term


    # Top Términos TIC
    if all_terms_flat:
        term_counts = Counter(all_terms_flat).most_common(15)
        if term_counts:
            terms_df_viz = pd.DataFrame(term_counts, columns=['Término', 'Frecuencia'])
            fig_terms = px.bar(
                terms_df_viz, x='Término', y='Frecuencia',
                title="Top 15 Términos TIC Encontrados",
                labels={'Frecuencia': 'Frecuencia', 'Término': ''},
                color='Frecuencia', color_continuous_scale='Viridis'
            )
            fig_terms.update_layout(xaxis={'categoryorder':'total descending'})
        else:
            fig_terms.update_layout(title="No hay términos suficientes para el gráfico Top")


    # Tendencias de Términos por Año
    if all_terms_flat and 'Fecha de inicio del contrato' in filtered_df.columns:
        trend_base = filtered_df[['Fecha de inicio del contrato', 'terminos_positivos']].dropna(subset=['Fecha de inicio del contrato']).copy()
        if not trend_base.empty:
            trend_base['Year'] = trend_base['Fecha de inicio del contrato'].dt.year
            # 'terminos_positivos' ya es una lista de strings
            trend_exploded = trend_base.explode('terminos_positivos').dropna(subset=['terminos_positivos'])
            trend_exploded = trend_exploded[trend_exploded['terminos_positivos'].str.strip() != ""]


            if not trend_exploded.empty:
                top_5_terms = [term for term, _ in Counter(all_terms_flat).most_common(5)]
                trend_filtered_for_plot = trend_exploded[trend_exploded['terminos_positivos'].isin(top_5_terms)]
                
                if not trend_filtered_for_plot.empty:
                    term_yearly_counts = trend_filtered_for_plot.groupby(['Year', 'terminos_positivos']).size().reset_index(name='Frecuencia')
                    term_yearly_counts = term_yearly_counts.sort_values('Year')
                    fig_trend = px.line(
                        term_yearly_counts, x='Year', y='Frecuencia', color='terminos_positivos',
                        title="Tendencias de los 5 Términos TIC Más Comunes por Año",
                        labels={'Frecuencia': 'Frecuencia Anual', 'Year': 'Año', 'terminos_positivos': 'Término'},
                        markers=True
                    )
                else:
                    fig_trend.update_layout(title="No hay datos de los top 5 términos para tendencias")
            else:
                fig_trend.update_layout(title="No hay términos válidos para explotar en tendencias")
        else:
            fig_trend.update_layout(title="No hay fechas válidas para tendencias de términos")


    # Co-ocurrencia de Términos
    if all_terms_flat:
        # Usar las listas de terminos_positivos directamente
        cooccur_base_lists = filtered_df['terminos_positivos'].dropna().tolist()
        cooccur_base_lists = [lst for lst in cooccur_base_lists if lst] # Filtrar listas vacías

        if cooccur_base_lists:
            top_10_terms_cooccur = [term for term, _ in Counter(all_terms_flat).most_common(10)]
            cooccurrence_matrix = pd.DataFrame(0, index=top_10_terms_cooccur, columns=top_10_terms_cooccur)

            from itertools import combinations
            for terms_list_in_contract in cooccur_base_lists:
                # Filtrar términos en la lista del contrato para que solo estén los top_10
                relevant_terms_in_contract = [term for term in terms_list_in_contract if term in top_10_terms_cooccur]
                # Pares únicos de términos relevantes dentro de este contrato
                for term1, term2 in combinations(set(relevant_terms_in_contract), 2): # Usar set para evitar duplicados si un término está varias veces en la lista
                    cooccurrence_matrix.loc[term1, term2] += 1
                    cooccurrence_matrix.loc[term2, term1] += 1
            
            # Opcional: Llenar diagonal con frecuencia total del término (del top 10)
            # for term in top_10_terms_cooccur:
            #    cooccurrence_matrix.loc[term, term] = Counter(all_terms_flat)[term]

            if not cooccurrence_matrix.empty:
                fig_cooccur = px.imshow(
                    cooccurrence_matrix,
                    title="Co-ocurrencia de los 10 Términos TIC Más Comunes",
                    labels=dict(x="Término 1", y="Término 2", color="Co-ocurrencias"),
                    color_continuous_scale='Viridis'
                )
                fig_cooccur.update_layout(height=600) # Ajustar tamaño para mejor visualización
            else:
                fig_cooccur.update_layout(title="No hay datos para matriz de co-ocurrencia")
        else:
            fig_cooccur.update_layout(title="No hay listas de términos válidas para co-ocurrencia")


    # Valor Promedio de Contratos por Término TIC
    if all_terms_flat and 'Importe DRC' in filtered_df.columns:
        value_term_base = filtered_df[['terminos_positivos', 'Importe DRC']].dropna(subset=['Importe DRC']).copy()
        if not value_term_base.empty:
            value_term_exploded = value_term_base.explode('terminos_positivos').dropna(subset=['terminos_positivos'])
            value_term_exploded = value_term_exploded[value_term_exploded['terminos_positivos'].str.strip() != ""]

            if not value_term_exploded.empty:
                term_value_agg = value_term_exploded.groupby('terminos_positivos')['Importe DRC'].agg(['mean', 'count']).reset_index()
                term_value_agg.rename(columns={'mean': 'Valor Promedio', 'count': 'Num Contratos', 'terminos_positivos': 'Término'}, inplace=True)
                
                # Filtrar términos con al menos N contratos (ej. 2) y tomar Top 15 por valor promedio
                term_value_filtered_plot = term_value_agg[term_value_agg['Num Contratos'] >= 1].sort_values('Valor Promedio', ascending=False).head(15)

                if not term_value_filtered_plot.empty:
                    fig_value_term = px.bar(
                        term_value_filtered_plot, x='Término', y='Valor Promedio',
                        color='Num Contratos',
                        title="Valor Promedio de Contratos por Término TIC (Top 15)",
                        labels={'Valor Promedio': 'Valor Promedio (MXN)', 'Término': '', 'Num Contratos': 'Núm. Contratos'},
                        color_continuous_scale='Blues', #Cambiado de Viridis para diferenciar
                        hover_data={'Valor Promedio': ':,.2f'}
                    )
                    fig_value_term.update_layout(xaxis={'categoryorder':'total descending'})
                else:
                    fig_value_term.update_layout(title="No hay suficientes datos de términos y valor para el gráfico")
            else:
                fig_value_term.update_layout(title="No hay términos válidos para explotar en análisis de valor")
        else:
            fig_value_term.update_layout(title="No hay importes válidos para análisis de valor por término")

    return fig_terms, fig_trend, fig_cooccur, fig_value_term


#Callbacks para actualizar la ventana de Redes de Contratacion
@app.callback(
    Output('network-graph', 'figure'),
    [Input('tic-dropdown', 'value'),
     Input('siglas-dropdown', 'value'),
     Input('gobierno-dropdown', 'value'),
     Input('importe-slider', 'value')]
)

#Ventana de Redes de Contratacion
def update_tab_network(tic_filter, selected_siglas, selected_gobierno, importe_range):
    filtered_df = filter_dataframe(df, tic_filter, selected_siglas, selected_gobierno, importe_range)

    empty_network_fig_layout = {
        "title": "No hay datos suficientes para crear la red",
        "showlegend": False,
        "xaxis": {"showgrid": False, "zeroline": False, "showticklabels": False},
        "yaxis": {"showgrid": False, "zeroline": False, "showticklabels": False},
        "plot_bgcolor": "rgba(0,0,0,0)", "paper_bgcolor": "rgba(0,0,0,0)"
    }

    if filtered_df.empty or \
       'Proveedor o contratista' not in filtered_df.columns or \
       'Siglas de la Institución' not in filtered_df.columns:
        return go.Figure().update_layout(**empty_network_fig_layout)

    # 1. Crear pares y agregar
    link_col = 'Importe DRC' if 'Importe DRC' in filtered_df.columns and filtered_df['Importe DRC'].notna().any() else 'Código del contrato'
    agg_func = 'sum' if link_col == 'Importe DRC' else 'count'
    
    # Asegurarse de no tener nulos en las columnas clave para la red
    network_data_prep = filtered_df.dropna(subset=['Proveedor o contratista', 'Siglas de la Institución', link_col])

    if network_data_prep.empty:
        empty_network_fig_layout["title"] = "No hay datos válidos (proveedor, institución, valor/conteo) para generar la red"
        return go.Figure().update_layout(**empty_network_fig_layout)

    edges_df = network_data_prep.groupby(['Proveedor o contratista', 'Siglas de la Institución']).agg(
        weight=(link_col, agg_func)
    ).reset_index()

    top_n = 50 # Limitar a las N relaciones más fuertes
    edges_df = edges_df.sort_values('weight', ascending=False).head(top_n)

    if edges_df.empty:
        empty_network_fig_layout["title"] = "No se encontraron relaciones significativas para mostrar en la red"
        return go.Figure().update_layout(**empty_network_fig_layout)

    # 2. Crear lista de nodos únicos
    proveedores_nodes = pd.unique(edges_df['Proveedor o contratista'])
    instituciones_nodes = pd.unique(edges_df['Siglas de la Institución'])
    all_graph_nodes = np.concatenate([proveedores_nodes, instituciones_nodes])
    # Eliminar duplicados si una entidad es tanto proveedor como institución (poco probable aquí)
    all_graph_nodes = pd.unique(all_graph_nodes) 
    
    node_map = {name: i for i, name in enumerate(all_graph_nodes)}
    
    # Determinar tipo de nodo para colorear (más robusto)
    node_types_map = {p: 'Proveedor' for p in proveedores_nodes}
    for i_node in instituciones_nodes:
        if i_node not in node_types_map: # Si no es ya un proveedor, es institución
            node_types_map[i_node] = 'Institución'
        # else: podría ser ambos, priorizar uno o manejar como "Mixto"

    # 3. Crear posiciones para los nodos (Layout Circular Simple)
    num_graph_nodes = len(all_graph_nodes)
    if num_graph_nodes == 0: # No debería pasar si edges_df no está vacío
        return go.Figure().update_layout(**empty_network_fig_layout)

    radius = 5 if num_graph_nodes > 1 else 0 # Evitar división por cero si hay un solo nodo
    angles = np.linspace(0, 2 * np.pi, num_graph_nodes, endpoint=False) if num_graph_nodes > 1 else [0]
    pos_x = radius * np.cos(angles)
    pos_y = radius * np.sin(angles)
    node_positions = {name: (pos_x[i], pos_y[i]) for i, name in enumerate(all_graph_nodes)}

    fig_network = go.Figure()

    # 4. Crear Trazas para los Enlaces (Edges)
    min_w, max_w = edges_df['weight'].min(), edges_df['weight'].max()
    
    for _, row in edges_df.iterrows():
        prov_name = row['Proveedor o contratista']
        inst_name = row['Siglas de la Institución']
        
        # Asegurarse que los nodos del enlace existen en el mapa de posiciones
        if prov_name not in node_positions or inst_name not in node_positions:
            continue # Saltar este enlace si uno de los nodos no está en la lista principal (raro si edges_df es la fuente)

        x0, y0 = node_positions[prov_name]
        x1, y1 = node_positions[inst_name]
        
        # Normalizar grosor de línea
        line_width = 1
        if max_w > min_w:
            line_width = 1 + 4 * (row['weight'] - min_w) / (max_w - min_w)
        elif max_w == min_w and max_w > 0 : # Todos los pesos son iguales y no cero
            line_width = 2 
        
        weight_label = f"${row['weight']:,.2f}" if link_col == 'Importe DRC' else f"{int(row['weight'])} contratos"
        fig_network.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode='lines',
            line=dict(width=line_width, color='rgba(180,180,180,0.6)'),
            hoverinfo='text',
            text=f"Relación: {prov_name} &harr; {inst_name}<br>Valor/Conteo: {weight_label}"
        ))

    # 5. Crear Traza para los Nodos
    node_x_coords = [node_positions[node][0] for node in all_graph_nodes]
    node_y_coords = [node_positions[node][1] for node in all_graph_nodes]
    
    node_hover_texts = []
    node_display_texts = []
    node_colors_list = []
    
    # Calcular grado del nodo
    node_degrees = Counter()
    for _, row in edges_df.iterrows():
        node_degrees[row['Proveedor o contratista']] += 1
        node_degrees[row['Siglas de la Institución']] += 1

    node_sizes_list = []

    for node_name in all_graph_nodes:
        node_type = node_types_map.get(node_name, "Desconocido")
        node_hover_texts.append(f"{node_type}: {node_name}<br>Conexiones: {node_degrees[node_name]}")
        node_display_texts.append(node_name[:15] + '...' if len(node_name) > 15 else node_name) # Acortar texto en nodo
        node_colors_list.append('firebrick' if node_type == 'Proveedor' else 'steelblue') # Colores distintos
        node_sizes_list.append(8 + node_degrees[node_name] * 2.5) # Tamaño basado en grado


    fig_network.add_trace(go.Scatter(
        x=node_x_coords, y=node_y_coords, mode='markers+text',
        text=node_display_texts, textposition="bottom center", textfont=dict(size=9),
        hoverinfo='text', hovertext=node_hover_texts,
        marker=dict(
            showscale=False, color=node_colors_list, size=node_sizes_list,
            line=dict(width=0.8, color='DarkSlateGrey')
        )
    ))

    # 6. Configurar Layout de la Figura
    layout_padding = 1 if num_graph_nodes > 1 else 1 # Espacio alrededor del círculo
    fig_network.update_layout(
        title=None, showlegend=False, hovermode='closest',
        margin=dict(b=10, l=5, r=5, t=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[min(pos_x)-layout_padding, max(pos_x)+layout_padding]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[min(pos_y)-layout_padding, max(pos_y)+layout_padding]),
        plot_bgcolor='rgba(248, 249, 250, 0.9)'
    )
    return fig_network

####### Callback para mostrar contratos al hacer clic en el gráfico de Top Términos TIC #######
@app.callback(
    Output('term-contracts-table-container', 'children'),
    [Input('top-terms-graph', 'clickData'),
     Input('tic-dropdown', 'value'),
     Input('siglas-dropdown', 'value'),
     Input('gobierno-dropdown', 'value'),
     Input('importe-slider', 'value')]
)


# Esta función se encarga de mostrar los contratos asociados al término TIC seleccionado en el gráfico
def display_term_contracts_table(click_data, tic_filter, selected_siglas, selected_gobierno, importe_range):
    ctx = dash.callback_context
    if not click_data or not click_data['points']:
        return html.P("Haz clic en una barra del gráfico 'Top Términos TIC' para ver los contratos asociados.",
                      style={'textAlign': 'center', 'marginTop': '20px'})

    try:
        selected_term = click_data['points'][0]['x']
    except (KeyError, IndexError, TypeError) as e:
        print(f"Error al extraer el término: {e}, click_data: {click_data}")
        return html.P("No se pudo obtener el término seleccionado del gráfico.",
                      style={'textAlign': 'center', 'marginTop': '20px', 'color': 'red'})

    filtered_df_global = filter_dataframe(df.copy(), tic_filter, selected_siglas, selected_gobierno, importe_range)

    if filtered_df_global.empty:
        return html.P(f"No hay datos para los filtros generales seleccionados al buscar contratos para el término '{selected_term}'.",
                      style={'textAlign': 'center', 'marginTop': '20px'})

    if 'terminos_positivos' not in filtered_df_global.columns:
        return html.P("La columna 'terminos_positivos' no se encuentra en los datos filtrados.",
                      style={'color': 'red', 'textAlign': 'center', 'marginTop': '20px'})

    term_contracts_df = filtered_df_global[
        filtered_df_global['terminos_positivos'].apply(lambda terms_list: isinstance(terms_list, list) and selected_term in terms_list)
    ].copy()

    if term_contracts_df.empty:
        return html.P(f"No se encontraron contratos para el término '{selected_term}' con los filtros actuales.",
                      style={'textAlign': 'center', 'marginTop': '20px'})

    cols_for_table_display = []
    data_for_table = pd.DataFrame()

    col_titulo_script = 'Título del contrato'
    col_institucion_script = 'Siglas de la Institución' # <--- Asegúrate que esta columna exista o quieras mostrarla
    col_importe_script = 'Importe DRC'
    col_fecha_inicio_script = 'Fecha de inicio del contrato'
    col_fecha_fin_script = 'Fecha de fin del contrato'
    col_anuncio_script = 'Dirección del anuncio'

    current_cols_for_selection = []

    if col_titulo_script in term_contracts_df.columns:
        current_cols_for_selection.append(col_titulo_script)
        cols_for_table_display.append({"name": "Título del Contrato", "id": col_titulo_script})

    # Añadir columna Institución si existe y se desea (como en la tabla de ejemplo de proveedor)
    if col_institucion_script in term_contracts_df.columns:
        current_cols_for_selection.append(col_institucion_script)
        cols_for_table_display.append({"name": "Institución", "id": col_institucion_script})


    if col_importe_script in term_contracts_df.columns:
        current_cols_for_selection.append(col_importe_script)
        term_contracts_df.loc[:, col_importe_script] = pd.to_numeric(term_contracts_df[col_importe_script], errors='coerce')
        cols_for_table_display.append({
            "name": "Importe DRC", "id": col_importe_script, "type": "numeric",
            "format": dash_table.Format.Format(scheme=dash_table.Format.Scheme.fixed, precision=2, group=True, symbol=dash_table.Format.Symbol.yes, symbol_prefix='$')
        })

    if col_fecha_inicio_script in term_contracts_df.columns:
        current_cols_for_selection.append(col_fecha_inicio_script)
        # Asegurar que la conversión a datetime se haga antes del strftime y maneje errores
        term_contracts_df.loc[:, col_fecha_inicio_script] = pd.to_datetime(term_contracts_df[col_fecha_inicio_script], errors='coerce').dt.strftime('%Y-%m-%d') # Cambiado a YYYY-MM-DD para que se vea como en la tabla de proveedores, o usa '%d/%m/%Y'
        cols_for_table_display.append({"name": "Fecha Inicio", "id": col_fecha_inicio_script})

    if col_fecha_fin_script in term_contracts_df.columns:
        current_cols_for_selection.append(col_fecha_fin_script)
        term_contracts_df.loc[:, col_fecha_fin_script] = pd.to_datetime(term_contracts_df[col_fecha_fin_script], errors='coerce').dt.strftime('%Y-%m-%d') # Cambiado a YYYY-MM-DD
        cols_for_table_display.append({"name": "Fecha Fin", "id": col_fecha_fin_script})

    warning_message_anuncio = None
    if col_anuncio_script in term_contracts_df.columns:
        # Usar un nombre de columna único para el markdown si es diferente de la tabla de proveedor
        term_contracts_df['Enlace_Anuncio_Term_MD'] = term_contracts_df[col_anuncio_script].apply(
            lambda x: f"[Ver Anuncio]({x})" if pd.notna(x) and str(x).strip().lower().startswith('http') else "N/A"
        )
        current_cols_for_selection.append('Enlace_Anuncio_Term_MD')
        cols_for_table_display.append({"name": "Anuncio", "id": "Enlace_Anuncio_Term_MD", 'presentation': 'markdown'})
    else:
        warning_message_anuncio = html.P("Advertencia: La columna 'Dirección del anuncio' no fue encontrada para esta tabla.", style={'color': 'orange', 'textAlign': 'center', 'fontSize': '0.9em'})


    if not current_cols_for_selection:
         return html.P(f"No hay columnas con datos para mostrar para el término '{selected_term}'.",
                      style={'textAlign': 'center', 'marginTop': '20px', 'color': 'red'})

    data_for_table = term_contracts_df[current_cols_for_selection]


    table = dash_table.DataTable(
        id='term-specific-contracts-table',
        columns=cols_for_table_display,
        data=data_for_table.to_dict('records'),
        # Estilos copiados/adaptados de 'provider-specific-contracts-table'
        style_table={'overflowX': 'auto', 'marginTop': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'width':'100%'},
        style_header={'backgroundColor': '#34495e', 'color': 'white', 'fontWeight': 'bold', 'textAlign': 'center', 'padding': '10px'}, # Coincide con tabla proveedor
        style_cell={
            'textAlign': 'left', 'padding': '8px',
            'minWidth': '120px', 'maxWidth': '400px', # Coincide con tabla proveedor
            'whiteSpace': 'normal', 'height': 'auto',
            'border': '1px solid #eee',
            'fontFamily': 'Arial, sans-serif', 'fontSize': '13px', # Coincide con tabla proveedor
            'verticalAlign': 'middle' # Añadido para consistencia
            },
        style_cell_conditional=[
            # Coincide con tabla proveedor
            {'if': {'column_id': col_titulo_script}, 'minWidth': '250px', 'fontWeight': 'bold'},
            {'if': {'column_id': col_importe_script}, 'textAlign': 'right', 'minWidth': '150px'},
            # Asegúrate que 'Enlace_Anuncio_Term_MD' es el id correcto si la columna se llama así
            {'if': {'column_id': 'Enlace_Anuncio_Term_MD'}, 'textAlign': 'center', 'minWidth': '100px'}
        ],
        style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#f9f9f9'}], # Coincide con tabla proveedor
        page_size=5,
        filter_action='native',
        sort_action='native',
    )

    return html.Div([
        # Título con estilo similar al de la tabla de proveedor
        html.H4(f"Contratos que incluyen el término: '{selected_term}'",
                style={'marginTop': '20px', 'marginBottom': '5px', 'textAlign': 'center', 'color': '#2c3e50'}), # Color oscuro como otros H4
        warning_message_anuncio if warning_message_anuncio else "",
        table
    ])



# Agregar CSS para estilizar mejor la aplicación
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Dashboard de Contratos TIC</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                background-color: #f8f9fa;
            }
            .metric-box {
                /* width: 23%; */ /* Ajustado por flex */
                flex: 1; /* Permite que las cajas crezcan y se encojan */
                margin: 0 10px; /* Espacio entre cajas */
                border-radius: 8px; /* Bordes más redondeados */
                box-shadow: 0 4px 8px rgba(0,0,0,0.05); /* Sombra más sutil */
                background: white;
                padding: 20px; /* Más padding */
                text-align: center; /* Centrar contenido de métricas */
            }
            .metric-box:first-child { margin-left: 0; }
            .metric-box:last-child { margin-right: 0; }

            .metric-box h4 {
                margin-top: 0;
                margin-bottom: 8px;
                font-size: 1em; /* Tamaño de fuente para el título de la métrica */
                color: #555;
            }
            .metric-box p {
                font-size: 1.5em; /* Tamaño de fuente para el valor de la métrica */
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 0;
            }

            h1, h3 {
                color: #2c3e50;
            }
            /* Estilos para Dash DataTable (pueden necesitar ajustes) */
            .dash-spreadsheet-container table {
                border-collapse: collapse;
                width: 100%;
            }
            .dash-spreadsheet-container th,
            .dash-spreadsheet-container td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            .dash-spreadsheet-container th {
                background-color: #f2f2f2;
                position: sticky; /* Encabezados pegajosos si la tabla tiene scroll */
                top: 0;
                z-index: 1;
            }
            /* Estilos para Tabs */
            .Tabs {
                border-bottom: 1px solid #ddd;
            }
            .Tab {
                padding: 10px 15px;
                cursor: pointer;
                border: 1px solid transparent;
                border-bottom: none;
                margin-right: 5px;
            }
            .Tab--selected {
                border-color: #ddd;
                border-bottom: 1px solid white; /* O el color de fondo del contenido */
                background-color: white; /* O el color de fondo del contenido */
                border-radius: 5px 5px 0 0;
                color: #2c3e50;
                font-weight: bold;
            }

        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True)
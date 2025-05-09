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

# Asumimos que 'df' ya está cargado con los datos de contratos clasificados TIC
# Si no es así, deberías cargar tus datos aquí:
# df = pd.read_csv("ruta_a_tus_datos.csv", encoding="latin1")

# Asegurarse de que las fechas estén en formato datetime




df = pd.read_csv("Solo_TIC2023-2005.csv", encoding="latin1")

df['Fecha de inicio del contrato'] = pd.to_datetime(df['Fecha de inicio del contrato'], dayfirst=True, errors='coerce')
df['Fecha de fin del contrato'] = pd.to_datetime(df['Fecha de fin del contrato'], dayfirst=True, errors='coerce')
df['Fecha de firma del contrato'] = pd.to_datetime(df['Fecha de firma del contrato'], errors='coerce')

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
siglas_list = df['Siglas de la Institución'].dropna().unique()
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
                max=round(df['Importe DRC'].max()) if 'Importe DRC' in df.columns else 1000000,
                step=100000,
                marks={i: f"${i/1000000:.1f}M" for i in range(0,
                       int(df['Importe DRC'].max()) if 'Importe DRC' in df.columns else 1000000,
                       1000000)},
                value=[0, round(df['Importe DRC'].max()) if 'Importe DRC' in df.columns else 1000000]
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

    elif tab == 'tab-2':
        return html.Div([
            html.Div([
                html.Div([
                    html.H3("Oportunidades por Proximidad de Vencimiento"),
                    dcc.Graph(id='opportunities-expiry-graph'),
                ], style={'width': '49%', 'display': 'inline-block'}),

                html.Div([
                    html.H3("Valor de Contratos Próximos a Vencer"),
                    dcc.Graph(id='value-expiry-graph'),
                ], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),
            ], style={'margin-bottom': '20px'}),

            html.Div([
                html.H3("Línea de Tiempo de Vencimientos de Contratos TIC"),
                dcc.Graph(id='timeline-graph'),
            ], style={'margin-bottom': '20px'}),

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
            html.Div([
                html.Div([
                    html.H3("Top Términos TIC Encontrados"),
                    dcc.Graph(id='top-terms-graph'),
                ], style={'width': '49%', 'display': 'inline-block'}),

                html.Div([
                    html.H3("Tendencias de Términos TIC por Año"),
                    dcc.Graph(id='terms-trend-graph'),
                ], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),
            ], style={'margin-bottom': '20px'}),

            html.Div([
                html.H3("Co-ocurrencia de Términos"),
                dcc.Graph(id='term-cooccurrence-graph'),
            ], style={'margin-bottom': '20px'}),

            html.Div([
                html.H3("Valor Promedio de Contratos por Término TIC"),
                dcc.Graph(id='term-value-graph'),
            ]),
        ])
    # elif tab == 'tab-6':
    #     return html.Div([
    #         html.H3("Análisis Detallado de Proveedores"),
    #         # Aquí irían los componentes para esta pestaña:
    #         # dcc.Graph(id='provider-detail-graph-1'),
    #         # html.Div(id='provider-table-container'),
    #         html.P("Contenido para el Análisis de Proveedores (aún por implementar).") # Placeholder
    #     ])
    elif tab == 'tab-network':
        return html.Div([
            html.H3("Redes de Relaciones Proveedor-Institución"),
            dcc.Graph(id='network-graph'),
            html.P("Visualiza que proveedores contratan con que insituciones. El tamaño del enlace representa el valor total de los contratos entre ellos (Top 50 relaciones mostradas)."),
        ])

# Función auxiliar para filtrar datos
def filter_dataframe(df, tic_filter, selected_siglas, selected_gobierno, importe_range):
    filtered_df = df.copy()

    # Filtrar por clasificación TIC
    if tic_filter == 'tic':
        filtered_df = filtered_df[filtered_df['es_TIC']]
    elif tic_filter == 'no_tic':
        filtered_df = filtered_df[~filtered_df['es_TIC']]

    # Filtrar por siglas de institución
    if selected_siglas and len(selected_siglas) > 0:
        filtered_df = filtered_df[filtered_df['Siglas de la Institución'].isin(selected_siglas)]

    # Filtrar por orden de gobierno
    if 'Orden de gobierno' in filtered_df.columns and selected_gobierno and len(selected_gobierno) > 0:
        filtered_df = filtered_df[filtered_df['Orden de gobierno'].isin(selected_gobierno)]

    # Filtrar por rango de importe
    if 'Importe DRC' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['Importe DRC'] >= importe_range[0]) &
                                 (filtered_df['Importe DRC'] <= importe_range[1])]

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
            xaxis=dict(title=""),
            yaxis=dict(title="")
        )

        metric_empty = html.Div([
            html.H4("Sin datos"),
            html.P("0")
        ], style={'textAlign': 'center', 'padding': '10px', 'background': '#f0f0f0', 'borderRadius': '5px'})

        return metric_empty, metric_empty, metric_empty, metric_empty, empty_fig, empty_fig, empty_fig, empty_fig

    # Métricas
    total_contratos = len(filtered_df)
    contratos_tic = filtered_df['es_TIC'].sum() if 'es_TIC' in filtered_df.columns else 0
    porcentaje_tic = (contratos_tic / total_contratos) * 100 if total_contratos > 0 else 0

    if 'Importe DRC' in filtered_df.columns:
        avg_importe = filtered_df['Importe DRC'].mean()
        total_importe = filtered_df['Importe DRC'].sum()
    else:
        avg_importe = 0
        total_importe = 0

    # Gráfica de tendencia de contratos TIC por año
    if not filtered_df.empty and 'Fecha de inicio del contrato' in filtered_df.columns:
        yearly_df = filtered_df.groupby([filtered_df['Fecha de inicio del contrato'].dt.year, 'es_TIC']).agg(
            count=('Código del contrato', 'count'),
            total_importe=('Importe DRC', 'sum') if 'Importe DRC' in filtered_df.columns else ('Código del contrato', 'count')
        ).reset_index()

        fig_trend = px.line(
            yearly_df[yearly_df['es_TIC'] == True] if 'es_TIC' in yearly_df.columns else yearly_df,
            x='Fecha de inicio del contrato',
            y='count',
            title="Evolución de Contratos TIC por Año",
            labels={'count': 'Número de Contratos', 'Fecha de inicio del contrato': 'Año'},
            markers=True,
            line_shape='linear'
        )
        fig_trend.update_traces(line=dict(color='#2c3e50', width=3), marker=dict(size=10))
    else:
        fig_trend = go.Figure()
        fig_trend.update_layout(title="No hay datos disponibles para mostrar la tendencia")

    # Gráfica de crecimiento de importe por categoría/año
    if not filtered_df.empty and 'Importe DRC' in filtered_df.columns and 'Fecha de inicio del contrato' in filtered_df.columns:
        # Agrupar por año y una categoría relevante (por ejemplo, Tipo de contratación)
        if 'Tipo de contratación' in filtered_df.columns:
            growth_df = filtered_df.groupby([filtered_df['Fecha de inicio del contrato'].dt.year, 'Tipo de contratación'])['Importe DRC'].sum().reset_index()

            fig_growth = px.area(
                growth_df,
                x='Fecha de inicio del contrato',
                y='Importe DRC',
                color='Tipo de contratación',
                title="Crecimiento de Importe por Tipo de Contratación",
                labels={'Importe DRC': 'Importe Total (MXN)', 'Fecha de inicio del contrato': 'Año'},
            )
        else:
            # Si no existe la columna de tipo de contratación, usar otra categorización o solo mostrar el total
            growth_df = filtered_df.groupby([filtered_df['Fecha de inicio del contrato'].dt.year])['Importe DRC'].sum().reset_index()

            fig_growth = px.area(
                growth_df,
                x='Fecha de inicio del contrato',
                y='Importe DRC',
                title="Crecimiento de Importe Total por Año",
                labels={'Importe DRC': 'Importe Total (MXN)', 'Fecha de inicio del contrato': 'Año'},
            )
    else:
        fig_growth = go.Figure()
        fig_growth.update_layout(title="No hay datos disponibles para mostrar el crecimiento de importe")

    # Gráfica de estacionalidad (patrones mensuales)
    if not filtered_df.empty and 'Fecha de inicio del contrato' in filtered_df.columns:
        # Crear un DataFrame para analizar la estacionalidad por mes
        season_df = filtered_df.groupby(filtered_df['Fecha de inicio del contrato'].dt.month).agg(
            count=('Código del contrato', 'count'),
            total_importe=('Importe DRC', 'sum') if 'Importe DRC' in filtered_df.columns else ('Código del contrato', 'count')
        ).reset_index()

        # Asignar nombres de meses
        month_names = {i: calendar.month_name[i] for i in range(1, 13)}
        season_df['Mes'] = season_df['Fecha de inicio del contrato'].map(month_names)

        # Ordenar por mes numérico
        season_df = season_df.sort_values('Fecha de inicio del contrato')

        # Usar la columna correcta para la visualización
        y_col = 'total_importe' if 'total_importe' in season_df.columns else 'count'
        y_title = 'Importe Total (MXN)' if y_col == 'total_importe' else 'Número de Contratos'

        fig_season = px.bar(
            season_df,
            x='Mes',
            y=y_col,
            title=f"Estacionalidad en la Contratación TIC (Análisis Mensual)",
            labels={y_col: y_title, 'Mes': ''},
            color=y_col,
            color_continuous_scale='Viridis'
        )

        # Personalizar para que se vean correctamente los meses en orden
        fig_season.update_xaxes(
            categoryorder='array',
            categoryarray=[calendar.month_name[i] for i in range(1, 13)]
        )
    else:
        fig_season = go.Figure()
        fig_season.update_layout(title="No hay datos disponibles para mostrar la estacionalidad")

    # Top 10 instituciones por gasto en TIC
    if not filtered_df.empty and 'Siglas de la Institución' in filtered_df.columns:
        # Si tenemos importe, usamos esa columna para el ranking
        if 'Importe DRC' in filtered_df.columns:
            top_inst = filtered_df.groupby('Siglas de la Institución')['Importe DRC'].sum().reset_index()
            top_inst = top_inst.sort_values('Importe DRC', ascending=False).head(10)

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
            # Si no tenemos importe, usamos el conteo de contratos
            top_inst = filtered_df.groupby('Siglas de la Institución').size().reset_index(name='count')
            top_inst = top_inst.sort_values('count', ascending=False).head(10)

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
        fig_top = go.Figure()
        fig_top.update_layout(title="No hay datos disponibles para mostrar las principales instituciones")

    # Formatear las métricas para mostrar
    metric_tic = html.Div([
        html.H4("Porcentaje TIC"),
        html.P(f"{porcentaje_tic:.1f}%")
    ], style={'textAlign': 'center', 'padding': '10px', 'background': '#f0f0f0', 'borderRadius': '5px'})

    metric_total = html.Div([
        html.H4("Total Contratos"),
        html.P(f"{total_contratos:,}")
    ], style={'textAlign': 'center', 'padding': '10px', 'background': '#f0f0f0', 'borderRadius': '5px'})

    metric_avg = html.Div([
        html.H4("Importe Promedio"),
        html.P(f"${avg_importe:,.2f} MXN")
    ], style={'textAlign': 'center', 'padding': '10px', 'background': '#f0f0f0', 'borderRadius': '5px'})

    metric_total_value = html.Div([
        html.H4("Valor Total"),
        html.P(f"${total_importe:,.2f} MXN")
    ], style={'textAlign': 'center', 'padding': '10px', 'background': '#f0f0f0', 'borderRadius': '5px'})

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
            xaxis=dict(title=""),
            yaxis=dict(title="")
        )
        empty_table = html.Div("No hay datos para mostrar")
        return empty_fig, empty_fig, empty_fig, empty_table

    # Oportunidades por proximidad de vencimiento (sin cambios aquí)
    if 'Oportunidad' in filtered_df.columns:
        opp_counts = filtered_df.groupby('Oportunidad').size().reset_index(name='count')

        #Definir un orden personalizado para las categorías
        order = ['Vencido', 'Urgente (< 3 meses)', 'Corto plazo (3-6 meses)',
                 'Medio plazo (6-12 meses)', 'Largo plazo (> 12 meses)']
        
        #Asegurar que todas las categorias estén presentes
        for cat in order:
            if cat not in opp_counts['Oportunidad'].values:
                opp_counts = pd.concat([opp_counts, pd.DataFrame({'Oportunidad': [cat], 'count': [0]})], ignore_index=True)

        #Ordenar segun el orden definido
        opp_counts['Oportunidad'] = pd.Categorical(opp_counts['Oportunidad'], categories=order, ordered=True)
        opp_counts = opp_counts.sort_values('Oportunidad')

        fig_opp = px.bar(
            opp_counts, 
            x='Oportunidad', 
            y='count', 
            title="Distribución de Contratos por Proximidad de Vencimiento", labels={'count': 'Número de Contratos', 'Oportunidad': ''}, color='Oportunidad', color_discrete_map={'Vencido': 'grey', 'Urgente (< 3 meses)': 'red', 'Corto plazo (3-6 meses)': 'orange', 'Medio plazo (6-12 meses)': 'blue', 'Largo plazo (> 12 meses)': 'green'})
    else:
        fig_opp = go.Figure()
        fig_opp.update_layout(title="No hay datos disponibles para mostrar oportunidades por vencimiento")

    # Valor de contratos próximos a vencer (sin cambios aquí)
    if 'Oportunidad' in filtered_df.columns and 'Importe DRC' in filtered_df.columns:
        value_by_expiry = filtered_df.groupby('Oportunidad')['Importe DRC'].sum().reset_index()

        #Usar el mismo oder que en la grafica
        order = ['Vencido', 'Urgente (< 3 meses)', 'Corto plazo (3-6 meses)',
                 'Medio plazo (6-12 meses)', 'Largo plazo (> 12 meses)']
        
        #Asegurar que todas las categorias estén presentes
        for cat in order:
            if cat not in value_by_expiry['Oportunidad'].values:
                value_by_expiry = pd.concat([value_by_expiry, pd.DataFrame({'Oportunidad': [cat], 'Importe DRC': [0]})], ignore_index=True)

        #Ordenar segun el orden definido       
        value_by_expiry['Oportunidad'] = pd.Categorical(value_by_expiry['Oportunidad'], categories=order, ordered=True)
        value_by_expiry = value_by_expiry.sort_values('Oportunidad')


        #Continuacion del grafico de valor de contratos proximos a vencer
        fig_value = px.bar(value_by_expiry, x='Oportunidad', y='Importe DRC', title="Valor Total de Contratos por Proximidad de Vencimiento", labels={'Importe DRC': 'Importe Total (MXN)', 'Oportunidad': ''}, color='Oportunidad', color_discrete_map={'Vencido': 'grey', 'Urgente (< 3 meses)': 'red', 'Corto plazo (3-6 meses)': 'orange', 'Medio plazo (6-12 meses)': 'blue', 'Largo plazo (> 12 meses)': 'green'})
    else:
        fig_value = go.Figure()
        fig_value.update_layout(title="No hay datos disponibles para mostrar el valor por vencimiento")

    # Línea de tiempo de vencimientos
    if 'Fecha de fin del contrato' in filtered_df.columns:
        now = datetime.now()
        timeline_df = filtered_df[(filtered_df['Fecha de fin del contrato'] > now - timedelta(days=90))]

        if not timeline_df.empty:
            #Agregar columna para tooltip
            timeline_df['tooltip_info'] = timeline_df.apply(
                lambda row: f"Contrato: {row['Título del contrato'] if 'Título del contrato' in timeline_df.columns else 'N/A'}<br>" +
                            f"Institución: {row['Siglas de la Institución'] if 'Siglas de la Institución' in timeline_df.columns else 'N/A'}<br>" +
                            f"Proveedor: {row['Proveedor o contratista'] if 'Proveedor o contratista' in timeline_df.columns else 'N/A'}<br>" +
                            f"Importe: ${row['Importe DRC']:,.2f} MXN" if 'Importe DRC' in timeline_df.columns else "Importe: N/A",
                axis=1
            )
            fig_timeline = px.scatter(
                timeline_df,
                x='Fecha de fin del contrato',
                y='Importe DRC' if 'Importe DRC' in timeline_df.columns else 'Siglas de la Institución',
                size='Importe DRC' if 'Importe DRC' in timeline_df.columns else None,
                color='Oportunidad' if 'Oportunidad' in timeline_df.columns else None,
                hover_name='Título del contrato' if 'Título del contrato' in timeline_df.columns else None,
                hover_data=['tooltip_info'],
                title="Línea de Tiempo de Vencimientos de Contratos",
                labels={
                    'Fecha de fin del contrato': 'Fecha de Vencimiento',
                    'Importe DRC': 'Importe (MXN)' if 'Importe DRC' in timeline_df.columns else '',
                    'Siglas de la Institución': 'Institución'
                },
                color_discrete_map={
                    'Vencido': 'grey',
                    'Urgente (< 3 meses)': 'red',
                    'Corto plazo (3-6 meses)': 'orange',
                    'Medio plazo (6-12 meses)': 'blue',
                    'Largo plazo (> 12 meses)': 'green'
                }
            )

            # **SOLUCIÓN 5: Comentar la línea de add_vline temporalmente**
            fig_timeline.add_vline(x=now, line_dash="dash", line_color="red")
            fig_timeline.add_annotation(
                x=now,
                y=1,
                yref="paper",
                text="Hoy",
                showarrow=False,
                yshift=10,
            )
            
        else:
            fig_timeline = go.Figure()
            fig_timeline.update_layout(title="No hay contratos próximos a vencer en el período seleccionado")
    else:
        fig_timeline = go.Figure()
        fig_timeline.update_layout(title="No hay datos disponibles para mostrar la línea de tiempo")

    # Tabla de oportunidades
    if not filtered_df.empty:
        # Filtrar solo contratos que vencen en los próximos 12 meses
        opp_df = filtered_df[(filtered_df['Tiempo hasta vencimiento (días)'] > 0) &
                             (filtered_df['Tiempo hasta vencimiento (días)'] <= 365)].copy()

        if not opp_df.empty:
            # Ordenar por tiempo hasta vencimiento
            opp_df = opp_df.sort_values('Tiempo hasta vencimiento (días)')

            # Seleccionar columnas relevantes para la tabla
            display_cols = ['Siglas de la Institución', 'Título del contrato', 'Proveedor o contratista',
                           'Fecha de fin del contrato', 'Tiempo hasta vencimiento (días)', 'Oportunidad']

            if 'Importe DRC' in opp_df.columns:
                display_cols.append('Importe DRC')

            # Asegurarse de que todas las columnas existen
            table_cols = [col for col in display_cols if col in opp_df.columns]

            # Limitar a 20 filas para mejor rendimiento
            opp_table_df = opp_df[table_cols].head(20)

            # Formatear fechas y valores numéricos para mejor visualización
            if 'Fecha de fin del contrato' in opp_table_df.columns:
                opp_table_df['Fecha de fin del contrato'] = opp_table_df['Fecha de fin del contrato'].dt.strftime('%d/%m/%Y')

            if 'Importe DRC' in opp_table_df.columns:
                opp_table_df['Importe DRC'] = opp_table_df['Importe DRC'].apply(lambda x: f"${x:,.2f} MXN")

            # Crear tabla con Dash DataTable
            opportunity_table = dash_table.DataTable(
                id='opportunity-table',
                columns=[{"name": col.replace('_', ' ').title(), "id": col} for col in opp_table_df.columns],
                data=opp_table_df.to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'padding': '10px',
                    'whiteSpace': 'normal',
                    'height': 'auto',
                },
                style_header={
                    'backgroundColor': '#2c3e50',
                    'color': 'white',
                    'fontWeight': 'bold'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#f9f9f9'
                    },
                    {
                        'if': {'filter_query': '{Oportunidad} contains "Urgente"'},
                        'backgroundColor': '#ffcccc'
                    },
                    {
                        'if': {'filter_query': '{Oportunidad} contains "Corto plazo"'},
                        'backgroundColor': '#fff2cc'
                    }
                ],
                page_size=10
            )
        else:
            opportunity_table = html.Div("No hay contratos próximos a vencer en los filtros seleccionados",
                                        style={'color': '#666', 'textAlign': 'center', 'padding': '20px'})
    else:
        opportunity_table = html.Div("No hay datos disponibles para mostrar oportunidades",
                                    style={'color': '#666', 'textAlign': 'center', 'padding': '20px'})

    return fig_opp, fig_value, fig_timeline, opportunity_table

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

    # Si no hay datos después de filtrar
    if filtered_df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No hay datos para los filtros seleccionados",
            xaxis=dict(title=""),
            yaxis=dict(title="")
        )

        empty_table = html.Div("No hay datos para mostrar")

        return empty_fig, empty_fig, empty_fig, empty_table

    # Market Share de Proveedores (Top 10)
    if 'Proveedor o contratista' in filtered_df.columns:
        if 'Importe DRC' in filtered_df.columns:
            # Por valor de contratos
            provider_share = filtered_df.groupby('Proveedor o contratista')['Importe DRC'].sum().reset_index()
            provider_share = provider_share.sort_values('Importe DRC', ascending=False).head(10)

            # Calcular porcentaje del total
            total_value = provider_share['Importe DRC'].sum()
            provider_share['Porcentaje'] = (provider_share['Importe DRC'] / total_value) * 100

            fig_share = px.pie(
                provider_share,
                values='Importe DRC',
                names='Proveedor o contratista',
                title="Market Share de Proveedores TIC (por Importe)",
                hole=0.4
            )
        else:
            # Por número de contratos
            provider_share = filtered_df.groupby('Proveedor o contratista').size().reset_index(name='count')
            provider_share = provider_share.sort_values('count', ascending=False).head(10)

            fig_share = px.pie(
                provider_share,
                values='count',
                names='Proveedor o contratista',
                title="Market Share de Proveedores TIC (por Número de Contratos)",
                hole=0.4
            )
    else:
        fig_share = go.Figure()
        fig_share.update_layout(title="No hay datos disponibles sobre proveedores")

    # Especialización de Proveedores
    if 'Proveedor o contratista' in filtered_df.columns and 'terminos_positivos' in filtered_df.columns:
        # Crear un diccionario para almacenar términos por proveedor
        provider_terms = {}

        # Iterar a través de los registros
        for _, row in filtered_df.iterrows():
            provider = row['Proveedor o contratista']
            terms = row['terminos_positivos'] if isinstance(row['terminos_positivos'], list) else []

            if provider not in provider_terms:
                provider_terms[provider] = []

            provider_terms[provider].extend(terms)

        # Encontrar el término más común para cada proveedor
        provider_specialization = []
        for provider, terms in provider_terms.items():
            if terms:
                term_counts = Counter(terms)
                most_common_term = term_counts.most_common(1)[0][0]
                term_count = term_counts.most_common(1)[0][1]
                provider_specialization.append({
                    'Proveedor': provider,
                    'Término Especialización': most_common_term,
                    'Frecuencia': term_count
                })

        if provider_specialization:
            # Convertir a DataFrame y tomar los top 15
            spec_df = pd.DataFrame(provider_specialization)
            spec_df = spec_df.sort_values('Frecuencia', ascending=False).head(15)

            fig_spec = px.bar(
                spec_df,
                x='Proveedor',
                y='Frecuencia',
                color='Término Especialización',
                title="Especialización de Proveedores TIC (Término Más Común)",
                labels={'Frecuencia': 'Frecuencia del Término', 'Proveedor': ''}
            )
            fig_spec.update_layout(xaxis={'categoryorder':'total descending'})
        else:
            fig_spec = go.Figure()
            fig_spec.update_layout(title="No hay datos disponibles para análisis de especialización")
    else:
        fig_spec = go.Figure()
        fig_spec.update_layout(title="No hay datos disponibles para análisis de especialización")

    # Duración Promedio de Contratos por Proveedor
    if 'Proveedor o contratista' in filtered_df.columns and 'Duración del contrato (días)' in filtered_df.columns:
        # Calcular duración promedio por proveedor
        duration_df = filtered_df.groupby('Proveedor o contratista')['Duración del contrato (días)'].agg(
            ['mean', 'count']).reset_index()

        # Renombrar columnas
        duration_df.columns = ['Proveedor o contratista', 'Duración Promedio', 'Número de Contratos']

        # Filtrar proveedores con al menos 2 contratos y tomar los top 15
        duration_df = duration_df[duration_df['Número de Contratos'] >= 2].sort_values(
            'Duración Promedio', ascending=False).head(15)

        if not duration_df.empty:
            fig_duration = px.bar(
                duration_df,
                x='Proveedor o contratista',
                y='Duración Promedio',
                color='Número de Contratos',
                title="Duración Promedio de Contratos por Proveedor (en días)",
                labels={'Duración Promedio': 'Días', 'Proveedor o contratista': ''},
                color_continuous_scale='Viridis'
            )
            fig_duration.update_layout(xaxis={'categoryorder':'total descending'})
        else:
            fig_duration = go.Figure()
            fig_duration.update_layout(title="No hay suficientes datos para análisis de duración")
    else:
        fig_duration = go.Figure()
        fig_duration.update_layout(title="No hay datos disponibles para análisis de duración")

    # Tabla de análisis de competidores
    if 'Proveedor o contratista' in filtered_df.columns:
        # Preparar análisis de competidores
        if 'Importe DRC' in filtered_df.columns:
            competitor_analysis = filtered_df.groupby('Proveedor o contratista').agg(
                total_contratos=('Código del contrato', 'count'),
                valor_total=('Importe DRC', 'sum'),
                valor_promedio=('Importe DRC', 'mean'),
                duracion_promedio=('Duración del contrato (días)', 'mean') if 'Duración del contrato (días)' in filtered_df.columns else ('Código del contrato', 'count'),
                instituciones_unicas=('Siglas de la Institución', lambda x: x.nunique())
            ).reset_index()
        else:
            competitor_analysis = filtered_df.groupby('Proveedor o contratista').agg(
                total_contratos=('Código del contrato', 'count'),
                duracion_promedio=('Duración del contrato (días)', 'mean') if 'Duración del contrato (días)' in filtered_df.columns else ('Código del contrato', 'count'),
                instituciones_unicas=('Siglas de la Institución', lambda x: x.nunique())
            ).reset_index()

        # Ordenar por total de contratos o valor total
        if 'valor_total' in competitor_analysis.columns:
            competitor_analysis = competitor_analysis.sort_values('valor_total', ascending=False).head(15)
        else:
            competitor_analysis = competitor_analysis.sort_values('total_contratos', ascending=False).head(15)

        # Formatear valores para mejor visualización
        if 'valor_total' in competitor_analysis.columns:
            competitor_analysis['valor_total'] = competitor_analysis['valor_total'].apply(lambda x: f"${x:,.2f} MXN")

        if 'valor_promedio' in competitor_analysis.columns:
            competitor_analysis['valor_promedio'] = competitor_analysis['valor_promedio'].apply(lambda x: f"${x:,.2f} MXN")

        if 'duracion_promedio' in competitor_analysis.columns:
            competitor_analysis['duracion_promedio'] = competitor_analysis['duracion_promedio'].apply(lambda x: f"{x:.0f} días")

        # Crear tabla con Dash DataTable
        competitor_table = dash_table.DataTable(
            id='competitor-table',
            columns=[{"name": col.replace('_', ' ').title(), "id": col} for col in competitor_analysis.columns],
            data=competitor_analysis.to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'whiteSpace': 'normal',
                'height': 'auto',
            },
            style_header={
                'backgroundColor': '#2c3e50',
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#f9f9f9'
                }
            ],
            page_size=15
        )
    else:
        competitor_table = html.Div("No hay datos disponibles para análisis de competidores",
                                    style={'color': '#666', 'textAlign': 'center', 'padding': '20px'})

    return fig_share, fig_spec, fig_duration, competitor_table

####### Callbacks para mostrar los contratos al hacer clic en el gráfico de proveedores #######
@app.callback(
    Output('provider-contracts-table-container', 'children'),
    [Input('provider-share-graph', 'clickData'),
     Input('tic-dropdown', 'value'),
     Input('siglas-dropdown', 'value'),
     Input('gobierno-dropdown', 'value'),
     Input('importe-slider', 'value')]
)
def display_provider_contracts_table(click_data, tic_filter, selected_siglas, selected_gobierno, importe_range):

    if click_data is None:
        return html.P("Seleccione un proveedor para poder ver los contratos asociados",
                      style={'textAlign': 'center', 'marginTop': '20px'})

    try:
        provider_name = click_data['points'][0]['label']
    except (KeyError, IndexError, TypeError):
        return html.P("No se pudo obtener el proveedor seleccionado",
                      style={'textAlign': 'center', 'marginTop': '20px', 'color': 'red'})

    filtered_df_general = filter_dataframe(df.copy(), tic_filter, selected_siglas, selected_gobierno, importe_range)

    if filtered_df_general.empty:
        return html.P("No hay datos para los filtros seleccionados",
                      style={'textAlign': 'center', 'marginTop': '20px'})

    if 'Proveedor o contratista' not in filtered_df_general.columns:
        return html.P("La Columna 'Proveedor o contratista' no se encuentra en los datos.", style={'color': 'red', 'textAlign': 'center', 'marginTop': '20px'})

    provider_contracts_df = filtered_df_general[filtered_df_general['Proveedor o contratista'] == provider_name]

    
    if provider_contracts_df.empty:
        return html.P(f"No se encontraron contratos para '{provider_name}' en los datos filtrados.",
                      style={'textAlign': 'center', 'marginTop': '20px'})
    


    col_titulo = 'Título del contrato'
    col_importe = 'Importe DRC'
    col_fecha_inicio = 'Fecha de inicio del contrato'
    col_fecha_fin = 'Fecha de fin del contrato'
    col_direccion_anuncio = 'Dirección del anuncio'

    
    if col_importe in provider_contracts_df.columns:
        provider_contracts_df[col_importe] = pd.to_numeric(provider_contracts_df[col_importe], errors='coerce')

    warning_message = None # Inicializar warning_message
    cols_for_table = [] # Inicializar cols_for_table

    if col_direccion_anuncio not in provider_contracts_df.columns:
        cols_to_select = [col_titulo, col_importe, col_fecha_inicio, col_fecha_fin]
        existing_cols_for_selection = [col for col in cols_to_select if col in provider_contracts_df.columns]
        cols_to_display_data = provider_contracts_df[existing_cols_for_selection].copy()

        if col_titulo in existing_cols_for_selection:
            cols_for_table.append({"name": "Título del Contrato", "id": col_titulo})
        if col_importe in existing_cols_for_selection:
            cols_for_table.append({
                "name": "Importe DRC", 
                "id": col_importe, 
                "type": "numeric", 
                "format": dash_table.Format.Format(symbol='$',precision=2, group=',')
            })
        if col_fecha_inicio in existing_cols_for_selection:
            cols_for_table.append({"name": "Fecha Inicio", "id": col_fecha_inicio})
        if col_fecha_fin in existing_cols_for_selection:
            cols_for_table.append({"name": "Fecha Fin", "id": col_fecha_fin})

        warning_message = html.P("Advertencia: La columna 'Dirección del anuncio' no fue encontrada.", style={'color': 'orange', 'textAlign': 'center'})
    else:
        provider_contracts_df['Enlace_Anuncio_MD'] = provider_contracts_df[col_direccion_anuncio].apply(
            lambda x: f"[Ver Anuncio]({x})" if pd.notna(x) and str(x).strip().lower().startswith('http') else "N/A"
        )
        cols_to_select = [col_titulo, col_importe, col_fecha_inicio, col_fecha_fin, 'Enlace_Anuncio_MD']
        existing_cols_for_selection = [col for col in cols_to_select if col in provider_contracts_df.columns or col == 'Enlace_Anuncio_MD']
        cols_to_display_data = provider_contracts_df[existing_cols_for_selection].copy()

        if col_titulo in existing_cols_for_selection:
            cols_for_table.append({"name": "Título del Contrato", "id": col_titulo})
        if col_importe in existing_cols_for_selection:
            cols_for_table.append({
                "name": "Importe DRC", 
                "id": col_importe, 
                "type": "numeric", 
                "format": dash_table.Format.Format(symbol='$', precision=2, group=',') 
            })
        if col_fecha_inicio in existing_cols_for_selection:
            cols_for_table.append({"name": "Fecha Inicio", "id": col_fecha_inicio})
        if col_fecha_fin in existing_cols_for_selection:
            cols_for_table.append({"name": "Fecha Fin", "id": col_fecha_fin})
        if 'Enlace_Anuncio_MD' in existing_cols_for_selection:
            cols_for_table.append({"name": "Dirección del Anuncio", "id": "Enlace_Anuncio_MD", 'presentation': 'markdown'})
        
        # warning_message = None # Ya está inicializado arriba

    if col_fecha_inicio in cols_to_display_data.columns:
        cols_to_display_data[col_fecha_inicio] = pd.to_datetime(cols_to_display_data[col_fecha_inicio], errors = 'coerce').dt.strftime('%d/%m/%Y')

    if col_fecha_fin in cols_to_display_data.columns:
        cols_to_display_data[col_fecha_fin] = pd.to_datetime(cols_to_display_data[col_fecha_fin], errors = 'coerce').dt.strftime('%d/%m/%Y')

    if not cols_for_table:
        return html.P(f"No hay columnas válidas para mostrar para el proveedor '{provider_name}'.",
                      style={'textAlign': 'center', 'marginTop': '20px', 'color': 'red'})

    table = dash_table.DataTable(
        id='provider-specific-contracts-table',
        columns=cols_for_table, # Esta la generas dinámicamente como ya lo haces
        data=cols_to_display_data.to_dict('records'),
        style_table={
            'overflowX': 'auto', # Permite scroll horizontal si la tabla es muy ancha
            'marginTop': '20px',
            'border': '1px solid #ddd', # Un borde sutil alrededor de la tabla
            'borderRadius': '5px', # Bordes redondeados para la tabla
        },
        style_header={
            'backgroundColor': '#34495e', # Un azul/gris oscuro para el encabezado
            'color': 'white',
            'fontWeight': 'bold',
            'textAlign': 'center', # Centrar texto del encabezado
            'border': '1px solid #34495e',
            'padding': '10px', # Más padding en el encabezado
        },
        style_cell={ # Estilo general para todas las celdas de datos
            'textAlign': 'left',
            'padding': '10px', # Buen padding para legibilidad
            'minWidth': '120px', # Ancho mínimo para todas las celdas
            'width': 'auto',     # Permitir que el ancho se ajuste al contenido
            'maxWidth': '400px', # Ancho máximo para evitar que una celda sea demasiado ancha
            'whiteSpace': 'normal', # Permite que el texto se ajuste en múltiples líneas
            'height': 'auto',      # Altura automática para ajustarse al contenido
            'border': '1px solid #eee', # Bordes más sutiles para las celdas
            'fontFamily': 'Arial, sans-serif', # Fuente más estándar
            'fontSize': '14px', # Tamaño de fuente legible
            'verticalAlign': 'middle', # Alinear texto verticalmente al medio
        },
        style_cell_conditional=[ # Estilos específicos por columna
            {
                'if': {'column_id': col_titulo}, # Asumiendo que col_titulo es 'Título del contrato'
                'textAlign': 'left',
                'minWidth': '250px', # Más ancho para el título
                'maxWidth': '500px',
                'fontWeight': 'bold', # Hacer el título un poco más prominente
            },
            {
                'if': {'column_id': col_importe}, # Asumiendo que col_importe es 'Importe DRC'
                'textAlign': 'right', # Alinear números a la derecha
                'minWidth': '150px',
                'fontFamily': 'monospace', # Fuente monoespaciada para números puede verse bien
            },
            {
                'if': {'column_id': col_fecha_inicio},
                'textAlign': 'center',
                'minWidth': '130px',
            },
            {
                'if': {'column_id': col_fecha_fin},
                'textAlign': 'center',
                'minWidth': '130px',
            },
            {
                'if': {'column_id': 'Enlace_Anuncio_MD'}, # La columna con el enlace Markdown
                'textAlign': 'center',
                'minWidth': '150px',
            }
        ],
        style_data_conditional=[
            {'if': {'row_index': 'odd'},
             'backgroundColor': '#f9f9f9'} # Alternar color de filas
        ],
        page_size=10,
        filter_action='native',
        sort_action='native',
    )

    return html.Div([
        html.H4(f"Contratos para: {provider_name}", style={'marginTop': '20px', 'marginBottom': '10px', 'textAlign': 'center'}),
        warning_message if warning_message else "", # Muestra el mensaje de advertencia si existe
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

    # Si no hay datos después de filtrar
    if filtered_df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No hay datos para los filtros seleccionados",
            xaxis=dict(title=""),
            yaxis=dict(title="")
        )

        return empty_fig, empty_fig, empty_fig

    # Mapa de distribución geográfica (simplificado por la falta de coordenadas)
    # En una implementación real, necesitaríamos mapear instituciones o proveedores a ubicaciones geográficas
    if 'Orden de gobierno' in filtered_df.columns:
        # Crear un mapa simplificado usando una gráfica de barras por orden de gobierno
        geo_df = filtered_df.groupby('Orden de gobierno').size().reset_index(name='count')

        fig_geo = px.bar(
            geo_df,
            x='Orden de gobierno',
            y='count',
            title="Distribución de Contratos TIC por Orden de Gobierno",
            labels={'count': 'Número de Contratos', 'Orden de gobierno': ''},
            color='Orden de gobierno'
        )

        # Mensaje informativo sobre la limitación
        fig_geo.add_annotation(
            x=0.5, y=0.9,
            xref="paper", yref="paper",
            text="Nota: Para un mapa geográfico completo se requieren datos de ubicación",
            showarrow=False,
            font=dict(size=12, color="gray")
        )
    else:
        fig_geo = go.Figure()
        fig_geo.update_layout(title="No hay datos disponibles para distribución geográfica")

    # Gasto TIC por Orden de Gobierno
    if 'Orden de gobierno' in filtered_df.columns and 'Importe DRC' in filtered_df.columns:
        gov_spending = filtered_df.groupby('Orden de gobierno')['Importe DRC'].sum().reset_index()
        gov_spending = gov_spending.sort_values('Importe DRC', ascending=False)

        fig_gov = px.pie(
            gov_spending,
            values='Importe DRC',
            names='Orden de gobierno',
            title="Distribución del Gasto TIC por Orden de Gobierno",
            hole=0.4
        )
    else:
        fig_gov = go.Figure()
        fig_gov.update_layout(title="No hay datos disponibles para análisis por orden de gobierno")

    # Tipo de Procedimiento por Región/Gobierno
    if 'Tipo Procedimiento' in filtered_df.columns and 'Orden de gobierno' in filtered_df.columns:
        # Agrupar por orden de gobierno y tipo de procedimiento
        proc_df = filtered_df.groupby(['Orden de gobierno', 'Tipo Procedimiento']).size().reset_index(name='count')

        fig_proc = px.bar(
            proc_df,
            x='Orden de gobierno',
            y='count',
            color='Tipo Procedimiento',
            title="Tipos de Procedimiento por Orden de Gobierno",
            labels={'count': 'Número de Contratos', 'Orden de gobierno': ''}
        )
    else:
        fig_proc = go.Figure()
        fig_proc.update_layout(title="No hay datos disponibles para análisis de procedimientos")

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

    # Figuras vacías por defecto
    empty_fig = go.Figure()
    empty_fig.update_layout(
            title="No hay datos para los filtros seleccionados",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
    fig_terms = empty_fig
    fig_trend = empty_fig
    fig_cooccur = empty_fig
    fig_value_term = empty_fig # Placeholder para la figura nueva


    if filtered_df.empty:
        return fig_terms, fig_trend, fig_cooccur, fig_value_term # Devuelve 4 figuras vacías

    # --- Cálculos (con validaciones) ---
    all_terms = [] # Inicializar lista para todos los términos

    # Función segura para obtener listas de términos
    def safe_eval_list(val):
        if isinstance(val, list): return val
        if isinstance(val, str):
            try:
                import ast
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list): return parsed
            except (ValueError, SyntaxError): pass
        return []

    # Top Términos TIC (requiere 'terminos_positivos')
    if 'terminos_positivos' in filtered_df.columns:
        terms_base = filtered_df['terminos_positivos'].dropna().apply(safe_eval_list)
        all_terms = [term for sublist in terms_base for term in sublist] # Aplanar la lista de listas

        if all_terms:
            # Contar frecuencia de términos
            term_counts = Counter(all_terms).most_common(15) # Top 15
            terms_df = pd.DataFrame(term_counts, columns=['Término', 'Frecuencia'])

            if not terms_df.empty:
                fig_terms = px.bar(
                    terms_df, x='Término', y='Frecuencia',
                    title="Top 15 Términos TIC Encontrados",
                    labels={'Frecuencia': 'Frecuencia', 'Término': ''},
                    color='Frecuencia', color_continuous_scale='Viridis'
                )
                fig_terms.update_layout(xaxis={'categoryorder':'total descending'}, xaxis_title='', yaxis_title='Frecuencia')


    # Tendencias de Términos por Año (requiere 'terminos_positivos' y fecha)
    if all_terms and 'Fecha de inicio del contrato' in filtered_df.columns:
         trend_base = filtered_df[['Fecha de inicio del contrato', 'terminos_positivos']].dropna().copy()
         trend_base['Year'] = trend_base['Fecha de inicio del contrato'].dt.year
         trend_base['terminos_list'] = trend_base['terminos_positivos'].apply(safe_eval_list)
         trend_exploded = trend_base.explode('terminos_list').dropna(subset=['terminos_list'])

         # Obtener los 5 términos más comunes en general para seguir su tendencia
         top_5_terms = [term for term, _ in Counter(all_terms).most_common(5)]

         # Filtrar el df explotado para incluir solo los top 5 términos
         trend_filtered = trend_exploded[trend_exploded['terminos_list'].isin(top_5_terms)]

         # Contar frecuencia por año y término
         term_yearly_counts = trend_filtered.groupby(['Year', 'terminos_list']).size().reset_index(name='Frecuencia')

         if not term_yearly_counts.empty:
              # Ordenar por año para el gráfico de línea
              term_yearly_counts = term_yearly_counts.sort_values('Year')
              fig_trend = px.line(
                  term_yearly_counts, x='Year', y='Frecuencia', color='terminos_list',
                  title="Tendencias de los 5 Términos TIC Más Comunes por Año",
                  labels={'Frecuencia': 'Frecuencia Anual', 'Year': 'Año', 'terminos_list': 'Término'},
                  markers=True
              )
              fig_trend.update_layout(xaxis_title='Año', yaxis_title='Frecuencia Anual')


    # Co-ocurrencia de Términos (requiere 'terminos_positivos')
    if all_terms:
        cooccur_base = filtered_df['terminos_positivos'].dropna().apply(safe_eval_list)
        # Tomar los 10 términos más comunes para la matriz
        top_10_terms = [term for term, _ in Counter(all_terms).most_common(10)]

        # Crear matriz de co-ocurrencia (inicializar con ceros)
        cooccurrence_matrix = pd.DataFrame(0, index=top_10_terms, columns=top_10_terms)

        # Llenar la matriz
        for terms_list in cooccur_base:
            relevant_terms = [term for term in terms_list if term in top_10_terms]
            # Iterar sobre pares únicos dentro de la lista de términos relevantes
            from itertools import combinations
            for term1, term2 in combinations(relevant_terms, 2):
                 if term1 != term2: # Asegurar que no sean el mismo término
                     cooccurrence_matrix.loc[term1, term2] += 1
                     cooccurrence_matrix.loc[term2, term1] += 1 # Matriz simétrica
            # Contar también la diagonal (ocurrencia individual dentro de estas listas)
            for term in relevant_terms:
                cooccurrence_matrix.loc[term, term] += 1


        if not cooccurrence_matrix.empty:
             fig_cooccur = px.imshow(
                 cooccurrence_matrix,
                 title="Co-ocurrencia de los 10 Términos TIC Más Comunes",
                 labels=dict(x="Término 1", y="Término 2", color="Co-ocurrencias"),
                 color_continuous_scale='Viridis'
             )
             # Ajustar tamaño para mejor visualización
             fig_cooccur.update_layout(height=600)


    # ---- Implementación Placeholder para term-value-graph ----
    # TODO: Implementar la lógica real aquí. Necesitarás 'terminos_positivos' y 'Importe DRC'.
    if all_terms and 'Importe DRC' in filtered_df.columns and filtered_df['Importe DRC'].notna().any():
        value_term_base = filtered_df[['terminos_positivos', 'Importe DRC']].dropna().copy()
        value_term_base['terminos_list'] = value_term_base['terminos_positivos'].apply(safe_eval_list)
        value_term_exploded = value_term_base.explode('terminos_list').dropna(subset=['terminos_list'])

        # Calcular valor promedio por término
        term_value_agg = value_term_exploded.groupby('terminos_list')['Importe DRC'].agg(['mean', 'count']).reset_index()
        term_value_agg.rename(columns={'mean': 'Valor Promedio', 'count': 'Num Contratos', 'terminos_list': 'Término'}, inplace=True)

        # Filtrar términos con al menos N contratos (ej. 3) y tomar Top 15 por valor promedio
        term_value_filtered = term_value_agg[term_value_agg['Num Contratos'] >= 3].sort_values('Valor Promedio', ascending=False).head(15)

        if not term_value_filtered.empty:
            fig_value_term = px.bar(
                term_value_filtered, x='Término', y='Valor Promedio',
                color='Num Contratos',
                title="Valor Promedio de Contratos por Término TIC (Top 15)",
                labels={'Valor Promedio': 'Valor Promedio (MXN)', 'Término': '', 'Num Contratos': 'Núm. Contratos'},
                color_continuous_scale='Viridis',
                hover_data={'Valor Promedio': ':,.2f'}
            )
            fig_value_term.update_layout(xaxis={'categoryorder':'total descending'}, xaxis_title='', yaxis_title='Valor Promedio (MXN)')
        else:
            # Si no hay suficientes datos después de filtrar N contratos
            fig_value_term = go.Figure()
            fig_value_term.update_layout(title="No hay suficientes datos para 'Valor Promedio por Término'")

    else:
        # Si faltan columnas necesarias
        fig_value_term = go.Figure()
        fig_value_term.update_layout(title="Faltan datos (términos o importe) para 'Valor Promedio por Término'")
    # ---- Fin Implementación Placeholder ----


    # --- FIX: Añadir la sentencia return faltante ---
    # Devuelve las 4 figuras, incluyendo el placeholder para la última
    return fig_terms, fig_trend, fig_cooccur, fig_value_term

#Callbacks para actualizar la nueva ventana de Redes de Contrtacion 
@app.callback(
    Output('network-graph', 'figure'),
    [Input('tic-dropdown', 'value'),
     Input('siglas-dropdown', 'value'),
     Input('gobierno-dropdown', 'value'),
     Input('importe-slider', 'value')]
)
def update_tab_network(tic_filter, selected_siglas, selected_gobierno, importe_range):
    filtered_df = filter_dataframe(df, tic_filter, selected_siglas, selected_gobierno, importe_range)

    #-----Preparacion de los datos para la red------
    if filtered_df.empty or 'Proveedor o contratista'  not in filtered_df.columns or 'Siglas de la Institución' not in filtered_df.columns:
        fig_network = go.Figure()
        fig_network.update_layout(title="No hay datos suficientes para crear la red", showlegend=False,
                                  xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        return fig_network
    
    # 1. Crear pares y agregar (usar importe si existe, si no, contar contratos)
    link_col = 'Importe DRC' if 'Importe DRC' in filtered_df.columns and filtered_df['Importe DRC'].notna().any() else 'Código del contrato'
    agg_func = 'sum' if link_col == 'Importe DRC' else 'count'

    # Asegurarse de no tener nulos en las columnas clave para la red
    network_data = filtered_df.dropna(subset=['Proveedor o contratista', 'Siglas de la Institución', link_col])

    if network_data.empty:
         fig_network = go.Figure()
         fig_network.update_layout(title="No hay datos válidos (proveedor, institución, valor/conteo) para generar la red", showlegend=False,
                                   xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                   yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                   plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
         return fig_network


    edges_df = network_data.groupby(['Proveedor o contratista', 'Siglas de la Institución']).agg(
        weight=(link_col, agg_func)
    ).reset_index()

    # Limitar a las N relaciones más fuertes para claridad (e.g., Top 50)
    top_n = 50
    edges_df = edges_df.sort_values('weight', ascending=False).head(top_n)

    if edges_df.empty:
         fig_network = go.Figure()
         fig_network.update_layout(title="No se encontraron relaciones significativas para mostrar en la red", showlegend=False,
                                   xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                   yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                   plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
         return fig_network


    # 2. Crear lista de nodos únicos (proveedores e instituciones de las relaciones top)
    proveedores = pd.unique(edges_df['Proveedor o contratista'])
    instituciones = pd.unique(edges_df['Siglas de la Institución'])
    all_nodes = np.concatenate([proveedores, instituciones])
    node_map = {name: i for i, name in enumerate(all_nodes)}
    node_types = ['Proveedor'] * len(proveedores) + ['Institución'] * len(instituciones)

    # 3. Crear posiciones para los nodos (Layout Circular Simple)
    num_nodes = len(all_nodes)
    radius = 5
    angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
    pos_x = radius * np.cos(angles)
    pos_y = radius * np.sin(angles)
    node_positions = {name: (pos_x[i], pos_y[i]) for i, name in enumerate(all_nodes)}

    # --- Creación de la Figura Plotly ---
    fig_network = go.Figure()

    # 4. Crear Trazas para los Enlaces (Edges)
    edge_x, edge_y = [], []
    edge_weights = []
    for _, row in edges_df.iterrows():
        prov_name = row['Proveedor o contratista']
        inst_name = row['Siglas de la Institución']
        x0, y0 = node_positions[prov_name]
        x1, y1 = node_positions[inst_name]
        edge_x.extend([x0, x1, None]) # None para separar líneas
        edge_y.extend([y0, y1, None])
        edge_weights.append(row['weight'])

    # Normalizar pesos para grosor de línea (evita líneas muy gruesas)
    min_weight = min(edge_weights) if edge_weights else 1
    max_weight = max(edge_weights) if edge_weights else 1
    if max_weight == min_weight: # Evitar división por cero si todos los pesos son iguales
         normalized_weights = [2] * len(edge_weights) # Grosor fijo
    else:
         normalized_weights = [1 + 4 * (w - min_weight) / (max_weight - min_weight) for w in edge_weights] # Escala de 1 a 5

    # Crear traza de enlaces (una sola traza para eficiencia)
    # Necesitamos repetir los pesos para cada segmento de línea (x0,y0 -> x1,y1)
    edge_trace_weights = []
    for w in normalized_weights:
        edge_trace_weights.extend([w, w, np.nan]) # Peso para x0, peso para x1, NaN

    # Debido a limitaciones en Scatter para ancho de línea variable por segmento,
    # crearemos trazas individuales para cada enlace para poder controlar el grosor.
    # Esto es menos eficiente pero funcional.
    for i, row in enumerate(edges_df.iterrows()):
        index, data = row
        prov_name = data['Proveedor o contratista']
        inst_name = data['Siglas de la Institución']
        x0, y0 = node_positions[prov_name]
        x1, y1 = node_positions[inst_name]
        weight_label = f"${data['weight']:,.2f}" if link_col == 'Importe DRC' else f"{data['weight']} contratos"

        fig_network.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode='lines',
            line=dict(width=normalized_weights[i], color='rgba(150,150,150,0.5)'), # Grosor basado en peso
            hoverinfo='text',
            text=f"Relación: {prov_name} - {inst_name}<br>Valor/Conteo: {weight_label}" # Tooltip del enlace
        ))


    # 5. Crear Traza para los Nodos
    node_x = [pos[0] for pos in node_positions.values()]
    node_y = [pos[1] for pos in node_positions.values()]
    node_text = [f"{node_type}: {name}" for name, node_type in zip(all_nodes, node_types)]
    node_colors = ['blue' if nt == 'Institución' else 'red' for nt in node_types] # Distinguir tipos

    # Calcular tamaño del nodo basado en el grado (número de conexiones en el Top N)
    node_degrees = {name: 0 for name in all_nodes}
    for _, row in edges_df.iterrows():
        node_degrees[row['Proveedor o contratista']] += 1
        node_degrees[row['Siglas de la Institución']] += 1

    node_sizes = [5 + node_degrees[name] * 2 for name in all_nodes] # Tamaño base + ajuste por grado


    fig_network.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text', # Mostrar marcador y nombre
        text=[name for name in all_nodes], # Mostrar nombre del nodo
        textposition="top center",
        hoverinfo='text',
        hovertext=node_text, # Tooltip detallado
        marker=dict(
            showscale=False,
            color=node_colors,
            size=node_sizes, # Tamaño basado en grado
            line_width=1,
            line_color='black'
        )
    ))

    # 6. Configurar Layout de la Figura
    fig_network.update_layout(
        title=None, # Mantener sin título para más espacio
        # titlefont_size=16, # LÍNEA ELIMINADA
        showlegend=False,
        hovermode='closest',
        margin=dict(b=10,l=5,r=5,t=10), # Márgenes ajustados
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[min(pos_x)-2, max(pos_x)+2]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[min(pos_y)-2, max(pos_y)+2]),
        plot_bgcolor='rgba(248, 249, 250, 1)' # Fondo ajustado
    )

    return fig_network
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
                width: 30%;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                background: white;
                padding: 15px;
            }
            h1, h3 {
                color: #2c3e50;
            }
            table {
                border-collapse: collapse;
                width: 100%;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
                position: sticky;
                top: 0;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
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
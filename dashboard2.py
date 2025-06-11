import dash
from dash import html, dcc, dash_table, no_update, State
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime, timedelta
import numpy as np
from collections import Counter
import calendar
import ast



#---------- Importaciones para BD y Seguridad --------
import os 
from dotenv import load_dotenv
from sqlalchemy import create_engine
from werkzeug.security import check_password_hash

# Carga el env 
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST =os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

# --- Carga de datos y preprocesamiento ---
df = pd.read_csv("contratos_tic_consolidados_llama3_colab (22).csv", encoding="utf-8")

column_mapping = {
    'CÃ³digo del contrato': 'Código del contrato',
    'TÃ­tulo del contrato': 'Título del contrato',
    'Tipo de contrataciÃ³n': 'Tipo de contratación',
    'DirecciÃ³n del anuncio': 'Dirección del anuncio',
    'Siglas de la InstituciÃ³n': 'Siglas de la Institución'
}
df.rename(columns=column_mapping, inplace=True)

if 'clasificacion_llm' in df.columns:
    df['es_TIC'] = df['clasificacion_llm'].apply(
        lambda x: True if isinstance(x, str) and x.strip().upper() == 'TIC' else False
    )
else:
    df['es_TIC'] = False
    print("Advertencia: La columna 'clasificacion_llm' no se encontró. 'es_TIC' se ha establecido a False para todos los registros.")

if 'subcategoria_tic' in df.columns:
    df['terminos_positivos'] = df['subcategoria_tic'].apply(
        lambda x: [x.strip()] if pd.notna(x) and isinstance(x, str) and x.strip() != "" else []
    )
else:
    df['terminos_positivos'] = pd.Series([[] for _ in range(len(df))])
    print("Advertencia: La columna 'subcategoria_tic' no se encontró. 'terminos_positivos' se ha establecido como listas vacías.")

if 'marca_tic' not in df.columns:
    df['marca_tic'] = pd.Series([None] * len(df))  # O pd.NA
    print("Advertencia: La columna 'marca_tic' no se encontró en el CSV. Se ha creado como una columna vacía.")

df['Fecha de inicio del contrato'] = pd.to_datetime(df['Fecha de inicio del contrato'], dayfirst=True, errors='coerce')
df['Fecha de fin del contrato'] = pd.to_datetime(df['Fecha de fin del contrato'], dayfirst=True, errors='coerce')
df['Fecha de firma del contrato'] = pd.to_datetime(df['Fecha de firma del contrato'], errors='coerce')

df['Duración del contrato (días)'] = (df['Fecha de fin del contrato'] - df['Fecha de inicio del contrato']).dt.days
df['Mes de inicio'] = df['Fecha de inicio del contrato'].dt.month
df['Año de inicio'] = df['Fecha de inicio del contrato'].dt.year
df['Mes de fin'] = df['Fecha de fin del contrato'].dt.month
df['Año de fin'] = df['Fecha de fin del contrato'].dt.year
df['Tiempo hasta vencimiento (días)'] = (df['Fecha de fin del contrato'] - datetime.now()).dt.days

df['Oportunidad'] = pd.cut(
    df['Tiempo hasta vencimiento (días)'],
    bins=[-float('inf'), 0, 90, 180, 365, float('inf')],
    labels=['Vencido', 'Urgente (< 3 meses)', 'Corto plazo (3-6 meses)', 'Medio plazo (6-12 meses)', 'Largo plazo (> 12 meses)']
)

if 'Siglas de la Institución' in df.columns:
    siglas_list = df['Siglas de la Institución'].dropna().unique()
else:
    siglas_list = []
    print("Advertencia: La columna 'Siglas de la Institución' no se encontró después del preprocesamiento.")

orden_gobierno_list = df['Orden de gobierno'].dropna().unique() if 'Orden de gobierno' in df.columns else []

if 'Estado' in df.columns:
    estado_list = sorted(df['Estado'].dropna().unique())
else:
    estado_list = []
    print("Advertencia: La columna 'Estado' no se encontró. El filtro de estado no estará disponible.")
# --- Instancia de la App ---
app = dash.Dash(__name__, suppress_callback_exceptions=True, prevent_initial_callbacks='initial_duplicate')
server = app.server

# --- Definición de Layouts ---
login_layout = html.Div([
    html.Div([
        html.H2("Iniciar Sesión", style={'textAlign': 'center'}),
        html.Div(id='login-error', style={'color': 'red', 'textAlign': 'center', 'margin-bottom': '10px'}),
        dcc.Input(id='username-input', type='text', placeholder='Usuario', style={'margin': '10px auto', 'display': 'block', 'width': '250px', 'padding': '8px'}),
        dcc.Input(id='password-input', type='password', placeholder='Contraseña', style={'margin': '10px auto', 'display': 'block', 'width': '250px', 'padding': '8px'}),
        html.Button('Ingresar', id='login-button', n_clicks=0, style={'margin': '20px auto', 'display': 'block', 'backgroundColor': '#2c3e50', 'color': 'white', 'padding': '10px 20px', 'border': 'none', 'borderRadius': '5px'})
    ], style={
        'width': '350px', 'margin': '100px auto', 'padding': '40px',
        'border': '1px solid #ddd', 'borderRadius': '10px', 'backgroundColor': 'white',
        'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'
    })
])

main_layout = html.Div([
    # NUEVO: Botón de Cerrar Sesión
    html.Div([
        html.Button('Cerrar Sesión', id='logout-button', n_clicks=0, style={
            'margin': '10px',
            'padding': '5px 10px',
            'color': '#c0392b',
            'backgroundColor': 'white',
            'border': '1px solid #c0392b',
            'borderRadius': '5px',
            'cursor': 'pointer'
        })
    ], style={'textAlign': 'right'}),

    html.H1(
        "Dashboard de Oportunidades en Contratos TIC Gubernamentales",
        style={'textAlign': 'center', 'color': '#2c3e50', 'margin-bottom': '20px'}
    ),
    html.Div([
        html.Div([
            html.Label("Filtrar por Clasificación TIC"),
            dcc.Dropdown(id='tic-dropdown', options=[{'label': 'Todos los contratos', 'value': 'todos'}, {'label': 'Solo contratos TIC', 'value': 'tic'}, {'label': 'Solo contratos no TIC', 'value': 'no_tic'}], value='tic'),
        ], style={'width': '23%', 'display': 'inline-block', 'margin-right': '2%'}),
        html.Div([
            html.Label("Seleccionar Institución"),
            dcc.Dropdown(id='siglas-dropdown', options=[{'label': sigla, 'value': sigla} for sigla in siglas_list], multi=True, value=[]),
        ], style={'width': '23%', 'display': 'inline-block', 'margin-right': '2%'}),
        html.Div([
            html.Label("Orden de Gobierno"),
            dcc.Dropdown(id='gobierno-dropdown', options=[{'label': orden, 'value': orden} for orden in orden_gobierno_list], multi=True, value=[]),
        ], style={'width': '23%', 'display': 'inline-block', 'margin-right': '2%'}),
        html.Div([
            html.Label("Seleccionar Estado"),
            dcc.Dropdown(id='estado-dropdown', options=[{'label': estado, 'value': estado} for estado in estado_list], multi=True, value=[]),
        ], style={'width': '23%', 'display': 'inline-block'}),
    ], style={'margin-bottom': '20px', 'backgroundColor': '#f9f9f9', 'padding': '15px', 'borderRadius': '5px'}),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Resumen y Tendencias', value='tab-1'),
        dcc.Tab(label='Oportunidades de Negocio', value='tab-2'),
        dcc.Tab(label='Análisis Competitivo', value='tab-3'),
        dcc.Tab(label='Distribución Geográfica', value='tab-4'),
        dcc.Tab(label='Análisis de Términos y Marcas TIC', value='tab-5'),
        dcc.Tab(label='Redes de Contratacion', value='tab-network'),
    ]),
    html.Div(id='tabs-content')
])


# Layout "Raíz" que decide qué página mostrar
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='session-store', storage_type='session'),
    html.Div(id='page-content')
])


# --- Callbacks de Login y Ruteo ---
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname'),
    Input('session-store', 'data')
)
def display_page(pathname, session_data):
    is_authenticated = session_data and session_data.get('authenticated', False)
    if not is_authenticated:
        return login_layout
    return main_layout

@app.callback(
    Output('session-store', 'data'),
    Output('url', 'pathname'),
    Output('login-error', 'children'),
    Input('login-button', 'n_clicks'),
    State('username-input', 'value'),
    State('password-input', 'value'),
    prevent_initial_call=True
)
def handle_login(n_clicks, username, password):
    if not username or not password:
        return no_update, no_update, "Por favor, ingrese usuario y contraseña."
    try:
        query = "SELECT password_hash FROM usuarios WHERE username = %(username)s;"
        user_data = pd.read_sql(query, engine, params={'username': username})
        if not user_data.empty:
            stored_password_hash = user_data['password_hash'].iloc[0]
            if check_password_hash(stored_password_hash, password):
                return {'authenticated': True}, '/', ""
        return no_update, no_update, "Usuario o contraseña incorrectos."
    except Exception as e:
        print(f"Error de autenticación en la base de datos: {e}")
        return no_update, no_update, "Error en el servidor. Intente más tarde."

# NUEVO: Callback para Cerrar Sesión
@app.callback(
    Output('session-store', 'data', allow_duplicate=True),
    Output('url', 'pathname', allow_duplicate=True),
    Input('logout-button', 'n_clicks'),
    prevent_initial_call=True
)
def handle_logout(n_clicks):
    # Al hacer clic, borra los datos de la sesión y redirige a /login
    return None, '/login'

@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_tab_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.Div([
                html.Div(id='metric-tic-percentage', className='metric-box'),
                html.Div(id='metric-total-contratos', className='metric-box'),
                html.Div(id='metric-avg-importe', className='metric-box'),
                html.Div(id='metric-total-value', className='metric-box'),
            ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '20px'}),
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
                ], style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'top'}),
                html.Div([
                    html.H3("Valor de Contratos Próximos a Vencer"),
                    dcc.Graph(id='value-expiry-graph'),
                ], style={'width': '49%', 'display': 'inline-block', 'float': 'right', 'vertical-align': 'top'}),
            ], style={'margin-bottom': '20px', 'overflow': 'hidden'}),
            html.Div(
                id='opportunity-category-contracts-table-container',
                style={'margin-top': '20px', 'margin-bottom': '20px', 'clear': 'both'}
            ),
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
            html.Div([
                html.Div([
                    html.H3("Market Share de Proveedores TIC"),
                    dcc.Graph(id='provider-share-graph'),
                ], style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'top'}),
                html.Div([
                    html.H3("Especialización de Proveedores"),
                    dcc.Graph(id='provider-specialization-graph'),
                ], style={'width': '49%', 'display': 'inline-block', 'float': 'right', 'vertical-align': 'top'}),
            ], style={'margin-bottom': '20px', 'overflow': 'hidden'}),
            html.Div(
                id='provider-contracts-table-container',
                style={'width': '100%', 'margin-top': '20px'}
            ),
            html.Div([
                html.H3("Duración Promedio de Contratos por Proveedor"),
                dcc.Graph(id='contract-duration-graph'),
            ], style={'margin-bottom': '20px', 'clear': 'both'}),
            html.Div([
                html.H3("Análisis de Competidores"),
                html.Div(id='competitor-table-container')
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
            # html.Div([
            # html.H3("Gasto TIC por Estado"),
            # dcc.Graph(id='state-spending-graph'), # Necesitaría su propio callback
            # ], style={'margin-bottom': '20px'}),
        ])
    elif tab == 'tab-5':
        return html.Div([
            html.Div([
                html.Div([
                    html.H3("Top Términos TIC Encontrados"),
                    dcc.Graph(id='top-terms-graph'),
                ], style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                html.Div([
                    html.H3("Tendencias de Términos TIC por Año"),
                    dcc.Graph(id='terms-trend-graph'),
                ], style={'width': '49%', 'display': 'inline-block', 'float': 'right', 'verticalAlign': 'top'}),
            ], style={'margin-bottom': '20px', 'overflow': 'hidden'}),
            html.Div(
                id='term-contracts-table-container',
                style={'width': '100%', 'margin-top': '20px', 'margin-bottom': '20px'}
            ),
            html.Div([
                html.Div([
                    html.H3("Top Marcas TIC por Frecuencia"),
                    dcc.Graph(id='top-marcas-graph'),
                ], style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                html.Div([
                    html.H3("Valor Promedio de Contratos por Término TIC"),
                    dcc.Graph(id='term-value-graph'),
                ], style={'width': '49%', 'display': 'inline-block', 'float': 'right', 'verticalAlign': 'top'}),
            ], style={'margin-bottom': '20px', 'clear': 'both', 'overflow': 'hidden'}),
            html.Div(
                id='marca-contracts-table-container',
                style={'width': '100%', 'margin-top': '20px', 'margin-bottom': '20px'}
            ),
        ])
    elif tab == 'tab-network':
        return html.Div([
            html.H3("Redes de Relaciones Proveedor-Institución"),
            dcc.Graph(id='network-graph', style={'height': '700px'}),
            html.P("Visualiza que proveedores contratan con que insituciones. El tamaño del enlace representa el valor total de los contratos entre ellos (Top 50 relaciones mostradas)."),
        ])


def filter_dataframe(df_original, tic_filter, selected_siglas, selected_gobierno, selected_estado):
    if df_original is None or not isinstance(df_original, pd.DataFrame):
        return pd.DataFrame()

    filtered_df = df_original.copy()

    if 'es_TIC' in filtered_df.columns:
        if tic_filter == 'tic':
            filtered_df = filtered_df[filtered_df['es_TIC'] == True]
        elif tic_filter == 'no_tic':
            filtered_df = filtered_df[filtered_df['es_TIC'] == False]
    elif tic_filter != 'todos':
        print(f"Advertencia: La columna 'es_TIC' no existe, no se puede filtrar por clasificación TIC '{tic_filter}'.")

    if 'Siglas de la Institución' in filtered_df.columns and selected_siglas:
        filtered_df = filtered_df[filtered_df['Siglas de la Institución'].isin(selected_siglas)]
    elif selected_siglas: # selected_siglas is not empty but column is missing
        print("Advertencia: La columna 'Siglas de la Institución' no existe, no se puede filtrar por institución.")

    if 'Orden de gobierno' in filtered_df.columns and selected_gobierno:
        filtered_df = filtered_df[filtered_df['Orden de gobierno'].isin(selected_gobierno)]
    elif selected_gobierno:
        print("Advertencia: La columna 'Orden de gobierno' no existe, no se puede filtrar por orden de gobierno.")

    if 'Estado' in filtered_df.columns and selected_estado:
        filtered_df = filtered_df[filtered_df['Estado'].isin(selected_estado)]
    elif selected_estado:
        print("Advertencia: La columna 'Estado' no existe, no se puede filtrar por estado.")

    return filtered_df


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
     Input('estado-dropdown', 'value')]
)
def update_tab1(tic_filter, selected_siglas, selected_gobierno, selected_estado):
    filtered_df = filter_dataframe(df, tic_filter, selected_siglas, selected_gobierno, selected_estado)

    if filtered_df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No hay datos para los filtros seleccionados",
            xaxis=dict(title="", showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(title="", showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        metric_empty_style = {'textAlign': 'center', 'padding': '10px', 'background': '#f0f0f0', 'borderRadius': '5px', 'flex': '1', 'margin': '5px'}
        metric_empty = html.Div([html.H4("Sin datos"), html.P("0")], style=metric_empty_style)
        return metric_empty, metric_empty, metric_empty, metric_empty, empty_fig, empty_fig, empty_fig, empty_fig

    total_contratos = len(filtered_df)
    contratos_tic = filtered_df['es_TIC'].sum() if 'es_TIC' in filtered_df.columns else 0
    porcentaje_tic = (contratos_tic / total_contratos) * 100 if total_contratos > 0 else 0
    avg_importe = filtered_df['Importe DRC'].mean() if 'Importe DRC' in filtered_df.columns and not filtered_df['Importe DRC'].dropna().empty else 0
    total_importe = filtered_df['Importe DRC'].sum() if 'Importe DRC' in filtered_df.columns and not filtered_df['Importe DRC'].dropna().empty else 0

    fig_trend = go.Figure().update_layout(title="No hay datos para tendencia de contratos")
    if 'Fecha de inicio del contrato' in filtered_df.columns and 'Código del contrato' in filtered_df.columns and 'es_TIC' in filtered_df.columns:
        yearly_df_prep = filtered_df.dropna(subset=['Fecha de inicio del contrato'])
        if not yearly_df_prep.empty:
            yearly_df = yearly_df_prep.groupby([yearly_df_prep['Fecha de inicio del contrato'].dt.year, 'es_TIC']).agg(
                count=('Código del contrato', 'count'),
                total_importe=('Importe DRC', 'sum') if 'Importe DRC' in filtered_df.columns else ('Código del contrato', 'count')
            ).reset_index().rename(columns={'Fecha de inicio del contrato': 'Año de inicio del contrato'})
            df_tic_trend = yearly_df[yearly_df['es_TIC'] == True]
            if not df_tic_trend.empty:
                fig_trend = px.line(
                    df_tic_trend, x='Año de inicio del contrato', y='count', title="Evolución de Contratos TIC por Año",
                    labels={'count': 'Número de Contratos', 'Año de inicio del contrato': 'Año'},
                    markers=True, line_shape='linear'
                )
                fig_trend.update_traces(line=dict(color='#2c3e50', width=3), marker=dict(size=10))
            else:
                fig_trend.update_layout(title="No hay contratos TIC para mostrar tendencia")
        else:
            fig_trend.update_layout(title="No hay fechas válidas para mostrar tendencia")

    fig_growth = go.Figure().update_layout(title="No hay datos para crecimiento de importe")
    if 'Importe DRC' in filtered_df.columns and 'Fecha de inicio del contrato' in filtered_df.columns:
        growth_df_prep = filtered_df.dropna(subset=['Fecha de inicio del contrato', 'Importe DRC'])
        if not growth_df_prep.empty:
            group_col = 'Tipo de contratación' if 'Tipo de contratación' in growth_df_prep.columns else None
            if group_col:
                growth_df = growth_df_prep.groupby(
                    [growth_df_prep['Fecha de inicio del contrato'].dt.year, group_col]
                )['Importe DRC'].sum().reset_index().rename(columns={'Fecha de inicio del contrato': 'Año Contrato'})
                if not growth_df.empty:
                    fig_growth = px.area(
                        growth_df, x='Año Contrato', y='Importe DRC', color=group_col,
                        title=f"Crecimiento de Importe por {group_col}",
                        labels={'Importe DRC': 'Importe Total (MXN)', 'Año Contrato': 'Año'}
                    )
                else:
                    fig_growth.update_layout(title=f"No hay datos para crecimiento de importe por {group_col}")
            else:
                growth_df = growth_df_prep.groupby(
                    growth_df_prep['Fecha de inicio del contrato'].dt.year
                )['Importe DRC'].sum().reset_index().rename(columns={'Fecha de inicio del contrato': 'Año Contrato'})
                if not growth_df.empty:
                    fig_growth = px.area(
                        growth_df, x='Año Contrato', y='Importe DRC', title="Crecimiento de Importe Total por Año",
                        labels={'Importe DRC': 'Importe Total (MXN)', 'Año Contrato': 'Año'}
                    )
                else:
                    fig_growth.update_layout(title="No hay datos para crecimiento de importe total por año")
        else:
            fig_growth.update_layout(title="No hay fechas o importes válidos para mostrar crecimiento")

    fig_season = go.Figure().update_layout(title="No hay datos para estacionalidad")
    if 'Fecha de inicio del contrato' in filtered_df.columns and 'Código del contrato' in filtered_df.columns:
        season_df_prep = filtered_df.dropna(subset=['Fecha de inicio del contrato'])
        if not season_df_prep.empty:
            season_df = season_df_prep.groupby(season_df_prep['Fecha de inicio del contrato'].dt.month).agg(
                count=('Código del contrato', 'count'),
                total_importe=('Importe DRC', 'sum') if 'Importe DRC' in filtered_df.columns else ('Código del contrato', 'count')
            ).reset_index().rename(columns={'Fecha de inicio del contrato': 'Mes Numérico'})
            month_names = {i: calendar.month_name[i] for i in range(1, 13)}
            season_df['Mes'] = season_df['Mes Numérico'].map(month_names)
            season_df = season_df.sort_values('Mes Numérico')
            y_col_season = 'total_importe' if 'total_importe' in season_df.columns and season_df['total_importe'].sum() > 0 else 'count'
            y_title_season = 'Importe Total (MXN)' if y_col_season == 'total_importe' else 'Número de Contratos'
            if not season_df.empty:
                fig_season = px.bar(
                    season_df, x='Mes', y=y_col_season, title="Estacionalidad en la Contratación TIC (Análisis Mensual)",
                    labels={y_col_season: y_title_season, 'Mes': ''},
                    color=y_col_season, color_continuous_scale='Viridis'
                )
                fig_season.update_xaxes(categoryorder='array', categoryarray=[calendar.month_name[i] for i in range(1, 13)])
            else:
                fig_season.update_layout(title="No hay datos agregados para estacionalidad")
        else:
            fig_season.update_layout(title="No hay fechas válidas para estacionalidad")

    fig_top = go.Figure().update_layout(title="No hay datos para top instituciones")
    if 'Siglas de la Institución' in filtered_df.columns:
        if 'Importe DRC' in filtered_df.columns and not filtered_df['Importe DRC'].dropna().empty:
            top_inst_prep = filtered_df.dropna(subset=['Siglas de la Institución', 'Importe DRC'])
            top_inst = top_inst_prep.groupby('Siglas de la Institución')['Importe DRC'].sum().reset_index().sort_values('Importe DRC', ascending=False).head(10)
            if not top_inst.empty:
                fig_top = px.bar(
                    top_inst, x='Siglas de la Institución', y='Importe DRC', title="Top 10 Instituciones por Gasto en TIC",
                    labels={'Importe DRC': 'Importe Total (MXN)', 'Siglas de la Institución': 'Institución'},
                    color='Importe DRC', color_continuous_scale='Viridis'
                )
            else:
                fig_top.update_layout(title="No hay datos de importe para top instituciones")
        elif 'Código del contrato' in filtered_df.columns:
            top_inst_prep = filtered_df.dropna(subset=['Siglas de la Institución'])
            top_inst = top_inst_prep.groupby('Siglas de la Institución')['Código del contrato'].count().reset_index(name='count').sort_values('count', ascending=False).head(10)
            if not top_inst.empty:
                fig_top = px.bar(
                    top_inst, x='Siglas de la Institución', y='count', title="Top 10 Instituciones por Número de Contratos TIC",
                    labels={'count': 'Número de Contratos', 'Siglas de la Institución': 'Institución'},
                    color='count', color_continuous_scale='Viridis'
                )
            else:
                fig_top.update_layout(title="No hay datos de conteo para top instituciones")
        else:
            fig_top.update_layout(title="Faltan columnas 'Importe DRC' o 'Código del contrato' para top instituciones")

    metric_style = {'textAlign': 'center', 'padding': '10px', 'background': '#f0f0f0', 'borderRadius': '5px', 'flex': '1', 'margin': '5px'}
    metric_tic = html.Div([html.H4("Porcentaje TIC"), html.P(f"{porcentaje_tic:.1f}%")], style=metric_style)
    metric_total = html.Div([html.H4("Total Contratos"), html.P(f"{total_contratos:,}")], style=metric_style)
    metric_avg = html.Div([html.H4("Importe Promedio"), html.P(f"${avg_importe:,.2f} MXN")], style=metric_style)
    metric_total_value = html.Div([html.H4("Valor Total"), html.P(f"${total_importe:,.2f} MXN")], style=metric_style)

    return metric_tic, metric_total, metric_avg, metric_total_value, fig_trend, fig_growth, fig_season, fig_top


@app.callback(
    [Output('opportunities-expiry-graph', 'figure'),
     Output('value-expiry-graph', 'figure'),
     Output('timeline-graph', 'figure'),
     Output('opportunity-table-container', 'children')],
    [Input('tic-dropdown', 'value'),
     Input('siglas-dropdown', 'value'),
     Input('gobierno-dropdown', 'value'),
     Input('estado-dropdown', 'value')]
)
def update_tab2(tic_filter, selected_siglas, selected_gobierno, selected_estado):
    filtered_df = filter_dataframe(df, tic_filter, selected_siglas, selected_gobierno, selected_estado)

    empty_fig_layout = {
        "title": "No hay datos para los filtros seleccionados",
        "xaxis": dict(showgrid=False, zeroline=False, showticklabels=False),
        "yaxis": dict(showgrid=False, zeroline=False, showticklabels=False),
        "plot_bgcolor": "rgba(0,0,0,0)", "paper_bgcolor": "rgba(0,0,0,0)"
    }
    if filtered_df.empty:
        empty_fig = go.Figure().update_layout(**empty_fig_layout)
        empty_table = html.Div("No hay datos para mostrar en la tabla.", style={'textAlign': 'center', 'padding': '20px'})
        return empty_fig, empty_fig, empty_fig, empty_table

    fig_opp = go.Figure().update_layout(title="No hay datos de 'Oportunidad' para gráfico")
    color_map_oportunidad = {
        'Vencido': 'grey', 'Urgente (< 3 meses)': 'red',
        'Corto plazo (3-6 meses)': 'orange', 'Medio plazo (6-12 meses)': 'blue',
        'Largo plazo (> 12 meses)': 'green'
    }
    order_oportunidad = list(color_map_oportunidad.keys())

    if 'Oportunidad' in filtered_df.columns and 'Código del contrato' in filtered_df.columns:
        opp_counts = filtered_df.groupby('Oportunidad', observed=False)['Código del contrato'].count().reset_index(name='count')
        for cat in order_oportunidad:
            if cat not in opp_counts['Oportunidad'].values:
                opp_counts = pd.concat([opp_counts, pd.DataFrame({'Oportunidad': [cat], 'count': [0]})], ignore_index=True)
        opp_counts['Oportunidad'] = pd.Categorical(opp_counts['Oportunidad'], categories=order_oportunidad, ordered=True)
        opp_counts = opp_counts.sort_values('Oportunidad')
        if not opp_counts.empty:
            fig_opp = px.bar(
                opp_counts, x='Oportunidad', y='count', title="Distribución de Contratos por Proximidad de Vencimiento",
                labels={'count': 'Número de Contratos', 'Oportunidad': ''},
                color='Oportunidad', color_discrete_map=color_map_oportunidad
            )
        else:
            fig_opp.update_layout(title="No hay datos agregados para oportunidades por vencimiento")

    fig_value = go.Figure().update_layout(title="No hay datos de 'Importe DRC' u 'Oportunidad' para gráfico")
    if 'Oportunidad' in filtered_df.columns and 'Importe DRC' in filtered_df.columns:
        value_by_expiry_prep = filtered_df.dropna(subset=['Importe DRC'])
        if not value_by_expiry_prep.empty:
            value_by_expiry = value_by_expiry_prep.groupby('Oportunidad', observed=False)['Importe DRC'].sum().reset_index()
            for cat in order_oportunidad:
                if cat not in value_by_expiry['Oportunidad'].values:
                    value_by_expiry = pd.concat([value_by_expiry, pd.DataFrame({'Oportunidad': [cat], 'Importe DRC': [0]})], ignore_index=True)
            value_by_expiry['Oportunidad'] = pd.Categorical(value_by_expiry['Oportunidad'], categories=order_oportunidad, ordered=True)
            value_by_expiry = value_by_expiry.sort_values('Oportunidad')
            if not value_by_expiry.empty:
                fig_value = px.bar(
                    value_by_expiry, x='Oportunidad', y='Importe DRC', title="Valor Total de Contratos por Proximidad de Vencimiento",
                    labels={'Importe DRC': 'Importe Total (MXN)', 'Oportunidad': ''},
                    color='Oportunidad', color_discrete_map=color_map_oportunidad
                )
            else:
                fig_value.update_layout(title="No hay datos agregados para valor por vencimiento")
        else:
            fig_value.update_layout(title="No hay importes válidos para valor por vencimiento")

    fig_timeline = go.Figure().update_layout(title="No hay datos para línea de tiempo de vencimientos")
    if 'Fecha de fin del contrato' in filtered_df.columns:
        now = datetime.now()
        timeline_df_prep = filtered_df[
            (filtered_df['Fecha de fin del contrato'] > now - timedelta(days=90)) &
            (filtered_df['Fecha de fin del contrato'].notna())
        ].copy()
        if not timeline_df_prep.empty:
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
                timeline_df_prep, x='Fecha de fin del contrato', y=y_col_timeline,
                size=size_col_timeline, color=color_col_timeline,
                hover_name=hover_name_col, hover_data=['tooltip_info'],
                title="Línea de Tiempo de Vencimientos de Contratos",
                labels={
                    'Fecha de fin del contrato': 'Fecha de Vencimiento',
                    y_col_timeline: 'Importe (MXN)' if y_col_timeline == 'Importe DRC' else 'Institución',
                    'Oportunidad': 'Categoría Oportunidad'
                },
                color_discrete_map=color_map_oportunidad
            )
            fig_timeline.add_vline(x=now, line_dash="dash", line_color="red")
            fig_timeline.add_annotation(x=now, y=1, yref="paper", text="Hoy", showarrow=False, yshift=10)
        else:
            fig_timeline.update_layout(title="No hay contratos próximos a vencer o vencidos recientemente")

    opportunity_table_content = html.Div("No hay datos disponibles para mostrar oportunidades en tabla.", style={'textAlign': 'center', 'padding': '20px'})
    if 'Tiempo hasta vencimiento (días)' in filtered_df.columns:
        opp_df = filtered_df[filtered_df['Tiempo hasta vencimiento (días)'] <= 365].copy()
        if not opp_df.empty:
            opp_df = opp_df.sort_values('Tiempo hasta vencimiento (días)')
            display_cols = ['Siglas de la Institución', 'Título del contrato', 'Proveedor o contratista',
                            'Fecha de fin del contrato', 'Tiempo hasta vencimiento (días)', 'Oportunidad']
            if 'Importe DRC' in opp_df.columns:
                display_cols.append('Importe DRC')

            table_cols_present = [col for col in display_cols if col in opp_df.columns]
            opp_table_df = opp_df[table_cols_present].head(20).copy()

            if 'Fecha de fin del contrato' in opp_table_df.columns:
                date_series = pd.to_datetime(opp_table_df['Fecha de fin del contrato'], errors='coerce')
                opp_table_df.loc[:, 'Fecha de fin del contrato'] = date_series.apply(lambda x: x.strftime('%d/%m/%Y') if pd.notnull(x) else 'N/A')
            if 'Importe DRC' in opp_table_df.columns:
                formatted_importes = opp_table_df['Importe DRC'].apply(lambda x: f"${x:,.2f} MXN" if pd.notna(x) else "N/A")
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


@app.callback(
    Output('opportunity-category-contracts-table-container', 'children'),
    [Input('opportunities-expiry-graph', 'clickData'),
     Input('tic-dropdown', 'value'),
     Input('siglas-dropdown', 'value'),
     Input('gobierno-dropdown', 'value'),
     Input('estado-dropdown', 'value')]
)
def display_opportunity_category_contracts_table(click_data, tic_filter, selected_siglas, selected_gobierno, selected_estado):
    ctx = dash.callback_context
    if not click_data or not click_data['points']:
        return html.P("Haz clic en una barra del gráfico 'Oportunidades por Proximidad de Vencimiento' para ver los contratos asociados.", style={'textAlign': 'center', 'marginTop': '20px'})

    try:
        selected_category = click_data['points'][0]['x']
    except (KeyError, IndexError, TypeError) as e:
        print(f"Error al extraer la categoría de oportunidad: {e}, click_data: {click_data}")
        return html.P("No se pudo obtener la categoría seleccionada del gráfico.", style={'textAlign': 'center', 'marginTop': '20px', 'color': 'red'})

    filtered_df_global = filter_dataframe(df.copy(), tic_filter, selected_siglas, selected_gobierno, selected_estado)
    if filtered_df_global.empty:
        return html.P(f"No hay datos para los filtros generales seleccionados al buscar contratos para la categoría '{selected_category}'.", style={'textAlign': 'center', 'marginTop': '20px'})
    if 'Oportunidad' not in filtered_df_global.columns:
        return html.P("La columna 'Oportunidad' no se encuentra en los datos filtrados.", style={'color': 'red', 'textAlign': 'center', 'marginTop': '20px'})

    category_contracts_df = filtered_df_global[filtered_df_global['Oportunidad'] == selected_category].copy()
    if category_contracts_df.empty:
        return html.P(f"No se encontraron contratos para la categoría '{selected_category}' con los filtros actuales.", style={'textAlign': 'center', 'marginTop': '20px'})

    col_titulo = 'Título del contrato'
    col_institucion = 'Siglas de la Institución'
    col_proveedor = 'Proveedor o contratista'
    col_importe = 'Importe DRC'
    col_inicio = 'Fecha de inicio del contrato'
    col_fin = 'Fecha de fin del contrato'
    col_anuncio = 'Dirección del anuncio'
    col_oportunidad_cat = 'Oportunidad'  # Renombrar para evitar confusión con la variable 'Oportunidad' de arriba
    md_col_anuncio = 'Enlace_Anuncio_Oportunidad_MD'

    current_cols = []
    cols_for_table_display = []

    if col_titulo in category_contracts_df.columns:
        current_cols.append(col_titulo)
        cols_for_table_display.append({"name": "Título del Contrato", "id": col_titulo})
    if col_institucion in category_contracts_df.columns:
        current_cols.append(col_institucion)
        cols_for_table_display.append({"name": "Institución", "id": col_institucion})
    if col_proveedor in category_contracts_df.columns:
        current_cols.append(col_proveedor)
        cols_for_table_display.append({"name": "Proveedor", "id": col_proveedor})
    if col_importe in category_contracts_df.columns:
        current_cols.append(col_importe)
        category_contracts_df[col_importe] = pd.to_numeric(category_contracts_df[col_importe], errors='coerce')
        cols_for_table_display.append({
            "name": "Importe DRC", "id": col_importe, "type": "numeric",
            "format": dash_table.Format.Format(scheme=dash_table.Format.Scheme.fixed, precision=2, group=True, symbol=dash_table.Format.Symbol.yes, symbol_prefix='$')
        })
    if col_inicio in category_contracts_df.columns:
        current_cols.append(col_inicio)
        category_contracts_df[col_inicio] = pd.to_datetime(category_contracts_df[col_inicio], errors='coerce').dt.strftime('%d/%m/%Y').fillna('N/A')
        cols_for_table_display.append({"name": "Fecha Inicio", "id": col_inicio})
    if col_fin in category_contracts_df.columns:
        current_cols.append(col_fin)
        category_contracts_df[col_fin] = pd.to_datetime(category_contracts_df[col_fin], errors='coerce').dt.strftime('%d/%m/%Y').fillna('N/A')
        cols_for_table_display.append({"name": "Fecha Fin", "id": col_fin})
    if col_oportunidad_cat in category_contracts_df.columns:
        current_cols.append(col_oportunidad_cat)
        cols_for_table_display.append({"name": "Categoría Oportunidad", "id": col_oportunidad_cat})

    warning_message_anuncio = None
    if col_anuncio in category_contracts_df.columns:
        category_contracts_df[md_col_anuncio] = category_contracts_df[col_anuncio].apply(
            lambda x: f"[Ver Anuncio]({x})" if pd.notna(x) and str(x).strip().lower().startswith('http') else "N/A"
        )
        current_cols.append(md_col_anuncio)
        cols_for_table_display.append({"name": "Anuncio", "id": md_col_anuncio, 'presentation': 'markdown'})
    else:
        warning_message_anuncio = html.P(f"Advertencia: La columna '{col_anuncio}' no fue encontrada.", style={'color': 'orange', 'textAlign': 'center', 'fontSize': '0.9em'})

    if not current_cols:
        return html.P(f"No hay columnas con datos para mostrar para la categoría '{selected_category}'.", style={'textAlign': 'center', 'marginTop': '20px', 'color': 'red'})

    if 'Tiempo hasta vencimiento (días)' in category_contracts_df.columns:
        category_contracts_df = category_contracts_df.sort_values(by='Tiempo hasta vencimiento (días)', ascending=True)

    data_for_table = category_contracts_df[current_cols]
    style_cell_conditional = [
        {'if': {'column_id': col_titulo}, 'minWidth': '200px', 'fontWeight': 'bold'},
        {'if': {'column_id': col_importe}, 'textAlign': 'right', 'minWidth': '130px'}
    ]
    if col_anuncio in category_contracts_df.columns:  # Check if original column existed to add style for markdown version
        style_cell_conditional.append({'if': {'column_id': md_col_anuncio}, 'textAlign': 'center', 'minWidth': '100px'})

    table = dash_table.DataTable(
        id='opportunity-category-specific-contracts-table',
        columns=cols_for_table_display,
        data=data_for_table.to_dict('records'),
        style_table={'overflowX': 'auto', 'marginTop': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'width': '100%'},
        style_header={'backgroundColor': '#34495e', 'color': 'white', 'fontWeight': 'bold', 'textAlign': 'center', 'padding': '10px'},
        style_cell={
            'textAlign': 'left', 'padding': '8px', 'minWidth': '100px', 'maxWidth': '350px',
            'whiteSpace': 'normal', 'height': 'auto', 'border': '1px solid #eee',
            'fontFamily': 'Arial, sans-serif', 'fontSize': '13px', 'verticalAlign': 'middle'
        },
        style_cell_conditional=style_cell_conditional,
        style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#f9f9f9'}],
        page_size=10,
        filter_action='native',
        sort_action='native'
    )
    return html.Div([
        html.H4(f"Contratos en la categoría de oportunidad: '{selected_category}'", style={'marginTop': '20px', 'marginBottom': '10px', 'textAlign': 'center', 'color': '#2c3e50'}),
        warning_message_anuncio if warning_message_anuncio else "",
        table
    ])


@app.callback(
    [Output('provider-share-graph', 'figure'),
     Output('provider-specialization-graph', 'figure'),
     Output('contract-duration-graph', 'figure'),
     Output('competitor-table-container', 'children')],
    [Input('tic-dropdown', 'value'),
     Input('siglas-dropdown', 'value'),
     Input('gobierno-dropdown', 'value'),
     Input('estado-dropdown', 'value')]
)
def update_tab3(tic_filter, selected_siglas, selected_gobierno, selected_estado):
    filtered_df = filter_dataframe(df, tic_filter, selected_siglas, selected_gobierno, selected_estado)
    empty_fig_layout = {"title": "No hay datos para los filtros seleccionados", "plot_bgcolor": "rgba(0,0,0,0)", "paper_bgcolor": "rgba(0,0,0,0)"}

    if filtered_df.empty:
        empty_fig = go.Figure().update_layout(**empty_fig_layout)
        empty_table = html.Div("No hay datos para mostrar en la tabla.", style={'textAlign': 'center', 'padding': '20px'})
        return empty_fig, empty_fig, empty_fig, empty_table

    fig_share = go.Figure().update_layout(title="No hay datos de proveedores")
    provider_col_name = 'Proveedor o contratista'
    if provider_col_name in filtered_df.columns:
        if 'Importe DRC' in filtered_df.columns and not filtered_df[provider_col_name].dropna().empty and not filtered_df['Importe DRC'].dropna().empty:
            provider_share = filtered_df.groupby(provider_col_name)['Importe DRC'].sum().reset_index().sort_values('Importe DRC', ascending=False).head(10)
            if not provider_share.empty:
                fig_share = px.pie(provider_share, values='Importe DRC', names=provider_col_name, title="Market Share de Proveedores TIC (por Importe)", hole=0.4)
        elif 'Código del contrato' in filtered_df.columns and not filtered_df[provider_col_name].dropna().empty:
            provider_share = filtered_df.groupby(provider_col_name)['Código del contrato'].count().reset_index(name='count').sort_values('count', ascending=False).head(10)
            if not provider_share.empty:
                fig_share = px.pie(provider_share, values='count', names=provider_col_name, title="Market Share de Proveedores TIC (por Número de Contratos)", hole=0.4)

    fig_spec = go.Figure().update_layout(title="No hay datos para especialización de proveedores")
    if provider_col_name in filtered_df.columns and 'terminos_positivos' in filtered_df.columns:
        df_spec_prep = filtered_df[filtered_df['terminos_positivos'].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()
        if not df_spec_prep.empty:
            # df_spec_prep['primer_termino'] = df_spec_prep['terminos_positivos'].apply(lambda x: x[0] if x else None) # No usado directamente
            if 'Importe DRC' in df_spec_prep.columns:
                top_providers_by_value = df_spec_prep.groupby(provider_col_name)['Importe DRC'].sum().nlargest(15).index
                spec_df_focus = df_spec_prep[df_spec_prep[provider_col_name].isin(top_providers_by_value)]
                provider_specialization_list = []
                for provider in top_providers_by_value:
                    provider_contracts = spec_df_focus[spec_df_focus[provider_col_name] == provider]
                    if not provider_contracts.empty:
                        all_terms_for_provider = [term for sublist in provider_contracts['terminos_positivos'] for term in sublist]
                        if all_terms_for_provider:
                            term_counts = Counter(all_terms_for_provider)
                            most_common = term_counts.most_common(1)[0]
                            provider_specialization_list.append({'Proveedor': provider, 'Término Especialización': most_common[0], 'Frecuencia': most_common[1]})
                if provider_specialization_list:
                    spec_df_viz = pd.DataFrame(provider_specialization_list)
                    fig_spec = px.bar(
                        spec_df_viz.sort_values('Frecuencia', ascending=False),
                        x='Proveedor', y='Frecuencia', color='Término Especialización',
                        title="Especialización de Proveedores TIC (Término Más Común para Top Proveedores)",
                        labels={'Frecuencia': 'Frecuencia del Término Principal', 'Proveedor': ''}
                    )
                    fig_spec.update_layout(xaxis={'categoryorder': 'total descending'})

    fig_duration = go.Figure().update_layout(title="No hay datos para duración de contratos")
    if provider_col_name in filtered_df.columns and 'Duración del contrato (días)' in filtered_df.columns:
        duration_df_prep = filtered_df.dropna(subset=[provider_col_name, 'Duración del contrato (días)'])
        if not duration_df_prep.empty:
            duration_df = duration_df_prep.groupby(provider_col_name)['Duración del contrato (días)'].agg(['mean', 'count']).reset_index()
            duration_df.columns = [provider_col_name, 'Duración Promedio', 'Número de Contratos']
            duration_df_filtered = duration_df[duration_df['Número de Contratos'] >= 2].sort_values('Duración Promedio', ascending=False).head(15)
            if not duration_df_filtered.empty:
                fig_duration = px.bar(
                    duration_df_filtered, x=provider_col_name, y='Duración Promedio', color='Número de Contratos',
                    title="Duración Promedio de Contratos por Proveedor (en días, Top 15)",
                    labels={'Duración Promedio': 'Días', provider_col_name: ''},
                    color_continuous_scale='Viridis'
                )
                fig_duration.update_layout(xaxis={'categoryorder': 'total descending'})

    competitor_table_content = html.Div("No hay datos para análisis de competidores.", style={'textAlign': 'center', 'padding': '20px'})
    if provider_col_name in filtered_df.columns and 'Código del contrato' in filtered_df.columns:
        agg_dict = {
            'total_contratos': ('Código del contrato', 'count'),
            'instituciones_unicas': ('Siglas de la Institución', lambda x: x.nunique() if 'Siglas de la Institución' in filtered_df.columns else 0)
        }
        if 'Importe DRC' in filtered_df.columns:
            agg_dict['valor_total'] = ('Importe DRC', 'sum')
            agg_dict['valor_promedio'] = ('Importe DRC', 'mean')
        if 'Duración del contrato (días)' in filtered_df.columns:
            agg_dict['duracion_promedio'] = ('Duración del contrato (días)', 'mean')

        competitor_analysis = filtered_df.groupby(provider_col_name).agg(**agg_dict).reset_index()
        sort_col = 'valor_total' if 'valor_total' in competitor_analysis.columns else 'total_contratos'
        competitor_analysis = competitor_analysis.sort_values(sort_col, ascending=False).head(15).copy()

        if 'valor_total' in competitor_analysis.columns:
            competitor_analysis['valor_total'] = competitor_analysis['valor_total'].apply(lambda x: f"${x:,.2f} MXN" if pd.notna(x) else "N/A")
        if 'valor_promedio' in competitor_analysis.columns:
            competitor_analysis['valor_promedio'] = competitor_analysis['valor_promedio'].apply(lambda x: f"${x:,.2f} MXN" if pd.notna(x) else "N/A")
        if 'duracion_promedio' in competitor_analysis.columns:
            competitor_analysis['duracion_promedio'] = competitor_analysis['duracion_promedio'].apply(lambda x: f"{x:.0f} días" if pd.notna(x) else "N/A")

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


@app.callback(
    Output('provider-contracts-table-container', 'children'),
    [Input('provider-share-graph', 'clickData'),
     Input('tic-dropdown', 'value'),
     Input('siglas-dropdown', 'value'),
     Input('gobierno-dropdown', 'value'),
     Input('estado-dropdown', 'value')]
)
def display_provider_contracts_table(click_data, tic_filter, selected_siglas, selected_gobierno, selected_estado):
    ctx = dash.callback_context
    if not click_data or not click_data['points']:
        return html.P("Haga clic en un proveedor en el gráfico de Market Share para ver sus contratos.", style={'textAlign': 'center', 'marginTop': '20px'})

    try:
        provider_name = click_data['points'][0]['label']
    except (KeyError, IndexError, TypeError):
        return html.P("No se pudo obtener el proveedor seleccionado del gráfico.", style={'textAlign': 'center', 'marginTop': '20px', 'color': 'red'})

    filtered_df_for_provider = filter_dataframe(df, tic_filter, selected_siglas, selected_gobierno, selected_estado)
    if filtered_df_for_provider.empty:
        return html.P(f"No hay datos para los filtros generales seleccionados al buscar contratos de '{provider_name}'.", style={'textAlign': 'center', 'marginTop': '20px'})

    provider_col_name = 'Proveedor o contratista'
    if provider_col_name not in filtered_df_for_provider.columns:
        return html.P(f"La columna '{provider_col_name}' no se encuentra en los datos filtrados.", style={'color': 'red', 'textAlign': 'center', 'marginTop': '20px'})

    provider_contracts_df = filtered_df_for_provider[filtered_df_for_provider[provider_col_name] == provider_name].copy()
    if provider_contracts_df.empty:
        return html.P(f"No se encontraron contratos para '{provider_name}' con los filtros actuales.", style={'textAlign': 'center', 'marginTop': '20px'})

    cols_for_table = []
    col_titulo_script = 'Título del contrato'
    col_importe_script = 'Importe DRC'
    col_fecha_inicio_script = 'Fecha de inicio del contrato'
    col_fecha_fin_script = 'Fecha de fin del contrato'
    col_anuncio_script = 'Dirección del anuncio'
    md_col_anuncio_script = 'Enlace_Anuncio_MD'

    current_cols_selection = []
    if col_titulo_script in provider_contracts_df.columns:
        current_cols_selection.append(col_titulo_script)
        cols_for_table.append({"name": "Título del Contrato", "id": col_titulo_script})
    if col_importe_script in provider_contracts_df.columns:
        current_cols_selection.append(col_importe_script)
        provider_contracts_df.loc[:, col_importe_script] = pd.to_numeric(provider_contracts_df[col_importe_script], errors='coerce')
        cols_for_table.append({
            "name": "Importe DRC", "id": col_importe_script, "type": "numeric",
            "format": dash_table.Format.Format(scheme=dash_table.Format.Scheme.fixed, precision=2, group=True, symbol=dash_table.Format.Symbol.yes, symbol_prefix='$')
        })
    if col_fecha_inicio_script in provider_contracts_df.columns:
        current_cols_selection.append(col_fecha_inicio_script)
        provider_contracts_df[col_fecha_inicio_script] = pd.to_datetime(provider_contracts_df[col_fecha_inicio_script], errors='coerce').dt.strftime('%d/%m/%Y').fillna('N/A')
        cols_for_table.append({"name": "Fecha Inicio", "id": col_fecha_inicio_script})
    if col_fecha_fin_script in provider_contracts_df.columns:
        current_cols_selection.append(col_fecha_fin_script)
        provider_contracts_df[col_fecha_fin_script] = pd.to_datetime(provider_contracts_df[col_fecha_fin_script], errors='coerce').dt.strftime('%d/%m/%Y').fillna('N/A')
        cols_for_table.append({"name": "Fecha Fin", "id": col_fecha_fin_script})

    warning_message_anuncio = None
    if col_anuncio_script in provider_contracts_df.columns:
        provider_contracts_df[md_col_anuncio_script] = provider_contracts_df[col_anuncio_script].apply(
            lambda x: f"[Ver Anuncio]({x})" if pd.notna(x) and str(x).strip().lower().startswith('http') else "N/A"
        )
        current_cols_selection.append(md_col_anuncio_script)
        cols_for_table.append({"name": "Anuncio", "id": md_col_anuncio_script, 'presentation': 'markdown'})
    else:
        warning_message_anuncio = html.P(f"Advertencia: La columna '{col_anuncio_script}' no fue encontrada.", style={'color': 'orange', 'textAlign': 'center'})

    if not cols_for_table:
        return html.P(f"No hay columnas válidas para mostrar para el proveedor '{provider_name}'.", style={'textAlign': 'center', 'marginTop': '20px', 'color': 'red'})

    cols_to_display_data = provider_contracts_df[current_cols_selection]
    table = dash_table.DataTable(
        id='provider-specific-contracts-table',
        columns=cols_for_table,
        data=cols_to_display_data.to_dict('records'),
        style_table={'overflowX': 'auto', 'marginTop': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'width': '100%'},
        style_header={'backgroundColor': '#34495e', 'color': 'white', 'fontWeight': 'bold', 'textAlign': 'center', 'padding': '10px'},
        style_cell={
            'textAlign': 'left', 'padding': '8px', 'minWidth': '120px', 'maxWidth': '400px',
            'whiteSpace': 'normal', 'height': 'auto', 'border': '1px solid #eee',
            'fontFamily': 'Arial, sans-serif', 'fontSize': '13px', 'verticalAlign': 'middle'
        },
        style_cell_conditional=[
            {'if': {'column_id': col_titulo_script}, 'minWidth': '250px', 'fontWeight': 'bold'},
            {'if': {'column_id': col_importe_script}, 'textAlign': 'right', 'minWidth': '150px'},
            {'if': {'column_id': md_col_anuncio_script}, 'textAlign': 'center', 'minWidth': '100px'}
        ],
        style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#f9f9f9'}],
        page_size=5,
        filter_action='native',
        sort_action='native'
    )
    return html.Div([
        html.H4(f"Contratos para: {provider_name}", style={'marginTop': '20px', 'marginBottom': '5px', 'textAlign': 'center'}),
        warning_message_anuncio if warning_message_anuncio else "",
        table
    ])


@app.callback(
    [Output('geo-distribution-graph', 'figure'),
     Output('government-level-graph', 'figure'),
     Output('procedure-type-graph', 'figure')],
    [Input('tic-dropdown', 'value'),
     Input('siglas-dropdown', 'value'),
     Input('gobierno-dropdown', 'value'),
     Input('estado-dropdown', 'value')]
)
def update_tab4(tic_filter, selected_siglas, selected_gobierno, selected_estado):
    filtered_df = filter_dataframe(df, tic_filter, selected_siglas, selected_gobierno, selected_estado)

    empty_fig_layout = {
        "xaxis": dict(showgrid=False, zeroline=False, showticklabels=False),
        "yaxis": dict(showgrid=False, zeroline=False, showticklabels=False),
        "plot_bgcolor": "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)"
    }

    if filtered_df.empty:
        empty_fig = go.Figure().update_layout(title="No hay datos para los filtros seleccionados", **empty_fig_layout)
        return empty_fig, empty_fig, empty_fig

    # --- fig_geo (Distribución Geográfica de Contratos) ---
    fig_geo = go.Figure()
    if selected_estado:
        geo_group_col = 'Estado'
        current_selection_for_title = ", ".join(selected_estado)
        dynamic_title_no_data = f"No hay datos de contratos para: {current_selection_for_title}"
    else:
        geo_group_col = 'Orden de gobierno'
        dynamic_title_no_data = "No hay datos para mostrar por Orden de gobierno"

    if geo_group_col in filtered_df.columns and \
       'Código del contrato' in filtered_df.columns and \
       not filtered_df[geo_group_col].dropna().empty:
        geo_df_grouped = filtered_df.groupby(geo_group_col, observed=True)['Código del contrato'].count().reset_index(name='count')
        if not geo_df_grouped.empty and geo_df_grouped['count'].sum() > 0:
            fig_geo = px.bar(geo_df_grouped, x=geo_group_col, y='count',
                             title=f"Distribución de Contratos por {geo_group_col}",
                             labels={'count': 'Número de Contratos', geo_group_col: ''},
                             color=geo_group_col)
            if geo_group_col == 'Orden de gobierno':
                fig_geo.add_annotation(x=0.5, y=-0.2, xref="paper", yref="paper", align="center",
                                       text="Nota: Para mapa geográfico real se requieren datos de latitud/longitud o códigos GeoJSON.",
                                       showarrow=False, font=dict(size=10, color="gray"))
            fig_geo.update_layout(xaxis={'categoryorder': 'total descending'})
        else:
            fig_geo.update_layout(title=dynamic_title_no_data, **empty_fig_layout)
    else:
        final_title_error = dynamic_title_no_data
        if not (geo_group_col in filtered_df.columns):
            final_title_error = f"Columna '{geo_group_col}' no encontrada en los datos filtrados."
        elif not ('Código del contrato' in filtered_df.columns):
            final_title_error = "Columna 'Código del contrato' no encontrada."
        fig_geo.update_layout(title=final_title_error, **empty_fig_layout)

    # --- fig_gov (Gasto TIC por Orden de Gobierno) ---
    fig_gov = go.Figure().update_layout(title="No hay datos para gasto por orden de gobierno", **empty_fig_layout)
    if 'Orden de gobierno' in filtered_df.columns and 'Importe DRC' in filtered_df.columns:
        gov_spending_prep = filtered_df.dropna(subset=['Importe DRC'])
        if not gov_spending_prep.empty:
            gov_spending = gov_spending_prep.groupby('Orden de gobierno')['Importe DRC'].sum().reset_index().sort_values('Importe DRC', ascending=False)
            if not gov_spending.empty and gov_spending['Importe DRC'].sum() > 0:
                fig_gov = px.pie(gov_spending, values='Importe DRC', names='Orden de gobierno', title="Distribución del Gasto TIC por Orden de Gobierno", hole=0.4)
            else:
                fig_gov.update_layout(title="No hay datos de importe para Orden de Gobierno", **empty_fig_layout)
        else:
            fig_gov.update_layout(title="No hay datos de importe para Orden de Gobierno", **empty_fig_layout)
    else:
        fig_gov.update_layout(title="Columnas requeridas no encontradas para Gasto por Orden de Gobierno", **empty_fig_layout)

    # --- fig_proc (Tipo de Procedimiento por Región/Grupo) ---
    fig_proc = go.Figure().update_layout(title="No hay datos para tipo de procedimiento", **empty_fig_layout)
    proc_group_col = 'Estado' if 'Estado' in filtered_df.columns and not filtered_df['Estado'].dropna().empty and selected_estado else 'Orden de gobierno'
    if 'Tipo Procedimiento' in filtered_df.columns and proc_group_col in filtered_df.columns and 'Código del contrato' in filtered_df.columns:
        proc_df_prep = filtered_df.dropna(subset=['Tipo Procedimiento', proc_group_col])
        if not proc_df_prep.empty:
            proc_df = proc_df_prep.groupby([proc_group_col, 'Tipo Procedimiento'], observed=True)['Código del contrato'].count().reset_index(name='count')
            if not proc_df.empty and proc_df['count'].sum() > 0:
                fig_proc = px.bar(proc_df, x=proc_group_col, y='count', color='Tipo Procedimiento',
                                  title=f"Tipos de Procedimiento por {proc_group_col}",
                                  labels={'count': 'Número de Contratos', proc_group_col: ''}, barmode='group')
                fig_proc.update_layout(xaxis={'categoryorder': 'total descending'})
            else:
                fig_proc.update_layout(title=f"No hay datos de Tipo de Procedimiento por {proc_group_col}", **empty_fig_layout)
        else:
            fig_proc.update_layout(title="Datos insuficientes (Tipo Procedimiento, Grupo) para el gráfico.", **empty_fig_layout)
    else:
        fig_proc.update_layout(title="Columnas requeridas para 'Tipo de Procedimiento por Grupo' no encontradas.", **empty_fig_layout)

    return fig_geo, fig_gov, fig_proc


@app.callback(
    [Output('top-terms-graph', 'figure'),
     Output('terms-trend-graph', 'figure'),
     Output('top-marcas-graph', 'figure'),
     Output('term-value-graph', 'figure')],
    [Input('tic-dropdown', 'value'),
     Input('siglas-dropdown', 'value'),
     Input('gobierno-dropdown', 'value'),
     Input('estado-dropdown', 'value')]
)
def update_tab5(tic_filter, selected_siglas, selected_gobierno, selected_estado):
    filtered_df = filter_dataframe(df, tic_filter, selected_siglas, selected_gobierno, selected_estado)

    empty_fig_layout_base = dict(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )

    fig_terms = go.Figure().update_layout(title="Cargando datos de términos...", **empty_fig_layout_base)
    fig_trend = go.Figure().update_layout(title="Cargando tendencias de términos...", **empty_fig_layout_base)
    fig_top_marcas = go.Figure().update_layout(title="Cargando datos de marcas...", **empty_fig_layout_base)
    fig_value_term = go.Figure().update_layout(title="Cargando valor por término...", **empty_fig_layout_base)

    if filtered_df.empty:
        no_data_title = "No hay datos para los filtros seleccionados"
        fig_terms.update_layout(title=no_data_title)
        fig_trend.update_layout(title=no_data_title)
        fig_top_marcas.update_layout(title=no_data_title)
        fig_value_term.update_layout(title=no_data_title)
        return fig_terms, fig_trend, fig_top_marcas, fig_value_term

    all_terms_flat = []
    if 'terminos_positivos' in filtered_df.columns:
        all_terms_flat = [
            term.strip()
            for sublist in filtered_df['terminos_positivos']
            if isinstance(sublist, list)
            for term in sublist
            if isinstance(term, str) and term.strip() != ""
        ]

        if all_terms_flat:
            term_counts_obj = Counter(all_terms_flat)
            term_counts = term_counts_obj.most_common(15)

            if term_counts:
                terms_df_viz = pd.DataFrame(term_counts, columns=['Término', 'Frecuencia'])
                fig_terms = px.bar(terms_df_viz, x='Término', y='Frecuencia',
                                   title="Top 15 Términos TIC Encontrados",
                                   labels={'Frecuencia': 'Frecuencia', 'Término': ''},
                                   color='Frecuencia', color_continuous_scale='Viridis')
                fig_terms.update_layout(xaxis={'categoryorder': 'total descending'})
            else:
                fig_terms.update_layout(title="No hay términos suficientes para el gráfico Top")

            if 'Fecha de inicio del contrato' in filtered_df.columns:
                trend_base = filtered_df[['Fecha de inicio del contrato', 'terminos_positivos']].copy()
                trend_base.dropna(subset=['Fecha de inicio del contrato'], inplace=True)
                trend_base = trend_base[trend_base['terminos_positivos'].apply(lambda x: isinstance(x, list) and len(x) > 0)]

                if not trend_base.empty:
                    trend_base['Year'] = trend_base['Fecha de inicio del contrato'].dt.year
                    trend_exploded = trend_base.explode('terminos_positivos')
                    trend_exploded = trend_exploded[trend_exploded['terminos_positivos'].apply(lambda x: isinstance(x, str) and x.strip() != "")]

                    if not trend_exploded.empty:
                        top_5_terms_for_trend = [tc[0] for tc in term_counts[:5]]
                        trend_filtered_for_plot = trend_exploded[trend_exploded['terminos_positivos'].isin(top_5_terms_for_trend)]
                        if not trend_filtered_for_plot.empty:
                            term_yearly_counts = trend_filtered_for_plot.groupby(['Year', 'terminos_positivos'], observed=True).size().reset_index(name='Frecuencia').sort_values('Year')
                            if not term_yearly_counts.empty:
                                fig_trend = px.line(term_yearly_counts, x='Year', y='Frecuencia', color='terminos_positivos',
                                                    title="Tendencias de los 5 Términos TIC Más Comunes por Año",
                                                    labels={'Frecuencia': 'Frecuencia Anual', 'Year': 'Año', 'terminos_positivos': 'Término'},
                                                    markers=True)
                            else:
                                fig_trend.update_layout(title="No hay datos agrupados para tendencias de términos")
                        else:
                            fig_trend.update_layout(title="No hay datos de los top 5 términos para tendencias")
                    else:
                        fig_trend.update_layout(title="No hay términos válidos después de procesar para tendencias")
                else:
                    fig_trend.update_layout(title="No hay datos base para tendencias (fechas/términos válidos)")
            else:
                fig_trend.update_layout(title="Falta columna de fecha para tendencias de términos")
        else:
            fig_terms.update_layout(title="No se encontraron términos TIC en los datos filtrados")
            fig_trend.update_layout(title="No hay tendencias de términos para mostrar (sin términos)")
    else:
        no_terms_col_msg = "Columna 'terminos_positivos' no encontrada"
        fig_terms.update_layout(title=no_terms_col_msg)
        fig_trend.update_layout(title=no_terms_col_msg)

    if 'marca_tic' in filtered_df.columns:
        marcas_validas = filtered_df['marca_tic'].dropna().astype(str)
        marcas_validas = marcas_validas[marcas_validas.str.strip() != ""]
        if not marcas_validas.empty:
            top_n_marcas = 15
            marcas_counts_series = marcas_validas.value_counts().nlargest(top_n_marcas)
            if not marcas_counts_series.empty:
                marcas_counts_df = marcas_counts_series.reset_index()
                marcas_counts_df.columns = ['Marca', 'Frecuencia']
                fig_top_marcas = px.bar(marcas_counts_df, x='Marca', y='Frecuencia',
                                        title=f"Top {top_n_marcas} Marcas TIC por Frecuencia",
                                        labels={'Frecuencia': 'Número de Contratos', 'Marca': 'Marca'},
                                        color='Frecuencia', color_continuous_scale='Plasma')
                fig_top_marcas.update_layout(xaxis={'categoryorder': 'total descending'})
            else:
                fig_top_marcas.update_layout(title="No hay datos de frecuencia de marcas para mostrar")
        else:
            fig_top_marcas.update_layout(title="No hay marcas TIC válidas en los datos filtrados")
    else:
        fig_top_marcas.update_layout(title="Columna 'marca_tic' no encontrada")

    if 'terminos_positivos' in filtered_df.columns and 'Importe DRC' in filtered_df.columns:
        if all_terms_flat:
            value_term_base = filtered_df[['terminos_positivos', 'Importe DRC']].copy()
            value_term_base.dropna(subset=['Importe DRC'], inplace=True)
            value_term_base = value_term_base[value_term_base['terminos_positivos'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
            if not value_term_base.empty:
                value_term_exploded = value_term_base.explode('terminos_positivos')
                value_term_exploded = value_term_exploded[value_term_exploded['terminos_positivos'].apply(lambda x: isinstance(x, str) and x.strip() != "")]
                if not value_term_exploded.empty:
                    term_value_agg = value_term_exploded.groupby('terminos_positivos', observed=True)['Importe DRC'].agg(['mean', 'count']).reset_index()
                    term_value_agg.rename(columns={'mean': 'Valor Promedio', 'count': 'Num Contratos', 'terminos_positivos': 'Término'}, inplace=True)
                    term_value_filtered_plot = term_value_agg[term_value_agg['Num Contratos'] >= 1].sort_values('Valor Promedio', ascending=False).head(15)
                    if not term_value_filtered_plot.empty:
                        fig_value_term = px.bar(term_value_filtered_plot, x='Término', y='Valor Promedio',
                                                color='Num Contratos', title="Valor Promedio de Contratos por Término TIC (Top 15)",
                                                labels={'Valor Promedio': 'Valor Promedio (MXN)', 'Término': '', 'Num Contratos': 'Núm. Contratos'},
                                                color_continuous_scale='Blues', hover_data={'Valor Promedio': ':,.2f'})
                        fig_value_term.update_layout(xaxis={'categoryorder': 'total descending'})
                    else:
                        fig_value_term.update_layout(title="No hay suficientes datos de valor por término para el gráfico")
                else:
                    fig_value_term.update_layout(title="No hay términos válidos para explotar en análisis de valor")
            else:
                fig_value_term.update_layout(title="No hay importes o términos válidos para análisis de valor")
        else:
            fig_value_term.update_layout(title="No se encontraron términos TIC para análisis de valor")
    else:
        msg_val = "Columnas requeridas faltantes"
        if 'terminos_positivos' not in filtered_df.columns:
            msg_val = "Columna 'terminos_positivos' no encontrada"
        elif 'Importe DRC' not in filtered_df.columns:
            msg_val = "Columna 'Importe DRC' no encontrada"
        fig_value_term.update_layout(title=f"Valor por Término: {msg_val}")

    for fig in [fig_terms, fig_trend, fig_top_marcas, fig_value_term]:
        if not fig.data:
            if "Cargando" in (fig.layout.title.text if fig.layout.title else ""): # type: ignore
                fig.update_layout(title="No hay datos disponibles para este gráfico", **empty_fig_layout_base)
            else: # Title was already set to a specific error/info message
                fig.update_layout(**empty_fig_layout_base) # Ensure empty layout is applied

    return fig_terms, fig_trend, fig_top_marcas, fig_value_term


@app.callback(
    Output('network-graph', 'figure'),
    [Input('tic-dropdown', 'value'),
     Input('siglas-dropdown', 'value'),
     Input('gobierno-dropdown', 'value'),
     Input('estado-dropdown', 'value')]
)
def update_tab_network(tic_filter, selected_siglas, selected_gobierno, selected_estado):
    filtered_df = filter_dataframe(df, tic_filter, selected_siglas, selected_gobierno, selected_estado)
    empty_network_fig_layout = {
        "title": "No hay datos suficientes para crear la red", "showlegend": False,
        "xaxis": {"showgrid": False, "zeroline": False, "showticklabels": False},
        "yaxis": {"showgrid": False, "zeroline": False, "showticklabels": False},
        "plot_bgcolor": "rgba(0,0,0,0)", "paper_bgcolor": "rgba(0,0,0,0)"
    }

    if filtered_df.empty or \
       'Proveedor o contratista' not in filtered_df.columns or \
       'Siglas de la Institución' not in filtered_df.columns:
        return go.Figure().update_layout(**empty_network_fig_layout)

    link_col = 'Importe DRC' if 'Importe DRC' in filtered_df.columns and filtered_df['Importe DRC'].notna().any() else 'Código del contrato'
    agg_func = 'sum' if link_col == 'Importe DRC' else 'count'

    network_data_prep = filtered_df.dropna(subset=['Proveedor o contratista', 'Siglas de la Institución', link_col])
    if network_data_prep.empty:
        empty_network_fig_layout["title"] = "No hay datos válidos (proveedor, institución, valor/conteo) para generar la red"
        return go.Figure().update_layout(**empty_network_fig_layout)

    edges_df = network_data_prep.groupby(['Proveedor o contratista', 'Siglas de la Institución']).agg(weight=(link_col, agg_func)).reset_index()
    top_n = 50
    edges_df = edges_df.sort_values('weight', ascending=False).head(top_n)

    if edges_df.empty:
        empty_network_fig_layout["title"] = "No se encontraron relaciones significativas para mostrar en la red"
        return go.Figure().update_layout(**empty_network_fig_layout)

    proveedores_nodes = pd.unique(edges_df['Proveedor o contratista'])
    instituciones_nodes = pd.unique(edges_df['Siglas de la Institución'])
    all_graph_nodes = pd.unique(np.concatenate([proveedores_nodes, instituciones_nodes]))
    # node_map = {name: i for i, name in enumerate(all_graph_nodes)} # Not directly used

    node_types_map = {p: 'Proveedor' for p in proveedores_nodes}
    for i_node in instituciones_nodes:
        if i_node not in node_types_map:  # Avoid overwriting if an entity is both
            node_types_map[i_node] = 'Institución'

    num_graph_nodes = len(all_graph_nodes)
    if num_graph_nodes == 0:
        return go.Figure().update_layout(**empty_network_fig_layout)

    radius = 5 if num_graph_nodes > 1 else 0
    angles = np.linspace(0, 2 * np.pi, num_graph_nodes, endpoint=False) if num_graph_nodes > 1 else [0]
    pos_x = radius * np.cos(angles)
    pos_y = radius * np.sin(angles)
    node_positions = {name: (pos_x[i], pos_y[i]) for i, name in enumerate(all_graph_nodes)}

    fig_network = go.Figure()
    min_w, max_w = edges_df['weight'].min(), edges_df['weight'].max()

    for _, row in edges_df.iterrows():
        prov_name = row['Proveedor o contratista']
        inst_name = row['Siglas de la Institución']
        if prov_name not in node_positions or inst_name not in node_positions:
            continue
        x0, y0 = node_positions[prov_name]
        x1, y1 = node_positions[inst_name]
        line_width = 1
        if max_w > min_w and max_w > 0: # Avoid division by zero if max_w == min_w
            line_width = 1 + 4 * (row['weight'] - min_w) / (max_w - min_w)
        elif max_w == min_w and max_w > 0: # All weights are same and non-zero
            line_width = 2
        weight_label = f"${row['weight']:,.2f}" if link_col == 'Importe DRC' else f"{int(row['weight'])} contratos"
        fig_network.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode='lines',
            line=dict(width=line_width, color='rgba(180,180,180,0.6)'),
            hoverinfo='text', text=f"Relación: {prov_name} &harr; {inst_name}<br>Valor/Conteo: {weight_label}"
        ))

    node_x_coords = [node_positions[node][0] for node in all_graph_nodes]
    node_y_coords = [node_positions[node][1] for node in all_graph_nodes]
    node_hover_texts = []
    node_display_texts = []
    node_colors_list = []
    node_degrees = Counter()
    for _, row in edges_df.iterrows():
        node_degrees[row['Proveedor o contratista']] += 1
        node_degrees[row['Siglas de la Institución']] += 1
    node_sizes_list = []

    for node_name in all_graph_nodes:
        node_type = node_types_map.get(node_name, "Desconocido")
        node_hover_texts.append(f"{node_type}: {node_name}<br>Conexiones: {node_degrees[node_name]}")
        node_display_texts.append(node_name[:15] + '...' if len(node_name) > 15 else node_name)
        node_colors_list.append('firebrick' if node_type == 'Proveedor' else 'steelblue')
        node_sizes_list.append(8 + node_degrees[node_name] * 2.5)

    fig_network.add_trace(go.Scatter(
        x=node_x_coords, y=node_y_coords, mode='markers+text',
        text=node_display_texts, textposition="bottom center", textfont=dict(size=9),
        hoverinfo='text', hovertext=node_hover_texts,
        marker=dict(showscale=False, color=node_colors_list, size=node_sizes_list, line=dict(width=0.8, color='DarkSlateGrey'))
    ))

    layout_padding = 1 if num_graph_nodes > 1 else 1
    fig_network.update_layout(
        title=None, showlegend=False, hovermode='closest',
        margin=dict(b=10, l=5, r=5, t=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[min(pos_x) - layout_padding, max(pos_x) + layout_padding] if pos_x.size >0 else [-1,1]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[min(pos_y) - layout_padding, max(pos_y) + layout_padding] if pos_y.size >0 else [-1,1]),
        plot_bgcolor='rgba(248, 249, 250, 0.9)'
    )
    return fig_network


@app.callback(
    Output('term-contracts-table-container', 'children'),
    [Input('top-terms-graph', 'clickData'),
     Input('tic-dropdown', 'value'),
     Input('siglas-dropdown', 'value'),
     Input('gobierno-dropdown', 'value'),
     Input('estado-dropdown', 'value')]
)
def display_term_contracts_table(click_data, tic_filter, selected_siglas, selected_gobierno, selected_estado):
    ctx = dash.callback_context
    if not click_data or not click_data['points']:
        return html.P("Haz clic en una barra del gráfico 'Top Términos TIC' para ver los contratos asociados.", style={'textAlign': 'center', 'marginTop': '20px'})

    try:
        selected_term = click_data['points'][0]['x']
    except (KeyError, IndexError, TypeError) as e:
        print(f"Error al extraer el término: {e}, click_data: {click_data}")
        return html.P("No se pudo obtener el término seleccionado del gráfico.", style={'textAlign': 'center', 'marginTop': '20px', 'color': 'red'})

    filtered_df_global = filter_dataframe(df.copy(), tic_filter, selected_siglas, selected_gobierno, selected_estado)
    if filtered_df_global.empty:
        return html.P(f"No hay datos para los filtros generales seleccionados al buscar contratos para el término '{selected_term}'.", style={'textAlign': 'center', 'marginTop': '20px'})
    if 'terminos_positivos' not in filtered_df_global.columns:
        return html.P("La columna 'terminos_positivos' no se encuentra en los datos filtrados.", style={'color': 'red', 'textAlign': 'center', 'marginTop': '20px'})

    term_contracts_df = filtered_df_global[
        filtered_df_global['terminos_positivos'].apply(lambda terms_list: isinstance(terms_list, list) and selected_term in terms_list)
    ].copy()

    if term_contracts_df.empty:
        return html.P(f"No se encontraron contratos para el término '{selected_term}' con los filtros actuales.", style={'textAlign': 'center', 'marginTop': '20px'})

    cols_for_table_display = []
    col_titulo_script = 'Título del contrato'
    col_institucion_script = 'Siglas de la Institución'
    col_importe_script = 'Importe DRC'
    col_fecha_inicio_script = 'Fecha de inicio del contrato'
    col_fecha_fin_script = 'Fecha de fin del contrato'
    col_anuncio_script = 'Dirección del anuncio'
    md_col_anuncio_term_script = 'Enlace_Anuncio_Term_MD'

    current_cols_for_selection = []
    if col_titulo_script in term_contracts_df.columns:
        current_cols_for_selection.append(col_titulo_script)
        cols_for_table_display.append({"name": "Título del Contrato", "id": col_titulo_script})
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
        term_contracts_df[col_fecha_inicio_script] = pd.to_datetime(term_contracts_df[col_fecha_inicio_script], errors='coerce').dt.strftime('%d/%m/%Y').fillna('N/A')
        cols_for_table_display.append({"name": "Fecha Inicio", "id": col_fecha_inicio_script, "type": "text"})
    if col_fecha_fin_script in term_contracts_df.columns:
        current_cols_for_selection.append(col_fecha_fin_script)
        term_contracts_df[col_fecha_fin_script] = pd.to_datetime(term_contracts_df[col_fecha_fin_script], errors='coerce').dt.strftime('%d/%m/%Y').fillna('N/A')
        cols_for_table_display.append({"name": "Fecha Fin", "id": col_fecha_fin_script, "type": "text"})

    warning_message_anuncio_term = None
    if col_anuncio_script in term_contracts_df.columns:
        term_contracts_df[md_col_anuncio_term_script] = term_contracts_df[col_anuncio_script].apply(
            lambda x: f"[Ver Anuncio]({x})" if pd.notna(x) and str(x).strip().lower().startswith('http') else "N/A"
        )
        current_cols_for_selection.append(md_col_anuncio_term_script)
        cols_for_table_display.append({"name": "Anuncio", "id": md_col_anuncio_term_script, 'presentation': 'markdown'})
    else:
        warning_message_anuncio_term = html.P(f"Advertencia: La columna '{col_anuncio_script}' no fue encontrada para esta tabla.", style={'color': 'orange', 'textAlign': 'center', 'fontSize': '0.9em'})

    if not current_cols_for_selection:
        return html.P(f"No hay columnas con datos para mostrar para el término '{selected_term}'.", style={'textAlign': 'center', 'marginTop': '20px', 'color': 'red'})

    data_for_table = term_contracts_df[current_cols_for_selection]
    table = dash_table.DataTable(
        id='term-specific-contracts-table',
        columns=cols_for_table_display,
        data=data_for_table.to_dict('records'),
        style_table={'overflowX': 'auto', 'marginTop': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'width': '100%'},
        style_header={'backgroundColor': '#34495e', 'color': 'white', 'fontWeight': 'bold', 'textAlign': 'center', 'padding': '10px'},
        style_cell={
            'textAlign': 'left', 'padding': '8px', 'minWidth': '120px', 'maxWidth': '400px',
            'whiteSpace': 'normal', 'height': 'auto', 'border': '1px solid #eee',
            'fontFamily': 'Arial, sans-serif', 'fontSize': '13px', 'verticalAlign': 'middle'
        },
        style_cell_conditional=[
            {'if': {'column_id': col_titulo_script}, 'minWidth': '250px', 'fontWeight': 'bold'},
            {'if': {'column_id': col_importe_script}, 'textAlign': 'right', 'minWidth': '150px'},
            {'if': {'column_id': md_col_anuncio_term_script}, 'textAlign': 'center', 'minWidth': '100px'}
        ],
        style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#f9f9f9'}],
        page_size=5,
        filter_action='native',
        sort_action='native'
    )
    return html.Div([
        html.H4(f"Contratos que incluyen el término: '{selected_term}'", style={'marginTop': '20px', 'marginBottom': '5px', 'textAlign': 'center', 'color': '#2c3e50'}),
        warning_message_anuncio_term if warning_message_anuncio_term else "",
        table
    ])


@app.callback(
    Output('marca-contracts-table-container', 'children'),
    [Input('top-marcas-graph', 'clickData'),
     Input('tic-dropdown', 'value'),
     Input('siglas-dropdown', 'value'),
     Input('gobierno-dropdown', 'value'),
     Input('estado-dropdown', 'value')]
)
def display_marca_contracts_table(click_data, tic_filter, selected_siglas, selected_gobierno, selected_estado):
    ctx = dash.callback_context
    if not click_data or not click_data['points']:
        return html.P("Haz clic en una barra del gráfico 'Top Marcas TIC' para ver los contratos asociados.", style={'textAlign': 'center', 'marginTop': '20px'})

    try:
        selected_marca = click_data['points'][0]['x']
    except (KeyError, IndexError, TypeError) as e:
        print(f"Error al extraer la marca: {e}, click_data: {click_data}")
        return html.P("No se pudo obtener la marca seleccionada del gráfico.", style={'textAlign': 'center', 'marginTop': '20px', 'color': 'red'})

    filtered_df_global = filter_dataframe(df.copy(), tic_filter, selected_siglas, selected_gobierno, selected_estado)
    if filtered_df_global.empty:
        return html.P(f"No hay datos para los filtros generales seleccionados al buscar contratos para la marca '{selected_marca}'.", style={'textAlign': 'center', 'marginTop': '20px'})
    if 'marca_tic' not in filtered_df_global.columns:
        return html.P("La columna 'marca_tic' no se encuentra en los datos filtrados.", style={'color': 'red', 'textAlign': 'center', 'marginTop': '20px'})

    marca_contracts_df = filtered_df_global[
        filtered_df_global['marca_tic'].astype(str) == str(selected_marca)
    ].copy()

    if marca_contracts_df.empty:
        return html.P(f"No se encontraron contratos para la marca '{selected_marca}' con los filtros actuales.", style={'textAlign': 'center', 'marginTop': '20px'})

    cols_for_table = []
    col_titulo_script = 'Título del contrato'
    col_institucion_script = 'Siglas de la Institución'
    col_importe_script = 'Importe DRC'
    col_fecha_inicio_script = 'Fecha de inicio del contrato'
    col_fecha_fin_script = 'Fecha de fin del contrato'
    col_anuncio_script = 'Dirección del anuncio'
    md_col_anuncio_marca_script = 'Enlace_Anuncio_Marca_MD'

    current_cols_for_selection = []
    if col_titulo_script in marca_contracts_df.columns:
        current_cols_for_selection.append(col_titulo_script)
        cols_for_table.append({"name": "Título del Contrato", "id": col_titulo_script})
    if col_institucion_script in marca_contracts_df.columns:
        current_cols_for_selection.append(col_institucion_script)
        cols_for_table.append({"name": "Institución", "id": col_institucion_script})
    if col_importe_script in marca_contracts_df.columns:
        current_cols_for_selection.append(col_importe_script)
        marca_contracts_df.loc[:, col_importe_script] = pd.to_numeric(marca_contracts_df[col_importe_script], errors='coerce')
        cols_for_table.append({
            "name": "Importe DRC", "id": col_importe_script, "type": "numeric",
            "format": dash_table.Format.Format(scheme=dash_table.Format.Scheme.fixed, precision=2, group=True, symbol=dash_table.Format.Symbol.yes, symbol_prefix='$')
        })
    if col_fecha_inicio_script in marca_contracts_df.columns:
        current_cols_for_selection.append(col_fecha_inicio_script)
        marca_contracts_df[col_fecha_inicio_script] = pd.to_datetime(marca_contracts_df[col_fecha_inicio_script], errors='coerce').dt.strftime('%d/%m/%Y').fillna('N/A')
        cols_for_table.append({"name": "Fecha Inicio", "id": col_fecha_inicio_script, "type": "text"})
    if col_fecha_fin_script in marca_contracts_df.columns:
        current_cols_for_selection.append(col_fecha_fin_script)
        marca_contracts_df[col_fecha_fin_script] = pd.to_datetime(marca_contracts_df[col_fecha_fin_script], errors='coerce').dt.strftime('%d/%m/%Y').fillna('N/A')
        cols_for_table.append({"name": "Fecha Fin", "id": col_fecha_fin_script, "type": "text"})

    warning_message_anuncio_marca = None
    if col_anuncio_script in marca_contracts_df.columns:
        marca_contracts_df[md_col_anuncio_marca_script] = marca_contracts_df[col_anuncio_script].apply(
            lambda x: f"[Ver Anuncio]({x})" if pd.notna(x) and str(x).strip().lower().startswith('http') else "N/A"
        )
        current_cols_for_selection.append(md_col_anuncio_marca_script)
        cols_for_table.append({"name": "Anuncio", "id": md_col_anuncio_marca_script, 'presentation': 'markdown'})
    else:
        warning_message_anuncio_marca = html.P(f"Advertencia: La columna '{col_anuncio_script}' no fue encontrada para esta tabla.", style={'color': 'orange', 'textAlign': 'center', 'fontSize': '0.9em'})

    if not current_cols_for_selection:
        return html.P(f"No hay columnas con datos para mostrar para la marca '{selected_marca}'.", style={'textAlign': 'center', 'marginTop': '20px', 'color': 'red'})

    data_for_table = marca_contracts_df[current_cols_for_selection]
    table = dash_table.DataTable(
        id='marca-specific-contracts-table',
        columns=cols_for_table,
        data=data_for_table.to_dict('records'),
        style_table={'overflowX': 'auto', 'marginTop': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'width': '100%'},
        style_header={'backgroundColor': '#34495e', 'color': 'white', 'fontWeight': 'bold', 'textAlign': 'center', 'padding': '10px'},
        style_cell={
            'textAlign': 'left', 'padding': '8px',
            'minWidth': '120px', 'maxWidth': '400px',
            'whiteSpace': 'normal', 'height': 'auto',
            'border': '1px solid #eee',
            'fontFamily': 'Arial, sans-serif', 'fontSize': '13px',
            'verticalAlign': 'middle'
        },
        style_cell_conditional=[
            {'if': {'column_id': col_titulo_script}, 'minWidth': '250px', 'fontWeight': 'bold'},
            {'if': {'column_id': col_importe_script}, 'textAlign': 'right', 'minWidth': '150px'},
            {'if': {'column_id': md_col_anuncio_marca_script}, 'textAlign': 'center', 'minWidth': '100px'}
        ],
        style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#f9f9f9'}],
        page_size=5,
        filter_action='native',
        sort_action='native',
    )
    return html.Div([
        html.H4(f"Contratos asociados a la marca: '{selected_marca}'", style={'marginTop': '20px', 'marginBottom': '5px', 'textAlign': 'center', 'color': '#2c3e50'}),
        warning_message_anuncio_marca if warning_message_anuncio_marca else "",
        table
    ])

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
                flex: 1; 
                margin: 0 10px; 
                border-radius: 8px; 
                box-shadow: 0 4px 8px rgba(0,0,0,0.05); 
                background: white;
                padding: 20px; 
                text-align: center; 
            }
            .metric-box:first-child { margin-left: 0; }
            .metric-box:last-child { margin-right: 0; }
            .metric-box h4 {
                margin-top: 0;
                margin-bottom: 8px;
                font-size: 1em; 
                color: #555;
            }
            .metric-box p {
                font-size: 1.5em; 
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 0;
            }
            h1, h3 {
                color: #2c3e50;
            }
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
                position: sticky; 
                top: 0;
                z-index: 1;
            }
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
                border-bottom: 1px solid white; 
                background-color: white;  
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
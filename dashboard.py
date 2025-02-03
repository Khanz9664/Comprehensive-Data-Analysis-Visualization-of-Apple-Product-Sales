from dash import Dash, dcc, html, Input, Output, callback
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc

# Load the data
data = pd.read_csv("apple_sales_2024.csv")

# Initialize the Dash app with Bootstrap theme
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Apple Sales Dashboard 2024"

#  Colors
CUSTOM_COLORS = {
    'background': '#0a0a0a',
    'card-bg': '#1a1a1a',
    'text': '#ffffff',
    'accent': '#00FF88',
    'header-gradient': 'linear-gradient(145deg, #0a0a0a 0%, #007BFF 100%)'
}

# Header with gradient
header = dbc.Row([
    dbc.Col([
        html.H1("Apple Product Analytics 2024", className="display-4 mb-4", style={'color': CUSTOM_COLORS['accent']}),
        html.P("Sales & Revenue Dashboard", className="lead", style={'color': CUSTOM_COLORS['text']}),
    ], className="text-center py-5", style={'background': CUSTOM_COLORS['header-gradient']})
])

# Key Metrics Cards
def create_metric_card(metric_id, title):
    return dbc.Card([
        dbc.CardBody([
            html.H5(title, className="card-title", style={'color': CUSTOM_COLORS['accent']}),
            html.H2("0", id=metric_id, className="card-text", style={'color': CUSTOM_COLORS['text']})
        ])
    ], className="shadow-lg", style={'backgroundColor': CUSTOM_COLORS['card-bg']})

metrics_row = dbc.Row([
    dbc.Col(create_metric_card("total-sales", "Total Sales (Units)"), md=3),
    dbc.Col(create_metric_card("total-revenue", "Total Revenue"), md=3),
    dbc.Col(create_metric_card("avg-revenue", "Avg. Revenue/State"), md=3),
    dbc.Col(create_metric_card("top-product", "Top Product"), md=3),
], className="mb-4 g-4")

# Controls with Checkboxes
controls = dbc.Card([
    dbc.CardBody([
        html.H4("Filters", className="card-title mb-4", style={'color': CUSTOM_COLORS['accent']}),
        dbc.Label("Select Region:", className="mb-2", style={'color': CUSTOM_COLORS['text']}),
        dcc.Dropdown(
            id='region-dropdown',
            options=[{'label': r, 'value': r} for r in data['Region'].unique()],
            value=data['Region'].unique()[0],
            clearable=False,
            className="mb-4",
            style={'backgroundColor': CUSTOM_COLORS['card-bg'], 'color': 'black'}
        ),
        dbc.Label("Select States:", className="mb-2", style={'color': CUSTOM_COLORS['text']}),
        dcc.Checklist(
            id='state-checklist',
            options=[],
            inline=False,
            className="mb-4",
            labelStyle={'display': 'block', 'color': CUSTOM_COLORS['text']},
            inputStyle={'marginRight': '10px', 'accentColor': CUSTOM_COLORS['accent']}
        ),
        html.Div(id='state-selection-info', className="small", style={'color': CUSTOM_COLORS['text']})
    ])
], className="shadow-lg", style={'backgroundColor': CUSTOM_COLORS['card-bg']})

app.layout = dbc.Container([
    header,
    metrics_row,
    dbc.Row([
        dbc.Col(controls, md=3, className="mb-4"),
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(
                    dcc.Graph(id='sales-graph', className="border-0"),
                    label="Sales Analysis",
                    tabClassName="flex-grow-1 text-center",
                    label_style={'color': CUSTOM_COLORS['accent']}
                ),
                dbc.Tab(
                    dcc.Graph(id='revenue-graph', className="border-0"),
                    label="Revenue Analysis",
                    tabClassName="flex-grow-1 text-center",
                    label_style={'color': CUSTOM_COLORS['accent']}
                )
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='product-mix', className="border-0"), md=6),
                dbc.Col(dcc.Graph(id='revenue-mix', className="border-0"), md=6)
            ], className="mt-4")
        ], md=9)
    ], className="g-4"),
    html.Footer([
        html.Div("Apple Inc. Sales Data 2024 | Dashboard by Shahid Ul Islam", 
                className="text-center py-3", style={'color': CUSTOM_COLORS['text']})
    ], className="mt-5", style={'borderTop': f"1px solid {CUSTOM_COLORS['accent']}"})
], fluid=True, className="py-4", style={'backgroundColor': CUSTOM_COLORS['background']})

# Callbacks for Checklist
@callback(
    [Output('state-checklist', 'options'),
     Output('state-checklist', 'value'),
     Output('state-selection-info', 'children')],
    Input('region-dropdown', 'value')
)
def update_states(region):
    states = data[data['Region'] == region]['State'].unique()
    options = [{'label': s, 'value': s} for s in states]
    stats = f"{len(states)} states available in {region}"
    return options, states, stats

# Callback Outputs 
@callback(
    [Output('sales-graph', 'figure'),
     Output('revenue-graph', 'figure'),
     Output('product-mix', 'figure'),
     Output('revenue-mix', 'figure'),
     Output('total-sales', 'children'),
     Output('total-revenue', 'children'),
     Output('avg-revenue', 'children'),
     Output('top-product', 'children')],
    [Input('region-dropdown', 'value'),
     Input('state-checklist', 'value')]
)
def update_all(region, states):
    if not states:
        states = data[data['Region'] == region]['State'].unique()
    
    filtered_data = data[(data['Region'] == region) & (data['State'].isin(states))]
    
    # Sales Analysis
    sales_fig = px.bar(
        filtered_data.melt(id_vars=['State'], 
                         value_vars=[c for c in data.columns if 'Sales' in c],
                         var_name='Product', value_name='Sales'),
        x='Product', y='Sales', color='State',
        template='plotly_dark',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title="Product Sales Distribution"
    ).update_layout(
        plot_bgcolor=CUSTOM_COLORS['card-bg'],
        paper_bgcolor=CUSTOM_COLORS['background'],
        font={'color': CUSTOM_COLORS['text']},
        barmode='group',
        height=500
    )
    
    # Revenue Analysis
    revenue_fig = px.line(
        filtered_data.groupby('State', as_index=False)['Services Revenue (in billion $)'].sum(),
        x='State', y='Services Revenue (in billion $)',
        template='plotly_dark',
        color_discrete_sequence=[CUSTOM_COLORS['accent']],
        title="Revenue Trend by State"
    ).update_layout(
        plot_bgcolor=CUSTOM_COLORS['card-bg'],
        paper_bgcolor=CUSTOM_COLORS['background'],
        font={'color': CUSTOM_COLORS['text']},
        height=500
    ).update_traces(mode='markers+lines')
    
    #revenue mix
    revenue_mix = px.pie(
        filtered_data.groupby('State', as_index=False)['Services Revenue (in billion $)'].sum(),
        names='State', values='Services Revenue (in billion $)',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title="Revenue Distribution by State"
    ).update_layout(
        plot_bgcolor=CUSTOM_COLORS['card-bg'],
        paper_bgcolor=CUSTOM_COLORS['background'],
        font={'color': CUSTOM_COLORS['text']},
        height=400,
        width=400
    )
    
    # Product Mix
    product_mix = px.pie(
        filtered_data.melt(value_vars=[c for c in data.columns if 'Sales' in c],
                         var_name='Product', value_name='Sales'),
        names='Product', values='Sales',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title="Product Sales Distribution by State"
    ).update_layout(
        plot_bgcolor=CUSTOM_COLORS['card-bg'],
        paper_bgcolor=CUSTOM_COLORS['background'],
        font={'color': CUSTOM_COLORS['text']},
        height=500,
        width=500
    )
    
    
    # Metrics Calculations
    total_sales = filtered_data[[c for c in data.columns if 'Sales' in c]].sum().sum()
    total_rev = filtered_data['Services Revenue (in billion $)'].sum()
    avg_rev = total_rev / len(states) if states else 0
    top_product = filtered_data[[c for c in data.columns if 'Sales' in c]].sum().idxmax()
    
    return (
        sales_fig, 
        revenue_fig,
        revenue_mix,
        product_mix,
        f"{total_sales:,.2f}M",
        f"${total_rev:,.2f}B",
        f"${avg_rev:,.2f}B",
        top_product.split(' ')[0]
    )

if __name__ == '__main__':
    app.run_server(debug=True)

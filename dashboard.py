from dash import Dash, dcc, html, Input, Output
import pandas as pd
import plotly.express as px

# Load the data
data = pd.read_csv("apple_sales_2024.csv")

# Initialize the Dash app
app = Dash(__name__)
app.title = "Apple Sales Dashboard 2024"

# Layout of the dashboard
app.layout = html.Div([
    html.Div([
        html.H1("Apple Product Sales and Revenue Dashboard", className="header-title"),
        html.P("Explore product sales and revenue data by region and state.", className="header-description"),
    ], className="header"),

    html.Div([
        html.Div([
            html.Label("Select a Region:", className="label"),
            dcc.Dropdown(
                id='region-dropdown',
                options=[{'label': region, 'value': region} for region in data['Region'].unique()],
                value=data['Region'].unique()[0],
                clearable=False,
                className="dropdown"
            ),

            html.Label("Select States:", className="label"),
            dcc.Checklist(
                id='state-checklist',
                options=[],  # Dynamically updated
                inline=False,
                className="checklist"
            ),

            # Display total revenue for the region
            html.Div(id='total-revenue', className="total-revenue")
        ], className="sidebar"),

        html.Div([
            dcc.Graph(id='sales-graph', className="graph"),
            dcc.Graph(id='revenue-graph', className="graph")
        ], className="main-content")
    ], className="dashboard-container")
], className="app-container")

# Add external CSS for themes and styles
app.css.append_css({
    "external_url": "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"
})

# Custom CSS for theming and styling
custom_css = """
    body {
        background-image: url('https://www.transparenttextures.com/patterns/asfalt-dark.png');
        background-color: #2c3e50;
    }

    .app-container {
        font-family: Arial, sans-serif;
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        padding: 20px;
        margin: 20px auto;
        max-width: 1200px;
    }

    .header {
        background-color: #3498db;
        color: white;
        padding: 20px;
        text-align: center;
        border-radius: 8px;
        margin-bottom: 20px;
    }

    .header-title {
        font-size: 2.5em;
        margin: 0;
    }

    .header-description {
        font-size: 1.2em;
        margin: 0;
    }

    .dashboard-container {
        display: flex;
        flex-direction: row;
        gap: 20px;
    }

    .sidebar {
        background-color: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        flex: 1;
    }

    .main-content {
        flex: 3;
        display: flex;
        flex-direction: column;
        gap: 20px;
    }

    .label {
        font-weight: bold;
        margin-bottom: 10px;
        display: block;
    }

    .dropdown {
        margin-bottom: 20px;
    }

    .checklist {
        margin-bottom: 20px;
    }

    .graph {
        background-color: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }

    .total-revenue {
        margin-top: 20px;
        font-size: 1.2em;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
    }
"""

# Attach custom CSS to the Dash app
app.index_string = f"""
<!DOCTYPE html>
<html>
    <head>
        <title>Apple Sales Dashboard 2024</title>
        <style>{custom_css}</style>
    </head>
    <body>
        <div id="react-entry-point">{{%app_entry%}}</div>
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
"""

# Callback to update state checklist options based on selected region
@app.callback(
    Output('state-checklist', 'options'),
    Input('region-dropdown', 'value')
)
def update_state_checklist(selected_region):
    states = data[data['Region'] == selected_region]['State'].unique()
    return [{'label': state, 'value': state} for state in states]

# Callback to update graphs and total revenue
@app.callback(
    [Output('sales-graph', 'figure'), Output('revenue-graph', 'figure'), Output('total-revenue', 'children')],
    [Input('region-dropdown', 'value'), Input('state-checklist', 'value')]
)
def update_dashboard(selected_region, selected_states):
    if not selected_states:
        selected_states = data[data['Region'] == selected_region]['State'].unique()

    # Filter data for the selected region and states
    filtered_data = data[(data['Region'] == selected_region) & (data['State'].isin(selected_states))]

    # Sales by product for the selected region and states
    sales_data = pd.melt(
        filtered_data,
        id_vars=['Region', 'State'],
        value_vars=[
            'iPhone Sales (in million units)',
            'iPad Sales (in million units)',
            'Mac Sales (in million units)',
            'Wearables (in million units)'
        ],
        var_name='Product',
        value_name='Sales'
    )

    sales_fig = px.bar(
        sales_data,
        x='Product',
        y='Sales',
        color='State',
        title=f'Sales by Product in {selected_region}',
        labels={'Sales': 'Sales (in million units)', 'Product': 'Product Type'},
        template='plotly_dark',
        barmode='group',
    )

    # Revenue by state in the region
    revenue_fig = px.bar(
        filtered_data,
        x='State',
        y='Services Revenue (in billion $)',
        title=f'Revenue by State in {selected_region}',
        color='State',
        labels={'Services Revenue (in billion $)': 'Revenue (in billion $)', 'State': 'State'},
        template='plotly_dark',
        barmode='stack'
    )

    # Calculate total revenue for the selected region
    total_revenue = filtered_data['Services Revenue (in billion $)'].sum()
    total_revenue_display = f"Total Services Revenue for {selected_region}: ${total_revenue:.2f} billion"

    return sales_fig, revenue_fig, total_revenue_display

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)


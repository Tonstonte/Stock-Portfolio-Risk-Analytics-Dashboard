# Stock Portfolio Risk Analytics Dashboard
# Google Colab Starter Notebook

# ===== INSTALLATION & IMPORTS =====
# Run this cell first in Google Colab
"""
!pip install yfinance plotly streamlit scipy
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ===== DATA COLLECTION FUNCTIONS =====

def fetch_stock_data(tickers, period='2y'):
    """
    Fetch stock data for given tickers
    """
    try:
        data = yf.download(tickers, period=period, auto_adjust=True)['Close']
        if isinstance(data, pd.Series):
            data = data.to_frame(tickers[0])
        return data.dropna()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def get_sample_portfolio():
    """
    Returns a sample portfolio for testing
    """
    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'JNJ', 'V', 'PG', 'UNH']

# ===== PORTFOLIO ANALYSIS FUNCTIONS =====

def calculate_returns(data):
    """
    Calculate daily returns from price data
    """
    return data.pct_change().dropna()

def calculate_portfolio_stats(returns, weights):
    """
    Calculate portfolio statistics
    """
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe_ratio = portfolio_return / portfolio_std
    
    return {
        'Annual Return': portfolio_return,
        'Annual Volatility': portfolio_std,
        'Sharpe Ratio': sharpe_ratio
    }

def calculate_var_cvar(returns, confidence_level=0.05):
    """
    Calculate Value at Risk and Conditional Value at Risk
    """
    var = np.percentile(returns, confidence_level * 100)
    cvar = returns[returns <= var].mean()
    
    return var, cvar

def calculate_maximum_drawdown(price_series):
    """
    Calculate maximum drawdown
    """
    cumulative = (1 + price_series.pct_change()).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

# ===== PORTFOLIO OPTIMIZATION =====

def portfolio_performance(weights, returns):
    """
    Calculate portfolio performance metrics
    """
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return portfolio_return, portfolio_std

def negative_sharpe(weights, returns):
    """
    Negative Sharpe ratio for optimization (we minimize)
    """
    p_ret, p_vol = portfolio_performance(weights, returns)
    return -(p_ret / p_vol)

def optimize_portfolio(returns, target_return=None):
    """
    Optimize portfolio using Modern Portfolio Theory
    """
    n_assets = len(returns.columns)
    args = (returns,)
    
    # Constraints
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Initial guess
    x0 = np.array([1/n_assets] * n_assets)
    
    if target_return:
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: portfolio_performance(x, returns)[0] - target_return}
        ]
        # Minimize volatility
        result = minimize(lambda x: portfolio_performance(x, returns)[1], x0, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
    else:
        # Maximize Sharpe ratio
        result = minimize(negative_sharpe, x0, args=args, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
    
    return result.x

def generate_efficient_frontier(returns, n_portfolios=50):
    """
    Generate efficient frontier
    """
    target_returns = np.linspace(returns.mean().min() * 252, returns.mean().max() * 252, n_portfolios)
    efficient_portfolios = []
    
    for target in target_returns:
        try:
            weights = optimize_portfolio(returns, target)
            ret, vol = portfolio_performance(weights, returns)
            efficient_portfolios.append([ret, vol, weights])
        except:
            continue
    
    return pd.DataFrame(efficient_portfolios, columns=['Return', 'Volatility', 'Weights'])

# ===== TECHNICAL INDICATORS =====

def calculate_sma(data, window):
    """Simple Moving Average"""
    return data.rolling(window=window).mean()

def calculate_rsi(data, window=14):
    """Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Bollinger Bands"""
    sma = calculate_sma(data, window)
    std = data.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band

# ===== MONTE CARLO SIMULATION =====

def monte_carlo_simulation(returns, weights, initial_value=10000, days=252, simulations=1000):
    """
    Monte Carlo simulation for portfolio performance
    """
    portfolio_returns = returns @ weights
    mean_return = portfolio_returns.mean()
    std_return = portfolio_returns.std()
    
    results = []
    
    for _ in range(simulations):
        # Generate random returns
        random_returns = np.random.normal(mean_return, std_return, days)
        # Calculate cumulative portfolio value
        portfolio_values = [initial_value]
        for ret in random_returns:
            portfolio_values.append(portfolio_values[-1] * (1 + ret))
        results.append(portfolio_values[-1])
    
    return np.array(results)

# ===== VISUALIZATION FUNCTIONS =====

def plot_price_chart(data, title="Stock Prices"):
    """
    Plot interactive price chart with Plotly
    """
    fig = go.Figure()
    
    for column in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[column],
            name=column,
            mode='lines'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def plot_correlation_heatmap(returns):
    """
    Plot correlation heatmap
    """
    correlation_matrix = returns.corr()
    
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu',
        title="Asset Correlation Matrix"
    )
    
    return fig

def plot_efficient_frontier(efficient_frontier_df, optimal_weights, returns):
    """
    Plot efficient frontier with optimal portfolio
    """
    fig = go.Figure()
    
    # Efficient frontier
    fig.add_trace(go.Scatter(
        x=efficient_frontier_df['Volatility'],
        y=efficient_frontier_df['Return'],
        mode='lines+markers',
        name='Efficient Frontier',
        line=dict(color='blue')
    ))
    
    # Optimal portfolio
    opt_ret, opt_vol = portfolio_performance(optimal_weights, returns)
    fig.add_trace(go.Scatter(
        x=[opt_vol],
        y=[opt_ret],
        mode='markers',
        name='Optimal Portfolio',
        marker=dict(color='red', size=12, symbol='star')
    ))
    
    fig.update_layout(
        title="Efficient Frontier",
        xaxis_title="Volatility (Risk)",
        yaxis_title="Expected Return",
        template='plotly_white'
    )
    
    return fig

def plot_portfolio_composition(weights, tickers):
    """
    Plot portfolio composition pie chart
    """
    fig = go.Figure(data=[go.Pie(
        labels=tickers,
        values=weights,
        hole=0.3
    )])
    
    fig.update_layout(
        title="Portfolio Composition",
        template='plotly_white'
    )
    
    return fig

# ===== MAIN ANALYSIS WORKFLOW =====

def run_portfolio_analysis():
    """
    Main function to run the complete portfolio analysis
    """
    print("🚀 Starting Portfolio Risk Analytics...")
    
    # 1. Get sample portfolio
    tickers = get_sample_portfolio()
    print(f"📊 Analyzing portfolio: {tickers}")
    
    # 2. Fetch data
    print("📈 Fetching stock data...")
    price_data = fetch_stock_data(tickers)
    
    if price_data is None:
        print("❌ Failed to fetch data")
        return
    
    # 3. Calculate returns
    returns = calculate_returns(price_data)
    print(f"✅ Data fetched: {len(price_data)} days of data")
    
    # 4. Equal weight portfolio
    equal_weights = np.array([1/len(tickers)] * len(tickers))
    
    # 5. Portfolio statistics
    portfolio_stats = calculate_portfolio_stats(returns, equal_weights)
    print("\n📋 Equal Weight Portfolio Statistics:")
    for key, value in portfolio_stats.items():
        print(f"  {key}: {value:.4f}")
    
    # 6. Risk metrics
    portfolio_returns = returns @ equal_weights
    var_95, cvar_95 = calculate_var_cvar(portfolio_returns, 0.05)
    max_dd = calculate_maximum_drawdown(portfolio_returns)
    
    print(f"\n⚠️  Risk Metrics:")
    print(f"  VaR (95%): {var_95:.4f}")
    print(f"  CVaR (95%): {cvar_95:.4f}")
    print(f"  Maximum Drawdown: {max_dd:.4f}")
    
    # 7. Portfolio optimization
    print("\n🎯 Optimizing portfolio...")
    optimal_weights = optimize_portfolio(returns)
    optimal_stats = calculate_portfolio_stats(returns, optimal_weights)
    
    print("📋 Optimized Portfolio Statistics:")
    for key, value in optimal_stats.items():
        print(f"  {key}: {value:.4f}")
    
    # 8. Generate visualizations
    print("\n📊 Generating visualizations...")
    
    # Price chart
    price_fig = plot_price_chart(price_data, "Portfolio Stock Prices")
    price_fig.show()
    
    # Correlation heatmap
    corr_fig = plot_correlation_heatmap(returns)
    corr_fig.show()
    
    # Portfolio composition
    comp_fig = plot_portfolio_composition(optimal_weights, tickers)
    comp_fig.show()
    
    # Efficient frontier
    print("🔄 Generating efficient frontier...")
    efficient_frontier = generate_efficient_frontier(returns, 20)
    ef_fig = plot_efficient_frontier(efficient_frontier, optimal_weights, returns)
    ef_fig.show()
    
    # 9. Monte Carlo simulation
    print("🎲 Running Monte Carlo simulation...")
    mc_results = monte_carlo_simulation(returns, optimal_weights, 10000, 252, 1000)
    
    plt.figure(figsize=(10, 6))
    plt.hist(mc_results, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(mc_results), color='red', linestyle='--', 
                label=f'Mean: ${np.mean(mc_results):,.0f}')
    plt.axvline(np.percentile(mc_results, 5), color='orange', linestyle='--',
                label=f'5th Percentile: ${np.percentile(mc_results, 5):,.0f}')
    plt.xlabel('Portfolio Value After 1 Year')
    plt.ylabel('Frequency')
    plt.title('Monte Carlo Simulation Results (1000 simulations)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("✅ Analysis complete!")
    
    return {
        'price_data': price_data,
        'returns': returns,
        'optimal_weights': optimal_weights,
        'tickers': tickers,
        'portfolio_stats': optimal_stats,
        'mc_results': mc_results
    }

# ===== STREAMLIT DEPLOYMENT TEMPLATE =====

def create_streamlit_app():
    """
    Template for Streamlit app - save this as app.py
    """
    streamlit_code = '''
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# [Include all the functions from above here]

st.set_page_config(page_title="Portfolio Risk Analytics", layout="wide")

st.title("🚀 Portfolio Risk Analytics Dashboard")

# Sidebar for user inputs
st.sidebar.header("Portfolio Configuration")

# Stock selection
default_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
selected_stocks = st.sidebar.multiselect(
    "Select Stocks", 
    options=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM", "JNJ", "V", "PG", "UNH"],
    default=default_stocks
)

# Time period
period = st.sidebar.selectbox("Time Period", ["1y", "2y", "3y", "5y"])

# Analysis button
if st.sidebar.button("Analyze Portfolio"):
    if selected_stocks:
        with st.spinner("Fetching data and running analysis..."):
            # Run analysis
            data = fetch_stock_data(selected_stocks, period)
            returns = calculate_returns(data)
            
            # Display results in tabs
            tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Risk Metrics", "Optimization", "Simulations"])
            
            with tab1:
                st.subheader("Price Performance")
                fig = plot_price_chart(data)
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Correlation Matrix")
                corr_fig = plot_correlation_heatmap(returns)
                st.plotly_chart(corr_fig, use_container_width=True)
            
            # Add other tabs...
    else:
        st.warning("Please select at least one stock.")
'''
    
    with open('streamlit_app_template.py', 'w') as f:
        f.write(streamlit_code)
    
    print("📝 Streamlit app template created as 'streamlit_app_template.py'")

# ===== RUN ANALYSIS =====

if __name__ == "__main__":
    # Run the analysis
    results = run_portfolio_analysis()
    
    # Create Streamlit template
    create_streamlit_app()
    
    print(f"\n🎉 Project complete! Key results:")
    print(f"  • Analyzed {len(results['tickers'])} stocks")
    print(f"  • Optimal Sharpe Ratio: {results['portfolio_stats']['Sharpe Ratio']:.4f}")
    print(f"  • Expected Annual Return: {results['portfolio_stats']['Annual Return']:.2%}")
    print(f"  • Annual Volatility: {results['portfolio_stats']['Annual Volatility']:.2%}")
    
    print(f"\n📋 Next steps:")
    print(f"  1. Copy this code to Google Colab")
    print(f"  2. Run the analysis and experiment with different stocks")
    print(f"  3. Use the Streamlit template to create your web app")
    print(f"  4. Deploy on Streamlit Cloud for sharing")

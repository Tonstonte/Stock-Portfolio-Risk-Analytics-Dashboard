import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ===== CACHE FUNCTIONS FOR PERFORMANCE =====

@st.cache_data
def fetch_stock_data(tickers, period='2y'):
    """Fetch stock data with caching"""
    try:
        data = yf.download(tickers, period=period, auto_adjust=True)['Close']
        if isinstance(data, pd.Series):
            data = data.to_frame(tickers[0])
        return data.dropna()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

@st.cache_data
def calculate_returns(data):
    """Calculate returns with caching"""
    return data.pct_change().dropna()

# ===== CORE FUNCTIONS =====

def calculate_portfolio_stats(returns, weights):
    """Calculate portfolio statistics"""
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe_ratio = portfolio_return / portfolio_std if portfolio_std > 0 else 0
    
    return {
        'Annual Return': portfolio_return,
        'Annual Volatility': portfolio_std,
        'Sharpe Ratio': sharpe_ratio
    }

def calculate_var_cvar(returns, confidence_level=0.05):
    """Calculate VaR and CVaR"""
    var = np.percentile(returns, confidence_level * 100)
    cvar = returns[returns <= var].mean()
    return var, cvar

def calculate_maximum_drawdown(returns):
    """Calculate maximum drawdown"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def portfolio_performance(weights, returns):
    """Calculate portfolio performance"""
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return portfolio_return, portfolio_std

def negative_sharpe(weights, returns):
    """Negative Sharpe for optimization"""
    p_ret, p_vol = portfolio_performance(weights, returns)
    return -(p_ret / p_vol) if p_vol > 0 else 0

def optimize_portfolio(returns):
    """Optimize portfolio for maximum Sharpe ratio"""
    n_assets = len(returns.columns)
    args = (returns,)
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    x0 = np.array([1/n_assets] * n_assets)
    
    try:
        result = minimize(negative_sharpe, x0, args=args, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        return result.x if result.success else x0
    except:
        return x0

def monte_carlo_simulation(returns, weights, initial_value=10000, days=252, simulations=1000):
    """Monte Carlo simulation"""
    portfolio_returns = returns @ weights
    mean_return = portfolio_returns.mean()
    std_return = portfolio_returns.std()
    
    results = []
    for _ in range(simulations):
        random_returns = np.random.normal(mean_return, std_return, days)
        portfolio_value = initial_value
        for ret in random_returns:
            portfolio_value *= (1 + ret)
        results.append(portfolio_value)
    
    return np.array(results)

# ===== VISUALIZATION FUNCTIONS =====

def plot_price_chart(data):
    """Interactive price chart"""
    fig = go.Figure()
    
    for column in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[column],
            name=column,
            mode='lines',
            hovertemplate='%{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Stock Price Performance",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

def plot_correlation_heatmap(returns):
    """Correlation heatmap"""
    correlation_matrix = returns.corr()
    
    fig = px.imshow(
        correlation_matrix,
        text_auto='.2f',
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Asset Correlation Matrix",
        height=500
    )
    
    return fig

def plot_portfolio_composition(weights, tickers):
    """Portfolio composition pie chart"""
    # Filter out very small weights for cleaner visualization
    min_weight = 0.01
    filtered_weights = []
    filtered_tickers = []
    other_weight = 0
    
    for i, weight in enumerate(weights):
        if weight >= min_weight:
            filtered_weights.append(weight)
            filtered_tickers.append(tickers[i])
        else:
            other_weight += weight
    
    if other_weight > 0:
        filtered_weights.append(other_weight)
        filtered_tickers.append('Others')
    
    fig = go.Figure(data=[go.Pie(
        labels=filtered_tickers,
        values=filtered_weights,
        hole=0.3,
        textinfo='label+percent',
        textposition='outside'
    )])
    
    fig.update_layout(
        title="Optimized Portfolio Composition",
        template='plotly_white',
        height=500
    )
    
    return fig

def plot_risk_return_scatter(returns, equal_weights, optimal_weights, tickers):
    """Risk-return scatter plot"""
    fig = go.Figure()
    
    # Individual assets
    annual_returns = returns.mean() * 252
    annual_vols = returns.std() * np.sqrt(252)
    
    fig.add_trace(go.Scatter(
        x=annual_vols,
        y=annual_returns,
        mode='markers+text',
        text=tickers,
        textposition='top center',
        name='Individual Assets',
        marker=dict(size=10, color='lightblue')
    ))
    
    # Equal weight portfolio
    eq_ret, eq_vol = portfolio_performance(equal_weights, returns)
    fig.add_trace(go.Scatter(
        x=[eq_vol],
        y=[eq_ret],
        mode='markers',
        name='Equal Weight Portfolio',
        marker=dict(size=15, color='blue', symbol='diamond')
    ))
    
    # Optimal portfolio
    opt_ret, opt_vol = portfolio_performance(optimal_weights, returns)
    fig.add_trace(go.Scatter(
        x=[opt_vol],
        y=[opt_ret],
        mode='markers',
        name='Optimized Portfolio',
        marker=dict(size=15, color='red', symbol='star')
    ))
    
    fig.update_layout(
        title="Risk-Return Profile",
        xaxis_title="Annual Volatility",
        yaxis_title="Annual Return",
        template='plotly_white',
        height=500
    )
    
    return fig

def plot_monte_carlo_results(mc_results, initial_value):
    """Monte Carlo simulation results"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=mc_results,
        nbinsx=50,
        name='Simulation Results',
        opacity=0.7
    ))
    
    # Add percentile lines
    percentiles = [5, 50, 95]
    colors = ['red', 'blue', 'green']
    
    for p, color in zip(percentiles, colors):
        value = np.percentile(mc_results, p)
        fig.add_vline(
            x=value,
            line_dash="dash",
            line_color=color,
            annotation_text=f"{p}th percentile: ${value:,.0f}"
        )
    
    fig.update_layout(
        title="Monte Carlo Simulation Results (1000 simulations, 1 year)",
        xaxis_title="Portfolio Value",
        yaxis_title="Frequency",
        template='plotly_white',
        height=500
    )
    
    return fig

# ===== STREAMLIT APP =====

def main():
    st.set_page_config(
        page_title="Portfolio Risk Analytics",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📈 Portfolio Risk Analytics Dashboard")
    st.markdown("Analyze portfolio performance, risk metrics, and optimization using Modern Portfolio Theory")
    
    # Sidebar configuration
    st.sidebar.header("🔧 Portfolio Configuration")
    
    # Predefined stock options
    all_stocks = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 
        'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'BAC', 'NFLX'
    ]
    
    selected_stocks = st.sidebar.multiselect(
        "Select Stocks for Analysis",
        options=all_stocks,
        default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
        help="Choose 3-10 stocks for optimal analysis"
    )
    
    time_period = st.sidebar.selectbox(
        "Analysis Time Period",
        options=['1y', '2y', '3y', '5y'],
        index=1,
        help="Historical data period for analysis"
    )
    
    initial_investment = st.sidebar.number_input(
        "Initial Investment ($)",
        min_value=1000,
        max_value=1000000,
        value=10000,
        step=1000
    )
    
    # Analysis button
    analyze_button = st.sidebar.button("🚀 Run Analysis", type="primary")
    
    if analyze_button or st.session_state.get('analyzed', False):
        if len(selected_stocks) < 3:
            st.error("Please select at least 3 stocks for meaningful analysis.")
            return
        
        st.session_state['analyzed'] = True
        
        with st.spinner("Fetching data and running analysis..."):
            # Fetch data
            price_data = fetch_stock_data(selected_stocks, time_period)
            
            if price_data is None:
                st.error("Failed to fetch stock data. Please try again.")
                return
            
            returns = calculate_returns(price_data)
            
            # Calculate portfolios
            equal_weights = np.array([1/len(selected_stocks)] * len(selected_stocks))
            optimal_weights = optimize_portfolio(returns)
            
            # Portfolio statistics
            equal_stats = calculate_portfolio_stats(returns, equal_weights)
            optimal_stats = calculate_portfolio_stats(returns, optimal_weights)
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", "⚡ Performance", "🎯 Optimization", 
            "⚠️ Risk Analysis", "🎲 Monte Carlo"
        ])
        
        with tab1:
            st.subheader("Portfolio Overview")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.plotly_chart(plot_price_chart(price_data), use_container_width=True)
            
            with col2:
                st.metric("Stocks Analyzed", len(selected_stocks))
                st.metric("Data Points", len(price_data))
                st.metric("Time Period", time_period)
                
                # Recent performance
                recent_returns = returns.tail(30).mean() * 252
                best_performer = recent_returns.idxmax()
                st.metric(
                    "Best Recent Performer", 
                    best_performer,
                    f"{recent_returns[best_performer]:.2%}"
                )
            
            st.plotly_chart(plot_correlation_heatmap(returns), use_container_width=True)
        
        with tab2:
            st.subheader("Performance Comparison")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Equal Weight Return",
                    f"{equal_stats['Annual Return']:.2%}",
                    help="Annualized return with equal weights"
                )
                st.metric(
                    "Equal Weight Volatility",
                    f"{equal_stats['Annual Volatility']:.2%}",
                    help="Annualized volatility with equal weights"
                )
                st.metric(
                    "Equal Weight Sharpe",
                    f"{equal_stats['Sharpe Ratio']:.3f}",
                    help="Risk-adjusted return measure"
                )
            
            with col2:
                st.metric(
                    "Optimized Return",
                    f"{optimal_stats['Annual Return']:.2%}",
                    f"{optimal_stats['Annual Return'] - equal_stats['Annual Return']:.2%}"
                )
                st.metric(
                    "Optimized Volatility",
                    f"{optimal_stats['Annual Volatility']:.2%}",
                    f"{optimal_stats['Annual Volatility'] - equal_stats['Annual Volatility']:.2%}"
                )
                st.metric(
                    "Optimized Sharpe",
                    f"{optimal_stats['Sharpe Ratio']:.3f}",
                    f"{optimal_stats['Sharpe Ratio'] - equal_stats['Sharpe Ratio']:.3f}"
                )
            
            with col3:
                improvement = (optimal_stats['Sharpe Ratio'] - equal_stats['Sharpe Ratio']) / equal_stats['Sharpe Ratio'] * 100
                st.metric(
                    "Sharpe Improvement",
                    f"{improvement:.1f}%",
                    help="Percentage improvement in risk-adjusted returns"
                )
                
                # Calculate potential value improvement
                potential_value = initial_investment * (1 + optimal_stats['Annual Return'])
                equal_value = initial_investment * (1 + equal_stats['Annual Return'])
                value_diff = potential_value - equal_value
                
                st.metric(
                    "Annual Value Gain",
                    f"${value_diff:,.0f}",
                    help="Additional value from optimization"
                )
            
            st.plotly_chart(
                plot_risk_return_scatter(returns, equal_weights, optimal_weights, selected_stocks),
                use_container_width=True
            )
        
        with tab3:
            st.subheader("Portfolio Optimization Results")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.plotly_chart(
                    plot_portfolio_composition(optimal_weights, selected_stocks),
                    use_container_width=True
                )
            
            with col2:
                st.subheader("Optimal Weights")
                weights_df = pd.DataFrame({
                    'Stock': selected_stocks,
                    'Weight': optimal_weights,
                    'Weight %': optimal_weights * 100
                }).sort_values('Weight', ascending=False)
                
                st.dataframe(
                    weights_df.style.format({
                        'Weight': '{:.4f}',
                        'Weight %': '{:.2f}%'
                    }),
                    use_container_width=True
                )
                
                # Investment allocation
                st.subheader(f"Investment Allocation (${initial_investment:,})")
                allocation_df = weights_df.copy()
                allocation_df['Amount'] = allocation_df['Weight'] * initial_investment
                allocation_df['Shares'] = 0  # Would need current prices to calculate
                
                st.dataframe(
                    allocation_df[['Stock', 'Amount', 'Weight %']].style.format({
                        'Amount': '${:,.0f}',
                        'Weight %': '{:.2f}%'
                    }),
                    use_container_width=True
                )
        
        with tab4:
            st.subheader("Risk Analysis")
            
            # Calculate risk metrics for both portfolios
            equal_portfolio_returns = returns @ equal_weights
            optimal_portfolio_returns = returns @ optimal_weights
            
            equal_var_95, equal_cvar_95 = calculate_var_cvar(equal_portfolio_returns, 0.05)
            optimal_var_95, optimal_cvar_95 = calculate_var_cvar(optimal_portfolio_returns, 0.05)
            
            equal_max_dd = calculate_maximum_drawdown(equal_portfolio_returns)
            optimal_max_dd = calculate_maximum_drawdown(optimal_portfolio_returns)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Equal Weight Portfolio")
                st.metric("VaR (95%)", f"{equal_var_95:.2%}", help="Daily Value at Risk")
                st.metric("CVaR (95%)", f"{equal_cvar_95:.2%}", help="Conditional Value at Risk")
                st.metric("Max Drawdown", f"{equal_max_dd:.2%}", help="Maximum historical loss")
                
                # Annual risk metrics
                annual_var = equal_var_95 * np.sqrt(252)
                annual_cvar = equal_cvar_95 * np.sqrt(252)
                st.metric("Annual VaR (95%)", f"{annual_var:.2%}")
                st.metric("Annual CVaR (95%)", f"{annual_cvar:.2%}")
            
            with col2:
                st.subheader("Optimized Portfolio")
                st.metric(
                    "VaR (95%)", 
                    f"{optimal_var_95:.2%}",
                    f"{optimal_var_95 - equal_var_95:.2%}"
                )
                st.metric(
                    "CVaR (95%)", 
                    f"{optimal_cvar_95:.2%}",
                    f"{optimal_cvar_95 - equal_cvar_95:.2%}"
                )
                st.metric(
                    "Max Drawdown", 
                    f"{optimal_max_dd:.2%}",
                    f"{optimal_max_dd - equal_max_dd:.2%}"
                )
                
                # Annual risk metrics
                annual_var_opt = optimal_var_95 * np.sqrt(252)
                annual_cvar_opt = optimal_cvar_95 * np.sqrt(252)
                st.metric("Annual VaR (95%)", f"{annual_var_opt:.2%}")
                st.metric("Annual CVaR (95%)", f"{annual_cvar_opt:.2%}")
            
            # Risk decomposition
            st.subheader("Risk Decomposition")
            individual_vols = returns.std() * np.sqrt(252)
            risk_contrib = pd.DataFrame({
                'Stock': selected_stocks,
                'Individual Volatility': individual_vols,
                'Weight in Portfolio': optimal_weights,
                'Risk Contribution': individual_vols * optimal_weights
            }).sort_values('Risk Contribution', ascending=False)
            
            st.dataframe(
                risk_contrib.style.format({
                    'Individual Volatility': '{:.2%}',
                    'Weight in Portfolio': '{:.2%}',
                    'Risk Contribution': '{:.4f}'
                }),
                use_container_width=True
            )
        
        with tab5:
            st.subheader("Monte Carlo Simulation")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Run Monte Carlo simulation
                mc_results = monte_carlo_simulation(
                    returns, optimal_weights, initial_investment, 252, 1000
                )
                
                st.plotly_chart(
                    plot_monte_carlo_results(mc_results, initial_investment),
                    use_container_width=True
                )
            
            with col2:
                st.subheader("Simulation Statistics")
                
                mean_value = np.mean(mc_results)
                median_value = np.median(mc_results)
                std_value = np.std(mc_results)
                
                st.metric("Mean Final Value", f"${mean_value:,.0f}")
                st.metric("Median Final Value", f"${median_value:,.0f}")
                st.metric("Standard Deviation", f"${std_value:,.0f}")
                
                # Probability metrics
                prob_loss = np.mean(mc_results < initial_investment) * 100
                prob_double = np.mean(mc_results > initial_investment * 2) * 100
                
                st.metric("Probability of Loss", f"{prob_loss:.1f}%")
                st.metric("Probability of Doubling", f"{prob_double:.1f}%")
                
                # Percentile analysis
                percentiles = [5, 25, 75, 95]
                st.subheader("Value Percentiles")
                for p in percentiles:
                    value = np.percentile(mc_results, p)
                    st.metric(f"{p}th Percentile", f"${value:,.0f}")
        
        # Download section
        st.subheader("📥 Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Portfolio weights CSV
            weights_csv = weights_df.to_csv(index=False)
            st.download_button(
                label="Download Portfolio Weights",
                data=weights_csv,
                file_name="portfolio_weights.csv",
                mime="text/csv"
            )
        
        with col2:
            # Historical data CSV
            price_csv = price_data.to_csv()
            st.download_button(
                label="Download Price Data",
                data=price_csv,
                file_name="historical_prices.csv",
                mime="text/csv"
            )
        
        with col3:
            # Monte Carlo results CSV
            mc_df = pd.DataFrame({'Simulation_Result': mc_results})
            mc_csv = mc_df.to_csv(index=False)
            st.download_button(
                label="Download MC Results",
                data=mc_csv,
                file_name="monte_carlo_results.csv",
                mime="text/csv"
            )
    
    else:
        # Landing page content
        st.markdown("""
        ## Welcome to Portfolio Risk Analytics! 🚀
        
        This dashboard helps you analyze and optimize your investment portfolio using advanced financial techniques:
        
        ### 🔍 What You Can Do:
        - **Portfolio Optimization**: Find the optimal asset allocation using Modern Portfolio Theory
        - **Risk Analysis**: Calculate VaR, CVaR, and maximum drawdown
        - **Performance Comparison**: Compare equal-weight vs optimized portfolios
        - **Monte Carlo Simulation**: Project future portfolio performance
        - **Correlation Analysis**: Understand asset relationships
        
        ### 📊 Key Features:
        - Interactive charts and visualizations
        - Real-time stock data from Yahoo Finance
        - Professional risk metrics
        - Downloadable results
        
        ### 🛠️ How to Use:
        1. Select 3-10 stocks from the sidebar
        2. Choose your analysis time period
        3. Set your initial investment amount
        4. Click "Run Analysis" to start
        
        **Ready to optimize your portfolio? Configure your settings in the sidebar and click "Run Analysis"!**
        """)
        
        # Sample analysis preview
        st.subheader("📈 Sample Analysis Preview")
        
        sample_data = {
            'Metric': [
                'Annual Return',
                'Annual Volatility', 
                'Sharpe Ratio',
                'Maximum Drawdown',
                'VaR (95%)'
            ],
            'Equal Weight': ['12.5%', '18.2%', '0.687', '-15.3%', '-2.1%'],
            'Optimized': ['14.8%', '16.9%', '0.876', '-12.1%', '-1.8%'],
            'Improvement': ['+2.3%', '-1.3%', '+27.5%', '+3.2%', '+0.3%']
        }
        
        sample_df = pd.DataFrame(sample_data)
        st.table(sample_df)

if __name__ == "__main__":
    main()

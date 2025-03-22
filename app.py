import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
import google.generativeai as genai
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

# Set page configuration
st.set_page_config(
    page_title="Financial Benchmarking Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 38px;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    .sub-header {
        font-size: 18px;
        font-weight: 400;
        color: #64748B;
        margin-top: 0;
        padding-top: 0;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 36px;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 14px;
        font-weight: 500;
        color: #64748B;
    }
    .section-header {
        font-size: 24px;
        font-weight: 600;
        color: #1E3A8A;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E2E8F0;
    }
    .stTabs {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .stExpander {
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #E2E8F0;
        font-size: 12px;
        color: #94A3B8;
        text-align: center;
    }
    .highlight {
        background-color: #F0F9FF;
        border-left: 4px solid #0EA5E9;
        padding: 10px 15px;
        margin: 20px 0;
    }
    .positive {
        color: #16A34A;
    }
    .negative {
        color: #DC2626;
    }
    .neutral {
        color: #64748B;
    }
</style>
""", unsafe_allow_html=True)

# Function to fetch related companies using Google Gemini API
def get_related_companies(ticker, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    prompt = f"List 5 major competitors of {ticker} company in the stock market. Return only the stock tickers in a comma-separated format with no additional text or explanation. For example: AAPL,MSFT,GOOGL"
    
    try:
        response = model.generate_content(prompt)
        competitors = response.text.strip().split(',')
        # Clean up the competitors list
        competitors = [comp.strip() for comp in competitors if comp.strip()]
        return competitors[:5]  # Limit to 5 competitors
    except Exception as e:
        st.error(f"Error fetching related companies: {str(e)}")
        return []

# Function to fetch company info
def get_company_info(ticker):
    try:
        company = yf.Ticker(ticker)
        info = company.info
        return info
    except Exception as e:
        st.error(f"Error fetching company info for {ticker}: {str(e)}")
        return {}

# Function to fetch stock data
def get_stock_data(ticker, period="10y"):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

# Function to calculate metrics
def calculate_metrics(data):
    if data.empty:
        return {}
    
    # Calculate daily returns
    data['Daily_Return'] = data['Close'].pct_change()
    
    # Calculate metrics
    current_price = data['Close'].iloc[-1]
    start_price = data['Close'].iloc[0]
    total_return = (current_price / start_price - 1) * 100
    
    # Annualized return
    days = (data.index[-1] - data.index[0]).days
    annualized_return = (((1 + total_return/100) ** (365.0/days)) - 1) * 100
    
    # Risk metrics
    volatility = data['Daily_Return'].std() * np.sqrt(252) * 100  # Annualized volatility
    sharpe = annualized_return / volatility if volatility > 0 else 0
    
    # Maximum drawdown
    data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod()
    data['Cumulative_Max'] = data['Cumulative_Return'].cummax()
    data['Drawdown'] = (data['Cumulative_Return'] / data['Cumulative_Max'] - 1) * 100
    max_drawdown = data['Drawdown'].min()
    
    return {
        "current_price": current_price,
        "start_price": start_price,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "data": data
    }

# Function to create performance chart
def create_performance_chart(data_dict, tickers):
    valid_tickers = [t for t in tickers if t in data_dict and not data_dict[t].empty]
    
    if not valid_tickers:
        return None
    
    fig = go.Figure()
    
    for ticker in valid_tickers:
        df = data_dict[ticker].copy()
        # Normalize to 100
        normalized = df['Close'] / df['Close'].iloc[0] * 100
        fig.add_trace(
            go.Scatter(
                x=df.index, 
                y=normalized, 
                name=ticker,
                mode='lines',
                line=dict(width=2),
                hovertemplate='%{x}<br>%{y:.2f}%<extra></extra>'
            )
        )
    
    fig.update_layout(
        title='Comparative Performance (Normalized to 100)',
        title_font=dict(size=20, family='Arial', color='#1E3A8A'),
        xaxis_title='Date',
        yaxis_title='Value (Starting at 100)',
        legend_title='Companies',
        template='plotly_white',
        height=500,
        hovermode='x unified',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(230, 230, 230, 0.8)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(230, 230, 230, 0.8)'
        ),
        plot_bgcolor='rgba(255, 255, 255, 0.9)',
        paper_bgcolor='rgba(255, 255, 255, 0.9)',
        margin=dict(l=40, r=40, t=80, b=40)
    )
    
    return fig

# Function to create returns chart
def create_returns_chart(metrics_dict, tickers):
    valid_tickers = [t for t in tickers if t in metrics_dict]
    
    if not valid_tickers:
        return None
    
    # Prepare data for the chart
    returns = [metrics_dict[t]["annualized_return"] for t in valid_tickers]
    volatilities = [metrics_dict[t]["volatility"] for t in valid_tickers]
    sharpes = [metrics_dict[t]["sharpe_ratio"] for t in valid_tickers]
    sizes = [abs(s) * 10 + 20 for s in sharpes]  # Scale for bubble size
    colors = ['green' if r > 0 else 'red' for r in returns]
    
    # Create figure
    fig = go.Figure()
    
    for i, ticker in enumerate(valid_tickers):
        fig.add_trace(
            go.Scatter(
                x=[volatilities[i]],
                y=[returns[i]],
                mode='markers+text',
                marker=dict(
                    size=sizes[i],
                    color=colors[i],
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                text=ticker,
                textposition="top center",
                name=ticker,
                hovertemplate=
                '<b>%{text}</b><br>' +
                'Return: %{y:.2f}%<br>' +
                'Volatility: %{x:.2f}%<br>' +
                'Sharpe Ratio: ' + f'{sharpes[i]:.2f}' +
                '<extra></extra>'
            )
        )
    
    # Add a line for Sharpe ratio = 1
    x_range = np.linspace(0, max(volatilities) * 1.1, 100)
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=x_range,
            mode='lines',
            line=dict(dash='dash', color='gray', width=1),
            name='Sharpe = 1',
            hoverinfo='skip'
        )
    )
    
    fig.update_layout(
        title='Risk-Return Analysis',
        title_font=dict(size=20, family='Arial', color='#1E3A8A'),
        xaxis_title='Volatility (Annualized %)',
        yaxis_title='Return (Annualized %)',
        template='plotly_white',
        height=500,
        hovermode='closest',
        showlegend=False,
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(230, 230, 230, 0.8)',
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(230, 230, 230, 0.8)',
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1
        ),
        plot_bgcolor='rgba(255, 255, 255, 0.9)',
        paper_bgcolor='rgba(255, 255, 255, 0.9)',
        margin=dict(l=40, r=40, t=80, b=40)
    )
    
    return fig

# Function to create drawdown chart
def create_drawdown_chart(metrics_dict, tickers):
    valid_tickers = [t for t in tickers if t in metrics_dict and "data" in metrics_dict[t]]
    
    if not valid_tickers:
        return None
    
    fig = go.Figure()
    
    for ticker in valid_tickers:
        df = metrics_dict[ticker]["data"]
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Drawdown'],
                name=ticker,
                mode='lines',
                line=dict(width=2),
                hovertemplate='%{x}<br>%{y:.2f}%<extra></extra>'
            )
        )
    
    fig.update_layout(
        title='Drawdown Analysis',
        title_font=dict(size=20, family='Arial', color='#1E3A8A'),
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        legend_title='Companies',
        template='plotly_white',
        height=500,
        hovermode='x unified',
        yaxis=dict(
            autorange="reversed",  # Inverting y-axis for better visualization of drawdowns
            showgrid=True,
            gridcolor='rgba(230, 230, 230, 0.8)'
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(230, 230, 230, 0.8)'
        ),
        plot_bgcolor='rgba(255, 255, 255, 0.9)',
        paper_bgcolor='rgba(255, 255, 255, 0.9)',
        margin=dict(l=40, r=40, t=80, b=40)
    )
    
    return fig

# Main application
def main():
    # Header
    st.markdown('<p class="main-header">Financial Benchmarking Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Compare performance metrics across industry competitors</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://api.placeholder.com/150/150", width=150)
        st.markdown("## Configuration")
        
        ticker = st.text_input("📌 Enter the stock ticker:", "AAPL")
        
        time_periods = {
            "1 Month": "1mo",
            "6 Months": "6mo",
            "1 Year": "1y",
            "3 Years": "3y",
            "5 Years": "5y",
            "10 Years": "10y",
            "Max": "max"
        }
        
        period = st.selectbox(
            "📅 Select time period:",
            options=list(time_periods.keys()),
            index=3
        )
        
        # Google Gemini API key
        api_key = st.text_input("🔑 Enter your Google Gemini API key:", 
                              value="your_api_key_here", 
                              type="password")
        
        st.markdown("---")
        
        # Advanced options
        with st.expander("⚙️ Advanced Settings"):
            include_benchmark = st.checkbox("Include market benchmark (S&P 500)", value=True)
            show_statistics = st.checkbox("Show detailed statistics", value=True)
            chart_type = st.radio("Chart style", ["Line", "Area", "Candlestick"])
            theme = st.selectbox("Chart theme", ["Light", "Dark", "Corporate"])
        
        analyze_button = st.button("🔍 Analyze", use_container_width=True)
    
    # Main content
    if analyze_button and ticker and api_key:
        # Initialize session state for storing data between reruns
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
            st.session_state.company_data = {}
            st.session_state.stock_data = {}
            st.session_state.metrics = {}
            st.session_state.competitors = []
            st.session_state.all_tickers = []
        
        # Create main layout
        main_col1, main_col2 = st.columns([2, 1])
        
        with main_col1:
            st.markdown(f'<p class="section-header">Analysis for {ticker.upper()}</p>', unsafe_allow_html=True)
        
        # Progress bar for data loading
        if not st.session_state.data_loaded:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Get company info
            status_text.text("Fetching company information...")
            company_info = get_company_info(ticker)
            progress_bar.progress(10)
            
            if company_info:
                company_name = company_info.get('shortName', ticker.upper())
                industry = company_info.get('industry', 'N/A')
                sector = company_info.get('sector', 'N/A')
                
                # Step 2: Fetch competitors
                status_text.text("Identifying competitors...")
                competitors = get_related_companies(ticker, api_key)
                progress_bar.progress(30)
                
                # Step 3: Add benchmark if requested
                all_tickers = [ticker] + competitors
                if include_benchmark:
                    all_tickers.append("SPY")  # S&P 500 ETF
                
                # Step 4: Fetch data for all tickers
                company_data = {}
                stock_data = {}
                metrics = {}
                
                for i, t in enumerate(all_tickers):
                    percent = 30 + (i / len(all_tickers)) * 60
                    status_text.text(f"Fetching data for {t}...")
                    progress_bar.progress(int(percent))
                    
                    # Get company info
                    company_data[t] = get_company_info(t)
                    
                    # Get stock data
                    stock_data[t] = get_stock_data(t, time_periods[period])
                    
                    # Calculate metrics
                    if not stock_data[t].empty:
                        metrics[t] = calculate_metrics(stock_data[t])
                    
                    time.sleep(0.5)  # Add a small delay for better UX
                
                # Save to session state
                st.session_state.company_data = company_data
                st.session_state.stock_data = stock_data
                st.session_state.metrics = metrics
                st.session_state.competitors = competitors
                st.session_state.all_tickers = all_tickers
                st.session_state.data_loaded = True
                
                status_text.text("Analysis complete!")
                progress_bar.progress(100)
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()
            else:
                st.error(f"Could not find information for ticker: {ticker}")
                return
        
        # Display company information
        company_info = st.session_state.company_data.get(ticker, {})
        company_name = company_info.get('shortName', ticker.upper())
        
        with main_col1:
            if 'longBusinessSummary' in company_info:
                st.markdown(f"### About {company_name}")
                st.markdown(f"<div class='highlight'>{company_info['longBusinessSummary']}</div>", unsafe_allow_html=True)
        
        with main_col2:
            if company_info:
                st.markdown("### Company Profile")
                profile_data = {
                    "Industry": company_info.get('industry', 'N/A'),
                    "Sector": company_info.get('sector', 'N/A'),
                    "Market Cap": f"${company_info.get('marketCap', 0)/1e9:.2f}B",
                    "Employees": f"{company_info.get('fullTimeEmployees', 0):,}",
                    "Country": company_info.get('country', 'N/A'),
                    "Website": company_info.get('website', 'N/A')
                }
                
                for key, value in profile_data.items():
                    st.markdown(f"**{key}**: {value}")
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "📊 Comparison", "💰 Financials", "📉 Risk Analysis"])
        
        with tab1:
            # Key metrics row
            if ticker in st.session_state.metrics:
                metrics_data = st.session_state.metrics[ticker]
                
                st.markdown("### Key Performance Indicators")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    current_price = metrics_data.get('current_price', 0)
                    st.markdown(f"<p class='metric-value'>${current_price:.2f}</p>", unsafe_allow_html=True)
                    st.markdown("<p class='metric-label'>Current Price</p>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    total_return = metrics_data.get('total_return', 0)
                    color_class = "positive" if total_return >= 0 else "negative"
                    st.markdown(f"<p class='metric-value {color_class}'>{total_return:.2f}%</p>", unsafe_allow_html=True)
                    st.markdown("<p class='metric-label'>Total Return</p>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col3:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    annual_return = metrics_data.get('annualized_return', 0)
                    color_class = "positive" if annual_return >= 0 else "negative"
                    st.markdown(f"<p class='metric-value {color_class}'>{annual_return:.2f}%</p>", unsafe_allow_html=True)
                    st.markdown("<p class='metric-label'>Annualized Return</p>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col4:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    volatility = metrics_data.get('volatility', 0)
                    st.markdown(f"<p class='metric-value neutral'>{volatility:.2f}%</p>", unsafe_allow_html=True)
                    st.markdown("<p class='metric-label'>Volatility (Annualized)</p>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Create performance chart
                st.markdown("### Historical Performance")
                
                # Using only ticker for individual performance
                perf_chart = create_performance_chart({ticker: st.session_state.stock_data[ticker]}, [ticker])
                if perf_chart:
                    st.plotly_chart(perf_chart, use_container_width=True)
                else:
                    st.warning("Not enough data to create performance chart.")
                
                # Statistics
                if show_statistics and ticker in st.session_state.metrics:
                    stats = st.session_state.metrics[ticker]
                    
                    st.markdown("### Detailed Statistics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Returns")
                        st.markdown(f"**Total Return:** {stats.get('total_return', 0):.2f}%")
                        st.markdown(f"**Annualized Return:** {stats.get('annualized_return', 0):.2f}%")
                        
                        # Calculate YTD return with timezone handling
                        if "data" in stats:
                            df = stats["data"]
                            last_day = df.index[-1]
                            
                            # Ensure year_start is timezone-aware matching df.index
                            year_start = pd.Timestamp(datetime(last_day.year, 1, 1), tz=df.index.tz)
                            
                            # Filter data for current year
                            year_data = df[df.index >= year_start]
                            
                            if not year_data.empty:
                                year_start_price = year_data['Close'].iloc[0]
                                ytd_return = (df['Close'].iloc[-1] / year_start_price - 1) * 100
                                st.markdown(f"**YTD Return:** {ytd_return:.2f}%")
                            else:
                                st.markdown("**YTD Return:** N/A")
                    
                    with col2:
                        st.markdown("#### Risk Metrics")
                        st.markdown(f"**Volatility (Annualized):** {stats.get('volatility', 0):.2f}%")
                        st.markdown(f"**Sharpe Ratio:** {stats.get('sharpe_ratio', 0):.2f}")
                        st.markdown(f"**Maximum Drawdown:** {stats.get('max_drawdown', 0):.2f}%")
        
        with tab2:
            st.markdown("### Comparative Analysis")
            
            # Display identified competitors
            competitors = st.session_state.competitors
            st.markdown(f"#### Competitors of {ticker.upper()}")
            
            if competitors and len(competitors) > 0:
                comp_cols = st.columns(min(len(competitors), 5))  # Limit to max 5 columns
                for i, comp in enumerate(competitors[:5]):  # Limit to 5 competitors
                    with comp_cols[i]:
                        comp_info = st.session_state.company_data.get(comp, {})
                        comp_name = comp_info.get('shortName', comp)
                        st.markdown(f"**{comp}**")
                        st.markdown(f"{comp_name}")
            else:
                st.info("No competitors identified for this company.")
            
            # Performance comparison chart
            st.markdown("#### Comparative Performance")
            
            perf_chart = create_performance_chart(st.session_state.stock_data, st.session_state.all_tickers)
            if perf_chart:
                st.plotly_chart(perf_chart, use_container_width=True)
            else:
                st.warning("Not enough data to create comparison chart.")
            
            # Risk-Return chart
            st.markdown("#### Risk-Return Analysis")
            
            risk_return_chart = create_returns_chart(st.session_state.metrics, st.session_state.all_tickers)
            if risk_return_chart:
                st.plotly_chart(risk_return_chart, use_container_width=True)
            else:
                st.warning("Not enough data to create risk-return chart.")
            
            # Summary metrics table
            st.markdown("#### Performance Metrics Comparison")
            
            metrics_df = pd.DataFrame(columns=["Company", "Current Price", "Total Return (%)", "Annual Return (%)", "Volatility (%)", "Sharpe Ratio", "Max Drawdown (%)"])
            
            for i, t in enumerate(st.session_state.all_tickers):
                if t in st.session_state.metrics:
                    m = st.session_state.metrics[t]
                    comp_info = st.session_state.company_data.get(t, {})
                    comp_name = comp_info.get('shortName', t)
                    
                    new_row = pd.DataFrame({
                        "Company": [comp_name],
                        "Ticker": [t],
                        "Current Price": [f"${m.get('current_price', 0):.2f}"],
                        "Total Return (%)": [f"{m.get('total_return', 0):.2f}%"],
                        "Annual Return (%)": [f"{m.get('annualized_return', 0):.2f}%"],
                        "Volatility (%)": [f"{m.get('volatility', 0):.2f}%"],
                        "Sharpe Ratio": [f"{m.get('sharpe_ratio', 0):.2f}"],
                        "Max Drawdown (%)": [f"{m.get('max_drawdown', 0):.2f}%"]
                    })
                    
                    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
            
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        with tab3:
            st.markdown("### Financial Analysis")
            
            # Fetch financial data if available
            if ticker in st.session_state.company_data and st.session_state.company_data[ticker]:
                fin_info = st.session_state.company_data[ticker]
                
                # Create financial summary
                fin_col1, fin_col2 = st.columns(2)
                
                with fin_col1:
                    st.markdown("#### Valuation Metrics")
                    
                    val_metrics = {
                        "Market Cap": f"${fin_info.get('marketCap', 0)/1e9:.2f}B",
                        "Enterprise Value": f"${fin_info.get('enterpriseValue', 0)/1e9:.2f}B",
                        "Trailing P/E": f"{fin_info.get('trailingPE', 'N/A')}",
                        "Forward P/E": f"{fin_info.get('forwardPE', 'N/A')}",
                        "PEG Ratio": f"{fin_info.get('pegRatio', 'N/A')}",
                        "Price/Sales": f"{fin_info.get('priceToSalesTrailing12Months', 'N/A')}",
                        "Price/Book": f"{fin_info.get('priceToBook', 'N/A')}",
                        "Enterprise Value/Revenue": f"{fin_info.get('enterpriseToRevenue', 'N/A')}",
                        "Enterprise Value/EBITDA": f"{fin_info.get('enterpriseToEbitda', 'N/A')}"
                    }
                    
                    for key, value in val_metrics.items():
                        st.markdown(f"**{key}:** {value}")
                
                with fin_col2:
                    st.markdown("#### Dividend & Yield")
                    
                    div_metrics = {
                        "Dividend Rate": f"${fin_info.get('dividendRate', 0):.2f}",
                        "Dividend Yield": f"{fin_info.get('dividendYield', 0)*100:.2f}%",
                        "Payout Ratio": f"{fin_info.get('payoutRatio', 0)*100:.2f}%",
                        "Ex-Dividend Date": fin_info.get('exDividendDate', 'N/A'),
                        "5 Year Avg Dividend Yield": f"{fin_info.get('fiveYearAvgDividendYield', 0):.2f}%"
                    }
                    
                    for key, value in div_metrics.items():
                        st.markdown(f"**{key}:** {value}")

                # Financial ratios comparison
                st.markdown("#### Key Financial Ratios")
                
                ratio_data = []
                ratio_columns = ["Ticker", "P/E Ratio", "PEG Ratio", "Price/Sales", "Price/Book", "ROE (%)", "ROA (%)", "Profit Margin (%)"]
                
                for t in st.session_state.all_tickers:
                    if t in st.session_state.company_data and st.session_state.company_data[t]:
                        t_info = st.session_state.company_data[t]
                        
                        ratio_data.append([
                            t,
                            t_info.get('trailingPE', 'N/A'),
                            t_info.get('pegRatio', 'N/A'),
                            t_info.get('priceToSalesTrailing12Months', 'N/A'),
                            t_info.get('priceToBook', 'N/A'),
                            t_info.get('returnOnEquity', 0) * 100 if t_info.get('returnOnEquity') else 'N/A',
                            t_info.get('returnOnAssets', 0) * 100 if t_info.get('returnOnAssets') else 'N/A',
                            t_info.get('profitMargins', 0) * 100 if t_info.get('profitMargins') else 'N/A'
                        ])
                
                if ratio_data:
                    ratio_df = pd.DataFrame(ratio_data, columns=ratio_columns)
                    st.dataframe(ratio_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Financial ratio data not available for comparison.")
                
                # Quarterly financials if available
                try:
                    quarterly_financials = yf.Ticker(ticker).quarterly_financials
                    if not quarterly_financials.empty:
                        st.markdown("#### Quarterly Financial Data")
                        st.dataframe(quarterly_financials)
                except:
                    st.info("Quarterly financial data not available.")
        
        with tab4:
            st.markdown("### Risk Analysis")
            
            # Drawdown chart
            st.markdown("#### Drawdown Analysis")
            st.write("This chart shows percentage decline from previous peak, highlighting maximum drawdowns.")
            
            drawdown_chart = create_drawdown_chart(st.session_state.metrics, st.session_state.all_tickers)
            if drawdown_chart:
                st.plotly_chart(drawdown_chart, use_container_width=True)
            else:
                st.warning("Not enough data to create drawdown chart.")
            
            # Volatility comparison
            st.markdown("#### Volatility Comparison")
            
            vol_data = []
            for t in st.session_state.all_tickers:
                if t in st.session_state.metrics:
                    vol_data.append({
                        'Ticker': t,
                        'Volatility': st.session_state.metrics[t].get('volatility', 0)
                    })
            
            if vol_data:
                vol_df = pd.DataFrame(vol_data)
                fig = px.bar(
                    vol_df, 
                    x='Ticker', 
                    y='Volatility',
                    title='Annualized Volatility (%)',
                    labels={'Volatility': 'Annualized Volatility (%)'},
                    template='plotly_white',
                    color='Volatility',
                    color_continuous_scale='Bluered_r'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Risk metrics table
            st.markdown("#### Risk Metrics Comparison")
            
            risk_data = []
            risk_columns = ["Ticker", "Volatility (%)", "Sharpe Ratio", "Max Drawdown (%)", "Beta", "Alpha (%)"]
            
            for t in st.session_state.all_tickers:
                if t in st.session_state.metrics and "SPY" in st.session_state.metrics and t != "SPY":
                    # Calculate beta and alpha if SPY data is available
                    t_data = st.session_state.metrics[t]["data"]
                    spy_data = st.session_state.metrics["SPY"]["data"]
                    
                    # Align dates
                    if not t_data.empty and not spy_data.empty:
                        common_dates = t_data.index.intersection(spy_data.index)
                        if len(common_dates) > 30:  # Need sufficient data points
                            t_returns = t_data.loc[common_dates]['Daily_Return'].values
                            spy_returns = spy_data.loc[common_dates]['Daily_Return'].values
                            
                            # Calculate beta (covariance / variance)
                            beta = np.cov(t_returns, spy_returns)[0, 1] / np.var(spy_returns)
                            
                            # Calculate alpha (annualized)
                            alpha = (st.session_state.metrics[t]['annualized_return'] - 
                                    beta * st.session_state.metrics["SPY"]['annualized_return'])
                        else:
                            beta = "N/A"
                            alpha = "N/A"
                    else:
                        beta = "N/A"
                        alpha = "N/A"
                    
                    risk_data.append([
                        t,
                        f"{st.session_state.metrics[t].get('volatility', 0):.2f}",
                        f"{st.session_state.metrics[t].get('sharpe_ratio', 0):.2f}",
                        f"{st.session_state.metrics[t].get('max_drawdown', 0):.2f}",
                        f"{beta:.2f}" if isinstance(beta, (int, float)) else beta,
                        f"{alpha:.2f}" if isinstance(alpha, (int, float)) else alpha
                    ])
            
            if risk_data:
                risk_df = pd.DataFrame(risk_data, columns=risk_columns)
                st.dataframe(risk_df, use_container_width=True, hide_index=True)
    
    # Footer
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown('**Disclaimer:** This app is for informational purposes only. The data provided should not be considered as financial advice.', unsafe_allow_html=True)
    st.markdown('Data source: Yahoo Finance | Last updated: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

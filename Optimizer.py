# # UTOR: FinTech Bootcamp - Project 1: Stock Portfolio Optimizer

# Import libraries
import numpy as np
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import plotly.express as px
import datetime
import plotly.graph_objects as go
import plotly.io as pio
from datetime import date, timedelta
import streamlit as st
from PIL import Image

import time
#import fix_yahoo_finance as fyf
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()

# Streamlit Setup

header = st.container()
data = st.container()
stock_graphs = st.container()
optimal_graphs = st.container()
analysis_graphs = st.container()




with header:

    banner2 = Image.open('AC22.png')
    #st.image(banner2)
    
    #col11, col12 = st.columns(2)
    
    #image = Image.open('arrow.png')   
    
    colb, colc = st.columns([1, 4.5])
    #colb.image(image, use_column_width=True)
    colb.image(banner2, width=100)
    
    #colc.markdown("<h1 style='text-align: left; color: Purple; padding-left: 0px; font-size: 60px'>ARROW-UP CAPITAL</h1>", unsafe_allow_html=True)
    colc.markdown("<h1 style='text-align: left; color: #008080; padding-left: 20px; font-size: 60px'><b>Ci Diversified Income<b></h1>", unsafe_allow_html=True)
   


    st.markdown("<h2 style='text-align: left; color: gray; padding-left: 0px; font-size: 50px'>Portfolio Optimizer</h2>", unsafe_allow_html=True)
    
    st.markdown(" ")
    
    #stock_list = ['AAPL', 'MSFT', 'UNH', 'JNJ', 'WMT', 'PG', 'JPM', 'V', 'CVX', 'HD', 'KO', 'MRK', 'DIS',
    #              'MCD', 'CSCO', 'VZ', 'CRM', 'NKE', 'AMGN', 'HON', 'INTC', 'IBM', 'AXP', 'GS', 'CAT', 'BA', 'MMM', 'TRV', 'DOW', 'WBA']
    

    stock_list = ['ABG.JO', 'ARI.JO', 'ANG.JO', 'APN.JO', 'BVT.JO', 'BTI.JO', 'CCO.JO', 'DSY.JO', 'EXX.JO', 'FSR.JO', 'GRT.JO', 
                      'IMP.JO', 'INP.JO', 'INL.JO', 'KIO.JO', 'LHC.JO', 'MNP.JO', 'MTN.JO', 'MCG.JO', 'NPN.JO', 'NED.JO', 'OMU.JO',
                      'RNI.JO', 'REM.JO', 'CFR.JO', 'RMH.JO', 'SLM.JO', 'SOL.JO', 'SHP.JO', 'SBK.JO', 'TBS.JO', 'VOD.JO', 'WHL.JO']


    
    selected_stocks = st.multiselect('Please choose your stocks', stock_list, default=['ABG.JO', 'NED.JO', 'SBK.JO'])
    
    selected_stocks = sorted(selected_stocks)
    
    col1, col2, col3 = st.columns(3)
    
    #theme1 = col4.selectbox('Theme', ["ggplot2", "seaborn", "simple_white"],index=1)
    out_of_sample = col3.number_input('Out of Sample', min_value=0, max_value=1260, step=1, value=252)
    
    #date1 = col1.date_input('Start Date',date(2008,1,1)).strftime("%Y/%m/%d")
    #date2 = col2.date_input('End Date', date(2022,9,30)).strftime("%Y/%m/%d")


    date1 = col1.date_input('Start Date',date(2008,1,1))
    date2 = col2.date_input('End Date', date(2023,5,31))
    
    


# ## User Input

# Yahoo finance API data input
#stocks = ['DIS', 'JNJ', 'HD', 'KO', 'NKE']
stocks = selected_stocks

stocks = sorted(stocks)

#benchmark = ['^DJI']
benchmark = '^JNX4.JO'

start_date = date1
end_date = date2

out_sample_days_held_back = out_of_sample

# Plotly graph themes
theme='seaborn'


# Set random seed - for optimal portfolio weight calculation
seed=42
np.random.seed(seed)

# Optimal portfolio calculation
number_opt_porfolios=10000

# Sharpe Calculation
rf = 0.01 # risk factor

init_investment = 10000



@st.cache_data
def get_data(stocks, start, end):
    dataframe = pdr.get_data_yahoo(stocks, start=start, end=end)
    dataframe = dataframe['Adj Close']
    return dataframe

    

@st.cache_data
def s_returns(stocks, df):
    stock_returns = pd.DataFrame()
    for stock in stocks:
        stock_returns[stock+'_Returns'] = df[stock].pct_change()
    stock_returns = stock_returns.dropna()
    return stock_returns


@st.cache_data
def b_returns(bm):
    benchmark_returns = pd.DataFrame()
    benchmark_returns['BM_Returns'] = bm[benchmark].pct_change()
    benchmark_returns = benchmark_returns.dropna()
    return benchmark_returns


#######################################################################################################################################
#######################################################################################################################################
 
########## write function ###########################

# ## Import and Clean Stock Data

# Import Stock data 

df = get_data(stocks, start_date, end_date)


@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

csv = convert_df(df)


with data:
    st.download_button(label="Download Stock Data", data=csv, file_name='stock_data.csv', mime='text/csv')


# Calculate Stocks peercentage change and slice dataframe

stock_returns = s_returns(stocks, df)


#Here we slice the dataframe to create backtesting data and out of sample data

count = df.shape[0] - out_sample_days_held_back

last_year_stock_returns = stock_returns.iloc[count:,:]
stock_returns_excl_ly = stock_returns.iloc[0:count,:]

# ## Import and Clean Benchmark Data


# Import Benchmark data

bm1 = get_data(benchmark, start_date, end_date)

bm = pd.DataFrame(bm1)

bm.rename(columns = {'Adj Close': benchmark}, inplace = True)


# Calculate Benchmarks peercentage change and slice dataframe

benchmark_returns = b_returns(bm)


#Here we slice the dataframe to create backtesting data and out of sample data

count = bm.shape[0] - out_sample_days_held_back

last_year_benchmark_returns = benchmark_returns.iloc[count:,:]
benchmark_returns_excl_ly = benchmark_returns.iloc[0:count,:]

# # Stock Dataframe graphs/Visualization

# Plot the retruns for each stock
@st.cache_data
def figure1(stock_returns):

    fig1 = go.Figure()

    for stock, col in enumerate(stock_returns.columns.tolist()):
        fig1.add_trace(
        go.Scatter(
            x=stock_returns.index,
            y=stock_returns[col],
            name=stocks[stock]
        ))
        
    fig1.update_layout(
        #title={
        #    'text': "Stock Returns",
        #},
        template=theme,
        xaxis=dict(autorange=True,
                title_text='Date'),
        yaxis=dict(autorange=True,
                title_text='Daily Returns'),
        margin=dict (l=0, r=0, t=0, b=0)
    )
    
    return fig1

figure1 = figure1(stock_returns)
figure1_data = convert_df(stock_returns)

# Plot a Boxplot of each stocks returns
@st.cache_data
def figure2(stock_returns):
    fig2 = go.Figure()

    for stock, col in enumerate(stock_returns.columns.tolist()):
        fig2.add_trace(
        go.Box(
            y=stock_returns[col],
            name=stocks[stock]
        ))


    fig2.update_layout(
        #title={
        #    'text': "Stock Returns Box Plot",
        #},
        template=theme,
        xaxis=dict(autorange=True,
                title_text='Stock'),
        yaxis=dict(autorange=True,
                title_text='Return Distribution'),
        margin=dict (l=0, r=0, t=0, b=0)
    )
    
    return fig2

figure2 = figure2(stock_returns)
figure2_data = convert_df(stock_returns)

# Calculate stocks cumulative returns


@st.cache_data
def cum_returns(stock_returns):
    cumulative_returns = (1 + stock_returns).cumprod()

    init_date = cumulative_returns.index[0] - timedelta(days=1)
    cumulative_returns.loc[init_date] = 1
    cumulative_returns = cumulative_returns.sort_index()
    return cumulative_returns


cumulative_returns = cum_returns(stock_returns)

# Plot the stocks cumulative returns
@st.cache_data
def figure3(cumulative_returns):
    fig3 = go.Figure()

    for stock, col in enumerate(cumulative_returns.columns.tolist()):
        fig3.add_trace(
        go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns[col],
            name=stocks[stock]
        ))


    fig3.update_layout(

        template=theme,
        xaxis=dict(autorange=True,
                title_text='Date'),
        yaxis=dict(autorange=True,
                title_text='Cumulative Return'),
        margin=dict (l=0, r=0, t=0, b=0)
    )
    return fig3

figure3 = figure3(cumulative_returns)
figure3_data = convert_df(cumulative_returns)


# Benchmark cumulative returns

cumulative_benchmark_returns = cum_returns(benchmark_returns)


# # Optimal Portfolio Weights Calculation


# Calculate Covariance matrix of the log stock returns. We use log as it produces marginally more accurate results.


@st.cache_data
def cov_matrix(df):
    cov_matrix = df.pct_change().apply(lambda x: np.log(1+x)).cov() 
    return cov_matrix

cov_matrix = cov_matrix(df)

# Calculate Correlation matrix of the log stock returns. We use log as it produces marginally more accurate results.
@st.cache_data
def corr_matrix(df):
    corr_matrix = df.pct_change().apply(lambda x: np.log(1+x)).corr() #read up and explain
    return corr_matrix

corr_matrix = corr_matrix(df)

# CAGR for individual companies 

dt = pd.to_datetime(start_date, format='%Y/%m/%d')
dt1 = pd.to_datetime(end_date, format='%Y/%m/%d')

yrs_full = ((dt1-dt).days)/365


@st.cache_data
def annual_returns(df):
    annual_returns = df.pct_change().apply(lambda x: (1+x)).cumprod().iloc[-1]**(1/yrs_full) - 1
    return annual_returns
    

bm_annual_returns = annual_returns(bm)

annual_returns = annual_returns(df)


@st.cache_data
def annual_std_dev(df):
    annual_std_dev = df.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(252))
    return annual_std_dev
    
bm_annual_std_dev = annual_std_dev(bm)

annual_std_dev = annual_std_dev(df)


#Concatenate the annual returns and standard deviation dataframes

risk_return = pd.concat([annual_returns, annual_std_dev], axis=1) # Creating a table for visualising returns and volatility of assets
risk_return.columns = ['Returns', 'Volatility']


# Setup lists to hold portfolio weights, returns and volatility

@st.cache_data
def efficient(stocks, annual_returns, cov_matrix):

    np.random.seed(seed)

    p_ret = [] # Define an empty array for portfolio returns
    p_vol = [] # Define an empty array for portfolio volatility
    p_weights = [] # Define an empty array for asset weights

    num_assets = len(stocks)
    num_portfolios = 10000


    # Calculate Portfolio weights for num_portfolios

    for portfolio in range(num_portfolios):
        weights = np.random.rand(num_assets)
        weights = weights/np.sum(weights)
        p_weights.append(weights)
        returns = np.dot(weights, annual_returns) # Returns are the product of individual expected returns of asset and its 
                                        # weights 
        p_ret.append(returns)
        var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()# Portfolio Variance
        sd = np.sqrt(var) # Daily standard deviation
        ann_sd = sd*np.sqrt(252) # Annual standard deviation = volatility
        p_vol.append(ann_sd)

    return p_ret, p_vol, p_weights

p_ret, p_vol, p_weights = efficient(stocks, annual_returns, cov_matrix)

# Insert the stock weights that correspon to the respective portfolio return and volatility

@st.cache_data
def portfolios1(stocks, p_ret, p_vol, p_weights):

    data = {'Returns':p_ret, 'Volatility':p_vol}

    for counter, symbol in enumerate(stocks):
        data[symbol+' weight'] = [w[counter] for w in p_weights]

    # Create portfolios dataframe to hold the portfolio weights of stocks, and portfolio return, volatility and sharpe ratio

    portfolios  = pd.DataFrame(data)

    portfolios['Sharpe'] = (portfolios['Returns']-rf)/portfolios['Volatility']
    
    return portfolios

portfolios = portfolios1(stocks, p_ret, p_vol, p_weights)

# Finding the optimal portfolio

optimal_risky_port = portfolios.iloc[(portfolios['Sharpe']).idxmax()]
    
bm_sharpe = (bm_annual_returns-rf)/bm_annual_std_dev

# Pie Chart of optimal portfolio stock weightings
@st.cache_data
def pie(optimal_risky_port):

    opt_port_df = pd.DataFrame(data={'Stocks': stocks, 'Weight': optimal_risky_port[2:-1].values})
    fig4 = px.pie(opt_port_df, values='Weight', names='Stocks',
                #title='Optimal Portfolio Stock Weighting',
                template=theme
                )
    fig4.update_traces(textposition='inside', textinfo='percent+label')
    fig4.update_layout(margin=dict (l=0, r=0, t=0, b=0))
    return fig4

figure4 = pie(optimal_risky_port)
figure4_data = convert_df(optimal_risky_port)

# create an optimal portfolio dataframe

@st.cache_data
def figure5(optimal_risky_port):
    optimal_port_df = pd.DataFrame(data={'Returns': optimal_risky_port[0], 'Volatility': optimal_risky_port[1]}, index=[0])
    #optimal_port_df

    # Plot efficient frontier with optimal portfolio

    fig5 = px.scatter(portfolios, x='Volatility', y='Returns', color='Sharpe',
                      #color_discrete_sequence=px.colors.qualitative.Antique,
                      color_continuous_scale=px.colors.sequential.Plasma
                      #title='Portfolio Efficient Frontier with Optimal Portfolio'
                      )
    fig5.add_trace(go.Scatter(x=optimal_port_df['Volatility'], y=optimal_port_df['Returns'], name='Optimal Portfolio',
                marker=dict(
                color='lightskyblue',
                size=20,
                )))

    fig5.update_layout(legend=dict(
        orientation="h",
        yanchor="top",
        y=1.1,
        xanchor="right",
        x=1
        ),
        template=theme,
        margin=dict (l=0, r=0, t=0, b=0)
    )
    
    return fig5

figure5 = figure5(optimal_risky_port)
figure5_data = convert_df(portfolios)

# Calculate Historical Optimal Portfolio Return using optimal weights

weights = optimal_risky_port[2:-1].values

weights_full = weights

portfolio_returns = stock_returns.dot(weights)

# Convert the historical optimal portfolio returns to a dataframe

port_returns = pd.DataFrame(portfolio_returns)
port_returns.columns = ['Portfolio_returns']

# Calculate the historical cumulative returns for the optimal portfolio

optimal_cumulative_returns = cum_returns(port_returns)


# Concatenate the optimal and benchmark daily returns

historic_daily_returns = pd.concat([benchmark_returns, port_returns], axis=1, join="inner")

historic_daily_returns['Date'] = historic_daily_returns.index
historic_daily_returns['Date'] = historic_daily_returns['Date'].dt.date

#Here we concatenate the historical optimal cumulative portfolio returns with that of the benchmark

historic_returns = pd.concat([cumulative_benchmark_returns, optimal_cumulative_returns], axis=1, join="inner")

historic_returns['Date'] = historic_returns.index
historic_returns['Date'] = historic_returns['Date'].dt.date

# Cumulative Historical Returns: Optimal Porfolio vs. Benchmark

@st.cache_data
def figure6(historic_returns):

    fig6 = go.Figure()

    fig6.add_trace(
        go.Scatter(
            x=historic_returns['Date'],
            y=historic_returns['Portfolio_returns'],
            name="Optimal Portfolio",
            line=dict(color="#33CFA5")
        ))

    fig6.add_trace(
        go.Scatter(
            x=historic_returns['Date'],
            y=historic_returns['BM_Returns'],
            name='Benchmark',
            line=dict(color="#bf00ff")
        ))


    fig6.update_layout(
        template=theme,
        xaxis=dict(autorange=True,
                title_text='Date'),
        yaxis=dict(autorange=True,
                title_text='Cumulative Returns'),
        margin=dict (l=0, r=0, t=0, b=0)
    )
    
    return fig6

figure6 = figure6(historic_returns)
figure6_data = convert_df(historic_returns)

@st.cache_data
def tables(init_investment, historic_returns, historic_daily_returns, optimal_risky_port, bm_annual_returns, bm_annual_std_dev, bm_sharpe):

    pm_start = init_investment
    bm_start = init_investment

    pm_end = round(init_investment * historic_returns['Portfolio_returns'][-1],2)
    bm_end = round(init_investment * historic_returns['BM_Returns'][-1],2)

    daily_pm_max_return = historic_daily_returns['Portfolio_returns'].max()
    daily_bm_max_retrun = historic_daily_returns['BM_Returns'].max()

    daily_pm_min_return = historic_daily_returns['Portfolio_returns'].min()
    daily_bm_min_retrun = historic_daily_returns['BM_Returns'].min()

    pm_return = optimal_risky_port[0]
    bm_return = bm_annual_returns[0]

    pm_vol = optimal_risky_port[1]
    bm_vol = bm_annual_std_dev[0]

    pm_sharpe = round(optimal_risky_port[-1],2)
    bm_sharpe = round(bm_sharpe,2)

    covariance = historic_daily_returns['Portfolio_returns'].cov(historic_daily_returns['BM_Returns'])
    variance = historic_daily_returns['BM_Returns'].var()
    pm_beta = round((covariance/variance),2)

    bm_beta = 1


    # Table of Descriptive Statistics

    head = ['<b>Statistic<b>', '<b>Optimal Portfolio<b>', '<b>Benchmark<b>']
    labels = ['<b>Initial Investment<b>', '<b>Ending Investment<b>', '<b>Max Daily Return<b>',
            '<b>Min Daily Return<b>', '<b>Return<b>', '<b>Volatility<b>', '<b>Sharpe Ratio<b>', '<b>Beta<b>']
    pf_stats = ['${:,}'.format(pm_start), '${:,}'.format(pm_end), '{:.2%}'.format(daily_pm_max_return), 
                '{:.2%}'.format(daily_pm_min_return), '{:.2%}'.format(pm_return), '{:.2%}'.format(pm_vol), pm_sharpe, pm_beta]
    bm_stats = ['${:,}'.format(bm_start),'${:,}'.format(bm_end), '{:.2%}'.format(daily_bm_max_retrun), 
                '{:.2%}'.format(daily_bm_min_retrun), '{:.2%}'.format(bm_return), '{:.2%}'.format(bm_vol), bm_sharpe, bm_beta]

    fig7 = go.Figure(data=[go.Table(
        header=dict(values=head,
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[labels, pf_stats, bm_stats],
                fill_color='lavender',
                align='left'))
    ])

    fig7.update_layout(margin=dict(l=0, r=0, b=0,t=0))
    
    return fig7

figure7 = tables(init_investment, historic_returns, historic_daily_returns, optimal_risky_port, bm_annual_returns, bm_annual_std_dev, bm_sharpe)

# Plot Daily Returns Optimal Portfolio vs. Benchmark

@st.cache_data
def beta(historic_daily_returns):

    fig8 = px.scatter(historic_daily_returns, x='BM_Returns', y='Portfolio_returns', 
                      )
    fig8.update_layout(template=theme, margin=dict (l=0, r=0, t=0, b=0))
    return fig8

figure8 = beta(historic_daily_returns)
figure8_data = convert_df(historic_daily_returns)


# # Calulate Optimal Portfolio weights using sliced dataframe and optimal portfolio performance for out of sample data

# Calculate Covariance matrix of the log stock returns. We use log as it produces marginally more accurate results.

@st.cache_data
def cov_matrix1(stock_returns_excl_ly):
    cov_matrix = stock_returns_excl_ly.apply(lambda x: np.log(1+x)).cov()
    return cov_matrix

cov_matrix = cov_matrix1(stock_returns_excl_ly)

# Calculate Correlation matrix of the log stock returns. We use log as it produces marginally more accurate results.
@st.cache_data
def corr_matrix1(stock_returns_excl_ly):
    corr_matrix = stock_returns_excl_ly.apply(lambda x: np.log(1+x)).corr()
    return corr_matrix

corr_matrix = corr_matrix1(stock_returns_excl_ly)
# CAGR for individual companies 

dt = pd.to_datetime(start_date, format='%Y/%m/%d')
dt2 = pd.to_datetime(stock_returns_excl_ly.index[-1], format='%Y-%m-%d')

yrs = ((dt2-dt).days)/365

@st.cache_data
def annual_returns1(stock_returns_excl_ly):
    annual_returns = stock_returns_excl_ly.apply(lambda x: (1+x)).cumprod().iloc[-1]**(1/yrs) - 1
    return annual_returns

bm_annual_returns = annual_returns1(benchmark_returns_excl_ly)
#bm_annual_returns = benchmark_returns_excl_ly.apply(lambda x: (1+x)).cumprod().iloc[-1]**(1/yrs) - 1

annual_returns = annual_returns1(stock_returns_excl_ly)
#annual_returns = stock_returns_excl_ly.apply(lambda x: (1+x)).cumprod().iloc[-1]**(1/yrs) - 1
#displayannual_returns

# Volatility is given by the annual standard deviation. We multiply by 252 because there are 252 trading days/year. Also
# We will use the log of the stock returns in our calculation as it produces mrginally more accurate results.

@st.cache_data
def annual_std_dev1(stock_returns_excl_ly):
    annual_std_dev = stock_returns_excl_ly.apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(252))
    return annual_std_dev


bm_annual_std_dev = annual_std_dev1(benchmark_returns_excl_ly)
#bm_annual_std_dev = benchmark_returns_excl_ly.apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(252))

annual_std_dev = annual_std_dev1(stock_returns_excl_ly)
#annual_std_dev = stock_returns_excl_ly.apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(252))

#Concatenate the annual returns and standard deviation dataframes

risk_return = pd.concat([annual_returns, annual_std_dev], axis=1) # Creating a table for visualising returns and volatility of assets
risk_return.columns = ['Returns', 'Volatility']


# Setup lists to hold portfolio weights, returns and volatility


####################################################################################################################
########################################################################################################################

p_ret, p_vol, p_weights = efficient(stocks, annual_returns, cov_matrix)
# Insert the stock weights that correspon to the respective portfolio return and volatility
######################################################################################################################

portfolios = portfolios1(stocks, p_ret, p_vol, p_weights)


optimal_risky_port = portfolios.iloc[(portfolios['Sharpe']).idxmax()]

bm_sharpe = (bm_annual_returns-rf)/bm_annual_std_dev

# Pie Chart of optimal portfolio stock weightings

figure9 = pie(optimal_risky_port)
figure9_data = convert_df(optimal_risky_port)

# Calculate Historical Optimal Portfolio Return using optimal weights

weights = optimal_risky_port[2:-1].values

weights_excl_ly = weights

portfolio_returns = stock_returns_excl_ly.dot(weights)

# Convert the historical optimal portfolio returns to a dataframe

port_returns = pd.DataFrame(portfolio_returns)
port_returns.columns = ['Portfolio_returns']


# Calculate the historical cumulative returns for the optimal portfolio

optimal_cumulative_returns = cum_returns(port_returns)

#Here we concatenate the historical optimal daily portfolio returns with that of the benchmark

sample_daily_historic_returns = pd.concat([benchmark_returns_excl_ly, port_returns], axis=1, join="inner")
sample_daily_historic_returns['Date'] = sample_daily_historic_returns.index
sample_daily_historic_returns['Date'] = sample_daily_historic_returns['Date'].dt.date


#Here we concatenate the historical optimal cumulative portfolio returns with that of the benchmark

sample_cumulative_historic_returns = pd.concat([cumulative_benchmark_returns, optimal_cumulative_returns], axis=1, join="inner")

sample_cumulative_historic_returns['Date'] = sample_cumulative_historic_returns.index
sample_cumulative_historic_returns['Date'] = sample_cumulative_historic_returns['Date'].dt.date


# ## Generate animated graph for out of sample data (Optimal Portfolio vs. Benchmark)

#Calculate portfolio returns for last year using updated weights

last_year_portfolio_returns = last_year_stock_returns.dot(weights_excl_ly)

# Convert calulated portfolio returns for last year into a dataframe

one_year_port_returns = pd.DataFrame(last_year_portfolio_returns)
one_year_port_returns.columns = ['Portfolio_returns']

# Calculate the cumulative returns for the optimal portfolio for the last year

one_year_cumulative_returns = cum_returns(one_year_port_returns)


one_year_benchmark_returns = cum_returns(last_year_benchmark_returns)

one_year_historic_daily_returns = pd.concat([last_year_benchmark_returns, one_year_port_returns], axis=1, join="inner")

one_year_historic_daily_returns['Date'] = one_year_historic_daily_returns.index
one_year_historic_daily_returns['Date'] = one_year_historic_daily_returns['Date'].dt.date


one_year_historic_returns = pd.concat([one_year_benchmark_returns, one_year_cumulative_returns], axis=1, join="inner")



one_year_historic_returns['Date'] = one_year_historic_returns.index
one_year_historic_returns['Date'] = one_year_historic_returns['Date'].dt.date


# Plot An animated graphs of the optimal vs. Benchmark cumulative returns

@st.cache_data
def animated(one_year_historic_returns):

    fig10 = go.Figure(
        layout=go.Layout(
            updatemenus=[dict(type="buttons", direction="left", x=0.1, y=1.13), ],
            xaxis=dict(autorange=True, 
                    title_text="Date"),
            yaxis=dict(autorange=True,
                    title_text="Returns"),
        ))

    # Add traces
    init = 0

    fig10.add_trace(
        go.Scatter(
            x=one_year_historic_returns['Date'][:init],
            y=one_year_historic_returns['Portfolio_returns'][:init],
            name='Optimal Portfolio Returns',
            line=dict(color="#33CFA5"),
            mode="lines",
            #line_shape = 'spline'
        ))


    fig10.add_trace(
        go.Scatter(
            x=one_year_historic_returns['Date'][:init],
            y=one_year_historic_returns['BM_Returns'][:init],
            name='Benchmark Returns',
            line=dict(color="#bf00ff"),
            mode="lines",
            #line_shape = 'spline'

        ))


    # Animation
    fig10.update(frames=[
        go.Frame(
            data=[
                go.Scatter(x=one_year_historic_returns['Date'][:k], y=one_year_historic_returns['Portfolio_returns'][:k]),
                go.Scatter(x=one_year_historic_returns['Date'][:k], y=one_year_historic_returns['BM_Returns'][:k])]

                
        )
        for k in range(init, one_year_historic_returns.shape[0]+1)])


    # Buttons
    fig10.update_layout(
        template=theme,
        margin=dict(l=0, r=0, b=0,t=0),
        updatemenus=[
            dict(
                buttons=list([
                    dict(label="Play",
                        method="animate",
                        args=[None, {"frame": {"duration": 10}}])
                ]))])
    
    return fig10

figure10 = animated(one_year_historic_returns)
figure10_data = convert_df(one_year_historic_returns)
#st.write(figure10)

#st.write(fig10)
#return fig10
#figs.append(fig10)

# ## In Sample and Out-of-Sample Descriptive Statistics

# Descriptive Statistics - Out of Sample Data
# one_year_historic_daily_returns
# one_year_historic_returns

@st.cache_data
def table2(init_investment, one_year_historic_returns, one_year_historic_daily_returns, yrs_full, yrs):

    pm_start = init_investment
    bm_start = init_investment

    pm_end = round(init_investment * one_year_historic_returns['Portfolio_returns'][-1],2)
    bm_end = round(init_investment * one_year_historic_returns['BM_Returns'][-1],2)

    daily_pm_max_return = one_year_historic_daily_returns['Portfolio_returns'].max()
    daily_bm_max_retrun = one_year_historic_daily_returns['BM_Returns'].max()

    daily_pm_min_return = one_year_historic_daily_returns['Portfolio_returns'].min()
    daily_bm_min_retrun = one_year_historic_daily_returns['BM_Returns'].min()

    one_yr = yrs_full - yrs

    pm_return = one_year_historic_daily_returns['Portfolio_returns'].apply(lambda x: (1+x)).cumprod().iloc[-1]**(1/one_yr) - 1
    bm_return = one_year_historic_daily_returns['BM_Returns'].apply(lambda x: (1+x)).cumprod().iloc[-1]**(1/one_yr) - 1

    pm_vol = one_year_historic_daily_returns['Portfolio_returns'].apply(lambda x: np.log(1+x)).std()*np.sqrt(252)
    bm_vol = one_year_historic_daily_returns['BM_Returns'].apply(lambda x: np.log(1+x)).std()*np.sqrt(252)

    pm_sharpe = round((pm_return-rf)/pm_vol,2)
    bm_sharpe = round((bm_return-rf)/bm_vol,2)

    covariance = one_year_historic_daily_returns['Portfolio_returns'].cov(one_year_historic_daily_returns['BM_Returns'])
    variance = one_year_historic_daily_returns['BM_Returns'].var()
    pm_beta = round((covariance/variance),2)

    bm_beta = 1


    # Table of Descriptive Statistics - Out of Sample Data

    head = ['<b>Statistic<b>', '<b>Optimal Portfolio<b>', '<b>Benchmark<b>']
    labels = ['<b>Initial Investment<b>', '<b>Ending Investment<b>', '<b>Max Daily Return<b>',
            '<b>Min Daily Return<b>', '<b>Return<b>', '<b>Volatility<b>', '<b>Sharpe Ratio<b>', '<b>Beta<b>']
    pf_stats = ['${:,}'.format(pm_start), '${:,}'.format(pm_end), '{:.2%}'.format(daily_pm_max_return), 
                '{:.2%}'.format(daily_pm_min_return), '{:.2%}'.format(pm_return), '{:.2%}'.format(pm_vol), pm_sharpe, pm_beta]
    bm_stats = ['${:,}'.format(bm_start), '${:,}'.format(bm_end), '{:.2%}'.format(daily_bm_max_retrun), 
                '{:.2%}'.format(daily_bm_min_retrun), '{:.2%}'.format(bm_return), '{:.2%}'.format(bm_vol), bm_sharpe, bm_beta]

    fig11 = go.Figure(data=[go.Table(
        header=dict(values=head,
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[labels, pf_stats, bm_stats],
                fill_color='lavender',
                align='left'))
    ])

    fig11.update_layout(margin=dict(l=0, r=0, b=0,t=0))
    
    return fig11

figure11 = table2(init_investment, one_year_historic_returns, one_year_historic_daily_returns, yrs_full, yrs)
#st.write(figure11)


#st.write(fig11)
#return fig11
#figs.append(fig11)

# Plot Daily Returns Optimal Portfolio vs. Benchmark - Out of Sample Data
##############################################################################################################
#fig12 = px.scatter(one_year_historic_daily_returns, x='BM_Returns', y='Portfolio_returns', 
#                title='Daily Returns Optimal Portfolio vs. Benchmark - Sample Data')
#fig12.update_layout(template=theme)
##############################################################################################################


figure12 = beta(one_year_historic_daily_returns)
figure12_data = convert_df(one_year_historic_daily_returns)
#st.write(figure12)

#st.write(fig12)
#return fig12
#figs.append(fig12)

#return figs

####### end function #################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
 




with stock_graphs:
    
    st.markdown(" ")
    st.markdown(" ")
    
    st.markdown("<h3 style='text-align: left; color: SteelBlue; padding-left: 0px; font-size: 50px'>Stock Analysis</h3>", unsafe_allow_html=True)
    
    graph_list = ['Daily Returns', 'Box Plots', 'Cumulative Returns', 'Efficient Frontier']
    graphs = st.selectbox('Stock Graphs', graph_list, index=0)
    
    if(graphs == 'Daily Returns'):
        #st.subheader('Daily Stock Returns')
        st.download_button(label="Download Graph Data", data=figure1_data, file_name='graph_data.csv', mime='text/csv')
        st.markdown("<h4 style='text-align: center; color: SlateGray;'>Daily Stock Returns</h4>", unsafe_allow_html=True)
        st.write(figure1)
    elif (graphs == 'Box Plots'):
        #st.subheader('Stock Return Distribution')
        st.download_button(label="Download Graph Data", data=figure2_data, file_name='graph_data.csv', mime='text/csv')
        st.markdown("<h4 style='text-align: center; color: SlateGray;'>Stock Return Distribution</h4>", unsafe_allow_html=True)
        st.write(figure2)
    elif (graphs == 'Cumulative Returns'):
        #st.subheader('Cumulative Stock Returns')
        st.download_button(label="Download Graph Data", data=figure3_data, file_name='graph_data.csv', mime='text/csv')
        st.markdown("<h4 style='text-align: center; color: SlateGray;'>Cumulative Stock Returns</h4>", unsafe_allow_html=True)
        st.write(figure3)
    elif (graphs == 'Efficient Frontier'):
        st.download_button(label="Download Graph Data", data=figure5_data, file_name='graph_data.csv', mime='text/csv')
        st.markdown("<h4 style='text-align: center; color: SlateGray;'>Portfolio Efficient Frontier</h4>", unsafe_allow_html=True)
        #st.subheader('Portfolio Efficient Frontier')
        st.write(figure5)
    else:
        st.error('Please Rerun')


with optimal_graphs:
    
    st.markdown(" ")
    st.markdown(" ")
    
    st.markdown("<h3 style='text-align: left; color: SteelBlue; padding-left: 0px; font-size: 50px'>Optimal Portfolio</h3>", unsafe_allow_html=True)
    
    graph_list2 = ['Pie Chart', 'Portfolio Returns', 'Beta', 'Descriptive Statistics']
    graphs2 = st.selectbox('Optimal Portfolio Graphs', graph_list2, index=0)
    
    if(graphs2 == 'Pie Chart'):
        #st.subheader('Optimal Portfolio Weights')
        st.download_button(label="Download Graph Data", data=figure4_data, file_name='graph_data.csv', mime='text/csv')
        st.markdown("<h4 style='text-align: center; color: SlateGray;'>Optimal Portfolio Weights</h4>", unsafe_allow_html=True)
        st.write(figure4)
    elif (graphs2 == 'Portfolio Returns'):
        #st.subheader('Optimal Portfolio Cumulative Return')
        st.download_button(label="Download Graph Data", data=figure6_data, file_name='graph_data.csv', mime='text/csv')
        st.markdown("<h4 style='text-align: center; color: SlateGray;'>Optimal Portfolio Cumulative Return</h4>", unsafe_allow_html=True)
        st.write(figure6)
    elif (graphs2 == 'Beta'):
        #st.subheader('Optimal Portfolio Beta')
        st.download_button(label="Download Graph Data", data=figure8_data, file_name='graph_data.csv', mime='text/csv')
        st.markdown("<h4 style='text-align: center; color: SlateGray;'>Optimal Portfolio Beta</h4>", unsafe_allow_html=True)
        st.write(figure8)
    elif (graphs2 == 'Descriptive Statistics'):
        st.markdown("<h4 style='text-align: center; color: SlateGray;'>Optimal Portfolio vs Benchmark Statistics</h4>", unsafe_allow_html=True)
        #st.subheader('Optimal Portfolio vs Benchmark Statistics')
        st.write(figure7)
    else:
        st.error('Please Rerun')
        
        
with analysis_graphs:
    
    st.markdown(" ")
    st.markdown(" ")
    st.markdown("<h3 style='text-align: left; color: SteelBlue; padding-left: 0px; font-size: 50px'>Out of Sample Statistics</h3>", unsafe_allow_html=True)
    
    graph_list3 = ['Sample Pie Chart', 'Sample Portfolio Returns', 'Sample Beta', 'Sample Descriptive Statistics']
    graphs2 = st.selectbox('Out of Sample Graphs', graph_list3, index=0)
    
    if(graphs2 == 'Sample Pie Chart'):
        #st.subheader('Optimal Portfolio Weights')
        st.download_button(label="Download Graph Data", data=figure9_data, file_name='graph_data.csv', mime='text/csv')
        st.markdown("<h4 style='text-align: center; color: SlateGray;'>Out of Sample Optimal Portfolio Weights</h4>", unsafe_allow_html=True)
        st.write(figure9)
    elif (graphs2 == 'Sample Portfolio Returns'):
        #st.subheader('Optimal Portfolio Cumulative Return')
        st.download_button(label="Download Graph Data", data=figure10_data, file_name='graph_data.csv', mime='text/csv')
        st.markdown("<h4 style='text-align: center; color: SlateGray;'>Out of Sample Portfolio Returns</h4>", unsafe_allow_html=True)
        st.write(figure10)
    elif (graphs2 == 'Sample Beta'):
        #st.subheader('Optimal Portfolio Beta')
        st.download_button(label="Download Graph Data", data=figure12_data, file_name='graph_data.csv', mime='text/csv')
        st.markdown("<h4 style='text-align: center; color: SlateGray;'>Out of Sample Optimal Portfolio Beta</h4>", unsafe_allow_html=True)
        st.write(figure12)
    elif (graphs2 == 'Sample Descriptive Statistics'):
        st.markdown("<h4 style='text-align: center; color: SlateGray;'>Out of Sample Optimal Portfolio vs Benchmark Statistics</h4>", unsafe_allow_html=True)
        #st.subheader('Optimal Portfolio vs Benchmark Statistics')
        st.write(figure11)
    else:
        st.error('Please Rerun')
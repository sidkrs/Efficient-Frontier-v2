import yfinance
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import webbrowser
import os
import pyfiglet
import time
from tabulate import tabulate
import sys
import plotly.graph_objects as go
from fpdf import FPDF
import zipfile
import binascii
import shutil



# Constants
INTEREST_RATE = 0.04 / 12  # 4% annual interest rate divided by 12 months
START_DATE = '2019-10-01'
END_DATE = '2024-09-02'

print(
    f"\nMonthly Interest Rate: {round(INTEREST_RATE*100, 2)}% ({INTEREST_RATE*12*100}% annually)\n"
)

TARGET = float(
    input("Enter a target return, enter a number as a percentage %: ")
) / 100  # Convert to decimal

# Get the benchmark ticker
benchmark_tickers = pd.DataFrame({
    "Number": [1, 2, 3, 4, 5, 6],
    "Benchmark": [
        "S&P 500", "Russell 1000", "Russell 2000",
        "Dow Jones Industrial Average", "NASDAQ Composite", "Russell 3000"
    ],
    "Ticker": ["^GSPC", "^RUI", "^RUT", "^DJI", "^IXIC", "^RUA"]
})
print('\n')
print(benchmark_tickers)
benchmark_choice = int(input("Enter a benchmark ticker number: "))

BENCHMARK = benchmark_tickers.iloc[benchmark_choice - 1, 2]

tickers = []
while True:
    ticker = input("Enter a ticker symbol (press enter again once finished): ")
    if ticker == "":
        break
    tickers.append(ticker.upper())


def get_data(tickers, start_date, end_date):
    data = yfinance.download(tickers, start=start_date,
                             end=end_date, interval='1mo')['Adj Close']
    benchmark_data = yfinance.download(BENCHMARK,
                                       start=start_date,
                                       end=end_date)['Adj Close']
    benchmark_data = benchmark_data.resample('MS').last()
    return data, benchmark_data


def calculate_returns(data):
    return data.pct_change().dropna()

def tmax_drawdown(cumulative_returns):
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown_value = drawdown.min()
    return max_drawdown_value

def get_stats(monthly_returns, interest_rate=INTEREST_RATE):
    average_returns = monthly_returns.mean()
    volatility = monthly_returns.std(ddof=0)
    correlation = monthly_returns.corr()
    cumulative_returns = (1 + monthly_returns).cumprod()
    max_drawdown_value = tmax_drawdown(cumulative_returns)
    sharpe_ratio = (average_returns - interest_rate) / volatility
    return average_returns, volatility, correlation, max_drawdown_value, sharpe_ratio


# Analytical solution for tangent portfolio
def analytical_tangent_portfolio(returns, cov_matrix, rf):
    n = len(returns)
    ones = np.ones(n)
    mu = returns - rf
    inv_cov = np.linalg.inv(cov_matrix)

    A = np.dot(np.dot(mu.T, inv_cov), ones)
    B = np.dot(np.dot(mu.T, inv_cov), mu)
    C = np.dot(np.dot(ones.T, inv_cov), ones)
    D = B * C - A * A

    weights = np.dot(inv_cov, mu) / A
    expected_return = np.dot(weights, returns)
    volatility = np.sqrt(np.dot(np.dot(weights, cov_matrix), weights))
    sharpe = (expected_return - rf) / volatility

    return weights, expected_return, volatility, sharpe


# Analytical solution for Global Minimum Variance Portfolio (GMVP)
def analytical_gmvp(cov_matrix):
    n = len(cov_matrix)
    ones = np.ones(n)
    inv_cov = np.linalg.inv(cov_matrix)

    weights = np.dot(inv_cov, ones) / np.dot(np.dot(ones.T, inv_cov), ones)
    return weights


def plot_efficient_frontier(results_df, tangent_return, tangent_volatility,
                            gmvp_return, gmvp_volatility, target_vol, TARGET,
                            tickers, volatility, average_returns,
                            INTEREST_RATE, START_DATE, END_DATE):
    # Set up Seaborn style
    sns.set_style("darkgrid")
    sns.set_palette("deep")

    # Create a custom color palette
    colors = sns.color_palette("viridis", as_cmap=True)

    # Plot the Efficient Frontier
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(results_df['Volatility'],
                          results_df['Return'],
                          c=results_df['Sharpe'],
                          cmap=colors,
                          alpha=0.7)
    plt.colorbar(scatter, label='Sharpe Ratio')
    plt.xlabel('Volatility (Standard Deviation)', fontsize=12)
    plt.ylabel('Expected Return', fontsize=12)
    plt.title('Efficient Frontier', fontsize=16)

    # Plot the Capital Market Line
    plt.plot([0, tangent_volatility, results_df['Volatility'].max()], [
        INTEREST_RATE, tangent_return, INTEREST_RATE +
        (tangent_return - INTEREST_RATE) / tangent_volatility *
        results_df['Volatility'].max()
    ],
             color='blue',
             linestyle='-',
             linewidth=2,
             label='Capital Market Line')

    # Plot the Tangent Portfolio, GMVP, Target Portfolio, and Assets
    plt.scatter(tangent_volatility,
                tangent_return,
                color='red',
                marker='*',
                s=200,
                label='Tangent Portfolio',
                edgecolors='black')
    plt.annotate(
        f'Tangent\nVol: {100*tangent_volatility:.2f}%\nReturn: {100*tangent_return:.2f}%',
        xy=(tangent_volatility, tangent_return),
        xytext=(10, 10),
        textcoords='offset points',
        ha='left',
        va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.scatter(gmvp_volatility,
                gmvp_return,
                color='green',
                marker='*',
                s=200,
                label='GMVP',
                edgecolors='black')
    plt.annotate(
        f'GMVP\nVol: {100*gmvp_volatility:.2f}%\nReturn: {gmvp_return*100:.2f}%',
        xy=(gmvp_volatility, gmvp_return),
        xytext=(10, -10),
        textcoords='offset points',
        ha='left',
        va='top',
        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.scatter(target_vol,
                TARGET,
                color='purple',
                marker='*',
                s=200,
                label='Target Portfolio',
                edgecolors='black')
    plt.annotate(
        f'Target\nVol: {100*target_vol:.2f}%\nReturn: {100*TARGET:.2f}%',
        xy=(target_vol, TARGET),
        xytext=(-10, 10),
        textcoords='offset points',
        ha='right',
        va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.scatter(0,
                INTEREST_RATE,
                color='blue',
                marker='*',
                s=200,
                label='Risk-Free Asset',
                edgecolors='black')

    for i in range(len(tickers)):
        plt.scatter(volatility.iloc[i],
                    average_returns.iloc[i],
                    marker='o',
                    s=100,
                    label=tickers[i],
                    edgecolors='black')

    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='upper left')
    plt.tight_layout()

    textstr = '\n'.join(
        (f'Interest Rate: {INTEREST_RATE*12*100:.2f}% (annual)',
         f'Start Date: {START_DATE}', f'End Date: {END_DATE}'))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.95,
             0.05,
             textstr,
             transform=plt.gca().transAxes,
             fontsize=9,
             verticalalignment='bottom',
             horizontalalignment='right',
             bbox=props)

    plt.savefig('Efficient_frontier.png', transparent=True)
    #plt.show()


def performance_vs_benchmark(monthly_returns, tangent_weights, gmvp_weights,
                             wt, wf, benchmark_returns):
    tangent_returns = monthly_returns.dot(tangent_weights)
    rf_returns = pd.Series([INTEREST_RATE] * len(monthly_returns),
                           index=monthly_returns.index)
    target_returns = wt * tangent_returns + wf * rf_returns
    gmvp_returns = monthly_returns.dot(gmvp_weights)
    tangent_cumulative = (1 + tangent_returns).cumprod()
    target_cumulative = (1 + target_returns).cumprod()
    benchmark_cumulative = (1 + benchmark_returns).cumprod()
    benchmark_name = yfinance.Ticker(BENCHMARK).info['shortName']
    gmvp_cumulative = (1 + gmvp_returns).cumprod()

    plt.figure(figsize=(12, 8))
    plt.plot(tangent_cumulative, label='Tangent Portfolio', color='red')
    plt.plot(target_cumulative, label='Target Portfolio', color='purple')
    plt.plot(benchmark_cumulative, label=f'{benchmark_name}', color='blue')
    plt.plot(gmvp_cumulative, label='GMVP Portfolio', color='green')
    plt.title(f'Performance vs. {benchmark_name}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()

    plt.savefig('Performance.png', transparent=True)
    #plt.show()

    # Create a figure
    fig = go.Figure()

    # Add the tangent portfolio trace
    fig.add_trace(
        go.Scatter(x=tangent_cumulative.index,
                   y=tangent_cumulative.values,
                   mode='lines',
                   name='Tangent Portfolio',
                   line=dict(color='red')))

    # Add the target portfolio trace
    fig.add_trace(
        go.Scatter(x=target_cumulative.index,
                   y=target_cumulative.values,
                   mode='lines',
                   name='Target Portfolio',
                   line=dict(color='purple')))

    # Add the GMVP portfolio trace
    fig.add_trace(
        go.Scatter(x=gmvp_cumulative.index,
                   y=gmvp_cumulative.values,
                   mode='lines',
                   name='GMVP Portfolio',
                   line=dict(color='green')))

    # Add the benchmark portfolio trace
    fig.add_trace(
        go.Scatter(x=benchmark_cumulative.index,
                   y=benchmark_cumulative.values,
                   mode='lines',
                   name=benchmark_name,
                   line=dict(color='blue')))

    # Update the layout
    fig.update_layout(title=f'Performance vs. {benchmark_name}',
                      xaxis_title='Date',
                      yaxis_title='Cumulative Returns',
                      legend=dict(x=0, y=1, traceorder='normal'),
                      width=800,
                      height=600)

    # Show the plot
    fig.show()

    return tangent_cumulative, target_cumulative, gmvp_cumulative, benchmark_cumulative 


def main():
    monthly_data, benchmark_data = get_data(tickers, START_DATE, END_DATE)

    monthly_returns = calculate_returns(monthly_data)
    benchmark_returns = calculate_returns(benchmark_data)
    average_returns, volatility, correlation, max_drawdown, sharpe_ratio = get_stats(
        monthly_returns)

    summary_data = pd.DataFrame({
        "Average Returns": average_returns,
        "Volatility": volatility,
        "Max Drawdown": max_drawdown,
        "Sharpe Ratio": sharpe_ratio
    })
    print('\nSummary Data:')
    print(summary_data.T)
    print('\nCorrelation Matrix:')
    print(correlation)

    cov_matrix = monthly_returns.cov(ddof=0)

    # Analytical solution for tangent portfolio
    tangent_weights, tangent_return, tangent_volatility, tangent_sharpe = analytical_tangent_portfolio(
        average_returns, cov_matrix, INTEREST_RATE)

    # Analytical solution for GMVP
    gmvp_weights = analytical_gmvp(cov_matrix)
    gmvp_return = np.dot(gmvp_weights, average_returns)
    gmvp_volatility = np.sqrt(
        np.dot(np.dot(gmvp_weights, cov_matrix), gmvp_weights))

    # Generate efficient frontier
    num_portfolios = 500000
    results = np.zeros((num_portfolios, 3))
    n_assets = len(tickers)

    for i in range(num_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)
        p_return = np.dot(weights, average_returns)
        p_volatility = np.sqrt(np.dot(np.dot(weights, cov_matrix), weights))
        results[i] = [
            p_return, p_volatility, (p_return - INTEREST_RATE) / p_volatility
        ]

    results_df = pd.DataFrame(results,
                              columns=['Return', 'Volatility', 'Sharpe'])

    target_vol = (TARGET - INTEREST_RATE) / (
        (tangent_return - INTEREST_RATE) / tangent_volatility)

    # get max drawdown of gmvp, tangent, target, benchmark


    portfolio_data = pd.DataFrame(
        {
            "Tangent Portfolio":
            [tangent_return, tangent_volatility, tangent_sharpe],
            "GMVP": [
                gmvp_return, gmvp_volatility,
                (gmvp_return - INTEREST_RATE) / gmvp_volatility
            ],
            "Target Portfolio":
            [TARGET, target_vol, (TARGET - INTEREST_RATE) / target_vol],
        },
        index=["Expected Return", "Volatility", "Sharpe Ratio"])

    print('\nPortfolio Data:')
    print(portfolio_data)

    print("\nTangent Portfolio Weights:")
    for asset, weight in zip(tickers, tangent_weights):
        print(f"{asset}: {weight:.4f}")

    print("\nGMVP Weights:")
    for asset, weight in zip(tickers, gmvp_weights):
        print(f"{asset}: {weight:.4f}")

    print("\nTarget Portfolio Weights:")
    rf = INTEREST_RATE
    rt = tangent_return
    rp = TARGET
    wf = (rt - rp) / (rt - rf)
    wt = 1 - wf
    for asset, weight in zip(tickers, tangent_weights):
        print(f"{asset}: {wt*weight:.4f}")
    print(f"Risk-Free Asset: {wf:.4f}")
    print('\n')

    plot_efficient_frontier(results_df, tangent_return, tangent_volatility,
                            gmvp_return, gmvp_volatility, target_vol, TARGET,
                            tickers, volatility, average_returns,
                            INTEREST_RATE, START_DATE, END_DATE)


    # Save the monthly returns to a csv file
    monthly_returns.to_csv('/Users/siddarthkerkar/Documents/Financial Risk Management/Assignment 1/stock_returns.csv')

    
    tangent_cumulative, target_cumulative, gmvp_cumulative, benchmark_cumulative = performance_vs_benchmark(
        monthly_returns, tangent_weights, gmvp_weights, wt, wf, benchmark_returns)

    gmvp_max_drawdown = tmax_drawdown(gmvp_cumulative)
    tangent_max_drawdown = tmax_drawdown(tangent_cumulative)
    target_max_drawdown = tmax_drawdown(target_cumulative)
    benchmark_max_drawdown = tmax_drawdown(benchmark_cumulative)

    portfolio_data = pd.DataFrame(
        {
            "Tangent Portfolio":
            [tangent_return, tangent_volatility, tangent_sharpe, tangent_max_drawdown],
            "GMVP": [
                gmvp_return, gmvp_volatility,
                (gmvp_return - INTEREST_RATE) / gmvp_volatility
            , gmvp_max_drawdown],
            "Target Portfolio":
            [TARGET, target_vol, (TARGET - INTEREST_RATE) / target_vol, target_max_drawdown],
        },
        index=["Expected Return", "Volatility", "Sharpe Ratio", "Max Drawdown"])



    # Open the text file in write mode
    with open('stats.txt', 'w') as f:
        # ASCII art header using pyfiglet
        f.write(pyfiglet.figlet_format("Portfolio Summary"))

        # Write the selected target Return
        f.write(f"Target Return: {TARGET*100:.2f}%\n")
        f.write(f"Interest Rate: {INTEREST_RATE*100:.2f}% (monthly)\n")
        f.write(
            f"Benchmark: {benchmark_tickers.iloc[benchmark_choice - 1, 1]}\n\n"
        )

        # Format summary data using tabulate
        f.write(pyfiglet.figlet_format("Summary Data", font="small"))
        summary_data_table = tabulate(summary_data.T,
                                      headers="keys",
                                      tablefmt="grid")
        f.write(summary_data_table + '\n')

        # Format correlation matrix
        f.write(pyfiglet.figlet_format("Correlation Matrix", font="small"))
        correlation_table = tabulate(correlation,
                                     headers="keys",
                                     tablefmt="grid")
        f.write(correlation_table + '\n')

        # Portfolio data
        f.write(pyfiglet.figlet_format("Portfolio Data", font="small"))
        portfolio_data_table = tabulate(portfolio_data,
                                        headers="keys",
                                        tablefmt="grid")
        f.write(portfolio_data_table + '\n')

        portfolio_weights_df = pd.DataFrame(
            {
                "Tangent Portfolio":
                list(tangent_weights) +
                [0],  # Risk-Free Asset weight is 0 for Tangent Portfolio
                "GMVP":
                list(gmvp_weights) +
                [0],  # Risk-Free Asset weight is 0 for GMVP
                "Target Portfolio":
                [wt * weight for weight in tangent_weights] +
                [wf]  # Target Portfolio includes risk-free weight
            },
            index=tickers +
            ["Risk-Free Asset"])  # Add "Risk-Free Asset" to index

        # Rename index and add pyfiglet header
        f.write(pyfiglet.figlet_format("Portfolio Weights", font="small"))

        # Format the portfolio weights DataFrame using tabulate
        portfolio_weights_table = tabulate(portfolio_weights_df.T,
                                           headers="keys",
                                           tablefmt="grid")
        f.write(portfolio_weights_table + '\n')

        # Print out the cumulative return performance of each portfolio
        f.write(pyfiglet.figlet_format("Portfolio Performance", font="small"))
    
        # Creating the data for tabulation as a list of lists
        data = [
            ["Benchmark", benchmark_cumulative.iloc[-1]],
            ["Tangent Portfolio", tangent_cumulative.iloc[-1]],
            ["GMVP", gmvp_cumulative.iloc[-1]],
            ["Target Portfolio", target_cumulative.iloc[-1]]
        ]
        
        # Using tabulate to create the table
        performance_table = tabulate(data, headers=["Portfolio", "Cumulative Return"], tablefmt="grid")
        
        f.write(performance_table + '\n')

        # Add disclaimer
    disclaimer = '''
    +--------------------------------------------------------------------------------------------------------------------------------------+
    |                                                               INVESTMENT DISCLAIMER                                                  |
    |                                                                                                                                      |
    | SK Investments provides investment advice with the goal of helping you achieve your financial objectives, but it is important to     |
    | understand that all investments carry risks, including the potential loss of principal. While we strive to recommend strategies that |
    | align with your goals, market fluctuations, economic conditions, and other factors may affect performance. Past performance is not   |
    | a guarantee of future results, and certain investments, such as leveraged products or international securities, may involve          |
    | additional risks. SK Investments is not liable for any losses incurred from following our advice, and we encourage you to fully      |
    | evaluate the risks before making any investment decisions.                                                                           |
    +--------------------------------------------------------------------------------------------------------------------------------------+
    '''

    with open('stats.txt', 'a') as f:
        f.write(disclaimer)


    # Open the text file in read mode
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, 'stats.txt')
    img1_path = os.path.join(current_directory, 'Efficient_frontier.png')
    img2_path = os.path.join(current_directory, 'Performance.png')

    # Open the file in the default web browser
    webbrowser.open(f"file://{file_path}")
    webbrowser.open(f"file://{img1_path}")
    webbrowser.open(f"file://{img2_path}")
    with open('stats.txt', 'r') as f:
        # Read and print the contents of the file
        print(f.read())

    # save as a zip file
    with zipfile.ZipFile('Portfolio.zip', 'w') as zipf:
        zipf.write('stats.txt')
        zipf.write('Efficient_frontier.png')
        zipf.write('Performance.png')
    



if __name__ == '__main__':
    main()

# Efficient-Frontier-v2

## Overview
This Python program performs efficient portfolio optimization based on Modern Portfolio Theory. It calculates the efficient frontier, determines the tangent portfolio, Global Minimum Variance Portfolio (GMVP), and a target return portfolio. The program also compares the performance of these portfolios against a chosen benchmark.

## Features
- Downloads historical stock data using yfinance
- Calculates returns, volatility, correlation, and other key statistics
- Generates the efficient frontier
- Determines the tangent portfolio and GMVP analytically
- Creates a target return portfolio
- Compares portfolio performance against a chosen benchmark
- Generates visualizations of the efficient frontier and portfolio performance
- Outputs detailed statistics and portfolio weights
- Creates a ZIP file with all results

## Requirements
- Python 3.x
- Required Python packages:
  - yfinance
  - pandas
  - numpy
  - seaborn
  - matplotlib
  - pyfiglet
  - tabulate
  - plotly
  - fpdf

## Usage
1. Run the script:
   ```
   python efficient_portfolio.py
   ```
2. Enter the required information when prompted:
   - Target return (as a percentage)
   - Benchmark ticker number
   - Stock tickers for portfolio optimization

3. The program will generate:
   - A text file (`stats.txt`) with detailed portfolio statistics
   - PNG images of the efficient frontier and performance comparison
   - A ZIP file containing all output files

4. Results will automatically open in your default web browser for viewing.

## Output
- `stats.txt`: Detailed portfolio statistics and weights
- `Efficient_frontier.png`: Visualization of the efficient frontier
- `Performance.png`: Performance comparison of portfolios vs benchmark
- `Portfolio.zip`: Compressed file containing all output files

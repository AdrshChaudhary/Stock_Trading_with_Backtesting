# Stock Trading with Backtesting

## Overview
The **Stock Trading with Backtesting** project provides a robust framework for algorithmic trading and backtesting strategies using Python. This web application allows users to analyze and simulate trading strategies with historical market data.

## Live Demo
You can access the live application [here](https://stock-trading-with-backtesting.onrender.com/).

## Features
- **User-Friendly Interface**: An intuitive web interface for easy navigation and interaction.
- **Real-Time Trading Simulation**: Test trading strategies using real-time data.
- **Backtesting Capabilities**: Evaluate strategies against historical data to gauge performance.
- **Performance Visualization**: Visual tools for analyzing trade results, including graphs and charts.

## Installation
To set up the project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AdrshChaudhary/Stock_Trading_with_Backtesting.git
   cd Stock_Trading_with_Backtesting

2. **Install the necessary dependencies:**
   ```bash
   pip install -r requirements.txt
   
3. **Configure your settings:**
   Modify the .env file to set your API keys and other parameters.
   
5. **Run the application:**
   ```bash
    python main.py

## Docker Setup
To run the application using Docker, follow these steps:

1. **Build the Docker image**:
   ```bash
   docker build -t stock-trading-backtesting .

2. **Run the Docker container:**:
   ```bash
   docker run -p 5000:5000 stock-trading-backtesting

3. **Access the application at http://localhost:5000 in your web browser.**


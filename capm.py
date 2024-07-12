from datetime import date
import numpy as np
import pandas as pd
import yfinance as yf

risk_free_return = 0.105

def capm(start, ticker1, ticker2):
    #Get stock data
    stock1 = yf.download(ticker1, start)
    #Get market data
    stock2 = yf.download(ticker2, start)


    #Resample for monthly data
    return_s1 = stock1.resample('ME').last()
    return_s2 = stock2.resample('ME').last()
    
    #Create a dataframe with the adjusted close
    data = pd.DataFrame({'s_adjclose' : return_s1['Adj Close'], 'm_adjclose': return_s2['Adj Close']}, index=return_s1.index)
    
    #Calc the stock and market retuens by computing log(n)/log(n-1)
    data[['s_returns','m_returns']] = np.log(data[['s_adjclose', 'm_adjclose']]/data[['s_adjclose', 'm_adjclose']].shift(1))
    
    #Drop null values
    data = data.dropna()

    #Generate covarience matrix
    covmat = np.cov(data["s_returns"], data["m_returns"])
    
    #Calc beta from matrix
    beta = covmat[0,1]/covmat[1,1]

    print("Beta from formula: ", beta)

    #Calc beta from regression
    beta, alpha = np.polyfit(data["m_returns"], data["s_returns"], deg=1)
    print("Beta from regression: ", beta)

    #Calc expected return
    expected_return = risk_free_return + beta*(data["m_returns"].mean()*12-risk_free_return)
    print("Expected Return: ",expected_return)

if __name__ == "__main__":
    capm("2024-01-01", 'bova11.sa', "itub4.SA")


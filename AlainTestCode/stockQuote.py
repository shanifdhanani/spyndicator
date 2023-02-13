
#https://twelvedata.com/docs#time-series
#This API call returns meta and time series for the requested instrument. Metaobject consists of general information about the requested symbol. 
#Time series is the array of objects ordered by time descending with Open, High, Low, Close prices. Non-currency instruments also include volume information.
# Symbol: ticker of the instrument. E.g. AAPL, EUR/USD, ETH/BTC, ...
# Interval: between two consecutive points in time series. Supports: 1min, 5min, 15min, 30min, 45min, 1h, 2h, 4h, 1day, 1week, 1month
# Output: number of data points to retrieve. Supports values in the range from 1 to 5000. Default 30 when no date parameters are set, otherwise set to maximum


from twelvedata import TDClient

class myQuote:
    
    def __init__(self, _symbol, _interval, _outputsize): 
        self.td = TDClient(apikey="47f35dc2431342ad974b8cca594e69e7")
        self.s=_symbol
        self.i=_interval
        self.o=_outputsize 

    def getQuote(self):
        self.ts=self.td.time_series( symbol=self.s, interval=self.i, outputsize=self.o)
        json_data = self.ts.as_json()
        return json_data




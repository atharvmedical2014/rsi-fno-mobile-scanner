
RSI BULLISH + BEARISH DIVERGENCE — NSE F&O MOBILE WEB
============================================

PURPOSE
-------
Mobile/browser version of the RSI Bullish Divergence scanner.

UNIVERSE
--------
- NSE individual F&O stocks ONLY.
- NIFTY / BANKNIFTY / FINNIFTY / MIDCPNIFTY are excluded.
- The app attempts to load the current F&O universe from NSE's permitted lot-size file.
- If NSE is temporarily unavailable, it tries a fallback F&O lot-size table.
- The count can change when NSE introduces/excludes contracts.

FEATURES
--------
- Monthly
- Weekly
- Daily
- 4 Hour
- RSI bullish divergence on closing basis
- Pivot left/right settings
- Previous-pivot search bars
- Minimum RSI difference
- Minimum price lower %
- STRICT BOTH RSI <30 checkbox
- Signal history / backtest years
- Mobile-friendly result table
- CSV download
- Parallel scanning
- Closed-candle confirmation

LOCAL TEST
----------
Install Python 3.12/3.13, then:

    pip install -r requirements.txt
    streamlit run app.py

Then open the shown local URL.

STREAMLIT COMMUNITY CLOUD DEPLOY
--------------------------------
1. Create a GitHub repository.
2. Upload these files:
   app.py
   requirements.txt
   runtime.txt

3. Open Streamlit Community Cloud.
4. Choose "New app".
5. Select your GitHub repo and app.py.
6. Deploy.

After deployment, Streamlit gives you one web link.
Open that link from Android, iPhone, PC or tablet.

IMPORTANT
---------
The app does not place trades and does not connect to a broker.
It is a scanner/backtest interface only.

F&O eligibility changes periodically. The app therefore refreshes the current
F&O list instead of permanently freezing an old count.


V2 - BULLISH + BEARISH
----------------------
GUI now has:
- Both
- Bullish
- Bearish

Bullish divergence:
- current price pivot low < previous price pivot low
- current RSI pivot > previous RSI pivot
- optional STRICT mode: both RSI pivots < 30

Bearish divergence:
- current price pivot high > previous price pivot high
- current RSI pivot < previous RSI pivot
- optional STRICT mode: both RSI pivots > 70

The result table includes a Type column: BULLISH / BEARISH.

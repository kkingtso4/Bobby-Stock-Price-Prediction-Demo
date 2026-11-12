# Bobby Stock Price Prediction Demo 📈

A complete Streamlit web application that compares the annual trading performance of **three simulated investors** (`$100,000` initial budget, trading on the **first trading day of each week** over the **most recent 365 calendar days**) using **5 years of Yahoo Finance daily stock data**:

1. **Investor 1 — SMA Crossover Strategy (`Short MA = 5`, `Long MA = 20`)**:
   - Uses only the daily stock data within the **most recent 365 calendar days**.
   - Buys on a **Golden Cross** (`SMA_5` crosses above `SMA_20`) and sells on a **Death Cross** (`SMA_5` crosses below `SMA_20`).
   - Scales trade volume dynamically based on signal criticality (`|SMA_5 - SMA_20| / Close`).
2. **Investor 2 — ML Linear Regression**:
   - Uses the **past 4 years of data** (excluding the most recent 365 calendar days) to train an OLS Linear Regression model mapping the **price movement of the past 5 days** (`x_1 ... x_5`) to the **average price movement of the next 5 days** (`y`).
   - Simulates weekly trading on the **most recent 365 calendar days** (`y_pred > 0` = Buy, `y_pred < 0` = Sell, volume scaled by `|y_pred|`).
   - Generates an automated **Win/Loss Explanation** at the end.
3. **Investor 3 — CNN Graphical Analysis**:
   - Uses the **past 4 years of data** to render **5-day Candlestick chart images (`32x32x3` RGB)** and labels each chart with **"Buy"** (`next 5-day avg movement > 0`) or **"Sell"** (`<= 0`) plus the **absolute effect size** (`|next 5-day avg movement|`).
   - Trains a dual-head 2D Convolutional Neural Network (`25,730` trainable parameters) and simulates weekly trades on the **most recent 365 calendar days**.
   - Generates an automated **Win/Loss Explanation** including all **hyperparameters used** and the **layer-by-layer number of parameters**.

---

## 🔗 How to Connect Your Google Account across Google Colab, GitHub, and Streamlit.io

### Step 1: Open in Google Colab (Connected to Your Google Account)
1. Sign in to [Google Colab](https://colab.research.google.com/) with your **Google Account**.
2. Upload `Bobby_Stock_Prediction_Colab.ipynb` (included in this bundle) or click **File -> Save a copy in Drive** so your project is stored in your Google Drive.
3. Run Cell 1 & Cell 2 in the Colab notebook to generate `app.py` and `requirements.txt` inside your Google Drive workspace.

### Step 2: Push from Colab to GitHub (Connected to Your Google Account)
1. Sign in to [GitHub](https://github.com/) (you can link your Google email address under **Settings -> Emails**) and create a new public repository named `bobby-stock-price-prediction-demo`.
2. In Google Colab, either:
   - Click **File -> Save a copy in GitHub** and select `bobby-stock-price-prediction-demo`, **OR**
   - Upload `app.py` and `requirements.txt` directly via **Add file -> Upload files** on your GitHub repository page, **OR**
   - Run Cell 4 in `Bobby_Stock_Prediction_Colab.ipynb` to push `app.py` and `requirements.txt` via Git CLI.

### Step 3: Deploy on Streamlit Community Cloud (`share.streamlit.io`) with Google Account Sign-In
1. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and click **Continue with Google** (or **Continue with GitHub**) to sign in with your Google Account.
2. Under **Workspace Settings -> Connected accounts**, connect your GitHub account.
3. Click **Create app -> Deploy a public app from GitHub**:
   - **Repository**: `<your-github-username>/bobby-stock-price-prediction-demo`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **Deploy!** Streamlit Community Cloud will automatically read `requirements.txt`, install `yfinance`, `scikit-learn`, `tensorflow-cpu`, and `plotly`, and launch your live public web app URL.

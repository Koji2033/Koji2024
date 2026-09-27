# iPad 株価チャート・売買シミュレーター

## できること

- 日本株・米国株などの株価データを Yahoo Finance 経由で取得
- ローソク足、出来高、移動平均線、RSI を表示
- 移動平均クロス戦略のバックテスト
- 当日終値でシグナル判定 → 翌取引日の始値で約定
- 手数料・スリッページを設定可能
- 資産曲線、最大ドローダウン、勝率、売買履歴を表示
- 指定した購入日・売却日・株数による1往復売買シミュレーション
- 価格データ・売買履歴をCSVで保存

## 銘柄コードの例

- トヨタ自動車: `7203.T`
- ソニーグループ: `6758.T`
- Apple: `AAPL`
- Microsoft: `MSFT`

---

# Streamlit Community CloudでiPadから使う

1. Streamlit Community CloudにGitHubアカウントでサインイン
2. Repository に `Koji2033/Koji2024` を指定
3. Branch は `main`
4. Main file path は `streamlit_app.py`
5. Deploy
6. 発行されたURLをiPadのSafariで開く
7. Safariの共有メニューから「ホーム画面に追加」

---

# Windows PCで起動して同じWi-Fi内のiPadから使う

## 1. Python環境を作る

```text
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## 2. 起動

`run_windows.bat` をダブルクリックするか、

```text
python -m streamlit run streamlit_app.py
```

を実行します。

## 3. PCのIPアドレスを確認

```text
ipconfig
```

PCが `192.168.1.25` の場合、iPadのSafariで

```text
http://192.168.1.25:8501
```

を開きます。

---

# 売買シミュレーションの考え方

移動平均クロスでは、当日終値でシグナルを確定し、翌取引日の始値で売買します。

これにより、当日終値を知ったうえで同日の始値に約定したことにする先読みを避けています。

## コスト

- 手数料: 売買金額に対する割合
- スリッページ: 表示価格と想定約定価格の差

買い: `始値 × (1 + スリッページ率)`

売り: `始値 × (1 - スリッページ率)`

---

# 注意事項

Yahoo Finance/yfinanceは証券取引所の公式リアルタイムフィードではありません。
データの遅延、欠損、価格調整、仕様変更があり得ます。

このアプリは学習・分析・バックテスト用途です。
実際の証券会社への注文は行いません。

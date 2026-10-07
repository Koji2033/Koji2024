# GitHub + Streamlit クラウド開発環境

## 構成
ChatGPT Work（クラウドで編集・検証） → GitHub（ソース管理） → Streamlit Community Cloud（アプリ実行）。ローカルPCは不要です。

GitHubアカウント：Koji2033
リポジトリ：Koji2033/Koji2024
ブランチ：streamlit-starter-20261008
エントリポイント：new_app/streamlit_app.py
依存関係：new_app/requirements.txt

既存のmainブランチには株価チャート用アプリがあります。新しいアプリはこのブランチのnew_app以下から作成します。

## Streamlit Community Cloudへの配置
1. https://share.streamlit.io/ を開き、GitHubアカウントKoji2033で接続する。
2. Create app（表示に応じてNew app）から既存リポジトリの配置を選ぶ。
3. Repository：Koji2033/Koji2024
4. Branch：streamlit-starter-20261008
5. Main file path：new_app/streamlit_app.py
6. Advanced settingsでPython 3.12を選択する。
7. Deployを実行し、発行されたstreamlit.appのURLで入力ボタンの動作を確認する。

Cloud側のアカウント認証・配置は、このファイル作成時点では未実施です。既存アプリの配置状態も未確認です。

## Workでの検証
Python 3.12の仮想環境でrequirements.txtを導入し、以下を実行します。

```bash
python -m pip install -r new_app/requirements.txt
python -m streamlit run new_app/streamlit_app.py --server.headless=true
```

Work内の起動URLは内部検証用です。スマートフォンなどから使うURLはCommunity Cloudで発行します。Workの実行プロセスを常時運用サービスとは扱いません。

## 更新
Workでアプリを編集・検証し、このブランチに保存します。配置したブランチの更新をCloudが取り込みます。Pythonパッケージの追加はnew_app/requirements.txtにも記載します。

## 秘密情報
APIキーはコードに書かず、Streamlit Community CloudのSecrets設定を使用します。

## 参照
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies

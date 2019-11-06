# Anonymous Voice Conversion

# 概要
Pix2Pixを音声データに応用することで声質変換を目指すプロジェクト（Work in Progress）。


# 変換結果（2019年10月30日　時点）

入力音声  １
<audio
    controls
    src="https://raw.githubusercontent.com/lilacs2039/Anonymous-VC/master/readme/input.wav">
        Your browser does not support the
        <code>audio</code> element.
</audio>


入力音声  ２
<audio
    controls
    src="/readme/input.wav">
        Your browser does not support the
        <code>audio</code> element.
</audio>

出力音声  
<audio
    controls
    src="readme/prediction.wav">
        Your browser does not support the
        <code>audio</code> element.
</audio>

教師音声  
<audio
    controls
    src="readme/target.wav">
        Your browser does not support the
        <code>audio</code> element.
</audio>

# 各Jupyter Notebookファイルについて
## 01_prestudy_convert-by-pyworld.ipynb
事前検証に使用したJupyter Notebook。

pyworldで抽出した音響特徴量を様々に修正してから音声を復元し、どのように音声が復元されるか確認した。

## 学習用Jupyter Notebook
### 02-1_train-with-pretraining.ipynb
学習に使用したJupyter Notebook　その１。
NoGANと呼ばれる事前学習テクニック（参考：https://www.fast.ai/2019/05/03/decrappify/ ）を使用して学習したが、NoGAN特有の、途中で学習が発散してしまう問題が発生したためうまくいかなかった。

GAN学習の前に、以下の手順で事前学習を行っている。
1. CriticなしでGeneratorの事前学習を行う
1. 事前学習したGeneratorを使用して、入力音声データセットすべてを変換して出力音声を作成し、Pickle形式で保存しておく
1. Pickle形式で保存しておいた出力音声と教師音声でCriticの事前学習を行う。

### 02-2_train-without-pretrain.ipynb
学習に使用したJupyter Notebook　その２。

Generator/Criticの事前学習はせずにGAN学習を行った。

### 03_output-without-pretrain.ipynb
「02-2_train-without-pretrain.ipynb」で求めたモデルを使用して、音声変換を行うためのJupyter Notebook。
上記の出力音声はこのJupyter Notebookで出力したもの。

# 学習について

- 変換先の話者の音声データのみ必要
  - 変換元・変換先の話者のパラレルデータは不要
- 声質匿名化（教師音声から話者情報を削ぎ落す）
  - pyworld（音声変換ライブラリ）によって教師音声データを音響特徴量（基本周波数・スペクトル包絡・非周期性指標）へ変換
  - 基本周波数を０で置き換える（ロボットのようなイメージの機会的な声質になる）などの変換を行うことで音声データから話者情報を削ぎ落す
  - 音響特徴量から音声を復元

- 声質匿名化した音声を入力データ、元の教師音声を対応する教師データとして学習を行う。
- 音声をPix2Pixモデルへ入れる前に音声をSTFTしてスペクトログラムと位相スペクトログラムへ変換し、２CHの画像として扱う。Pix2Pixモデルの入出力サイズ（行・列のサイズ）は実行時に動的に決定するため、音声の長さは任意（batchsize=1の場合）。
- 推論
  - 変換元の話者音声を声質匿名化してから、学習したモデルで推論を行う。







# インストール

以下のコマンドをプロジェクトを作成したいフォルダから実行すること

```bash
git clone --depth 1 https://github.com/lilacs2039/VoicePix2Pix
cd VoicePix2Pix
pip install -r requirements.txt

```

OGG・MP3形式の音声を扱いたい場合はffmpegをダウンロード・配置・パスを通しておくこと（音声ファイルはlibrosaを通して扱っているが、librosaは内部的にaudioreadを使用し、audioreadはffmpegを使用するため）。

https://www.ffmpeg.org/download.html






# Multimodal_crypto_prediction
This Repo records the code of multimodal Bitcoin prediction with deep learning methods.

## Abstract:
This research investigates the influence of multimodality on diverse deep learning architectures in the context of short-term cryptocurrency forecasting. We introduce CryptoBERT, an advanced sentiment analysis model tailored for cryptocurrency, and apply it to analyze an extensive dataset comprising 700,000+ social media posts and crypto news articles. Our exploration involves LSTM, CNN-LSTM, and transformer models. Noteworthy outcomes indicate the LSTM model achieves the highest accuracy (MAPE = 0.0377), showing a notable 5 percent improvement with multimodal inputs. The transformer, while not yielding superior predictions, demonstrates a 20 percent enhancement with multimodality. We meticulously interpret these findings and provide detailed insights into their underlying reasons. Additionally, we outline future efforts to refine our models and further enhance our predictive framework.

### Data
The crypto-based data are queried by using [Cryptocompare API](https://min-api.cryptocompare.com/)

The crypto-related news data are queried from Nasdaq (One need to subscribe the data)

The crypto-related historical social media data are queried from Kaggle public dataset [Kaggle](https://www.kaggle.com/datasets/kaushiksuresh147/bitcoin-tweets)
### CryptoBERT
For the detailed information about this pre-trained model: please check [CryptoBERT](https://huggingface.co/ElKulako/cryptobert)

### Models
Please check the **Models** folder.
### Results
| Model       | Results (MAE) | Results (MAPE) | Modal Fusion           |
|-------------|---------------|----------------|------------------------|
| LSTM        | 837.6         | 0.0377         | Coin + News + Media   |
|             | 1264.96       | 0.0545         | Coin + News            |
|             | 997.65        | 0.0510         | Coin + Media           |
| CNN-LSTM    | 908.86        | 0.0417         | Coin + News + Media   |
|             | 1272.83       | 0.0594         | Coin + News            |
|             | --            | --             | Coin + Media           |
| TFT         | 4523.8740     | 0.1936         | Coin + News + Media   |
|             | 5583.4273     | 0.2368         | Coin + News            |
|             | 5521.1431     | 0.2262         | Coin + Media           |


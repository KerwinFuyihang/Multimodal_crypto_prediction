# Multimodal_crypto_prediction
This Repo records the code of multimodal Bitcoin prediction with deep learning methods.

## Abstract:
Cryptocurrency investments, increasingly recognized for their promising returns, grapple with inherent volatility, posing significant prediction challenges. Although sentiment analysis can augment forecasting models, its accuracy limitations and the relatively uncharted domain of deep learning models suggest missed opportunities for enhancement. Addressing this gap, our research introduces a novel Dual Attention Mechanism for multimodal fusion. This mechanism integrates temporal data from diverse modalities, demonstrating a 20\% performance enhancement in the forecasting capabilities of our pretrained LSTM model. Further, we leveraged CryptoBERT, a sentiment analysis tool specifically for the cryptocurrency sector. This tool, trained by an extensive dataset comprising over 700,000 social media entries and cryptocurrency news articles, represents a targeted approach to understanding market sentiment dynamics.

### Data
The crypto-based data are queried by using [Cryptocompare API](https://min-api.cryptocompare.com/)

The crypto-related news data are queried from Nasdaq (One need to subscribe the data)

The crypto-related historical social media data are queried from Kaggle public dataset [Kaggle](https://www.kaggle.com/datasets/kaushiksuresh147/bitcoin-tweets)

#### Data Descriptor of Our Processed Dataset

| Predictor       | Explanation                                                                                     |
|-----------------|-------------------------------------------------------------------------------------------------|
| Opening price   | Price at which the Cryptocurrency began trading at the start of the given day.                |
| Closing price   | Price at which the Cryptocurrency ended trading at the end of the given day.                  |
| Highest/Lowest price | Highest/Lowest price at which the Cryptocurrency was traded on the given day.                 |
| Volumefrom      | The quantity of shares bought by buyers on the given day.                                      |
| Volumeto        | The quantity of shares sold by sellers on the given day.                                       |
| News            | The predictor represented the result of the analysis of news contents on the given day.        |
| Media           | The predictor represented the result of the sentiment analysis of social media contents on the given day. Positive Media index meant social media critics expected the price to rise. Negative Media index meant social media critics expected the price to drop. |

### CryptoBERT
It was built by further training the vinai's bertweet-base language model on the cryptocurrency domain, using a corpus of over 3.2M unique cryptocurrency-related social media posts.
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


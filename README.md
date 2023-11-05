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
<img width="800" height = '350' alt="Model" src="https://github.com/KerwinFuyihang/Multimodal_crypto_prediction/assets/109135319/4ab29964-d7d0-4085-bcd1-0adb289ef398">

In this study, we designed a end-to-end Dual Attention Mechanism for Multimodal Temporal Data Fusion. We firstly introduce the unimodal input attention module, which aims to reconstruct the inputs by deploying the attention weights. Then, we introduce the other attention module to do the cross-modal data fusion. At last, we transfer the new data input to a end-to-end LSTM model to do the final forecasting. For the details of the code, Please check the **Models** folder.
### Results
#### Comparative Result
| Model      | MAE      | MAE*     | IsMultimodal |
|------------|----------|----------|--------------|
| DAM-LSTM   | 719.82   | 431.86   | Yes          |
| LSTM       | 837.6    | 491.58   | Yes          |
|            | 863.63   | 501.26   | No           |
| CNN-LSTM   | 908.86   | 550.73   | Yes          |
|            | 887.65   | 500.15   | No           |
| TFT        | 4523.8740| 4320.2724| Yes          |
|            | 6324.7640| 6158.1635| No           |

#### Ablation Result

| MAE    | Attention Layer combination       |
|--------|-----------------------------------|
| 837.6  | No any Attention module           |
| 790.47 | No intra-modal attention module   |
| 891.25 | No cross-modal attention module   |
| 719.82 | Dual Attention module             |


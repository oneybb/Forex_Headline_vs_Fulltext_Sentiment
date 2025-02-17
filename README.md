# Problem Statement

In the fast-paced forex market, traders often rely on headlines for quick decisions without reading full articles. However, some news agencies may exaggerate headlines to capture attention, leading to sentiment misalignment with the article's content. Headlines are short in nature and may oversimplify or omit nuances present in the full article, distorting sentiment. This discrepancy can contribute to market inefficiencies, increased volatility, and misguided trading decisions.

This project investigates the discrepancy between news headline sentiment and full-text sentiment. The study uses a manually labeled dataset of headline sentiment and evaluates its alignment against full-text sentiment, which is predicted using a fine-tuned FinBERT model. By treating full-text sentiment as a reference point, this analysis aims to assess the accuracy and reliability of forex news headlines in reflecting broader article sentiment.


# Dataset Preparation

The dataset contains news headlines (“title”) and full text content(“text”) relevant to key forex pairs: AUD-USD, EUR-CHF, EUR-USD, GBP-USD, and USD-JPY. The data was extracted from renowned platforms such as Forex Live and FXstreet over a period of 86 days, from January to May 2023.
"true_sentiment" is manually labeled based on how traders would likely react to a headline, it took consideration of how the headlines impacted currency movements at the time of publication rather than just its wording. It reflects real-world trader sentiment upon reading the news headline. As it is in text, I encoded it to numerical values for modelling.

(Source:https://paperswithcode.com/dataset/forex-news-annotated-dataset-for-sentiment)


# Model Choice & Fine-tuning

In this analysis, I fine-tuned a FinBERT model on news headlines to predict their true
sentiment. I then used this fine-tuned model to classify sentiment on full-text articles with adjustment to text length. The results of full-text sentiment are then compared against headline sentiment in a confusion matrix to assess the reliability in headlines.

## Why BERT (FinBERT)
BERT, a transformer-based model, outperforms LSTMs and CNNs in NLP by leveraging
bidirectional context to analyze both past and future words. Fine-tuning a pre-trained BERT model also requires less data and computing power, making it ideal for the dataset with only 2,291 rows.
FinBERT, specifically pre-trained on financial texts, excels at finance-specific sentiment analysis, avoiding misinterpretations common in general NLP models. For example, it can correctly identify "bearish" as negative in finance, whereas a general model might classify it as neutral.

## Fine-tuning FinBERT
While FinBERT is strong in financial sentiment analysis, it may not fully capture Forex-specific nuances. To improve alignment with market dynamics, it is fine-tuned on headlines against true sentiment labels.
Fine-tuning leverages the AdamW optimizer and Cross-Entropy Loss. AdamW efficiently optimizes large transformers by dynamically adjusting learning rates and preventing overfitting through weight decay. Cross-Entropy Loss, ideal for multi-class classification (Negative, Neutral, Positive), minimizes errors by comparing predicted logits with actual sentiment labels.
The training loop starts with model.train(), updating all layers. Each epoch processes mini-batches of tokenized headlines, generates logits for sentiment classes, and computes Cross-Entropy Loss. Gradients are back propagated (loss.backward()), AdamW updates weights, and optimizer.zero_grad() clears previous gradients to ensure stable learning.

## Applying Headline-trained Model on Full-text
After tuning the FinBERT model from headlines, I used it to classify sentiment for full-text. Since headline and the full text of a news article often share similar vocabulary, the model is transferable.
However, longer texts dilute sentiment, as neutral or mixed content reduces intensity. To mitigate this, I used length-normalized sentiment adjustment (Amplayo et al., 2019), which segments long text into headline-sized chunks and weights sections based on their contribution to overall sentiment

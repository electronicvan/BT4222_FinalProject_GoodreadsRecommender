# BT4222_FinalProject_GoodreadsRecommender

**BT4222 Group 12 — Source Code & Dataset Index**

Personalised Book Recommendation System for Goodreads

Github Repo Link: [https://github.com/electronicvan/BT4222\_FinalProject\_GoodreadsRecommender](https://github.com/electronicvan/BT4222_FinalProject_GoodreadsRecommender)

**1\. Datasets (CSV)**

Cleaned datasets derived from the Goodreads Book Graph Dataset (children's books subset). All files hosted on Google Drive.

| Filename | Description | Link |
| :---- | :---- | :---- |
| books\_df\_cleaned.csv | Cleaned book metadata: title, authors, description, format, num\_pages, publication\_year, language, popular\_shelves. Used as the source for all item-level features. | [Link](https://drive.google.com/file/d/12Kni02HxnY-O4L49m3_wjvHObdmKg5QP/view?usp=sharing) |
| interactions\_df\_cleaned.csv | Cleaned user-book shelf interactions with event\_time. 10M+ implicit positive signals. Primary training signal for all models. | [Link](https://drive.google.com/file/d/1I9hvivJWPDncA2fj5Aj5UyBH-JUMyhp8/view?usp=sharing) |
| reviews\_df\_cleaned.csv | Cleaned detailed user reviews. Excluded from modelling: only 7.3% of interactions have a review, and reviews are post-outcome events unavailable at inference time. Retained for reference. | [Link](https://drive.google.com/file/d/1GKrJtmnBiQBBiaolYmXyZI94Vqau_dGr/view?usp=sharing) |

**2\. Notebooks**

All notebooks run via Google Colab in sequence. Notebooks 01-03 handle data preparation; 04-11 cover sequential models; 12-15 cover within-user NCF; 16-19 cover the global temporal split experiment.

| Filename | Description | Link |
| :---- | :---- | :---- |
| 01\_EDA.ipynb | Initial exploration of the Goodreads children's books dataset. Covers interaction distributions, user/book statistics, and data quality assessment. | [Link](https://drive.google.com/file/d/1uwCEWSxFxC3aaP5DiPX5jom5QRxqyEK5/view?usp=sharing) |
| 02\_Clean\_Preprocess.ipynb | Cleans raw data: removes low-signal users, standardises author fields, collapses format and language categories, assigns event\_time. | [Link](https://drive.google.com/file/d/1UWqc4tTcfZy173ElHJV6EnKoDuLrL33b/view?usp=sharing) |
| 03\_Feature\_Engineering.ipynb | Initial feature engineering pipeline. Generates SBERT semantic embeddings (item\_semantic\_embedding\_static, 384-dim), user-item matching signals, item temporal popularity, and user history features. | [Link](https://drive.google.com/file/d/1u7rMhJnG_89HQ-nCP2BMPfwWfPMO6oLG/view?usp=sharing) |
| 03\_Updated\_Feature\_Engineering.ipynb | Refined feature engineering with improved shelf tag exclusion (80+ filters), corrected temporal leakage prevention, and updated description truncation. Produces the final feature set used in all models. | [Link](https://drive.google.com/file/d/11PWuY9NpkaGzu6z3ZYxBOaK_443KIJik/view?usp=sharing) |
| 04\_SequentialLSTM\_Ksplit\_Baseline.ipynb | Baseline LSTM with Cross-Entropy loss. Encodes each item as book embedding \+ SBERT embedding, processed via single LSTM layer, scored via FFN. | [Link](https://colab.research.google.com/drive/1EjCaAdHYYndNEWsVtJRsxizXSk94MmNW?usp=sharing) |
| 05\_BPR\_SequentialLSTM\_Ksplit\_Baseline.ipynb | LSTM with BPR loss and hard negative sampling from future sequence interactions, with popular-books fallback. | [Link](https://colab.research.google.com/drive/1tM83EPoza1g9l-py8G9xtKhZFyKPnnEr?usp=sharing) |
| 06\_HybridLoss\_SequentialLSTM\_Ksplit\_Baseline.ipynb | LSTM with hybrid CE (0.6) \+ BPR (0.4) loss. Introduces sigmoid gating on static user embeddings to balance long-term and short-term preferences. | [Link](https://colab.research.google.com/drive/14w09tpqbOCXRdP3NX9W2PhbnetObHkLK?usp=sharing) |
| 07\_FAISS\_HybridLoss\_SequentialLSTM\_Ksplit\_Baseline.ipynb | Hybrid CE (0.6) \+ BPR (0.4) LSTM with FAISS-based semantic negative sampling. Fallback to popular-books then random sampling. | [Link](https://colab.research.google.com/drive/1Fg_OxvWxJVpY1qvyM0wmSyHoHs7cHV1f?usp=sharing) |
| 08\_FAISS\_AlternativeHybridLoss\_SequentialLSTM\_Ksplit\_Baseline.ipynb | Same as notebook 07 with alternative loss weighting CE (0.4) \+ BPR (0.6). Tests whether higher BPR weight improves ranking under FAISS negatives. | [Link](https://colab.research.google.com/drive/1O19GVs7Pg2vHIlqRcGPOmwuHLmlcYsAl?usp=sharing) |
| 09\_PoorIntegratedFeature\_SequentialLSTM\_Ksplit\_Baseline.ipynb | Enriched LSTM adding user-item matching, temporal popularity, and user history features. Documents performance degradation from unscaled numerical inputs. | [Link](https://colab.research.google.com/drive/1v2Y6yLbMD84SRjNSJG5VhFrf2jF2AG1I?usp=sharing) |
| 10\_RefinedFeatureRepresentation\_SequentialLSTM\_Ksplit\_Baseline.ipynb | Corrected enriched LSTM with log-scaling, clipping, and standardisation. Uses modality-specific projection blocks (Linear-\>LayerNorm-\>GELU-\>Dropout). Best LSTM result: HR@10 \= 0.9448. | [Link](https://colab.research.google.com/drive/10lYG4QkkObuMQrWBbZKB-K3DE_0N2urU?usp=sharing) |
| 11\_Transformer\_Sequential\_Ksplit\_Model.ipynb | Transformer-based sequential recommender with multi-modal feature fusion, causal self-attention (2 layers, 4 heads), learnable \[CLS\] token, and gated user fusion. Best overall: HR@10 \= 0.9547. | [Link](https://colab.research.google.com/drive/1LKyMZFuuw5FFD5tKuYMi16Vcq0eJ98l9?usp=sharing) |
| 12\_Within\_User\_Temporal\_split\_NCF.ipynb | Produces within-user temporal splits (70/15/15 by sequence position) used by all NCF models. Outputs App1\_train.pkl, App1\_val.pkl, App1\_test.pkl. | [Link](https://drive.google.com/file/d/1KOtaHkSMLnlm8Cs7u1lcU2QDc210AF3u/view?usp=sharing) |
| 13\_NCF\_ID\_Only.ipynb | ID-only NCF baseline using only 32-dim user/book embeddings via MLP. Negatives from top-1000 popular unseen books. Full architecture in notebook. | [Link](https://drive.google.com/file/d/1nw8c8vhlO2X8eJZjoqUj4xdf0H0p9OsT/view?usp=sharing) |
| 14\_NCF\_Feature\_Enhanced\_Stages1and2.ipynb | Feature-Enhanced NCF Stages 1 and 2\. Stage 1 adds SBERT embeddings; Stage 2 adds structured metadata and temporal popularity features. | [Link](https://colab.research.google.com/drive/1amlvKmixIWTHckcM_ZuDFpNLGvZBYlEB?usp=sharing) |
| 15\_NCF\_Feature\_Enhanced\_Stage3.ipynb | Feature-Enhanced NCF Stage 3\. Adds user-item matching and user history signals on top of Stage 2\. | [Link](https://drive.google.com/file/d/1bhm2qZQMet0mkZW1g35TxunC-GPLpo4V/view?usp=sharing) |
| 16\_Global\_Temporal\_Split.ipynb | Produces a global time-based train/val/test split to address cross-user temporal leakage. Outputs App2 split files. | [Link](https://colab.research.google.com/drive/1a5eqOQuWYeC4fOi6hKO1VwLKwbFoxQBV?usp=sharing) |
| 17\_Global\_Temporal\_NCF\_ID\_Only.ipynb | ID-only NCF retrained on global temporal split. Reports performance by user activity segment (Low/Medium/High). | [Link](https://colab.research.google.com/drive/19X3c4JFbu84ZNPPd5nwsNl_f6Cj8Ueyp?usp=sharing) |
| 18\_Global\_Temporal\_NCF\_Feature\_Enhanced\_Stages1and2.ipynb | Feature-Enhanced NCF Stages 1 and 2 retrained on global temporal split. | [Link](https://colab.research.google.com/drive/1G7EzJRpDBqcLNRIy_5RH-6M6uqgWxZJs?usp=sharing) |
| 19\_Global\_Temporal\_NCF\_Feature\_Enhanced\_Stage3.ipynb | Feature-Enhanced NCF Stage 3 retrained on global temporal split. | [Link](https://colab.research.google.com/drive/1H5qfh5kdDtfZuj_T3w1LGeL76OC2zt4U?usp=sharing) |

**3\. Other Files**

Intermediary data splits, evaluation user samples, and saved model weights used across notebooks.

| Filename | Description | Link |
| :---- | :---- | :---- |
| App1\_train.pkl | Within-user temporal training split (70%). Used by NCF notebooks 13-15. | [Link](https://drive.google.com/file/d/1eO531XoF9qHaHYvR5BYmrX0oXuHgHlGi/view?usp=sharing) |
| App1\_val.pkl | Within-user temporal validation split (15%). Used by NCF notebooks 13-15. | [Link](https://drive.google.com/file/d/1DCdgoEwWxHphoPhShpv9llYsHDLKFoXJ/view?usp=sharing) |
| App1\_test.pkl | Within-user temporal test split (15%). Used by NCF notebooks 13-15. | [Link](https://drive.google.com/file/d/1M5mQHBRNpYWiZ3ifQ3l3bCFetLqVdXf-/view?usp=sharing) |
| App2\_train.pkl | Global temporal training split. Used by NCF notebooks 17-19. | [Link](https://drive.google.com/file/d/1Qk2LWyxTtnHyW3ILL21VLOgmeV5IXBMG/view?usp=sharing) |
| App2\_val.pkl | Global temporal validation split (full). Used by NCF notebooks 17-19. | [Link](https://drive.google.com/file/d/1Bu03Rn6beMj9tqdrxpkV4UNY11FuWm1x/view?usp=sharing) |
| App2\_val\_ws.pkl | Global temporal validation split filtered to warm-start users and items only. | [Link](https://drive.google.com/file/d/1e-z3soe10N6JfGcsbtBOjJ4XkGsZhS4V/view?usp=sharing) |
| App2\_test.pkl | Global temporal test split (full). Used by NCF notebooks 17-19. | [Link](https://drive.google.com/file/d/15A4HCkxJYIvXhLZnTRAn5DPtduIq4v4o/view?usp=sharing) |
| App2\_test\_ws.pkl | Global temporal test split filtered to warm-start users and items only. | [Link](https://drive.google.com/file/d/1vOHm6yuEqEylroguub-1X-RQtaxqXt6W/view?usp=sharing) |
| final\_updated\_baseline\_df.pkl | Fully engineered feature table produced by 03\_Updated\_Feature\_Engineering.ipynb. Loaded by LSTM and Transformer notebooks. | [Link](https://drive.google.com/file/d/1ExoYGumAJIaMnA-UC-hR74Q65CqXlVXx/view?usp=sharing) |
| eval\_sample\_users.csv | Fixed 5,000-user sample used for consistent evaluation across within-user NCF models (notebooks 13-15). | [Link](https://drive.google.com/file/d/1PmqexhYLNg--YbhNQGyNFUsalfPBzpBK/view?usp=sharing) |
| val\_eval\_sample\_users.csv | Fixed user sample for validation evaluation in within-user NCF experiments. | [Link](https://drive.google.com/file/d/1zU3k8ceVimWfprjlkk0F5Hxff7cvW6vR/view?usp=sharing) |
| global\_temporal\_test\_eval\_sample\_5k\_users.csv | Fixed 5,000-user sample for test evaluation in global temporal NCF experiments (notebooks 17-19). | [Link](https://drive.google.com/file/d/1Ln1S3HS1Fb1XGdip6P7OXUM5v7e88-fE/view?usp=sharing) |
| global\_temporal\_val\_eval\_sample\_users.csv | Fixed user sample for validation evaluation in global temporal NCF experiments. | [Link](https://drive.google.com/file/d/1zuyFio8ibA_hylYf0r1W9ijhNnougtg9/view?usp=sharing) |
| global\_temporal\_best\_ncf\_idonly.pt | Saved best model weights for ID-only NCF trained on global temporal split (notebook 17). | [Link](https://drive.google.com/file/d/1mKv0ZFW0d76JzixS8HSzlvgY6XgVA0Qo/view?usp=sharing) |
| global\_temporal\_best\_enhanced\_ncf\_stage1.pt | Saved best model weights for Feature-Enhanced NCF Stage 1 on global temporal split (notebook 18). | [Link](https://drive.google.com/file/d/19q8b1AGV0HPw7TWIlKRzOiRpiZEuPfyA/view?usp=sharing) |
| global\_temporal\_best\_enhanced\_ncf\_stage2.pt | Saved best model weights for Feature-Enhanced NCF Stage 2 on global temporal split (notebook 18). | [Link](https://drive.google.com/file/d/1EH-PJpWQhymuzIxdNAGW0nr6BOyP_2nl/view?usp=sharing) |
| global\_temporal\_best\_enhanced\_ncf\_stage3.pt | Saved best model weights for Feature-Enhanced NCF Stage 3 on global temporal split (notebook 19). | [Link](https://drive.google.com/file/d/1gDHJAkUxSWQS1uu9CoCCCL_5NB7e_hQI/view?usp=sharing) |


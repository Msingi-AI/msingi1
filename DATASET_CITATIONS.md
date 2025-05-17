# Dataset Citations for Msingi1 Training Data

## Primary Data Sources

The Msingi1 language model was trained on a combined corpus created from the following two primary sources:

### 1. Swahili Corpus
Masasi, Noel; Masua, Bernard (2024), "Swahili Corpus", Mendeley Data, V2, doi: 10.17632/d4yhn5b9n6.2

### 2. Helsinki Corpus of Swahili (HCS-NA-v2)
Arvi Hurskainen (2004). Helsinki Corpus of Swahili. 2nd edition: Helsinki Corpus of Swahili, Version 2.0 (HCS 2.0) 2004-09-30. University of Helsinki, Institute for Asian and African Studies.


### 3. Swahili Wikipedia 2021 (30K)
Wikimedia Foundation. (2021). Swahili Wikipedia. Retrieved 2021 from https://sw.wikipedia.org/

### 4. Swahili Community 2023
Various Swahili news and community websites. (2023). Collected from sources including Mwananchi.co.tz, BBC Swahili, VOA Swahili, and Vodacom Tanzania.

### 5. Swahili Bible (ONEN14)
The Bible Society of Tanzania. (2014). Swahili Bible (ONEN14). Bible Society of Tanzania.
## Combined Dataset Statistics

The combined dataset used for training Msingi1 consists of:
- Total words: 60,898,742
- Total lines: 2,568,006
- Training set: 54,960,319 words (90.25%)
- Validation set: 5,938,423 words (9.75%)
- Average words per line: 23.71

## Data Processing

The raw data from both sources was processed using custom Python scripts to:
1. Clean and normalize text
2. Remove duplicates
3. Filter out very short segments
4. Shuffle and split into training and validation sets

## Usage Notes

This combined dataset was created for research purposes. All rights to the original data belong to their respective creators. If you use this dataset or derivatives in your research, please cite both original sources as listed above.

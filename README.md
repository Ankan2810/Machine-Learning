# Machine-Learning

## Sentiment Analysis using PhoBERT

This project demonstrates how to perform sentiment analysis on Vietnamese text using the PhoBERT model. The application fine-tunes the PhoBERT model on a custom dataset and uses it to predict the sentiment of new text inputs.

### Requirements

- Python 3.7+
- PyTorch
- Transformers
- Datasets
- Pandas
- NumPy

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Ankan2810/Machine-Learning.git
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Usage

1. Preprocess the dataset:
    ```sh
    python src/sentiment_analysis.py --preprocess
    ```

2. Fine-tune the PhoBERT model:
    ```sh
    python src/sentiment_analysis.py --train
    ```

3. Predict sentiment of new text:
    ```sh
    python src/sentiment_analysis.py --predict "Your text here"
    ```

4. Train on Google Colab:
    ```sh
    upload folder Machine-Learning to Google Drive
    import ML.ipynb file into colab
    Train
    Pray that it works
    ```

### Files

- `src/sentiment_analysis.py`: Main script for preprocessing, training, and predicting.
- `src/models/phobert_model.py`: Contains the PhoBERT model class.
- `src/data/dataset.csv`: The dataset used for training and testing.
- `src/utils/preprocessing.py`: Utility functions for text preprocessing.

### License

This project is licensed under the MIT License.



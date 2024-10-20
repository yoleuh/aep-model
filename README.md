# Safety Classification Project

For the AEP Safety Observation Challenge at HackOHI/O 2024

This project uses machine learning to classify safety comments into high priority and standard priority categories. It fine-tunes a deep learning transformer-based neural network ([DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased), adapting it to the specific task of safety observation classification.

## Project Structure

- `safety_classification.py`: Script for training the model, evaluating its performance, and generating results for the entire dataset.
- `classify_new_inputs.py`: Interactive script for classifying new safety comments using the trained model.
- `requirements.txt`: List of Python packages required for the project.
- `data/data.csv`: Input data file.

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/yoleuh/aep-model.git
   cd aep-model
   ```

2. Create a virtual environment (optional but recommended):

   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training the Model and Generating Results

1. Run the training and classification script:
   ```
   python3 safety_classification.py
   ```
2. This script will:
   - Train the model on the included dataset
   - Evaluate its performance
   - Save the model and tokenizer in the `saved_model` and `saved_tokenizer` directories respectively
   - Classify all comments in the original dataset
   - Save the results (including classifications) to `results/output_data.csv`

### Interactively Classifying New Inputs

1. After training the model, you can use it to classify new safety comments interactively:
   ```
   python3 classify_new_inputs.py
   ```
2. This script will:
   - Load the trained model and tokenizer
   - Prompt you to enter safety comments
   - Display the classification (High Priority or Standard Priority) and confidence score for each entered comment
3. Enter your safety comments one at a time and press Enter to see the classification result.
4. Type 'quit' to exit the program.

### Training on Your Own Data

If you want to train the model on your own dataset:

1. Prepare your data:

   - Create a CSV file with at least two columns: one for the safety comments and one for the priority labels.
   - Name your columns exactly as they are in the original dataset or update the column names in the `safety_classification.py` script.

2. Replace the existing data:

   - Move your CSV file to the `data/` directory.
   - Rename it to `data.csv` or update the file path in `safety_classification.py`.

3. Adjust the script if necessary:

   - If your data structure differs significantly, you may need to modify the data loading and preprocessing steps in `safety_classification.py`.

4. Run the training script:

   ```
   python3 safety_classification.py
   ```

5. The script will train on your new data, evaluate the model, and save the results.

## Notes

- The scripts are set to use CPU for computations. If you have a GPU and want to use it, set USE_CPU to False in both scripts.

  ```python
  # Configurable variables
  USE_CPU = False
  ```

- Be cautious with the included data and any personal data you use. Do not share it or use it with public AI services for privacy concerns.

- This project utilizes the DistilBERT base model (uncased) for sequence classification. DistilBERT is a smaller, faster version of BERT, developed by Hugging Face. DistilBERT is a transformers model, pretrained on the same corpus as BERT in a self-supervised fashion, using the BERT base model as a teacher. It's designed for tasks that use the whole sentence to make decisions, such as sequence classification, token classification, or question answering.

## Troubleshooting

If you encounter any issues:

- Ensure all dependencies are correctly installed.
- Verify that you have sufficient disk space for saving the model.
- Make sure you've run `safety_classification.py` before trying to use `classify_new_inputs.py`.
- If using custom data, check that your CSV file is formatted correctly and the column names match those expected by the script.
- Training the model will take a while, and will be physically demanding on your machines RAM. If this is an issue, the MAX_STEPS variable can be decreased, or another model can be used.
- Do not use ctrl-c to end to process early even if you think it is done, after the evaluation results, the model, tokenizer, and results will be saved.

For any other issues, please open an issue in the project repository or email one of us.

## Contributors

Created by:

- Reid Ammer (ammer.5@osu.edu)
- Brian Tan (tan.1220@osu.edu)

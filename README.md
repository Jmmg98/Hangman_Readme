# Advanced Hangman Solver

## Purpose and Overview

This project focuses on designing an advanced Hangman-solving algorithm leveraging state-of-the-art machine learning techniques and statistical methods. By integrating N-gram probabilities and a Hybrid Transformer Model, the solution achieves highly accurate letter predictions for varying word lengths and complexities. The goal is to combine statistical context with deep learning for robust and adaptive predictions.

---


## 1. N-Gram Probabilities in Hangman
To enhance the predictive capabilities of the Hangman solution, I incorporated n-gram probabilities. N-grams are subsequences of 
𝑛
n items from a given sequence. In this context, the items are characters from words in a dictionary. By leveraging n-grams, the algorithm identifies patterns and predicts missing letters based on contextual information.

How N-Gram Probabilities Are Calculated
N-gram probabilities are computed by analyzing sequences of characters from words in a training dictionary. These probabilities are then used to predict the likelihood of a specific letter appearing in a missing position within a word.

### N-Gram Construction
To build the n-grams, I iterated through each word in the training dictionary, extracting subsequences of varying lengths (
1
1-gram, 
2
2-gram, etc.). The counts of each n-gram are stored in a dictionary for later use.


### N-Gram Probability Calculation
Given a partially revealed word with known and unknown characters (represented by dots .), the algorithm computes the probabilities for the missing letters as follows:

#### Identify Missing Positions:

Determine the indices of unknown characters in the word.


#### Weighting by N-Gram Order

Assign higher weights to longer n-grams (e.g., 5-grams are weighted more than 2-grams), as they provide more contextual information.

Use dynamic weights for flexibility:

alpha_n = 0.5 / (n + 1), if n > 5



#### Iterate Over N-Grams:

For each n-gram in the word, calculate the probabilities of missing letters using counts from the precomputed n-gram dictionary.
If there is one missing letter, iterate over possible letters and compute their probabilities.
If there are two missing letters, consider combinations of letters and compute joint probabilities.
Normalization:

Normalize the counts to ensure they sum to 1, converting them into probabilities.
Formula:
For a missing letter 
𝑙
l in a word segment 
𝑆
S, the probability is calculated as:

P(l|S) = Count of l in n-gram S / Total counts of all letters in S



### Dynamic Weighting

N-grams of different lengths provide varying levels of contextual accuracy. For example:

- **Unigrams** provide basic letter frequency information.
- **Bigrams and trigrams** capture more contextual patterns.

To balance these contributions, weights `α_n` are applied dynamically:

- **Longer n-grams** are weighted more heavily.
- For `n > 5`, weights decrease gradually as:
 α_n = 0.5 / (n + 1)


### Advantages of N-Gram Probabilities

- **Contextual Awareness:**  
  N-grams capture character dependencies, enabling the algorithm to infer missing letters based on patterns.

- **Flexibility:**  
  The approach handles varying word lengths and numbers of missing letters.

- **Scalability:**  
  Dynamic weighting ensures effective use of higher-order n-grams without overfitting.

By integrating these probabilities into the prediction process, the algorithm achieves robust letter predictions, significantly improving the accuracy of Hangman guesses.

## 2. Hybrid Transformer Model: Advanced Language Modeling with N-Grams

The Hybrid Transformer Model combines advanced language modeling techniques with statistical n-gram probabilities and missing character embeddings to optimize letter predictions during the Hangman game.

### Combining Masked Word Inputs with N-Gram Probabilities and Missing Character Embeddings

A critical part of the strategy involves combining three sources of information to guide predictions effectively during the Hangman game:

#### Masked Word Input (`masked_word`)

- Represents the current state of the word being guessed, with:
  - Known letters encoded as their respective indices (0-25 for letters a-z).
  - Unknown letters represented by a special index (26 for `_` or `.`, indicating missing characters).
- This input allows the model to understand which positions in the word are filled and which remain uncertain, providing structural context.

#### N-Gram Probabilities

- Precomputed n-gram probabilities from multiple dictionaries (`words_250000_train.txt`, `words_not_contained.txt`, and `nltk.words`) are combined.
- These probabilities provide statistical insights into the likelihood of specific letters appearing in the missing positions, based on historical patterns of letter co-occurrence.

#### Missing Character Embeddings (`missing_letters`)

- A one-hot encoded representation of the missing or guessed letters is passed as additional input to the model.
- This feature helps penalize already guessed letters during predictions and guides the model to prioritize unexplored options.

### How These Inputs are Combined in the Model

#### Input Preparation

1. **Masked Word (`input_masked_word`):**
   - Each character in the word is converted into its corresponding index:
     - `a = 0`, `b = 1`, ..., `z = 25`, `_` or `.` = `26`.
   - Example: The word `_p_ple` would be encoded as `[26, 15, 26, 15, 11, 4]`.

2. **Missing Letters (`input_missing_letters`):**
   - Missing or guessed letters are encoded into a one-hot vector of size 26 (for a-z).
   - This ensures the model can recognize which letters are already eliminated from consideration.

3. **N-Gram Probabilities (`n_gram_probs`):**
   - Computed for all 26 letters, these probabilities serve as a context-aware guide to predict the next most probable letter based on statistical patterns in the training data.

#### Model Integration

1. **Masked Word and Positional Encoding:**
   - The masked word indices are embedded into a high-dimensional vector space using the embedding layer.
   - Positional encoding is added to the embeddings, allowing the model to recognize the relative positions of the letters within the word.

2. **Transformer Encoder:**
   - The encoded masked word is passed through the Transformer architecture, which captures long-range dependencies and semantic patterns in the word structure.

3. **Missing Character Features:**
   - The one-hot representation of missing letters is processed through a linear layer (`fc_miss_chars`), transforming it into a vector that aligns with the Transformer’s hidden space.

4. **N-Gram Features:**
   - The n-gram probabilities are mapped into the hidden space using another linear layer (`fc_ngram`).

5. **Feature Combination:**
   - The outputs from the Transformer encoder, missing character features, and n-gram probabilities are combined with learnable weights (`alpha` and `beta`):

   ```python
   combined_features = (
       self.alpha * transformer_output.mean(dim=1) + 
       self.beta * miss_char_features +
       ngram_features.unsqueeze(1)
   )```
    - This hybrid feature vector ensures that predictions are informed by structural, statistical, and contextual cues.
6. **Prediction Output:**
   - The final combined features are passed through a fully connected layer (fc_out) to produce the probability distribution over all possible letters (a-z).

#### Training and Validation Plots Summary

These plots illustrate the learning dynamics of the Hybrid Transformer Model during training:

1. **Training Loss**
   - **Trend**: The loss starts high (above 3.5) and decreases rapidly in the first 5 epochs, stabilizing around epoch 8.
   - **Conclusion**: A steady decline in loss indicates effective learning and convergence.
     
![Transformer_training_loss](https://github.com/user-attachments/assets/49676a59-ff10-47ed-9f52-d0c61be4ecf6)

2. **Validation Accuracy**
   - **Trend**: Accuracy improves sharply in early epochs, surpassing 70% by epoch 2 and stabilizing above 90% by epoch 6.
   - **Conclusion**: High, stable accuracy demonstrates strong generalization to unseen data.
     ![Transformer_validation_accuracy](https://github.com/user-attachments/assets/2b2ce3d4-93a7-465c-9bee-36a46f6d2000)


##### Key Takeaways
- **Efficient Learning**: Rapid improvements early in training highlight the model's ability to quickly capture patterns.
- **Convergence**: Plateauing loss and accuracy indicate optimal performance has been reached.
- **Balanced Training**: The model generalizes well without overfitting, making it reliable for diverse Hangman scenarios.


#### Why This Strategy is Effective

1. **Context Awareness:**
    - The masked word input provides structural context, ensuring predictions respect the known letter placements.
2. **Statistical Guidance:**
    - The n-gram probabilities ensure that the model prioritizes letters that are statistically likely to occur in the given context.
3. **Avoiding Repetition:**
    - The missing character embeddings penalize previously guessed letters, reducing redundancy and improving efficiency.
4. **Flexibility:**
    - The combination of these features allows the model to adapt to different word lengths, linguistic patterns, and contexts, making it robust across a wide range of test cases.

This multi-faceted approach is a core reason for the high accuracy achieved during simulations and real-world testing.


## 3. Word Length Distribution and Model Optimization

### Word Length Distribution

The histogram below provides insights into the distribution of word lengths in the dataset. Most words fall between 8 and 16 letters. This analysis was crucial in understanding the dataset and guided the modeling approach.

![word_length_histogram](https://github.com/user-attachments/assets/f7cc5723-6584-479c-9284-794f13a52de3)

### Model Optimization

#### Initial Experiments
- The analysis of the word length distribution led to experiments where multiple models were trained for specific ranges of word lengths (e.g., short words and long words).
- Each model was fine-tuned to focus on a particular subset of the dataset.

#### Final Model
- The best results were achieved with a single model trained on all word lengths. This approach allowed the model to generalize better across varying word lengths.

---

## 4. Using AWS for Training

### Why AWS?
Due to the computational complexity of training the Hangman model locally with the entire dataset, the training process was containerized using Docker and executed on AWS via Amazon Elastic Container Service (ECS). This setup allowed for efficient and scalable training, especially when processing large dictionaries and performing simulations.

### Training Script
- The training script (`train_hybrid_model_AWS.py`) was specifically adapted for the AWS environment.
- The containerized approach ensured portability and consistency across different computational environments.

### Key Benefits of Using AWS
1. **Scalability**: The ECS service allowed the use of multiple instances for faster training.
2. **Efficiency**: Leveraging powerful cloud hardware significantly reduced training time compared to local machines.
3. **Portability**: The Docker container ensured that the training setup was consistent across local and cloud environments.


## 5. Performance Evaluation and Simulations with Optimized Model Configuration

To evaluate the model's effectiveness, I performed local simulations of the Hangman game. These simulations focused on three specific metrics to analyze the model's performance:

### Metrics for Model Evaluation

#### 1. Correct Predictions by Word Length
- **Objective:** Track how well the model predicts letters across words of varying lengths.
- **Visualization:** See the file `Correct_Predictions_by_Word_Length_modelcount_min4_max30_missing3.png`.
![Correct_Predictions_by_Word_Length_modelcount_min4_max30_missing3](https://github.com/user-attachments/assets/ff4a1860-2a24-469f-9fe8-a27db1bf10be)

#### 2. Correct Predictions by Position
- **Objective:** Monitor the accuracy of predictions based on the position of the guessed letter in the word.
- **Visualization:** See the file `Correct_Predictions_by_Position_dict_results_ngram_min4_max30_missing3.png`.
![Correct_Predictions_by_Position_dict_results_ngram_min4_max30_missing3](https://github.com/user-attachments/assets/987982f8-8741-443c-8b5e-11dd76cd5ec8)

#### 3. Correct Predictions by Remaining Letters
- **Objective:** Evaluate how accuracy changes as fewer letters remain unguessed.
- **Visualization:** See the file `Correct_Predictions_by_Remaining_Letters_dict_results_ngram_f_min4_max30_missing3.png`.
![Correct_Predictions_by_Remaining_Letters_dict_results_ngram_f_min4_max30_missing3](https://github.com/user-attachments/assets/60c16971-de44-4b1b-b3a5-3b5e69373c8c)

### Model Configuration for Performance Optimization

To achieve high accuracy during the Hangman game, the model is configured to operate under specific criteria, defined by the `self.config` dictionary:

#### Explanation of Parameters

##### `min_length`
- This parameter sets the minimum length of words for which the model is allowed to make predictions.
- Words shorter than this length are excluded to ensure sufficient contextual information is available for accurate predictions.

##### `max_length`
- This parameter defines the maximum word length the model will process.
- Very long words are excluded to balance computational efficiency and model performance.

##### `max_missing`
- Specifies the maximum number of letters that can be missing (represented by underscores) in the word at any given point.
- If the number of missing letters exceeds this threshold, the model is not used, as its prediction accuracy tends to decline with insufficient context.

##### `model`
- Points to the specific pre-trained model (`self.loaded_model2`) used for predictions.
- This allows for flexibility in switching between models, especially if multiple models are trained for different ranges of word lengths or complexities.

### Why These Parameters Are Crucial

#### 1. Focused Predictions
By limiting the model's predictions to words within a specific length range (`min_length` to `max_length`), the model avoids working on cases where it may have insufficient context (short words) or excessive complexity (long words).

#### 2. Contextual Awareness
Limiting the number of missing letters (`max_missing`) ensures that the model only attempts predictions where there is enough revealed context to make an informed guess.

#### 3. Flexibility and Adaptability
The `model` parameter allows for the use of specialized models for different scenarios. For example:
- A model trained on shorter words could be used for `min_length=4`, `max_length=8`.
- Another model, like `self.loaded_model2`, is configured to handle a broader range of word lengths (`min_length=4`, `max_length=30`).

#### 4. Optimized Resource Allocation
The configuration ensures that computational resources are focused on scenarios where the model is most likely to succeed, enhancing overall efficiency and effectiveness.

### Combining Performance Insights with Optimized Configuration

The evaluation of the model's performance is directly influenced by this configuration. For example:

#### Correct Predictions by Word Length
The range of word lengths (`min_length=4` to `max_length=30`) ensures that the model operates within its optimal performance range.

#### Correct Predictions by Position
The restriction on the number of missing letters (`max_missing=3`) ensures that enough context is available, which improves the model’s accuracy for positional predictions.

#### Correct Predictions by Remaining Letters
The configuration minimizes instances where too many letters remain unguessed, helping the model make more focused and efficient predictions.



## 6. Final Strategy: Leveraging Multiple N-gram Probabilities and a Transformer Model

The final strategy for predicting the missing letters in the Hangman game revolves around a hybrid approach that combines n-gram probabilities from multiple dictionaries and the output of a trained Transformer model. Here's a breakdown of the approach and the rationale behind its effectiveness:

### Multi-Source N-gram Probabilities

#### Datasets Used for N-gram Computation

##### `words_250000_train.txt`
- This dataset contains 250,000 words and serves as the primary reference for n-gram probabilities.
- It represents the most likely context for the Hangman game.

##### `words_not_contained.txt`
- A custom dataset containing over 180,000 words, none of which overlap with `words_250000_train.txt`.
- By including this dataset, the model generalizes better to out-of-dictionary words, improving robustness.

##### Words from the `nltk` Library
- The `nltk` word corpus includes words that might overlap with the other two datasets but also introduces new words.
- It expands the linguistic diversity of the training context, helping the model handle rare or unseen words.

#### Combining Probabilities
- N-gram probabilities are calculated for each dataset up to `n=30` using the `get_n_gram_prob` method.
- Probabilities from all three datasets are summed to create a robust and balanced probability distribution:
  - **Primary Context:** Frequent patterns from the primary dataset are emphasized.
  - **Generalization:** Rare patterns from additional datasets improve adaptability.

#### Advantages of Multi-Source N-gram Probabilities
- **Reduced Overfitting:** The combined approach prevents the model from being overly reliant on any single dataset.
- **Improved Robustness:** Frequent patterns are emphasized, and rare patterns provide adaptability for edge-case words.

---

### Integration with the Transformer Model

#### When the Transformer Model is Used
- The Transformer model is applied when the word length and the number of missing letters satisfy configuration criteria (`min_length`, `max_length`, `max_missing`).

#### Role of the Transformer Model
- The pre-trained Hybrid Transformer Model adds contextual information that n-grams alone cannot provide, especially for complex patterns or longer words.

#### How N-gram Probabilities Are Used with the Transformer
- N-gram probabilities are passed as additional inputs to the model.
- The output probabilities from the Transformer model (`output_probs_model`) are added back to the combined n-gram probabilities to enhance predictions.

#### Advantages of Integrating the Transformer Model
- **Global Dependencies:** The Transformer captures long-range dependencies and semantic patterns.
- **Local Context:** N-grams provide statistical insights that the Transformer leverages.
- **Hybrid Approach:** Combining outputs ensures that predictions benefit from both statistical and deep learning capabilities.

---

### Why This Strategy Works

#### Breadth of Context
- The use of multiple dictionaries ensures the model is not overly reliant on any single source of linguistic patterns.
- Predictions become more adaptable to diverse and unseen words.

#### Depth of Understanding
- The Transformer model adds semantic understanding to complement the statistical patterns captured by n-grams.

#### Hybrid Optimization
- Combining n-gram probabilities and Transformer outputs optimizes both precision (accuracy on common patterns) and recall (generalization to edge cases).

#### Real-World Performance
- The method performed best in terms of accuracy across test sets during simulations.
- It consistently predicted correct letters under the constraints of the Hangman game.

---

### Justification of Model Usage

- The Transformer model is not merely a fallback but an integral part of the strategy.
- It is used only when there is sufficient context to make its predictions meaningful (as defined by the `self.config` parameters).
- Its predictions are blended with n-gram probabilities, ensuring the model is both efficient and reliable.

#### Final Remarks
This balanced approach leverages the strengths of each component, achieving a high success rate in the Hangman game while remaining computationally feasible.


## 7. Advanced Strategies and Techniques in the Hangman Model

In addition to the Hybrid Transformer Model and its integration of masked word inputs, n-gram probabilities, and missing character embeddings, the Hangman solution incorporates several advanced strategies to optimize performance. These include heuristic-based letter selection, reinforcement learning, and evaluations with alternative models. Below is a detailed explanation:

### Heuristic-Based Letter Selection for Unknown Words

#### Motivation
- When no letters in the target word are revealed, the model leverages word length-specific heuristics to prioritize letters for guessing.
- This approach blends letter frequencies from the dataset with heuristic patterns observed in common English usage.

#### Method

##### Letter Frequency by Word Length
- Letters are scored based on their occurrence in the dataset, specifically for words matching the current word length.
- For example, if the target word is 7 characters long, the model focuses on letters that frequently appear in 7-letter words.

##### Common Word Patterns
- The function boosts scores for letters that appear in frequently used words of the same length.

##### General English Letter Frequency
- Letters are further scored based on their overall frequency in the English language, with high-frequency letters like `e`, `t`, and `a` receiving priority.

#### Output
- A sorted list of letters (excluding already guessed letters) is returned, optimizing the initial guesses.

#### Example
- For a 6-letter word, the function may return a list like: `['e', 'a', 'o', 't', 'n', 'i']`.

---

### Alternative Models: Transformer, LSTM, GRU, and Reinforcement Learning

#### Experiments with LSTM and GRU

1. **LSTM and GRU Performance**:
   - Models based on Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU) were tested as alternatives to the Transformer.
   - While LSTMs and GRUs excel at capturing sequential dependencies, they performed less effectively than the Transformer in Hangman simulations.

2. **Why Transformer Performed Better**:
   - **Self-Attention**: The Transformer’s self-attention mechanism allowed it to capture relationships between distant letters in words.
   - **Efficiency**: The parallelized computations in Transformers made training faster for larger datasets.

---

### Reinforcement Learning Model: Combining Exploration and Learned Strategies

#### Overview
- The RL agent operates by learning optimal guessing strategies based on feedback from its interactions with the Hangman game.
- It uses a Deep Q-Network (DQN) to approximate the value of actions (i.e., guessing specific letters) in different game states.

---

#### Key Components

##### State Representation
- The current state of the game (e.g., the partially guessed word, guessed letters, attempts left) is encoded numerically.
- **Example**: For the word `_p_l_`, the state might be represented as `[26, 15, 11, 26, 26]`, where missing letters are represented by `26`.

##### Action Space
- The RL agent can choose any of the 26 letters (a to z) as its next action.

---

#### Reward System in Reinforcement Learning for Hangman

##### 1. Reward Modifier Based on Letter Ranking
- **Purpose**: Reward the agent for selecting high-probability letters based on precomputed n-gram probabilities and heuristics.
- **Implementation**:
  - If the guessed letter is among the top recommended letters, it receives a positive reward based on its rank:
    - Example: If the guessed letter is the top-ranked letter, it gets a reward of `1.0`. For the second-ranked letter, it gets `0.5`, and so on.
  - If the guessed letter is not in the top recommendations, a small penalty of `-0.15` is applied.

##### 2. Rewards for Correct Guesses
- **Purpose**: Reward the agent for correctly guessing letters in the target word.
- **Implementation**:
  - A base reward of `0.5` is given for any correct guess.
  - Additional reward is proportional to the number of letters revealed relative to the word length:
    ```
    reward increment = (letters revealed so far / total letters in the word) * 3
    ```
  - If the entire word is guessed correctly, an additional reward is given based on the word length.

##### 3. Penalties for Incorrect Guesses
- **Purpose**: Discourage the agent from making incorrect guesses.
- **Implementation**:
  - Penalty is proportional to the number of failed attempts:
    ```
    penalty = -0.15 * (max attempts - remaining attempts)
    ```
  - If the agent runs out of attempts, a larger penalty is applied.

##### 4. Small Baseline Reward
- **Purpose**: Provide stability in training and avoid overly conservative actions.
- **Implementation**:
  - A small reward of `0.05` is added to every step, regardless of the guess.

---

#### Combined Reward Formula
At each step, the total reward is calculated as:
    - reward = base reward or penalty + index_reward + 0.05




###### Exploration vs. Exploitation
- The agent balances exploration (guessing new letters) with exploitation (using the learned policy) via an epsilon-greedy strategy.
- Over time, the agent focuses more on exploitation as it learns from the game.

---

##### Training

1. **Replay Memory**:
   - The agent is trained over thousands of games using a replay memory to store past transitions (`state, action, reward, next_state`).
   - This ensures that the agent learns from a wide variety of scenarios.

2. **Policy Network Updates**:
   - The policy network is updated using backpropagation to minimize the loss between predicted and actual Q-values.

---

##### Example of Learning
1. During training, the agent encounters a word like `_a___a`.
2. Computes a Q-value for each possible action (letter).
3. Selects the letter with the highest Q-value (e.g., `n`) unless exploration is chosen.
4. Updates its policy based on whether the guess was correct or not.

---

##### Visualization of Learning Progress
- The RL agent's performance is monitored across epochs using metrics like:
  - Number of remaining attempts.
  - Success rates.
  - Steps to guess the word.
- Plots (e.g., training progress, attempts left) provide insights into the agent's learning curve.
![RL_results_f](https://github.com/user-attachments/assets/bb12c0b8-596d-4f9f-abd8-d695533bcb4c)

#!/usr/bin/env python3
"""
Neural Network Model of Two-Digit Addition: Testing Dehaene's Triple-Code Model

This script implements a neural network model to test the hypothesis that number
presentation format (digits vs. words) affects arithmetic processing due to
transcoding costs, as proposed by Dehaene's (1992) Triple-Code Model.

Author: [Your Name]
Course: PSY 360
Date: [Date]

Usage:
    python addition_transcoding_model.py

    Or with your own data:
    python addition_transcoding_model.py --data your_data.csv

Requirements:
    pip install numpy pandas matplotlib seaborn torch scikit-learn scipy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional
import argparse
import warnings
warnings.filterwarnings('ignore')

# Check if PyTorch is available, otherwise use numpy-based implementation
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Using NumPy-based implementation.")


# =============================================================================
# PART 1: PROBLEM GENERATION AND ENCODING
# =============================================================================

class ProblemGenerator:
    """Generate two-digit addition problems with carry/no-carry classification."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        
    def generate_problem_set(self, n_per_condition: int = 10) -> pd.DataFrame:
        """
        Generate a balanced set of problems across 4 conditions.
        
        Conditions:
        - No-Carry + Digit format
        - Carry + Digit format  
        - No-Carry + Word format
        - Carry + Word format
        """
        problems = []
        
        # Generate no-carry problems (units sum < 10)
        no_carry = []
        while len(no_carry) < n_per_condition:
            a = np.random.randint(10, 100)
            b = np.random.randint(10, 100)
            if (a % 10) + (b % 10) < 10:  # No carry condition
                if a != b:  # Exclude ties
                    no_carry.append((a, b, a + b, False))
        
        # Generate carry problems (units sum >= 10)
        carry = []
        while len(carry) < n_per_condition:
            a = np.random.randint(10, 100)
            b = np.random.randint(10, 100)
            if (a % 10) + (b % 10) >= 10:  # Carry condition
                if a != b:  # Exclude ties
                    carry.append((a, b, a + b, True))
        
        # Create dataframe
        for num1, num2, answer, has_carry in no_carry + carry:
            problems.append({
                'num1': num1,
                'num2': num2,
                'answer': answer,
                'has_carry': has_carry
            })
        
        return pd.DataFrame(problems)
    
    @staticmethod
    def has_carry(num1: int, num2: int) -> bool:
        """Check if adding two numbers requires a carry operation."""
        return (num1 % 10) + (num2 % 10) >= 10


class NumberEncoder:
    """
    Encode numbers in different formats following Dehaene's Triple-Code Model.
    
    Implements two encoding schemes:
    1. Digit format (Visual-Arabic Code): Compact positional encoding
    2. Word format (Verbal Code): Morpheme-based distributed encoding
    """
    
    # Mapping for word encoding
    DECADES = ['twenty', 'thirty', 'forty', 'fifty', 
               'sixty', 'seventy', 'eighty', 'ninety']
    UNITS = ['zero', 'one', 'two', 'three', 'four',
             'five', 'six', 'seven', 'eight', 'nine']
    TEENS = ['ten', 'eleven', 'twelve', 'thirteen', 'fourteen',
             'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']
    
    def __init__(self):
        self.digit_input_size = 4  # [tens1, units1, tens2, units2]
        self.word_input_size = 36  # 2 numbers × (8 decades + 10 units)
        
    def encode_digit_format(self, num1: int, num2: int) -> np.ndarray:
        """
        Encode two numbers in digit format (Visual-Arabic Code).
        
        This is a compact, positional encoding where each digit is directly
        represented by its numerical value, normalized to [0, 1].
        
        Example: 48 + 37 → [0.4, 0.8, 0.3, 0.7]
        """
        encoding = np.array([
            num1 // 10,  # tens digit of first number
            num1 % 10,   # units digit of first number
            num2 // 10,  # tens digit of second number
            num2 % 10    # units digit of second number
        ], dtype=np.float32) / 9.0  # Normalize to [0, 1]
        
        return encoding
    
    def encode_word_format(self, num1: int, num2: int) -> np.ndarray:
        """
        Encode two numbers in word format (Verbal Code).
        
        This uses a morpheme-based encoding where decade words and unit words
        are represented separately using one-hot encoding.
        
        Example: 48 ("forty-eight") → 
            decades: [0,0,1,0,0,0,0,0] (forty is index 2)
            units:   [0,0,0,0,0,0,0,0,1,0] (eight is index 8)
        
        The key insight: this encoding obscures the positional/magnitude
        relationship that is transparent in the digit format.
        """
        def encode_single_number(n: int) -> np.ndarray:
            """Encode a single two-digit number as morpheme features."""
            decade_idx = (n // 10) - 2  # 20→0, 30→1, ..., 90→7
            unit_idx = n % 10
            
            # One-hot encode decade (8 options: 20-90)
            decade_onehot = np.zeros(8, dtype=np.float32)
            if 0 <= decade_idx < 8:
                decade_onehot[decade_idx] = 1.0
            
            # One-hot encode unit (10 options: 0-9)
            unit_onehot = np.zeros(10, dtype=np.float32)
            unit_onehot[unit_idx] = 1.0
            
            return np.concatenate([decade_onehot, unit_onehot])
        
        # Concatenate encodings for both numbers
        encoding = np.concatenate([
            encode_single_number(num1),
            encode_single_number(num2)
        ])
        
        return encoding  # 36-dimensional vector
    
    def encode_answer(self, answer: int, method: str = 'normalized') -> np.ndarray:
        """
        Encode the answer for network output.
        
        Methods:
        - 'normalized': Single value normalized by max possible answer
        - 'digits': Three separate digits [hundreds, tens, units]
        """
        if method == 'normalized':
            # Max answer for two-digit addition: 99 + 99 = 198
            return np.array([answer / 198.0], dtype=np.float32)
        elif method == 'digits':
            hundreds = answer // 100
            tens = (answer % 100) // 10
            units = answer % 10
            return np.array([hundreds, tens, units], dtype=np.float32) / 9.0
        else:
            raise ValueError(f"Unknown encoding method: {method}")


# =============================================================================
# PART 2: NEURAL NETWORK MODEL
# =============================================================================

if TORCH_AVAILABLE:
    class AdditionNetwork(nn.Module):
        """
        Feedforward neural network for learning two-digit addition.
        
        Architecture:
        - Input layer (size depends on encoding format)
        - Hidden layer(s) with ReLU activation
        - Output layer (predicted sum)
        """
        
        def __init__(self, input_size: int, hidden_size: int = 64, 
                     num_hidden_layers: int = 2, output_size: int = 1):
            super().__init__()
            
            layers = []
            
            # Input to first hidden
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            
            # Additional hidden layers
            for _ in range(num_hidden_layers - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.ReLU())
            
            # Output layer
            layers.append(nn.Linear(hidden_size, output_size))
            
            self.network = nn.Sequential(*layers)
            
        def forward(self, x):
            return self.network(x)


class NumpyAdditionNetwork:
    """
    NumPy-based neural network (fallback if PyTorch unavailable).
    
    Simple feedforward network with one hidden layer.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights with Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(hidden_size)
        self.W3 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b3 = np.zeros(output_size)
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        return self.z3
    
    def backward(self, x, y, output, learning_rate=0.001):
        m = x.shape[0]
        
        # Output layer gradient
        dz3 = output - y
        dW3 = self.a2.T @ dz3 / m
        db3 = np.mean(dz3, axis=0)
        
        # Second hidden layer gradient
        da2 = dz3 @ self.W3.T
        dz2 = da2 * (self.z2 > 0)
        dW2 = self.a1.T @ dz2 / m
        db2 = np.mean(dz2, axis=0)
        
        # First hidden layer gradient
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)
        dW1 = x.T @ dz1 / m
        db1 = np.mean(dz1, axis=0)
        
        # Update weights
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train_step(self, x, y, learning_rate=0.001):
        output = self.forward(x)
        loss = np.mean((output - y) ** 2)
        self.backward(x, y, output, learning_rate)
        return loss


# =============================================================================
# PART 3: MODEL TRAINING AND EVALUATION
# =============================================================================

class TranscodingExperiment:
    """
    Main experiment class that trains networks on different formats
    and extracts difficulty measures.
    """
    
    def __init__(self, hidden_size: int = 64, num_hidden_layers: int = 2,
                 learning_rate: float = 0.001, seed: int = 42):
        self.encoder = NumberEncoder()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.learning_rate = learning_rate
        self.seed = seed
        
        # Results storage
        self.results = {}
        
    def generate_training_data(self, n_samples: int = 1000) -> Dict:
        """Generate training data for both formats."""
        np.random.seed(self.seed)
        
        data = {
            'digit': {'X': [], 'y': [], 'problems': []},
            'word': {'X': [], 'y': [], 'problems': []}
        }
        
        for _ in range(n_samples):
            num1 = np.random.randint(10, 100)
            num2 = np.random.randint(10, 100)
            answer = num1 + num2
            has_carry = ProblemGenerator.has_carry(num1, num2)
            
            # Digit format
            data['digit']['X'].append(self.encoder.encode_digit_format(num1, num2))
            data['digit']['y'].append(self.encoder.encode_answer(answer))
            data['digit']['problems'].append({
                'num1': num1, 'num2': num2, 'answer': answer, 'has_carry': has_carry
            })
            
            # Word format
            data['word']['X'].append(self.encoder.encode_word_format(num1, num2))
            data['word']['y'].append(self.encoder.encode_answer(answer))
            data['word']['problems'].append({
                'num1': num1, 'num2': num2, 'answer': answer, 'has_carry': has_carry
            })
        
        # Convert to arrays
        for fmt in ['digit', 'word']:
            data[fmt]['X'] = np.array(data[fmt]['X'])
            data[fmt]['y'] = np.array(data[fmt]['y'])
            
        return data
    
    def train_network_pytorch(self, X: np.ndarray, y: np.ndarray, 
                              n_epochs: int = 500) -> Tuple:
        """Train a network using PyTorch."""
        input_size = X.shape[1]
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Create network
        torch.manual_seed(self.seed)
        network = AdditionNetwork(input_size, self.hidden_size, 
                                  self.num_hidden_layers)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(network.parameters(), lr=self.learning_rate)
        
        # Training loop
        losses = []
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            outputs = network(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        return network, losses
    
    def train_network_numpy(self, X: np.ndarray, y: np.ndarray,
                            n_epochs: int = 500) -> Tuple[NumpyAdditionNetwork, List[float]]:
        """Train a network using NumPy (fallback)."""
        input_size = X.shape[1]
        
        np.random.seed(self.seed)
        network = NumpyAdditionNetwork(input_size, self.hidden_size)
        
        losses = []
        for epoch in range(n_epochs):
            loss = network.train_step(X, y, self.learning_rate)
            losses.append(loss)
            
        return network, losses
    
    def train_network(self, X: np.ndarray, y: np.ndarray, 
                      n_epochs: int = 500) -> Tuple:
        """Train a network using available backend."""
        if TORCH_AVAILABLE:
            return self.train_network_pytorch(X, y, n_epochs)
        else:
            return self.train_network_numpy(X, y, n_epochs)
    
    def evaluate_by_condition(self, network, X: np.ndarray, y: np.ndarray,
                              problems: List[Dict]) -> Dict:
        """
        Evaluate network performance separately for carry/no-carry conditions.
        
        Returns mean squared error (difficulty measure) for each condition.
        """
        if TORCH_AVAILABLE and isinstance(network, nn.Module):
            network.eval()
            with torch.no_grad():
                predictions = network(torch.FloatTensor(X)).numpy()
        else:
            predictions = network.forward(X)
        
        # Calculate per-problem error
        errors = (predictions - y) ** 2
        
        # Separate by condition
        results = {'carry': [], 'no_carry': []}
        for i, prob in enumerate(problems):
            if prob['has_carry']:
                results['carry'].append(errors[i, 0])
            else:
                results['no_carry'].append(errors[i, 0])
        
        return {
            'carry_mean_error': np.mean(results['carry']),
            'carry_std_error': np.std(results['carry']),
            'no_carry_mean_error': np.mean(results['no_carry']),
            'no_carry_std_error': np.std(results['no_carry']),
            'carry_errors': results['carry'],
            'no_carry_errors': results['no_carry']
        }
    
    def run_experiment(self, n_train: int = 2000, n_test: int = 500,
                       n_epochs: int = 500) -> Dict:
        """
        Run the full experiment: train networks on both formats,
        evaluate on test set, and compare predictions.
        """
        print("=" * 60)
        print("NEURAL NETWORK MODEL OF TRANSCODING COSTS")
        print("Testing Dehaene's Triple-Code Model")
        print("=" * 60)
        
        # Generate data
        print("\n[1] Generating training and test data...")
        train_data = self.generate_training_data(n_train)
        
        # Use different seed for test data
        self.seed += 1000
        test_data = self.generate_training_data(n_test)
        self.seed -= 1000
        
        results = {}
        
        # Train and evaluate for each format
        for fmt in ['digit', 'word']:
            print(f"\n[2] Training {fmt.upper()} format network...")
            
            network, losses = self.train_network(
                train_data[fmt]['X'], 
                train_data[fmt]['y'],
                n_epochs
            )
            
            print(f"    Final training loss: {losses[-1]:.6f}")
            
            # Evaluate on test set
            eval_results = self.evaluate_by_condition(
                network,
                test_data[fmt]['X'],
                test_data[fmt]['y'],
                test_data[fmt]['problems']
            )
            
            results[fmt] = {
                'network': network,
                'training_losses': losses,
                'evaluation': eval_results
            }
            
            print(f"    Test error (no-carry): {eval_results['no_carry_mean_error']:.6f}")
            print(f"    Test error (carry):    {eval_results['carry_mean_error']:.6f}")
        
        self.results = results
        return results
    
    def get_model_predictions(self) -> pd.DataFrame:
        """
        Extract model predictions in a format ready for comparison with human data.
        
        Returns DataFrame with columns:
        - format: 'digit' or 'word'
        - condition: 'carry' or 'no_carry'
        - difficulty: mean squared error (proxy for RT)
        """
        predictions = []
        
        for fmt in ['digit', 'word']:
            eval_results = self.results[fmt]['evaluation']
            
            predictions.append({
                'format': fmt,
                'condition': 'no_carry',
                'difficulty': eval_results['no_carry_mean_error'],
                'difficulty_std': eval_results['no_carry_std_error']
            })
            predictions.append({
                'format': fmt,
                'condition': 'carry',
                'difficulty': eval_results['carry_mean_error'],
                'difficulty_std': eval_results['carry_std_error']
            })
        
        return pd.DataFrame(predictions)


# =============================================================================
# PART 4: COMPARISON WITH HUMAN DATA
# =============================================================================

class HumanModelComparison:
    """
    Compare neural network predictions with human behavioral data.
    """
    
    def __init__(self, model_results: pd.DataFrame, human_data: Optional[pd.DataFrame] = None):
        self.model_results = model_results
        self.human_data = human_data
        
    def generate_synthetic_human_data(self) -> pd.DataFrame:
        """
        Generate synthetic human data based on typical effects found in literature.
        
        Based on Campbell & Fugelsang (2001) findings:
        - Word format increases RT by ~200-400ms
        - Carry operations increase RT by ~150-300ms
        - Possible interaction effect
        
        Replace this with your actual data!
        """
        print("\n[!] Using SYNTHETIC human data for demonstration.")
        print("    Replace with your actual participant data.\n")
        
        np.random.seed(42)
        
        # Base RT around 2000ms for simple digit problems
        base_rt = 2000
        format_effect = 350  # ms added for word format
        carry_effect = 200   # ms added for carry
        interaction = 100    # additional ms for word + carry
        
        data = []
        n_participants = 15
        n_trials_per_condition = 10
        
        for pid in range(n_participants):
            participant_offset = np.random.normal(0, 200)
            
            for _ in range(n_trials_per_condition):
                # Digit, No-Carry
                data.append({
                    'participant': pid,
                    'format': 'digit',
                    'condition': 'no_carry',
                    'rt': base_rt + participant_offset + np.random.normal(0, 150)
                })
                
                # Digit, Carry
                data.append({
                    'participant': pid,
                    'format': 'digit',
                    'condition': 'carry',
                    'rt': base_rt + carry_effect + participant_offset + np.random.normal(0, 180)
                })
                
                # Word, No-Carry
                data.append({
                    'participant': pid,
                    'format': 'word',
                    'condition': 'no_carry',
                    'rt': base_rt + format_effect + participant_offset + np.random.normal(0, 200)
                })
                
                # Word, Carry (with interaction)
                data.append({
                    'participant': pid,
                    'format': 'word',
                    'condition': 'carry',
                    'rt': base_rt + format_effect + carry_effect + interaction + 
                          participant_offset + np.random.normal(0, 220)
                })
        
        return pd.DataFrame(data)
    
    def load_human_data(self, filepath: str) -> pd.DataFrame:
        """
        Load human data from CSV file.
        
        Handles the following column format:
        - SubjectID: participant ID
        - Format: 'digit' or 'word' (or 'Digit'/'Word')
        - Complexity: 'carry' or 'no_carry' (or 'Carry'/'No-Carry'/'NoCarry')
        - RT_ms: reaction time in milliseconds
        - Accuracy: 1 or 0 (correct/incorrect)
        
        Also handles: ConditionCode, N1, N2, CorrectAnswer, UserAnswer, WasEdited
        """
        df = pd.read_csv(filepath)
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Create standardized columns
        # Map SubjectID -> participant
        if 'SubjectID' in df.columns:
            df['participant'] = df['SubjectID']
        elif 'subjectid' in df.columns.str.lower().tolist():
            df['participant'] = df[df.columns[df.columns.str.lower() == 'subjectid'][0]]
        
        # Map Format -> format (standardize to lowercase 'digit' or 'word')
        if 'Format' in df.columns:
            df['format'] = df['Format'].str.lower().str.strip()
        elif 'format' in df.columns.str.lower().tolist():
            col_name = df.columns[df.columns.str.lower() == 'format'][0]
            df['format'] = df[col_name].str.lower().str.strip()
        
        # Map Complexity -> condition (standardize to 'carry' or 'no_carry')
        if 'Complexity' in df.columns:
            df['condition'] = df['Complexity'].str.lower().str.strip().replace({
                'carry': 'carry',
                'no-carry': 'no_carry',
                'nocarry': 'no_carry',
                'no_carry': 'no_carry'
            })
        elif 'complexity' in df.columns.str.lower().tolist():
            col_name = df.columns[df.columns.str.lower() == 'complexity'][0]
            df['condition'] = df[col_name].str.lower().str.strip().replace({
                'carry': 'carry',
                'no-carry': 'no_carry',
                'nocarry': 'no_carry',
                'no_carry': 'no_carry'
            })
        elif 'has_carry' in df.columns.str.lower().tolist():
            col_name = df.columns[df.columns.str.lower() == 'has_carry'][0]
            df['condition'] = df[col_name].apply(
                lambda x: 'carry' if x else 'no_carry'
            )
        
        # Map RT_ms -> rt
        if 'RT_ms' in df.columns:
            df['rt'] = df['RT_ms']
        elif 'rt_ms' in df.columns.str.lower().tolist():
            col_name = df.columns[df.columns.str.lower() == 'rt_ms'][0]
            df['rt'] = df[col_name]
        elif 'rt' in df.columns.str.lower().tolist():
            col_name = df.columns[df.columns.str.lower() == 'rt'][0]
            df['rt'] = df[col_name]
        
        # Map Accuracy -> accuracy
        if 'Accuracy' in df.columns:
            df['accuracy'] = df['Accuracy']
        elif 'accuracy' in df.columns.str.lower().tolist():
            col_name = df.columns[df.columns.str.lower() == 'accuracy'][0]
            df['accuracy'] = df[col_name]
        
        # Keep other useful columns if present
        if 'N1' in df.columns:
            df['num1'] = df['N1']
        if 'N2' in df.columns:
            df['num2'] = df['N2']
        if 'CorrectAnswer' in df.columns:
            df['correct_answer'] = df['CorrectAnswer']
        if 'UserAnswer' in df.columns:
            df['user_answer'] = df['UserAnswer']
        
        # Print summary of loaded data
        print(f"    Loaded {len(df)} trials from {df['participant'].nunique()} participants")
        print(f"    Conditions: {df['condition'].unique()}")
        print(f"    Formats: {df['format'].unique()}")
        print(f"    Mean RT: {df['rt'].mean():.1f} ms")
        if 'accuracy' in df.columns:
            print(f"    Mean Accuracy: {df['accuracy'].mean()*100:.1f}%")
        
        return df
    
    def aggregate_human_data(self, df: pd.DataFrame, correct_only: bool = True) -> pd.DataFrame:
        """
        Aggregate human data by format and condition.
        
        Args:
            df: Raw trial-level data
            correct_only: If True, only include correct trials in RT analysis (standard practice)
        """
        if correct_only and 'accuracy' in df.columns:
            n_total = len(df)
            df_filtered = df[df['accuracy'] == 1].copy()
            n_correct = len(df_filtered)
            print(f"    Filtering for correct trials: {n_correct}/{n_total} ({100*n_correct/n_total:.1f}%)")
        else:
            df_filtered = df.copy()
            if correct_only:
                print("    Warning: No 'accuracy' column found, using all trials")
        
        # Aggregate RT
        agg = df_filtered.groupby(['format', 'condition'])['rt'].agg(['mean', 'std', 'count'])
        agg = agg.reset_index()
        agg.columns = ['format', 'condition', 'rt_mean', 'rt_std', 'n']
        
        return agg
    
    def analyze_accuracy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze accuracy by condition."""
        if 'accuracy' not in df.columns:
            print("    No accuracy column found, skipping accuracy analysis")
            return None
        
        acc_agg = df.groupby(['format', 'condition'])['accuracy'].agg(['mean', 'std', 'count'])
        acc_agg = acc_agg.reset_index()
        acc_agg.columns = ['format', 'condition', 'accuracy_mean', 'accuracy_std', 'n']
        
        print("\n--- ACCURACY BY CONDITION ---")
        for _, row in acc_agg.iterrows():
            print(f"  {row['format'].capitalize():6s} + {row['condition'].replace('_', ' '):10s}: "
                  f"{row['accuracy_mean']*100:.1f}% correct (n={row['n']})")
        
        return acc_agg
    
    def compute_correlation(self, human_agg: pd.DataFrame) -> Dict:
        """
        Compute correlation between model difficulty and human RT.
        """
        # Merge model and human data
        merged = pd.merge(
            self.model_results,
            human_agg,
            on=['format', 'condition']
        )
        
        # Compute correlation
        r, p = stats.pearsonr(merged['difficulty'], merged['rt_mean'])
        
        return {
            'correlation': r,
            'p_value': p,
            'merged_data': merged
        }
    
    def run_anova_on_human_data(self, df: pd.DataFrame) -> Dict:
        """
        Run 2x2 repeated measures ANOVA on human RT data.
        
        Note: This is a simplified between-subjects ANOVA.
        For true repeated measures, use statsmodels or R.
        """
        # Aggregate by participant first
        participant_means = df.groupby(['participant', 'format', 'condition'])['rt'].mean().reset_index()
        
        # Get condition means
        digit_nocarry = participant_means[
            (participant_means['format'] == 'digit') & 
            (participant_means['condition'] == 'no_carry')
        ]['rt'].values
        
        digit_carry = participant_means[
            (participant_means['format'] == 'digit') & 
            (participant_means['condition'] == 'carry')
        ]['rt'].values
        
        word_nocarry = participant_means[
            (participant_means['format'] == 'word') & 
            (participant_means['condition'] == 'no_carry')
        ]['rt'].values
        
        word_carry = participant_means[
            (participant_means['format'] == 'word') & 
            (participant_means['condition'] == 'carry')
        ]['rt'].values
        
        # Main effect of format
        digit_all = np.concatenate([digit_nocarry, digit_carry])
        word_all = np.concatenate([word_nocarry, word_carry])
        format_t, format_p = stats.ttest_ind(digit_all, word_all)
        
        # Main effect of carry
        nocarry_all = np.concatenate([digit_nocarry, word_nocarry])
        carry_all = np.concatenate([digit_carry, word_carry])
        carry_t, carry_p = stats.ttest_ind(nocarry_all, carry_all)
        
        # Interaction: Compare format effect in carry vs no-carry
        format_effect_nocarry = word_nocarry - digit_nocarry
        format_effect_carry = word_carry - digit_carry
        interaction_t, interaction_p = stats.ttest_ind(
            format_effect_nocarry, format_effect_carry
        )
        
        return {
            'format_effect': {
                't': format_t, 'p': format_p,
                'digit_mean': np.mean(digit_all),
                'word_mean': np.mean(word_all)
            },
            'carry_effect': {
                't': carry_t, 'p': carry_p,
                'nocarry_mean': np.mean(nocarry_all),
                'carry_mean': np.mean(carry_all)
            },
            'interaction': {
                't': interaction_t, 'p': interaction_p,
                'format_effect_in_nocarry': np.mean(format_effect_nocarry),
                'format_effect_in_carry': np.mean(format_effect_carry)
            }
        }


# =============================================================================
# PART 5: VISUALIZATION
# =============================================================================

class ResultsVisualizer:
    """Create publication-ready figures for the results."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 5)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-whitegrid')
        
    def plot_model_predictions(self, model_results: pd.DataFrame, 
                               save_path: Optional[str] = None):
        """Plot model difficulty predictions as bar chart."""
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Prepare data for plotting
        pivot = model_results.pivot(index='condition', columns='format', values='difficulty')
        pivot = pivot.reindex(['no_carry', 'carry'])
        
        x = np.arange(2)
        width = 0.35
        
        bars1 = ax.bar(x - width/2, pivot['digit'], width, label='Digit Format',
                       color='#2ecc71', edgecolor='black')
        bars2 = ax.bar(x + width/2, pivot['word'], width, label='Word Format',
                       color='#e74c3c', edgecolor='black')
        
        ax.set_ylabel('Model Difficulty (MSE)', fontsize=12)
        ax.set_xlabel('Problem Type', fontsize=12)
        ax.set_title('Neural Network Model Predictions', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['No Carry', 'Carry'])
        ax.legend()
        
        # Add value labels
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_human_vs_model(self, model_results: pd.DataFrame,
                            human_agg: pd.DataFrame,
                            save_path: Optional[str] = None):
        """Plot side-by-side comparison of model and human data."""
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Model predictions
        ax1 = axes[0]
        pivot_model = model_results.pivot(index='condition', columns='format', values='difficulty')
        pivot_model = pivot_model.reindex(['no_carry', 'carry'])
        
        x = np.arange(2)
        width = 0.35
        
        ax1.bar(x - width/2, pivot_model['digit'], width, label='Digit',
                color='#2ecc71', edgecolor='black')
        ax1.bar(x + width/2, pivot_model['word'], width, label='Word',
                color='#e74c3c', edgecolor='black')
        
        ax1.set_ylabel('Difficulty (MSE)', fontsize=12)
        ax1.set_xlabel('Problem Type', fontsize=12)
        ax1.set_title('Model Predictions', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['No Carry', 'Carry'])
        ax1.legend()
        
        # Human data
        ax2 = axes[1]
        pivot_human = human_agg.pivot(index='condition', columns='format', values='rt_mean')
        pivot_human = pivot_human.reindex(['no_carry', 'carry'])
        
        pivot_human_std = human_agg.pivot(index='condition', columns='format', values='rt_std')
        pivot_human_std = pivot_human_std.reindex(['no_carry', 'carry'])
        
        ax2.bar(x - width/2, pivot_human['digit'], width, label='Digit',
                color='#2ecc71', edgecolor='black',
                yerr=pivot_human_std['digit']/np.sqrt(15), capsize=3)
        ax2.bar(x + width/2, pivot_human['word'], width, label='Word',
                color='#e74c3c', edgecolor='black',
                yerr=pivot_human_std['word']/np.sqrt(15), capsize=3)
        
        ax2.set_ylabel('Reaction Time (ms)', fontsize=12)
        ax2.set_xlabel('Problem Type', fontsize=12)
        ax2.set_title('Human Data', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['No Carry', 'Carry'])
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_learning_curves(self, results: Dict, save_path: Optional[str] = None):
        """Plot training loss curves for both formats."""
        fig, ax = plt.subplots(figsize=(8, 5))
        
        epochs = range(1, len(results['digit']['training_losses']) + 1)
        
        ax.plot(epochs, results['digit']['training_losses'], 
                label='Digit Format', color='#2ecc71', linewidth=2)
        ax.plot(epochs, results['word']['training_losses'], 
                label='Word Format', color='#e74c3c', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Training Loss (MSE)', fontsize=12)
        ax.set_title('Learning Curves by Format', fontsize=14, fontweight='bold')
        ax.legend()
        ax.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_correlation(self, merged_data: pd.DataFrame, correlation: float,
                         p_value: float, save_path: Optional[str] = None):
        """Plot correlation between model difficulty and human RT."""
        fig, ax = plt.subplots(figsize=(6, 5))
        
        colors = {'digit': '#2ecc71', 'word': '#e74c3c'}
        markers = {'no_carry': 'o', 'carry': 's'}
        
        for _, row in merged_data.iterrows():
            ax.scatter(row['difficulty'], row['rt_mean'],
                      c=colors[row['format']], 
                      marker=markers[row['condition']],
                      s=150, edgecolor='black', linewidth=1.5,
                      label=f"{row['format'].capitalize()}, {row['condition'].replace('_', ' ').title()}")
        
        # Add regression line
        z = np.polyfit(merged_data['difficulty'], merged_data['rt_mean'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(merged_data['difficulty'].min(), merged_data['difficulty'].max(), 100)
        ax.plot(x_line, p(x_line), 'k--', alpha=0.5, linewidth=2)
        
        ax.set_xlabel('Model Difficulty (MSE)', fontsize=12)
        ax.set_ylabel('Human RT (ms)', fontsize=12)
        ax.set_title(f'Model-Human Correlation: r = {correlation:.3f}, p = {p_value:.3f}',
                    fontsize=14, fontweight='bold')
        
        # Create custom legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig


# =============================================================================
# PART 6: MAIN EXECUTION
# =============================================================================

def print_summary_statistics(experiment, comparison, human_agg, anova_results, corr_results):
    """Print a comprehensive summary of results."""
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    # Model results
    print("\n--- MODEL PREDICTIONS ---")
    model_preds = experiment.get_model_predictions()
    for _, row in model_preds.iterrows():
        print(f"  {row['format'].capitalize():6s} + {row['condition'].replace('_', ' '):10s}: "
              f"MSE = {row['difficulty']:.6f}")
    
    # Calculate model effects
    digit_nocarry = model_preds[(model_preds['format']=='digit') & 
                                (model_preds['condition']=='no_carry')]['difficulty'].values[0]
    digit_carry = model_preds[(model_preds['format']=='digit') & 
                              (model_preds['condition']=='carry')]['difficulty'].values[0]
    word_nocarry = model_preds[(model_preds['format']=='word') & 
                               (model_preds['condition']=='no_carry')]['difficulty'].values[0]
    word_carry = model_preds[(model_preds['format']=='word') & 
                             (model_preds['condition']=='carry')]['difficulty'].values[0]
    
    model_format_effect = ((word_nocarry + word_carry)/2) - ((digit_nocarry + digit_carry)/2)
    model_carry_effect = ((digit_carry + word_carry)/2) - ((digit_nocarry + word_nocarry)/2)
    model_interaction = (word_carry - word_nocarry) - (digit_carry - digit_nocarry)
    
    print(f"\n  Model Format Effect:      {model_format_effect:+.6f}")
    print(f"  Model Carry Effect:       {model_carry_effect:+.6f}")
    print(f"  Model Interaction:        {model_interaction:+.6f}")
    
    # Human results
    print("\n--- HUMAN DATA ---")
    for _, row in human_agg.iterrows():
        print(f"  {row['format'].capitalize():6s} + {row['condition'].replace('_', ' '):10s}: "
              f"RT = {row['rt_mean']:.1f} ms (SD = {row['rt_std']:.1f})")
    
    # Statistical tests
    print("\n--- STATISTICAL TESTS (Human Data) ---")
    print(f"  Main Effect of Format:")
    print(f"    t = {anova_results['format_effect']['t']:.3f}, "
          f"p = {anova_results['format_effect']['p']:.4f}")
    print(f"    Digit mean: {anova_results['format_effect']['digit_mean']:.1f} ms")
    print(f"    Word mean:  {anova_results['format_effect']['word_mean']:.1f} ms")
    
    print(f"\n  Main Effect of Carry:")
    print(f"    t = {anova_results['carry_effect']['t']:.3f}, "
          f"p = {anova_results['carry_effect']['p']:.4f}")
    print(f"    No-carry mean: {anova_results['carry_effect']['nocarry_mean']:.1f} ms")
    print(f"    Carry mean:    {anova_results['carry_effect']['carry_mean']:.1f} ms")
    
    print(f"\n  Format × Carry Interaction:")
    print(f"    t = {anova_results['interaction']['t']:.3f}, "
          f"p = {anova_results['interaction']['p']:.4f}")
    print(f"    Format effect in no-carry: {anova_results['interaction']['format_effect_in_nocarry']:.1f} ms")
    print(f"    Format effect in carry:    {anova_results['interaction']['format_effect_in_carry']:.1f} ms")
    
    # Model-Human correlation
    print("\n--- MODEL-HUMAN CORRELATION ---")
    print(f"  Pearson r = {corr_results['correlation']:.3f}")
    print(f"  p-value   = {corr_results['p_value']:.4f}")
    
    if corr_results['p_value'] < 0.05:
        print("  → Significant correlation! Model predictions align with human data.")
    else:
        print("  → Non-significant correlation (but N=4 conditions limits power).")
    
    print("\n" + "=" * 60)


def main(data_path: Optional[str] = None, output_dir: str = '.'):
    """Main execution function."""
    
    # Run neural network experiment
    experiment = TranscodingExperiment(
        hidden_size=64,
        num_hidden_layers=2,
        learning_rate=0.001,
        seed=42
    )
    
    results = experiment.run_experiment(
        n_train=2000,
        n_test=500,
        n_epochs=500
    )
    
    model_predictions = experiment.get_model_predictions()
    
    # Set up comparison with human data
    comparison = HumanModelComparison(model_predictions)
    
    # Load or generate human data
    if data_path:
        print(f"\n[3] Loading human data from: {data_path}")
        human_data = comparison.load_human_data(data_path)
        
        # Analyze accuracy
        accuracy_results = comparison.analyze_accuracy(human_data)
    else:
        human_data = comparison.generate_synthetic_human_data()
        accuracy_results = None
    
    human_agg = comparison.aggregate_human_data(human_data, correct_only=True)
    
    # Run statistical analyses
    print("[4] Running statistical analyses...")
    anova_results = comparison.run_anova_on_human_data(human_data)
    corr_results = comparison.compute_correlation(human_agg)
    
    # Print summary
    print_summary_statistics(experiment, comparison, human_agg, anova_results, corr_results)
    
    # Create visualizations
    print("\n[5] Creating visualizations...")
    viz = ResultsVisualizer()
    
    fig1 = viz.plot_model_predictions(model_predictions, 
                                       f'{output_dir}/model_predictions.png')
    fig2 = viz.plot_human_vs_model(model_predictions, human_agg,
                                    f'{output_dir}/human_vs_model.png')
    fig3 = viz.plot_learning_curves(results,
                                     f'{output_dir}/learning_curves.png')
    fig4 = viz.plot_correlation(corr_results['merged_data'],
                                 corr_results['correlation'],
                                 corr_results['p_value'],
                                 f'{output_dir}/correlation.png')
    
    plt.show()
    
    print("\n[✓] Analysis complete!")
    print(f"    Figures saved to: {output_dir}/")
    
    return {
        'experiment': experiment,
        'model_predictions': model_predictions,
        'human_data': human_data,
        'human_agg': human_agg,
        'anova_results': anova_results,
        'correlation_results': corr_results
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Neural Network Model of Transcoding Costs in Mental Arithmetic'
    )
    parser.add_argument('--data', type=str, default=None,
                        help='Path to CSV file with human RT data')
    parser.add_argument('--output', type=str, default='.',
                        help='Directory to save output figures')
    
    args = parser.parse_args()
    
    results = main(data_path=args.data, output_dir=args.output)
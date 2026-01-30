#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Markov Chain Brute Force Tool - Educational Edition
Advanced statistical modeling tool for sequence generation and analysis

Author: Security Research Team
License: MIT
Version: 1.0.0

WARNING: This tool is for EDUCATIONAL PURPOSES ONLY.
DO NOT use for unauthorized access or illegal activities.
"""

import argparse
import sys
import os
import time
import json
import pickle
import logging
import hashlib
import random
import string
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Optional imports with fallbacks
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("⚠️  NumPy not found. Some features will be limited.")

try:
    from termcolor import colored, cprint
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False
    # Fallback color functions
    def colored(text, color=None, on_color=None, attrs=None):
        return text
    def cprint(text, color=None, on_color=None, attrs=None):
        print(text)

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class MarkovChain:
    """
    Markov Chain model for sequence generation.
    Supports variable order (1 to n) and multiple generation strategies.
    """
    
    def __init__(self, order: int = 2, smoothing: float = 0.01):
        """
        Initialize Markov Chain model.
        
        Args:
            order: Number of previous states to consider (1-10)
            smoothing: Laplace smoothing factor for unseen transitions
        """
        self.order = max(1, min(10, order))
        self.smoothing = smoothing
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.transition_probs = {}
        self.charset = set()
        self.trained = False
        self.start_states = []
        self.corpus_size = 0
        self.entropy = 0.0
        
    def train(self, sequences: List[str], verbose: bool = False):
        """
        Train the Markov model on a corpus of sequences.
        
        Args:
            sequences: List of training sequences
            verbose: Print training progress
        """
        if verbose:
            cprint(f"🔄 Training Markov Chain (Order: {self.order})...", "cyan")
        
        self.corpus_size = len(sequences)
        
        # Build character set
        for seq in sequences:
            self.charset.update(seq)
        
        if verbose:
            print(f"   📊 Corpus size: {self.corpus_size:,} sequences")
            print(f"   🔤 Character set: {len(self.charset)} unique chars")
        
        # Count transitions
        for seq in sequences:
            # Add start marker
            padded = '^' * self.order + seq + '$'
            
            # Record starting states
            if len(seq) >= self.order:
                self.start_states.append(seq[:self.order])
            
            # Count all n-grams
            for i in range(len(padded) - self.order):
                state = padded[i:i+self.order]
                next_char = padded[i+self.order]
                self.transitions[state][next_char] += 1
        
        # Calculate probabilities with smoothing
        self._calculate_probabilities()
        
        # Calculate entropy
        self._calculate_entropy()
        
        self.trained = True
        
        if verbose:
            cprint(f"   ✅ Training completed!", "green")
            print(f"   📈 Model entropy: {self.entropy:.3f} bits")
    
    def _calculate_probabilities(self):
        """Calculate transition probabilities with Laplace smoothing."""
        vocab_size = len(self.charset) + 2  # +2 for start/end markers
        
        for state, next_chars in self.transitions.items():
            total_count = sum(next_chars.values())
            smoothed_total = total_count + (vocab_size * self.smoothing)
            
            self.transition_probs[state] = {}
            for char in self.charset | {'^', '$'}:
                count = next_chars.get(char, 0)
                prob = (count + self.smoothing) / smoothed_total
                self.transition_probs[state][char] = prob
    
    def _calculate_entropy(self):
        """Calculate Shannon entropy of the model."""
        if not self.transition_probs:
            return
        
        total_entropy = 0.0
        num_states = 0
        
        for state, probs in self.transition_probs.items():
            state_entropy = 0.0
            for prob in probs.values():
                if prob > 0:
                    state_entropy -= prob * (HAS_NUMPY and np.log2(prob) or 
                                           (prob and (lambda x: __import__('math').log2(x))(prob) or 0))
            total_entropy += state_entropy
            num_states += 1
        
        self.entropy = total_entropy / num_states if num_states > 0 else 0.0
    
    def generate_random(self, length: int, start_state: Optional[str] = None) -> Tuple[str, float]:
        """
        Generate a sequence using random walk.
        
        Args:
            length: Desired sequence length
            start_state: Starting state (random if None)
        
        Returns:
            Tuple of (generated sequence, probability)
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Initialize state
        if start_state is None and self.start_states:
            state = random.choice(self.start_states)
        elif start_state:
            state = start_state[:self.order]
        else:
            state = '^' * self.order
        
        sequence = state.replace('^', '')
        total_prob = 1.0
        
        # Generate sequence
        for _ in range(length - len(sequence)):
            if state not in self.transition_probs:
                # Fallback to random character
                next_char = random.choice(list(self.charset))
                prob = 1.0 / len(self.charset)
            else:
                probs = self.transition_probs[state]
                # Remove end marker for non-terminal generation
                valid_probs = {k: v for k, v in probs.items() if k != '$'}
                
                if not valid_probs:
                    break
                
                # Weighted random selection
                chars = list(valid_probs.keys())
                weights = list(valid_probs.values())
                
                if HAS_NUMPY:
                    weights = np.array(weights)
                    weights /= weights.sum()
                    next_char = np.random.choice(chars, p=weights)
                else:
                    # Manual weighted choice
                    total = sum(weights)
                    weights = [w/total for w in weights]
                    rand = random.random()
                    cumsum = 0
                    next_char = chars[0]
                    for char, weight in zip(chars, weights):
                        cumsum += weight
                        if rand <= cumsum:
                            next_char = char
                            break
                
                prob = probs.get(next_char, self.smoothing)
            
            sequence += next_char
            total_prob *= prob
            
            # Update state
            state = (state + next_char)[-self.order:]
        
        return sequence, total_prob
    
    def generate_greedy(self, length: int, start_state: Optional[str] = None) -> Tuple[str, float]:
        """
        Generate sequence by always selecting most probable next character.
        
        Args:
            length: Desired sequence length
            start_state: Starting state
        
        Returns:
            Tuple of (sequence, probability)
        """
        if not self.trained:
            raise ValueError("Model not trained.")
        
        if start_state is None and self.start_states:
            state = random.choice(self.start_states)
        elif start_state:
            state = start_state[:self.order]
        else:
            state = '^' * self.order
        
        sequence = state.replace('^', '')
        total_prob = 1.0
        
        for _ in range(length - len(sequence)):
            if state not in self.transition_probs:
                next_char = random.choice(list(self.charset))
                prob = 1.0 / len(self.charset)
            else:
                probs = self.transition_probs[state]
                valid_probs = {k: v for k, v in probs.items() if k != '$'}
                
                if not valid_probs:
                    break
                
                # Select most probable
                next_char = max(valid_probs, key=valid_probs.get)
                prob = valid_probs[next_char]
            
            sequence += next_char
            total_prob *= prob
            state = (state + next_char)[-self.order:]
        
        return sequence, total_prob
    
    def generate_beam_search(self, length: int, beam_width: int = 5, 
                            num_sequences: int = 10) -> List[Tuple[str, float]]:
        """
        Generate sequences using beam search for top-k most probable.
        
        Args:
            length: Target sequence length
            beam_width: Number of candidates to maintain
            num_sequences: Number of final sequences to return
        
        Returns:
            List of (sequence, probability) tuples
        """
        if not self.trained:
            raise ValueError("Model not trained.")
        
        # Initialize beam with start states
        if self.start_states:
            beam = [(random.choice(self.start_states), 1.0, 
                    random.choice(self.start_states)[-self.order:]) 
                   for _ in range(min(beam_width, len(self.start_states)))]
        else:
            initial_state = '^' * self.order
            beam = [('', 1.0, initial_state)]
        
        # Beam search
        for pos in range(length):
            candidates = []
            
            for seq, prob, state in beam:
                if len(seq) >= length:
                    candidates.append((seq, prob, state))
                    continue
                
                if state not in self.transition_probs:
                    # Expand with random characters
                    for char in list(self.charset)[:beam_width]:
                        new_seq = seq + char
                        new_prob = prob * (1.0 / len(self.charset))
                        new_state = (state + char)[-self.order:]
                        candidates.append((new_seq, new_prob, new_state))
                else:
                    probs = self.transition_probs[state]
                    valid_probs = {k: v for k, v in probs.items() if k != '$'}
                    
                    # Get top-k next characters
                    sorted_probs = sorted(valid_probs.items(), 
                                        key=lambda x: x[1], reverse=True)
                    
                    for char, char_prob in sorted_probs[:beam_width]:
                        new_seq = seq + char
                        new_prob = prob * char_prob
                        new_state = (state + char)[-self.order:]
                        candidates.append((new_seq, new_prob, new_state))
            
            # Select top beam_width candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[:beam_width]
            
            # Early termination if all sequences at target length
            if all(len(seq) >= length for seq, _, _ in beam):
                break
        
        # Return top num_sequences
        results = [(seq, prob) for seq, prob, _ in beam if len(seq) >= length]
        return results[:num_sequences]
    
    def save_model(self, filepath: str):
        """Save trained model to file."""
        model_data = {
            'order': self.order,
            'smoothing': self.smoothing,
            'transitions': dict(self.transitions),
            'transition_probs': dict(self.transition_probs),
            'charset': list(self.charset),
            'start_states': self.start_states,
            'corpus_size': self.corpus_size,
            'entropy': self.entropy,
            'trained': self.trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        cprint(f"✅ Model saved to: {filepath}", "green")
    
    def load_model(self, filepath: str):
        """Load trained model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.order = model_data['order']
        self.smoothing = model_data['smoothing']
        self.transitions = defaultdict(lambda: defaultdict(int), model_data['transitions'])
        self.transition_probs = model_data['transition_probs']
        self.charset = set(model_data['charset'])
        self.start_states = model_data['start_states']
        self.corpus_size = model_data['corpus_size']
        self.entropy = model_data['entropy']
        self.trained = model_data['trained']
        
        cprint(f"✅ Model loaded from: {filepath}", "green")


class SequenceGenerator:
    """
    Manages sequence generation with various strategies.
    """
    
    def __init__(self, model: MarkovChain, mode: str = 'random'):
        """
        Initialize generator.
        
        Args:
            model: Trained Markov Chain model
            mode: Generation mode ('random', 'greedy', 'beam')
        """
        self.model = model
        self.mode = mode
        self.generated = []
        self.stats = {}
    
    def generate_batch(self, num: int, length: int, min_length: Optional[int] = None,
                      max_length: Optional[int] = None, beam_width: int = 5,
                      threads: int = 1, verbose: bool = False) -> List[Tuple[str, float]]:
        """
        Generate a batch of sequences.
        
        Args:
            num: Number of sequences to generate
            length: Target length
            min_length: Minimum length filter
            max_length: Maximum length filter
            beam_width: Beam width for beam search
            threads: Number of threads
            verbose: Show progress
        
        Returns:
            List of (sequence, probability) tuples
        """
        if verbose:
            cprint(f"🎲 Generating {num} sequences (mode: {self.mode})...", "cyan")
        
        results = []
        
        if self.mode == 'beam':
            # Beam search generates multiple at once
            num_batches = (num + beam_width - 1) // beam_width
            for _ in range(num_batches):
                batch = self.model.generate_beam_search(length, beam_width, beam_width)
                results.extend(batch)
                if len(results) >= num:
                    break
        else:
            # Use threading for random/greedy
            if threads > 1:
                with ThreadPoolExecutor(max_workers=threads) as executor:
                    if self.mode == 'greedy':
                        futures = [executor.submit(self.model.generate_greedy, length) 
                                 for _ in range(num)]
                    else:  # random
                        futures = [executor.submit(self.model.generate_random, length) 
                                 for _ in range(num)]
                    
                    if verbose:
                        print("   ", end="")
                    
                    for i, future in enumerate(as_completed(futures)):
                        try:
                            result = future.result()
                            results.append(result)
                            
                            if verbose and (i + 1) % max(1, num // 20) == 0:
                                progress = (i + 1) / num * 100
                                print(f"\r   {'━' * int(progress // 2.5)}{'░' * (40 - int(progress // 2.5))} {progress:.0f}%", end="")
                        except Exception as e:
                            logging.error(f"Generation error: {e}")
                    
                    if verbose:
                        print()
            else:
                # Single-threaded
                for i in range(num):
                    if self.mode == 'greedy':
                        result = self.model.generate_greedy(length)
                    else:
                        result = self.model.generate_random(length)
                    results.append(result)
                    
                    if verbose and (i + 1) % max(1, num // 20) == 0:
                        progress = (i + 1) / num * 100
                        print(f"\r   {'━' * int(progress // 2.5)}{'░' * (40 - int(progress // 2.5))} {progress:.0f}%", end="")
                
                if verbose:
                    print()
        
        # Filter by length if specified
        if min_length or max_length:
            results = [(seq, prob) for seq, prob in results 
                      if (not min_length or len(seq) >= min_length) and
                         (not max_length or len(seq) <= max_length)]
        
        # Sort by probability (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Take only requested number
        results = results[:num]
        
        self.generated = results
        self._calculate_stats()
        
        if verbose:
            cprint(f"   ✅ Generated {len(results)} sequences!", "green")
        
        return results
    
    def _calculate_stats(self):
        """Calculate statistics for generated sequences."""
        if not self.generated:
            return
        
        sequences = [seq for seq, _ in self.generated]
        probs = [prob for _, prob in self.generated]
        
        self.stats = {
            'total': len(sequences),
            'unique': len(set(sequences)),
            'avg_length': sum(len(s) for s in sequences) / len(sequences),
            'avg_probability': sum(probs) / len(probs) if probs else 0,
            'max_probability': max(probs) if probs else 0,
            'min_probability': min(probs) if probs else 0,
        }


class BruteForceSimulator:
    """
    Simulates brute force testing (LOCAL ONLY, for educational purposes).
    """
    
    def __init__(self, test_hash: Optional[str] = None):
        """
        Initialize simulator.
        
        Args:
            test_hash: Target hash for simulation (SHA256)
        """
        self.test_hash = test_hash
        self.attempts = 0
        self.found = False
        self.matched_sequence = None
    
    def test_sequence(self, sequence: str) -> bool:
        """
        Test a sequence against the target (simulation only).
        
        Args:
            sequence: Sequence to test
        
        Returns:
            True if match found
        """
        self.attempts += 1
        
        if self.test_hash:
            # Hash the sequence and compare
            seq_hash = hashlib.sha256(sequence.encode()).hexdigest()
            if seq_hash == self.test_hash:
                self.found = True
                self.matched_sequence = sequence
                return True
        
        return False
    
    def test_batch(self, sequences: List[Tuple[str, float]], verbose: bool = False) -> Optional[str]:
        """
        Test a batch of sequences.
        
        Args:
            sequences: List of (sequence, probability) tuples
            verbose: Show progress
        
        Returns:
            Matched sequence if found
        """
        if verbose:
            cprint("🔍 Testing sequences (simulation)...", "cyan")
        
        for i, (seq, prob) in enumerate(sequences):
            if self.test_sequence(seq):
                if verbose:
                    cprint(f"   ✅ Match found: {seq} (attempt #{self.attempts})", "green")
                return seq
            
            if verbose and (i + 1) % max(1, len(sequences) // 10) == 0:
                print(f"   Tested: {i + 1}/{len(sequences)}")
        
        if verbose:
            cprint(f"   ❌ No match found in {self.attempts} attempts", "yellow")
        
        return None


class OutputFormatter:
    """
    Handles output formatting and export.
    """
    
    @staticmethod
    def print_sequences(sequences: List[Tuple[str, float]], top_n: int = 10, 
                       verbose: bool = False):
        """Print sequences to console."""
        print()
        cprint("📝 Generated Sequences:", "cyan", attrs=['bold'])
        print()
        
        if verbose:
            # Detailed output
            for i, (seq, prob) in enumerate(sequences[:top_n], 1):
                print(f"{i:3}. {colored(seq, 'yellow')} "
                     f"(prob: {colored(f'{prob:.6e}', 'green')}, "
                     f"len: {len(seq)})")
        else:
            # Simple output
            for i, (seq, prob) in enumerate(sequences[:top_n], 1):
                print(f"{i:3}. {seq} (probability: {prob:.6e})")
        
        if len(sequences) > top_n:
            print(f"\n... and {len(sequences) - top_n} more sequences")
    
    @staticmethod
    def print_statistics(stats: Dict, model: MarkovChain):
        """Print generation statistics."""
        print()
        cprint("📈 Statistics:", "cyan", attrs=['bold'])
        print(f"   Total generated: {stats.get('total', 0):,}")
        print(f"   Unique sequences: {stats.get('unique', 0):,}")
        print(f"   Average length: {stats.get('avg_length', 0):.2f}")
        print(f"   Average probability: {stats.get('avg_probability', 0):.6e}")
        print(f"   Model entropy: {model.entropy:.3f} bits")
        print(f"   Character set size: {len(model.charset)}")
    
    @staticmethod
    def export_csv(sequences: List[Tuple[str, float]], filepath: str):
        """Export sequences to CSV."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("sequence,probability,length\n")
            for seq, prob in sequences:
                f.write(f'"{seq}",{prob},{len(seq)}\n')
        
        cprint(f"✅ Exported to CSV: {filepath}", "green")
    
    @staticmethod
    def export_json(sequences: List[Tuple[str, float]], stats: Dict, 
                   model: MarkovChain, filepath: str):
        """Export sequences and metadata to JSON."""
        data = {
            'model_info': {
                'order': model.order,
                'charset_size': len(model.charset),
                'training_size': model.corpus_size,
                'entropy': model.entropy,
                'smoothing': model.smoothing
            },
            'sequences': [
                {'sequence': seq, 'probability': prob, 'length': len(seq)}
                for seq, prob in sequences
            ],
            'statistics': stats,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        cprint(f"✅ Exported to JSON: {filepath}", "green")
    
    @staticmethod
    def visualize_matrix(model: MarkovChain, output_path: Optional[str] = None):
        """Visualize transition matrix (requires matplotlib)."""
        if not HAS_MATPLOTLIB:
            cprint("⚠️  Matplotlib not available. Skipping visualization.", "yellow")
            return
        
        if not model.trained:
            cprint("⚠️  Model not trained. Cannot visualize.", "yellow")
            return
        
        cprint("📊 Generating transition matrix visualization...", "cyan")
        
        # Sample a subset of states for visualization
        states = list(model.transition_probs.keys())[:min(20, len(model.transition_probs))]
        chars = sorted(list(model.charset))[:min(20, len(model.charset))]
        
        if HAS_NUMPY:
            matrix = np.zeros((len(states), len(chars)))
            
            for i, state in enumerate(states):
                for j, char in enumerate(chars):
                    matrix[i, j] = model.transition_probs.get(state, {}).get(char, 0)
            
            plt.figure(figsize=(12, 10))
            plt.imshow(matrix, cmap='YlOrRd', aspect='auto')
            plt.colorbar(label='Probability')
            plt.xlabel('Next Character')
            plt.ylabel('Current State')
            plt.title(f'Markov Chain Transition Matrix (Order {model.order})')
            plt.xticks(range(len(chars)), chars, rotation=45)
            plt.yticks(range(len(states)), [s.replace('^', '⌃') for s in states])
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=150)
                cprint(f"✅ Visualization saved: {output_path}", "green")
            else:
                plt.show()
            
            plt.close()
        else:
            cprint("⚠️  NumPy required for visualization.", "yellow")


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"markov_brute_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout) if verbose else logging.NullHandler()
        ]
    )
    
    logging.info("="*60)
    logging.info("Markov Chain Brute Force Tool - Session Started")
    logging.info("="*60)


def load_training_data(filepath: str, verbose: bool = False) -> List[str]:
    """
    Load training data from file.
    
    Args:
        filepath: Path to training file
        verbose: Print loading info
    
    Returns:
        List of training sequences
    """
    if verbose:
        cprint(f"📂 Loading training data: {filepath}", "cyan")
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            sequences = [line.strip() for line in f if line.strip()]
        
        if verbose:
            cprint(f"   ✅ Loaded {len(sequences):,} sequences", "green")
        
        logging.info(f"Loaded {len(sequences)} sequences from {filepath}")
        
        return sequences
    
    except FileNotFoundError:
        cprint(f"❌ Error: File not found: {filepath}", "red")
        logging.error(f"File not found: {filepath}")
        sys.exit(1)
    except Exception as e:
        cprint(f"❌ Error loading file: {e}", "red")
        logging.error(f"Error loading file: {e}")
        sys.exit(1)


def detect_charset(sequences: List[str]) -> str:
    """
    Auto-detect character set from sequences.
    
    Returns:
        Character set name ('alpha', 'alphanum', 'full')
    """
    all_chars = set(''.join(sequences))
    
    has_lower = any(c in string.ascii_lowercase for c in all_chars)
    has_upper = any(c in string.ascii_uppercase for c in all_chars)
    has_digit = any(c in string.digits for c in all_chars)
    has_special = any(c in string.punctuation for c in all_chars)
    
    if has_special:
        return 'full'
    elif has_digit:
        return 'alphanum'
    elif has_lower or has_upper:
        return 'alpha'
    else:
        return 'custom'


def get_charset(charset_type: str, custom_chars: Optional[str] = None) -> Set[str]:
    """Get character set based on type."""
    if charset_type == 'alpha':
        return set(string.ascii_letters)
    elif charset_type == 'alphanum':
        return set(string.ascii_letters + string.digits)
    elif charset_type == 'full':
        return set(string.ascii_letters + string.digits + string.punctuation)
    elif charset_type == 'custom' and custom_chars:
        return set(custom_chars)
    else:
        return set(string.ascii_letters + string.digits)


def easy_mode():
    """
    Interactive easy mode for beginners.
    """
    cprint("\n🎮 Welcome to Easy Mode!", "cyan", attrs=['bold'])
    cprint("I'll guide you step-by-step. Let's get started! 🚀\n", "cyan")
    
    # Step 1: Training file
    print(colored("Step 1:", "yellow", attrs=['bold']) + " Do you have a training file?")
    print("   (A text file with passwords, text, or sequences)")
    has_file = input("   Enter Y/N: ").strip().upper()
    
    train_file = None
    if has_file == 'Y':
        train_file = input("   📁 Enter file path: ").strip()
        while not os.path.exists(train_file):
            cprint(f"   ❌ File not found: {train_file}", "red")
            train_file = input("   Try another path (or 'skip' to use default): ").strip()
            if train_file.lower() == 'skip':
                train_file = None
                break
    
    if not train_file:
        cprint("   ℹ️  Using built-in sample data", "cyan")
        # Create sample data
        sample_data = [
            "password123", "password1", "123456", "qwerty123",
            "admin123", "letmein", "welcome123", "monkey123"
        ]
        train_file = "temp_sample.txt"
        with open(train_file, 'w') as f:
            f.write('\n'.join(sample_data))
    
    # Step 2: Order
    print(f"\n{colored('Step 2:', 'yellow', attrs=['bold'])} How complex should the model be?")
    print("   1 = Simple (fast, less accurate)")
    print("   2 = Balanced (recommended) ⭐")
    print("   3 = Complex (slower, more accurate)")
    order_choice = input("   Enter 1, 2, or 3 [2]: ").strip() or "2"
    order = int(order_choice) if order_choice in ['1', '2', '3'] else 2
    
    # Step 3: Length
    print(f"\n{colored('Step 3:', 'yellow', attrs=['bold'])} How long should the sequences be?")
    length_input = input("   Enter length (e.g., 8) [8]: ").strip() or "8"
    length = int(length_input) if length_input.isdigit() else 8
    
    # Step 4: Number
    print(f"\n{colored('Step 4:', 'yellow', attrs=['bold'])} How many sequences to generate?")
    num_input = input("   Enter number (e.g., 100) [100]: ").strip() or "100"
    num = int(num_input) if num_input.isdigit() else 100
    
    # Step 5: Mode
    print(f"\n{colored('Step 5:', 'yellow', attrs=['bold'])} Generation mode:")
    print("   1 = Random (varied results)")
    print("   2 = Smart (most probable) ⭐")
    mode_choice = input("   Enter 1 or 2 [2]: ").strip() or "2"
    mode = 'beam' if mode_choice == '2' else 'random'
    
    # Confirmation
    print(f"\n{colored('Summary:', 'cyan', attrs=['bold'])}")
    print(f"   Training file: {train_file}")
    print(f"   Order: {order}")
    print(f"   Length: {length}")
    print(f"   Number: {num}")
    print(f"   Mode: {mode}")
    
    confirm = input(f"\n{colored('Ready to start?', 'green', attrs=['bold'])} (Y/N) [Y]: ").strip().upper() or "Y"
    
    if confirm != 'Y':
        cprint("\n👋 Okay, exiting. Come back anytime!", "cyan")
        sys.exit(0)
    
    return {
        'train_file': train_file,
        'order': order,
        'length': length,
        'num': num,
        'mode': mode,
        'verbose': True
    }


def print_banner():
    """Print tool banner."""
    banner = """
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║  🔐 MARKOV CHAIN BRUTE FORCE TOOL v1.0                         ║
║                                                                ║
║  Advanced Statistical Sequence Generation & Analysis           ║
║  Educational & Research Use Only                               ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
"""
    cprint(banner, "cyan", attrs=['bold'])
    
    cprint("⚠️  WARNING: EDUCATIONAL USE ONLY", "yellow", attrs=['bold'])
    cprint("    This tool is for learning, research, and ethical testing.", "yellow")
    cprint("    Unauthorized access attempts are ILLEGAL.\n", "yellow")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Markov Chain Brute Force Tool - Educational Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Easy mode (recommended for beginners)
  python markov_brute.py --easy

  # Train and generate
  python markov_brute.py --train-file passwords.txt --order 2 --length 8 --num 100

  # Use saved model
  python markov_brute.py --load-model model.pkl --length 10 --num 500

  # Beam search for best results
  python markov_brute.py --train-file data.txt --mode beam --beam-width 5

  # Export to CSV
  python markov_brute.py --train-file data.txt --output results.csv --format csv

Note: Always use responsibly and ethically! 🎓
        """
    )
    
    # Data input
    parser.add_argument('--train-file', type=str,
                       help='Path to training data file')
    parser.add_argument('--load-model', type=str,
                       help='Load pre-trained model from file')
    parser.add_argument('--save-model', type=str,
                       help='Save trained model to file')
    
    # Model parameters
    parser.add_argument('--order', type=int, default=2,
                       help='Markov chain order (1-10, default: 2)')
    parser.add_argument('--smoothing', type=float, default=0.01,
                       help='Laplace smoothing factor (default: 0.01)')
    
    # Generation parameters
    parser.add_argument('--length', type=int, default=8,
                       help='Target sequence length (default: 8)')
    parser.add_argument('--min-length', type=int,
                       help='Minimum sequence length')
    parser.add_argument('--max-length', type=int,
                       help='Maximum sequence length')
    parser.add_argument('--num', type=int, default=100,
                       help='Number of sequences to generate (default: 100)')
    
    # Generation mode
    parser.add_argument('--mode', type=str, default='random',
                       choices=['random', 'greedy', 'beam'],
                       help='Generation mode (default: random)')
    parser.add_argument('--beam-width', type=int, default=5,
                       help='Beam width for beam search (default: 5)')
    
    # Character set
    parser.add_argument('--charset', type=str, default='auto',
                       choices=['auto', 'alpha', 'alphanum', 'full', 'custom'],
                       help='Character set type (default: auto)')
    parser.add_argument('--custom-chars', type=str,
                       help='Custom character set')
    
    # Output options
    parser.add_argument('--output', type=str,
                       help='Output file path')
    parser.add_argument('--format', type=str, default='text',
                       choices=['text', 'csv', 'json'],
                       help='Output format (default: text)')
    parser.add_argument('--top-n', type=int, default=10,
                       help='Number of top sequences to display (default: 10)')
    
    # Testing (simulation only)
    parser.add_argument('--test-hash', type=str,
                       help='Test hash for simulation (SHA256)')
    
    # Performance
    parser.add_argument('--threads', type=int, default=4,
                       help='Number of threads for generation (default: 4)')
    
    # UI options
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output with details')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize transition matrix (requires matplotlib)')
    parser.add_argument('--easy', action='store_true',
                       help='Easy mode with guided setup')
    
    # Ethical confirmation
    parser.add_argument('--educational-only', action='store_true',
                       help='Confirm educational use (required)')
    
    # Testing
    parser.add_argument('--run-tests', action='store_true',
                       help='Run unit tests')
    
    return parser.parse_args()


def run_unit_tests():
    """Run built-in unit tests."""
    cprint("\n🧪 Running Unit Tests...\n", "cyan", attrs=['bold'])
    
    passed = 0
    failed = 0
    
    # Test 1: Model initialization
    try:
        model = MarkovChain(order=2)
        assert model.order == 2
        assert model.smoothing == 0.01
        cprint("✅ Test 1: Model initialization", "green")
        passed += 1
    except Exception as e:
        cprint(f"❌ Test 1 failed: {e}", "red")
        failed += 1
    
    # Test 2: Training
    try:
        model = MarkovChain(order=2)
        sequences = ["password", "pass123", "admin123"]
        model.train(sequences)
        assert model.trained
        assert len(model.charset) > 0
        cprint("✅ Test 2: Model training", "green")
        passed += 1
    except Exception as e:
        cprint(f"❌ Test 2 failed: {e}", "red")
        failed += 1
    
    # Test 3: Random generation
    try:
        model = MarkovChain(order=2)
        sequences = ["password", "pass123", "admin123"]
        model.train(sequences)
        seq, prob = model.generate_random(8)
        assert len(seq) <= 8
        assert 0 < prob <= 1
        cprint("✅ Test 3: Random generation", "green")
        passed += 1
    except Exception as e:
        cprint(f"❌ Test 3 failed: {e}", "red")
        failed += 1
    
    # Test 4: Greedy generation
    try:
        model = MarkovChain(order=2)
        sequences = ["password", "pass123", "admin123"]
        model.train(sequences)
        seq, prob = model.generate_greedy(8)
        assert len(seq) <= 8
        cprint("✅ Test 4: Greedy generation", "green")
        passed += 1
    except Exception as e:
        cprint(f"❌ Test 4 failed: {e}", "red")
        failed += 1
    
    # Test 5: Beam search
    try:
        model = MarkovChain(order=2)
        sequences = ["password", "pass123", "admin123"]
        model.train(sequences)
        results = model.generate_beam_search(8, beam_width=3, num_sequences=5)
        assert len(results) <= 5
        cprint("✅ Test 5: Beam search", "green")
        passed += 1
    except Exception as e:
        cprint(f"❌ Test 5 failed: {e}", "red")
        failed += 1
    
    # Test 6: Model save/load
    try:
        model1 = MarkovChain(order=2)
        sequences = ["password", "pass123"]
        model1.train(sequences)
        model1.save_model("test_model.pkl")
        
        model2 = MarkovChain()
        model2.load_model("test_model.pkl")
        assert model2.trained
        assert model2.order == 2
        
        os.remove("test_model.pkl")
        cprint("✅ Test 6: Model persistence", "green")
        passed += 1
    except Exception as e:
        cprint(f"❌ Test 6 failed: {e}", "red")
        failed += 1
    
    # Test 7: Batch generation
    try:
        model = MarkovChain(order=2)
        sequences = ["password", "pass123", "admin123"]
        model.train(sequences)
        generator = SequenceGenerator(model, mode='random')
        results = generator.generate_batch(10, length=8)
        assert len(results) <= 10
        cprint("✅ Test 7: Batch generation", "green")
        passed += 1
    except Exception as e:
        cprint(f"❌ Test 7 failed: {e}", "red")
        failed += 1
    
    # Test 8: Statistics calculation
    try:
        model = MarkovChain(order=2)
        sequences = ["password", "pass123", "admin123"]
        model.train(sequences)
        generator = SequenceGenerator(model)
        generator.generate_batch(10, length=8)
        assert 'total' in generator.stats
        assert 'unique' in generator.stats
        cprint("✅ Test 8: Statistics calculation", "green")
        passed += 1
    except Exception as e:
        cprint(f"❌ Test 8 failed: {e}", "red")
        failed += 1
    
    # Summary
    print()
    cprint("="*60, "cyan")
    cprint(f"Test Results: {passed} passed, {failed} failed", 
           "green" if failed == 0 else "yellow", attrs=['bold'])
    cprint("="*60, "cyan")
    
    return failed == 0


def main():
    """Main entry point."""
    
    # Print banner
    print_banner()
    
    # Parse arguments
    args = parse_arguments()
    
    # Run tests if requested
    if args.run_tests:
        success = run_unit_tests()
        sys.exit(0 if success else 1)
    
    # Easy mode
    if args.easy:
        easy_params = easy_mode()
        # Merge with args
        args.train_file = easy_params['train_file']
        args.order = easy_params['order']
        args.length = easy_params['length']
        args.num = easy_params['num']
        args.mode = easy_params['mode']
        args.verbose = easy_params['verbose']
        args.educational_only = True
    
    # Ethical confirmation
    if not args.educational_only and not args.easy:
        cprint("\n⚠️  ETHICAL USE CONFIRMATION REQUIRED", "yellow", attrs=['bold'])
        print("This tool is for educational and research purposes only.")
        confirm = input("Do you confirm ethical use only? (yes/no): ").strip().lower()
        
        if confirm != 'yes':
            cprint("\n❌ Ethical confirmation required. Exiting.", "red")
            sys.exit(1)
        
        args.educational_only = True
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Start timing
    start_time = time.time()
    
    try:
        # Initialize model
        if args.load_model:
            cprint(f"\n📦 Loading model from: {args.load_model}", "cyan")
            model = MarkovChain()
            model.load_model(args.load_model)
        else:
            # Create new model
            model = MarkovChain(order=args.order, smoothing=args.smoothing)
            
            # Load training data
            if args.train_file:
                sequences = load_training_data(args.train_file, args.verbose)
            else:
                cprint("\n⚠️  No training file provided. Using sample data.", "yellow")
                sequences = [
                    "password123", "password1", "123456789", "qwerty123",
                    "admin123", "letmein", "welcome123", "monkey123",
                    "dragon123", "master123"
                ]
            
            # Train model
            model.train(sequences, verbose=args.verbose)
            
            # Save model if requested
            if args.save_model:
                model.save_model(args.save_model)
        
        # Initialize generator
        generator = SequenceGenerator(model, mode=args.mode)
        
        # Generate sequences
        results = generator.generate_batch(
            num=args.num,
            length=args.length,
            min_length=args.min_length,
            max_length=args.max_length,
            beam_width=args.beam_width,
            threads=args.threads,
            verbose=args.verbose
        )
        
        # Test simulation (if hash provided)
        if args.test_hash:
            simulator = BruteForceSimulator(test_hash=args.test_hash)
            match = simulator.test_batch(results, verbose=args.verbose)
            
            if match:
                cprint(f"\n🎉 SUCCESS! Match found: {match}", "green", attrs=['bold'])
            else:
                cprint(f"\n❌ No match found in {args.num} attempts.", "yellow")
        
        # Output results
        if args.format == 'text':
            OutputFormatter.print_sequences(results, top_n=args.top_n, verbose=args.verbose)
            OutputFormatter.print_statistics(generator.stats, model)
        
        # Export if requested
        if args.output:
            if args.format == 'csv':
                OutputFormatter.export_csv(results, args.output)
            elif args.format == 'json':
                OutputFormatter.export_json(results, generator.stats, model, args.output)
            else:
                # Default to text file
                with open(args.output, 'w', encoding='utf-8') as f:
                    for seq, prob in results:
                        f.write(f"{seq}\n")
                cprint(f"✅ Exported to: {args.output}", "green")
        
        # Visualization
        if args.visualize:
            viz_path = args.output.replace('.csv', '.png').replace('.json', '.png') if args.output else None
            OutputFormatter.visualize_matrix(model, output_path=viz_path)
        
        # End timing
        elapsed_time = time.time() - start_time
        
        print()
        cprint("="*60, "cyan")
        cprint(f"⏱️  Total execution time: {elapsed_time:.2f}s", "cyan")
        cprint("✅ Done! Happy learning! 🎉", "green", attrs=['bold'])
        cprint("="*60, "cyan")
        
        # Logging
        logging.info(f"Session completed successfully in {elapsed_time:.2f}s")
        logging.info(f"Generated {len(results)} sequences")
        logging.info("="*60)
    
    except KeyboardInterrupt:
        cprint("\n\n⚠️  Interrupted by user. Exiting gracefully...", "yellow")
        logging.warning("Session interrupted by user")
        sys.exit(0)
    
    except Exception as e:
        cprint(f"\n❌ Error: {str(e)}", "red", attrs=['bold'])
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
        
        if args.verbose:
            import traceback
            print()
            traceback.print_exc()
        
        sys.exit(1)


if __name__ == "__main__":
    main()

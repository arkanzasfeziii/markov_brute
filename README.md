# Markov Chain Sequence Generator - Educational Edition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Educational Tool](https://img.shields.io/badge/Purpose-Educational-red)](#disclaimer)

An advanced educational tool demonstrating **Markov Chain** models for sequence generation and probabilistic analysis.  
This project is designed for learning concepts in probability, statistical modeling, machine learning, and sequence prediction.

**Strictly for educational and authorized research purposes only.**

## Disclaimer

> **WARNING**: This tool is intended **solely for educational purposes**, ethical security research, and authorized testing.  
> Any use for unauthorized access, illegal activities, or malicious intent is strictly prohibited and may violate laws.  
> The authors and contributors assume no liability for misuse.

## Features

- Variable-order Markov chains (order 1–10)
- Multiple generation strategies:
  - Random walk
  - Greedy (most probable path)
  - Beam search (top-k candidates)
- Laplace smoothing for robust probability estimation
- Shannon entropy calculation
- Multi-threaded batch generation
- Interactive "Easy Mode" for beginners
- Export results to Text, CSV, or JSON
- Transition matrix visualization (requires Matplotlib)
- Model save/load functionality
- Built-in unit tests
- Comprehensive logging

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/markov-sequence-generator.git
   cd markov-sequence-generator
   ```
2. (Optional) Install dependencies for full features:
   ```bash
   pip install -r requirements.txt
   ```
   Basic functionality works without extra packages. NumPy, termcolor, and Matplotlib are optional but recommended.
   Usage
Run the tool directly:
```bash
python markov_brute.py --help
```
Easy Mode (Recommended for beginners)
```bash
python markov_brute.py --easy
```
Common Examples

1. Train on a password list and generate sequences:Bash
   ```bash
   python markov_brute.py --train-file passwords.txt --order 3 --length 10 --num 1000 --mode beam --beam-width 10 --verbose
   ```
2. Load a saved model and generate:
   ```bash
   python markov_brute.py --load-model my_model.pkl --length 12 --num 500 --mode greedy
   ```
3. Export results:Bash
   ```bash
   python markov_brute.py --train-file data.txt --num 1000 --output results.csv --format csv
   ```
4. Visualize transition matrix:
   ```bash
   python markov_brute.py --train-file data.txt --visualize
   ```
   See --help for all available options.
   
Sample Training Data
A small sample dataset is included in the script for quick testing. For real use, provide your own text file with one sequence per line.

Contributing
Contributions are welcome! Feel free to:

Open issues for bugs or feature requests
Submit pull requests with improvements
Improve documentation

Please follow standard GitHub flow and keep the educational focus.

Acknowledgments

Built for educational demonstration of Markov chain concepts
Inspired by statistical modeling techniques in security research


Remember: Use responsibly and ethically. Knowledge is power — use it wisely.

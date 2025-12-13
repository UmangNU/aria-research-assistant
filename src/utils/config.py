# src/utils/config.py
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    
    # Model Settings
    LLM_MODEL = "gpt-4-turbo-preview"
    EMBEDDING_MODEL = "text-embedding-3-large"
    
    # RL Hyperparameters
    DQN_LEARNING_RATE = 0.001
    DQN_GAMMA = 0.99
    DQN_EPSILON_START = 1.0
    DQN_EPSILON_END = 0.01
    DQN_EPSILON_DECAY = 0.995
    BATCH_SIZE = 32
    REPLAY_BUFFER_SIZE = 10000
    
    # Bandit Settings
    BANDIT_ALPHA_PRIOR = 1.0
    BANDIT_BETA_PRIOR = 1.0
    
    # System Settings
    MAX_TOKENS = 4000
    TEMPERATURE = 0.7
    NUM_AGENTS = 9
    
    # Paths
    DATA_DIR = "data"
    MODELS_DIR = "models"
    RESULTS_DIR = "experiments/results"

config = Config()
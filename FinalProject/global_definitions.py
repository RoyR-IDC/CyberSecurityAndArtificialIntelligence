import os

# Change according to current user
USER = 'harel'

if USER == 'roy':
    PROJECT_PATH = r'/Users/royrubin/PycharmProjects/CyberSecurityAndArtificialIntelligence/FinalProject/'
elif USER == 'harel':
    PROJECT_PATH = r'/Users/harel/Documents/IDC/CyberAI/EX2/CyberSecurityAndArtificialIntelligence/FinalProject/'

# relative paths
DATA_DIR_PATH = os.path.join(PROJECT_PATH, 'Data/')
WORD_BANK_DIR_PATH = os.path.join(PROJECT_PATH, 'WordBank/')

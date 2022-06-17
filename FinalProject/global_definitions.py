import os

# Change according to current user
USER = 'roy'

if USER == 'roy':
    PROJECT_PATH = r'/Users/royrubin/PycharmProjects/CyberSecurityAndArtificialIntelligence/FinalProject/'
else:
    PROJECT_PATH = r''

# relative paths
DATA_DIR_PATH = os.path.join(PROJECT_PATH, 'CleanedFormattedData/')
WORD_BANK_DIR_PATH = os.path.join(PROJECT_PATH, 'WordBank/')

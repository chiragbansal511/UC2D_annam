set -e
KAGGLE_COMPETITION="soil-classification"

if ! command -v kaggle &> /dev/null; then
    echo "Kaggle CLI not found. Install it using: pip install kaggle"
    exit 1
fi

if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "Kaggle API token not found at ~/.kaggle/kaggle.json"
    exit 1
fi

mkdir -p data/raw
mkdir -p data/processed

kaggle competitions download -c "$KAGGLE_COMPETITION" -p data/raw --force
unzip -o data/raw/*.zip -d data/processed
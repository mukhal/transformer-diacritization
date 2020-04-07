# transformer-diacritization
Arabic diacritization with causal self-attention Transfromers

## Steps to reproduce

### Clone the repo
`git clone https://github.com/mohammadKhalifa/transformer-diacritization.git`

### Download Tashkeela 
```
cd transformer-diacritization/
wget https://sourceforge.net/projects/tashkeela/files/latest/download

mv download tashkeela.zip
mkdir data
unzip -q tashkeela.zip -d data/
mv data/Tashkeela-arabic-diacritized-text-utf8-0.3/ data/tashkeela 
rm data/tashkeela/doc/ -r 
```

### Preprocessing
```
python scripts/preprocess_data.py --corpus data/tashkeela/texts.txt/
python scripts/prepare_diacritization_dataset.py --corpus-dir data/preprocessed --outdir data/tashkeela/bin

```




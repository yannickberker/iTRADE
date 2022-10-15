#!/bin/bash
shopt -s globstar

# install required dev tools
if false; then
    pip install -r requirements-dev.txt
fi

cd "$(dirname "$0")/.." || exit
echo "$PWD"
sleep 5

# record requirements
(grep -v "==" requirements.txt && pip freeze --exclude-editable) | sponge requirements.txt

# format text files
find . -maxdepth 1 -type f -exec dos2unix --quiet {} +
find itrade -type f -exec dos2unix --quiet {} +
find util -type f -exec dos2unix --quiet {} +

# format code
isort ./**/*.py
black . --quiet

# lint code
cffconvert --validate
pylint ./**/*.py
bandit . --recursive --quiet --configfile pyproject.toml
flake8
mypy .
pycodestyle
pydocstyle
pylama

# test code
pytest --quiet

# check datasets
[ -d "itrade-data" ] && find itrade-data/Images -type f | grep -v -P "^itrade-data/Images/(.*_(CE|V\d_DS\d))/\1_P(\d)_[A-O]\d{2}_(midz|proj)_224x224.tif$"
[ -d "itrade-data" ] && find itrade-data/Metabolic -type f | grep -v -P "^itrade-data/Metabolic/(.*_V\d_DS\d)/\1_P(\d)_H104-03N\2[-A-D]\d{2}.txt$"

# check for sensitive information and to-dos
grep --color -I -r --exclude-dir={.*,*.egg-info,*-results} -i -P 'I[0-9]{3}_[0-9]{3}|http://(?!(\(|localhost))|[_]dEvEl|[T]ODO|[x]xxxx'
find . | grep --color "\bbug"

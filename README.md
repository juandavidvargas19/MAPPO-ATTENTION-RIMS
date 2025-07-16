## Installation

Clone the repository and install the requirements:

```bash
#Download conda installer here: https://www.anaconda.com/download/success
bash $anacoda_file
conda create --name RIMS python=3.11.7
conda activate RIMS
git clone https://github.com/juandavidvargas19/MAPS_PROJECT.git
cd MAPS_PROJECT
pip install -r requirements.txt
pip install "ray[cpp]" 
git clone -b main https://github.com/deepmind/meltingpot
cd meltingpot
pip install --editable .[dev]
```

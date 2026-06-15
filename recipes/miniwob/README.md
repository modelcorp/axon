# MiniWoB

Web-interaction agent on the BrowserGym MiniWoB subset.

```bash
# One-time setup
cd $HOME
git clone https://github.com/Farama-Foundation/miniwob-plusplus.git
cd $HOME/miniwob-plusplus
git checkout -f 7fd85d71a4b60325c6585396ec4f48377d049838
export MINIWOB_URL="file://$HOME/miniwob-plusplus/miniwob/html/miniwob/"

pip install browsergym
playwright install chromium
```

## Run

```bash
# build the dataset, then train (paths resolve via AXON_DIR, and MINIWOB_URL must be set as above)
cd recipes/miniwob
python data.py
./train_fsdp.sh
```

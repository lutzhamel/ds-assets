# ds-assets
This repository contains the assets for the data science notebooks in the ds-notes repository.

The content in the assets folder is made accessible in notebooks through the following
preamble,
```python
###### Config #####
import sys, os, platform
if os.path.isdir("ds-assets"):
  !cd ds-assets && git pull
else:
  !git clone https://github.com/lutzhamel/ds-assets.git
colab = True if 'google.colab' in os.sys.modules else False
system = platform.system() # "Windows", "Linux", "Darwin"
home = "ds-assets/assets/"
sys.path.append(home) 
```


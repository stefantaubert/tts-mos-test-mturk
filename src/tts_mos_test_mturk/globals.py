from pathlib import Path

import matplotlib as mpl
from matplotlib.font_manager import FontProperties

LATEX_TEXT_WIDTH = 5.90666



# https://www.ctan.org/tex-archive/fonts/cm/ps-type1/bakoma/otf/

# cmmi10 is for italic text

# cmsy10 is for \times
fpath = Path(mpl.get_data_path(), "fonts/ttf/cmsy10.ttf")
# cmr10 is for normal text
# r=regular
fpath = Path("cmsl12.otf")
fpath_cmr10 = Path(mpl.get_data_path(), "fonts/ttf/cmr10.ttf")
fpath = Path("cmr12.ttf")
fpath_nimbus = Path("NimbusSanL-Regu.ttf")
fpath = None
fpath_cmr12 = Path("cmr12.otf")
# print(fpath)
# /home/mi/.local/share/virtualenvs/python-tnykqW7H/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf/cmr10.ttf
LATEX_FONT = fpath_cmr10
LATEX_FONT_SIZE = 10

LEGEND_FONT = FontProperties(
  family="serif",
  style="normal",
  variant="normal",
  weight="normal",
  stretch="normal",
  size=8,
  fname=LATEX_FONT,
)

{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pyxtal as pyl\n",
    "\n",
    "from pyxtal.symmetry import Group, site_symmetry\n",
    "from visualisation_stat import *\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "path_json = '/home/jyin/workspace/gene_data/direct_space_data.json'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'stat_R2_D2_from_json' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[3], line 1\u001b[0m\n\u001b[0;32m----> 1\u001b[0m stat \u001b[38;5;241m=\u001b[39m \u001b[43mstat_R2_D2_from_json\u001b[49m(path_json)\n",
      "\u001b[0;31mNameError\u001b[0m: name 'stat_R2_D2_from_json' is not defined"
     ]
    }
   ],
   "source": [
    "stat = stat_R2_D2_from_json(path_json)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "aiida",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

# A deep learning based graph-transformer for whole slide image classification

# Introduction

# Usage
## 1. Graph Construction
### (a) Patch Tiling
```
python src/tile_WSI.py -s 512 -e 0 -j 32 -B 50 -M 20 -o <full_patch_to_output_folder> "full_path_to_input_slides/*/*.svs"
```
Mandatory parameters:
<li>-s is tile_size: 512 (512x512 pixel tiles)</li>
<li>-e is overlap, 0 (no overlap between adjacent tiles). Important: the overlap is defined as "the number of extra pixels to add to each interior edge of a tile". Which means that the final tile size is s + 2.e. So to get a 512px tile with a 50% overlap, you need to set s to 256 and e to 128. Also, tile from the edges of the slide will be smaller (since up to two sides have no "interior" edge)</li>
<li>-j is number of threads: 32</li>
<li>-B is Max Percentage of Background allowed: 50% (tiles removed if background percentage above this value)</li>
<li>-o is the path were the output images must be saved</li>
<li>-M set to -1 by default to tile the image at all magnifications. Set it to the value of the desired magnification to tile only at that magnification and save space</li>

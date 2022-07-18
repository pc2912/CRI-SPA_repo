# CRI-SPA_repo
Handle CRI-SPA data


### Notebooks <br/>
**Plate_Shuffling.ipynb** <br/>
Instruction to shuffle colonies when scaling from 384 to 1536 with a pinning robot. <br/>
**Read_Plates.ipynb** <br/>
Extract and process data from images. <br/>
**Screen Comparisons.ipynb** <br/>
Assess screens reproducibility. <br/>
**Networks.ipynb**
Go Enrichment Analysis in clusters of genes. <br/>
Draws networks of enriched terms. <br/>

### Supporting Functions <br/>
Colony_Analysis.py, Plate_Shuffling.py <br/>

### Supporting Files <br/>
**Individual_Colony_Data.csv:** Data for individual colonies, extracted with Read_Plates.ipynb  <br/>
**GA[...].csv:** stands for Gene Analysis, data extracted and processed from screen several screen repeats. <br/>
**GA1_2_4.csv:** data extracted from screen several screen repeats, pooled and processed after pooling. <br/>
**gene2go, go.obo:** For Go Enrichment Analysis. <br/>
**Plate_Map.txt:** Stores gene positions in our KO library. <br/>

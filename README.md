# CRI-SPA_repo
Handle CRI-SPA data


### Notebooks <br/>
**Plate_Shuffling.ipynb** <br/>
Instruction to shuffle colonies when scaling from 384 to 1536 with a pinning robot. <br/>
**Read_Plates.ipynb** <br/>
Extract and process data from images. <br/>
**Screen Comparisons.ipynb** <br/>
Assess screens reproducibility. <br/>
**Networks.ipynb** <br/>
Go Enrichment Analysis in clusters of genes. <br/>
Draws networks of enriched terms. <br/>
**Liquid Confirmation.ipynb** <br/>
Plot the fluorescence of hit strains grown in liquid SC.<br/>


### Supporting Functions <br/>
Colony_Analysis.py, Plate_Shuffling.py <br/>

### Supporting Files <br/>
**QD1.csv:** (Quadruplicate) Size and Yellowness Data. Data extracted from pictures in screen_sample/Screen1_9.9.21/ with Read_Plates.ipynb <br/>
**GA1.csv:** (Gene Analysis). Data stores the mean and std of colonies for each gene, data extracted from pictures in screen_sample/Screen1_9.9.21/ with Read_Plates.ipynb  <br/>
**GA1_2_4.csv:** data extracted from several screen repeats, pooled and processed after pooling. <br/>

### Data <br/>
**screen_sample/Screen1_9.9.21/** Images (24H) for screen1<br/>
**BTX_chess/H48.png** Image used to compare BY_Ref and BY_Btx in Fig 3A<br/>
**Liquid_Data/** Fluorescent Data for Hit Strains in liquid SC <br/>
**Liquid_Data/BTX_stability.23.4.10** Fluorescent Data Confirming BTX stability for a week (Fig. S7) <br/>

**gene2go, go.obo:** For Go Enrichment Analysis. <br/>
**Plate_Map.txt:** Stores gene positions in our KO library. <br/>

